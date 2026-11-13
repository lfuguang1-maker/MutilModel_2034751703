"""
MultiModalPlotter: 多模态统一绘图编排器。

职责
----
统一编排 YOLOMM / RTDETRMM 在 train / val / predict 三阶段、detect / segment / pose / obb /
semseg / classify 六任务的可视化输出，消灭约 20 个入口里重复的"拆模态→坐标处理→拼并排→落图"逻辑。

架构
----
    MultiModalPlotter（编排层 — 本文件）
      ├─ split_modalities / visualize_x_to_3ch / concat_side_by_side / prepare_rgb_for_display
      │     （vis.py 工具层 — 复用现有）
      ├─ _transform_for_side_by_side（坐标变换层 — 本文件，并排图标注复制）
      └─ plot_images_mm（绘图层 — plotting_mm.py）
            └─ Annotator（原语层 — plotting.py 复用，不改）

设计原则
--------
- 单一入口：所有多模态绘图经 plot_batch 进入。
- 明确契约：所有坐标输入必须归一化 [0,1]（predict 阶段用 result.boxes.xywhn）。
- 模态判定集中：modality_mode 枚举（dual / rgb_only / x_only），不再各入口自行 if-else。
- 命名统一：resolve_x_modality_name + build_filename，三阶段文件名一致。
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.plotting_mm import plot_images_mm

# vis.py 工具函数（复用，不重写）
from ultralytics.models.utils.multimodal.vis import (
    concat_side_by_side,
    ensure_batch_idx_long,
    extract_precision_contract_from_batch,
    overlay_metric_depth_for_display,
    prepare_rgb_for_display,
    split_modalities,
    visualize_x_to_3ch,
)


ModalityMode = Literal["dual", "rgb_only", "x_only"]
Stage = Literal["train", "val", "predict"]
Source = Literal["labels", "pred"]


class MultiModalPlotter:
    """多模态统一绘图编排器。

    所有多模态可视化（train_batch / val_batch / pred）经此类进入，保证：
      1. 单模态/多模态行为一致
      2. 各任务（bbox/mask/kpts/obb/sem_mask）并排图标注完整
      3. train/val/predict 文件命名一致
      4. 坐标处理唯一正确实现

    Examples:
        >>> plotter = MultiModalPlotter(save_dir, on_plot=trainer.on_plot)
        >>> plotter.plot_batch(
        ...     stage="val", source="labels", images=batch["img"],
        ...     batch_idx=batch["batch_idx"], cls=batch["cls"].squeeze(-1),
        ...     bboxes=batch["bboxes"], xch=3, x_modality_name="thermal",
        ...     precision_contract=extract_precision_contract_from_batch(batch),
        ...     modality_mode="dual", paths=batch["im_file"],
        ...     names=self.names, batch_ni=ni,
        ... )
    """

    def __init__(self, save_dir: Union[str, Path], on_plot: Optional[Callable] = None):
        """初始化绘图器。

        Args:
            save_dir: 输出目录。
            on_plot: 保存后的回调（W&B/ClearML 集成用）。
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.on_plot = on_plot

    # ==================================================================
    # 主入口
    # ==================================================================
    def plot_batch(
        self,
        stage: Stage,
        source: Source,
        images: torch.Tensor,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        contract,
        data: Optional[dict] = None,
        precision_contract: Optional[Dict[str, str]] = None,
        *,
        masks: Optional[torch.Tensor] = None,
        kpts: Optional[torch.Tensor] = None,
        obb: Optional[torch.Tensor] = None,
        sem_masks: Optional[torch.Tensor] = None,
        depth_maps: Optional[torch.Tensor] = None,
        confs: Optional[torch.Tensor] = None,
        paths: Optional[List[str]] = None,
        names: Optional[Dict[int, str]] = None,
        batch_ni: Optional[int] = None,
        pred_base: Optional[str] = None,
        threaded: bool = False,
        # 标注显示控制
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """统一绘图主入口（Wave 4 契约驱动）。

        每个槽位一张面板（标题=模态名），加一张全槽位并排图：
          - 输出文件 ``_<name>.jpg`` + ``_multimodal.jpg``
        渲染按通道数：3ch 彩图；1ch 灰度/伪彩；其它逐通道平铺。

        Args:
            contract: MultiModalChannelContract（通道真相）。
            data: data.yaml dict（precision / fill 覆盖查询用）。
            precision_contract: 按模态名的精度 dict（缺省 legacy）。
            images: 多模态张量 [B, fused_ch, H, W]。
            （其余标注参数同旧版；坐标必须归一化 [0,1]。）
        """
        from ultralytics.models.utils.multimodal.vis import (
            split_modalities,
            prepare_rgb_for_display,
            visualize_x_to_3ch,
            overlay_metric_depth_for_display,
        )

        batch_idx = ensure_batch_idx_long(batch_idx)
        if images.shape[1] != contract.fused_ch:
            raise ValueError(
                f"plot_batch: 输入 {images.shape[1]}ch 与契约不符（fused={contract.fused_ch}）"
            )

        precision_contract = precision_contract or {}
        data = data or {}

        # 拆槽位 → 每槽渲染 display 张量
        slot_tensors = split_modalities(images, contract=contract)
        visuals = []
        for slot, name, sl, t_slot in zip(contract.slots, contract.modality_names, contract.slices, slot_tensors):
            precision = precision_contract.get(name, "legacy")
            ch = t_slot.shape[1]
            if ch == 3 and contract.fill_kind(data=data, name=name) == "visible":
                vis_t = prepare_rgb_for_display(t_slot, precision=precision)
            else:
                vis_t = visualize_x_to_3ch(
                    t_slot, colorize=False, x_modality=name, precision=precision,
                )
                if ch not in (1, 3):
                    # 其它通道数：逐通道平铺到 3ch 画布（重复取前3/复制）
                    vis_t = t_slot[:, :3, :, :] if ch > 3 else t_slot.repeat(1, 3, 1, 1)
                    vis_t = visualize_x_to_3ch(vis_t, colorize=False, x_modality=name, precision=precision)
            if depth_maps is not None:
                vis_t = overlay_metric_depth_for_display(vis_t, depth_maps)
            visuals.append((slot, name, vis_t))

        # 标注张量（obb 优先）
        bboxes_eff = bboxes
        is_obb = False
        if obb is not None and (isinstance(obb, torch.Tensor) and obb.numel() > 0):
            bboxes_eff = obb
            is_obb = True

        # 每槽一张
        for slot, name, vis_t in visuals:
            self._plot_single(
                vis_t, batch_idx, cls, bboxes_eff, is_obb,
                masks, kpts, sem_masks, confs, paths, names,
                modal_tag=name, modality=name, source=source,
                stage=stage, batch_ni=batch_ni, pred_base=pred_base,
                x_modality_name=name, threaded=threaded,
                show_box_labels=show_box_labels, legend=legend,
            )

        # 并排一张
        if len(visuals) > 1:
            vis_list = [v for _, _, v in visuals]
            self._plot_side_by_side(
                vis_list, batch_idx, cls, bboxes_eff, is_obb,
                masks, kpts, sem_masks, confs, paths, names,
                source=source, stage=stage, batch_ni=batch_ni,
                pred_base=pred_base, modality_tags=[n for _, n, _ in visuals],
                threaded=threaded, show_box_labels=show_box_labels, legend=legend,
            )

    def _plot_single(
        self,
        visual: torch.Tensor,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        is_obb: bool,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        sem_masks: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
        paths: Optional[List[str]],
        names: Optional[Dict[int, str]],
        modal_tag: str,
        modality: str,
        source: Source,
        stage: Stage,
        batch_ni: Optional[int],
        pred_base: Optional[str],
        x_modality_name: str,
        threaded: bool,
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """单模态视图：直接调 plot_images_mm。"""
        fname = self.save_dir / self.build_filename(
            stage, source, modality, x_modality_name, batch_ni, pred_base
        )
        # paths 按模态重命名后缀
        paths_mm = self._suffix_paths(paths, modality, x_modality_name) if paths else paths

        plot_images_mm(
            images=visual,
            batch_idx=batch_idx,
            cls=cls,
            bboxes=bboxes,
            coords_normalized=True,
            masks=masks,
            kpts=kpts,
            sem_masks=sem_masks,
            confs=confs,
            paths=paths_mm,
            fname=str(fname),
            names=names,
            on_plot=self.on_plot,
            modal_tag=modal_tag,
            threaded=threaded,
            show_box_labels=show_box_labels,
            legend=legend,
        )

    # ==================================================================
    # 并排视图
    # ==================================================================
    def _plot_side_by_side(
        self,
        visuals: List[torch.Tensor],
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        is_obb: bool,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        sem_masks: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
        paths: Optional[List[str]],
        names: Optional[Dict[int, str]],
        source: Source,
        stage: Stage,
        batch_ni: Optional[int],
        pred_base: Optional[str],
        modality_tags: List[str],
        threaded: bool,
        show_box_labels: bool = True,
        legend: bool = False,
    ) -> None:
        """并排图（Wave 4：N 槽位横向拼接）。"""
        side_visual = visuals[0]
        for v in visuals[1:]:
            side_visual = torch.cat([side_visual, v], dim=3)  # W 维拼接

        # 坐标变换：单图域 → N 联画布
        t = self._transform_for_side_by_side(batch_idx, cls, bboxes, None, kpts, confs, n_panels=len(visuals))

        sem_side = None
        if sem_masks is not None:
            sem_side = sem_masks
            for _ in range(len(visuals) - 1):
                sem_side = torch.cat([sem_side, sem_masks], dim=2)

        fname = self.save_dir / self.build_filename(
            stage, source, "multimodal", "+".join(modality_tags), batch_ni, pred_base
        )
        paths_mm = self._suffix_paths(paths, "multimodal", "+".join(modality_tags)) if paths else paths

        plot_images_mm(
            images=side_visual,
            batch_idx=t["batch_idx"],
            cls=t["cls"],
            bboxes=t["bboxes"],
            coords_normalized=True,
            masks=None,
            kpts=t["kpts"],
            sem_masks=sem_side,
            confs=t["confs"],
            paths=paths_mm,
            fname=str(fname),
            names=names,
            on_plot=self.on_plot,
            modal_tag="Multimodal",
            threaded=threaded,
            show_box_labels=show_box_labels,
            legend=legend,
        )
    def _transform_for_side_by_side(
        self,
        batch_idx: torch.Tensor,
        cls: torch.Tensor,
        bboxes: torch.Tensor,
        masks: Optional[torch.Tensor],
        kpts: Optional[torch.Tensor],
        confs: Optional[torch.Tensor],
        n_panels: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """单图域归一化坐标 → 并排图归一化坐标（N → n_panels*N，Wave 4 泛化）。

        并排图坐标系（总宽度归一化到 [0,1]）：
          左半 RGB: x_center∈[0,0.5], w∈[0,0.5]
          右半 X:   x_center∈[0.5,1.0], w∈[0,0.5]

        各标注变换：
          - bbox [N,4]→[2N,4]: 前 N 左半(x*=0.5,w*=0.5), 后 N 右半(x=x*0.5+0.5,w*=0.5)
          - mask [N,H,W]→[2N,H,2W]: 前 N mask 在左半, 后 N mask 在右半（避免 plot_images_mm resize 拉伸）
          - kpts [N,K,3]→[2N,K,3]: x 坐标 *=0.5（左）/ x*0.5+0.5（右）
          - obb [N,5]→[2N,5]: 同 bbox，角度 r 不变
          - batch_idx/cls/confs: 复制为 2N

        Args:
            bboxes: 归一化 xywh [N,4] 或 xywhr [N,5]。

        Returns:
            dict 含 batch_idx/cls/bboxes/masks/kpts/confs，均为 2N 长度。
        """
        # 防御性类型归一化：调用方（val dataloader / predict pipeline）可能传入 numpy
        batch_idx = torch.as_tensor(batch_idx, dtype=torch.long)
        cls = torch.as_tensor(cls)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        if masks is not None:
            masks = torch.as_tensor(masks)
        if kpts is not None:
            kpts = torch.as_tensor(kpts)
        if confs is not None:
            confs = torch.as_tensor(confs)

        n = n_panels if n_panels and n_panels > 0 else 2
        if bboxes.numel() == 0:
            # 空标注，直接复制 batch_idx/cls
            return {
                "batch_idx": torch.cat([batch_idx] * n, dim=0),
                "cls": torch.cat([cls] * n, dim=0),
                "bboxes": bboxes,
                "masks": masks,
                "kpts": kpts,
                "confs": torch.cat([confs] * n, dim=0) if confs is not None else None,
            }

        # ===== N 面板泛化（Wave 4）=====
        n = n_panels if n_panels and n_panels > 0 else 2
        step = 1.0 / n
        bboxes_parts = []
        for p_i in range(n):
            b = bboxes.clone()
            b[:, 0] = b[:, 0] * step + p_i * step
            b[:, 2] *= step
            bboxes_parts.append(b)
        bboxes_side = torch.cat(bboxes_parts, dim=0)

        masks_side = masks
        if masks is not None and masks.numel() > 0:
            mask_parts = []
            zero_width = masks.shape[2] * (n - 1)
            zero = torch.zeros(masks.shape[0], masks.shape[1], zero_width,
                               dtype=masks.dtype, device=masks.device)
            for p_i in range(n):
                if p_i == 0:
                    mask_parts.append(torch.cat([masks, zero], dim=2))
                else:
                    left = torch.zeros(masks.shape[0], masks.shape[1], masks.shape[2] * p_i,
                                       dtype=masks.dtype, device=masks.device)
                    right = torch.zeros(masks.shape[0], masks.shape[1], masks.shape[2] * (n - 1 - p_i),
                                        dtype=masks.dtype, device=masks.device)
                    mask_parts.append(torch.cat([left, masks, right], dim=2))
            masks_side = torch.cat(mask_parts, dim=0)

        kpts_side = kpts
        if kpts is not None and kpts.numel() > 0:
            kpts_parts = []
            for p_i in range(n):
                k = kpts.clone()
                k[..., 0] = k[..., 0] * step + p_i * step
                kpts_parts.append(k)
            kpts_side = torch.cat(kpts_parts, dim=0)

        batch_idx_side = torch.cat([batch_idx] * n, dim=0)
        cls_side = torch.cat([cls] * n, dim=0)
        return {
            "batch_idx": batch_idx_side,
            "cls": cls_side,
            "bboxes": bboxes_side,
            "masks": masks_side,
            "kpts": kpts_side,
            "confs": torch.cat([confs] * n, dim=0) if confs is not None else None,
        }

    # ==================================================================
    # 命名与模态判定（静态方法）
    # ==================================================================
    @staticmethod
    def resolve_x_modality_name(
        modality_param: Optional[str],
        data: Optional[dict],
        x_modality_name_attr: Optional[str] = None,
    ) -> str:
        """统一解析 X 模态名（解决 train/val/predict 命名不一致）。

        优先级：
          1. x_modality_name_attr（train/val 的 self.x_modality_name）
          2. data.yaml 的 modality_used / x_modality
          3. modality_param 本身（若非 RGB/dual/x 泛化词）
          4. 兜底 "x"

        Args:
            modality_param: 入口传入的 modality 字符串（如 "thermal"/"RGB"/"X"）。
            data: 数据集配置 dict。
            x_modality_name_attr: 已解析的 x_modality_name 属性。

        Returns:
            真实模态名（如 "thermal"/"depth"），而非泛化的 "x"。
        """
        # 处理 list 类型（x_modality_name_attr 可能来自配置，如 ['rgb', 'sar']）
        attr = x_modality_name_attr
        if isinstance(attr, list):
            non_rgb = [v for v in attr if str(v).lower() != "rgb"]
            attr = non_rgb[0] if non_rgb else None
        if attr and str(attr).lower() not in ("unknown", ""):
            return str(attr).lower()

        if data:
            for key in ("modality_used", "x_modality", "modality"):
                val = data.get(key)
                if val is None:
                    continue
                # 处理 list 类型：modality_used 如 ["rgb", "sar"]，取非 RGB 的第一个作为 X 模态
                if isinstance(val, list):
                    non_rgb = [v for v in val if str(v).lower() != "rgb"]
                    if non_rgb:
                        val = non_rgb[0]
                    else:
                        continue
                if str(val).lower() not in ("rgb", "dual", "x", ""):
                    return str(val).lower()

        if modality_param:
            mp = str(modality_param).lower()
            if mp not in ("rgb", "dual", "x", ""):
                return mp

        return "x"

    @staticmethod
    def build_filename(
        stage: Stage,
        source: Source,
        modality: Literal["rgb", "x", "multimodal"],
        x_modality_name: str,
        batch_ni: Optional[int] = None,
        pred_base: Optional[str] = None,
    ) -> str:
        """构建输出文件名（保持现有命名约定，W&B glob 兼容）。

        规则：
          - predict: pred_{base}_labels_{mod}.jpg
          - train/val: {stage}_batch{ni}_{source}_{mod}.jpg

        其中 modality="x" 时替换为 x_modality_name（如 thermal）。

        Args:
            stage: train / val / predict。
            source: labels / pred。
            modality: rgb / x / multimodal。
            x_modality_name: X 模态名。
            batch_ni: train/val batch 序号。
            pred_base: predict 基础名。

        Returns:
            文件名字符串。
        """
        mod_str = x_modality_name if modality == "x" else modality
        if stage == "predict":
            if pred_base is None:
                raise ValueError("predict 阶段需提供 pred_base")
            return f"pred_{pred_base}_labels_{mod_str}.jpg"
        if batch_ni is None:
            raise ValueError(f"{stage} 阶段需提供 batch_ni")
        return f"{stage}_batch{batch_ni}_{source}_{mod_str}.jpg"

    @staticmethod
    def determine_modality_mode(modality: Optional[str], is_single_modal: bool = False) -> ModalityMode:
        """统一模态模式判定（解决各入口 if-else 不一致）。

        Args:
            modality: 入口的 self.modality 字段。可能值：
                - None / 空 / "dual": 双模态
                - "RGB" / "rgb": RGB 单模态
                - "X" / "thermal" / "depth" / ...: X 单模态
            is_single_modal: predict 阶段的单模态标志（可选，作为辅助判定）。

        Returns:
            "dual" / "rgb_only" / "x_only"
        """
        if modality is None or str(modality).strip() == "" or str(modality).lower() == "dual":
            return "dual"
        m = str(modality).lower()
        if m == "rgb":
            return "rgb_only"
        # "x" 或具体模态名（thermal/depth/ir...）均视为 X 单模态
        return "x_only"

    # ==================================================================
    # 辅助
    # ==================================================================
    @staticmethod
    def _suffix_paths(
        paths: List[str], modality: str, x_modality_name: str
    ) -> List[str]:
        """给 paths 加模态后缀（仅用于网格图文件名标注，不影响实际读取）。"""
        mod_str = x_modality_name if modality == "x" else modality
        return [p.replace(".jpg", f"_{mod_str}.jpg") for p in paths]
