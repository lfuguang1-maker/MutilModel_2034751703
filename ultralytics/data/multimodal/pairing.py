# Ultralytics Multimodal Inference - Explicit Pairing Resolver
# Resolves explicit RGB+X inputs into paired sample specifications
# Version: v2.0 (Breaking Change - 废弃隐式source参数)
# Date: 2026-01-13

from pathlib import Path
from typing import List, Dict, Union
from ultralytics.utils import LOGGER
from ultralytics.nn.mm import MultiModalSourceMatcher


class PairingResolver:
    """
    多模态推理显式配对解析器（新API）

    职责：
    - 接收显式的 rgb_source 和 x_source 参数
    - 验证输入合法性和文件存在性
    - 生成统一的配对样本规格

    新API设计：
    - 强制要求同时提供 rgb_source 和 x_source
    - 支持单对、批量推理
    - 不再支持隐式列表 [rgb, x] 格式（breaking change）
    """

    def __init__(self, x_modality: str = "unknown", verbose: bool = True):
        """
        初始化配对解析器

        Args:
            x_modality: X模态类型名称（如 'thermal', 'depth', 'ir'等）
            verbose: 是否输出详细日志
        """
        self.x_modality = x_modality
        self.verbose = verbose

    def resolve(
        self,
        rgb_source: Union[str, Path, List[Union[str, Path]], None] = None,
        x_source: Union[str, Path, List[Union[str, Path]], None] = None,
        strict_match: bool = True
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """
        解析显式RGB和X模态输入为配对样本列表（支持单模态推理）

        Args:
            rgb_source: RGB图像源（可为None表示缺失）
                - 单图: '/path/to/rgb.jpg' 或 Path('/path/to/rgb.jpg')
                - 目录: '/path/to/rgb_dir'（自动扫描目录内所有图像文件）
                - 批量: ['/path/rgb1.jpg', '/path/rgb2.jpg']
                - 缺失: None (将使用零填充)
            x_source: X模态图像源（可为None表示缺失）
                - 单图: '/path/to/thermal.jpg'
                - 目录: '/path/to/thermal_dir'
                - 批量: ['/path/thermal1.jpg', '/path/thermal2.jpg']
                - 缺失: None (将使用零填充)
            strict_match: 批量推理时的匹配策略（默认严格）
                - True: 严格模式，目录必须完全匹配，列表必须等长
                - False: 宽松模式，目录取交集，列表仍要求等长

        Returns:
            配对样本规格列表 [
                {
                    'id': 'sample_001',
                    'rgb_path': Path('/path/rgb.jpg') or None,
                    'x_path': Path('/path/thermal.jpg') or None,
                    'x_modality': 'thermal'
                },
                ...
            ]

        Raises:
            ValueError: 输入格式不合法或数量不匹配
            FileNotFoundError: 文件不存在
        """
        # --- 单模态推理处理 ---
        if rgb_source is None and x_source is not None:
            return self._resolve_single_modality(x_source, modality="x")

        if rgb_source is not None and x_source is None:
            return self._resolve_single_modality(rgb_source, modality="rgb")

        # --- 双模态推理 ---
        # 目录批量：source=[rgb_dir, x_dir]
        if isinstance(rgb_source, (str, Path)) and isinstance(x_source, (str, Path)):
            rgb_p, x_p = Path(rgb_source), Path(x_source)
            if rgb_p.is_dir() and x_p.is_dir():
                matcher = MultiModalSourceMatcher(rgb_source, x_source, strict_match=strict_match)
                pairs = matcher.match()
                samples = [
                    self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                    for i, (rp, xp) in enumerate(pairs)
                ]
                if self.verbose:
                    LOGGER.info(f"目录批量配对完成: {len(samples)} 对有效样本")
                return samples

        # 列表批量：rgb_source=[...], x_source=[...]
        if isinstance(rgb_source, list) and isinstance(x_source, list):
            pairs = MultiModalSourceMatcher.match_lists(rgb_source, x_source, strict_match=strict_match)
            samples = [
                self._create_sample_spec(rgb_path=rp, x_path=xp, sample_idx=i)
                for i, (rp, xp) in enumerate(pairs)
            ]
            if self.verbose:
                LOGGER.info(f"列表批量配对完成: {len(samples)} 对有效样本")
            return samples

        # 单对推理（str/Path对str/Path）: 使用现有逻辑
        rgb_list = self._normalize_to_list(rgb_source, "rgb_source")
        x_list = self._normalize_to_list(x_source, "x_source")

        # 验证数量匹配
        if len(rgb_list) != len(x_list):
            raise ValueError(
                f"RGB和X模态数量不匹配：\n"
                f"  rgb_source: {len(rgb_list)} 张\n"
                f"  x_source: {len(x_list)} 张\n"
                f"请确保两者数量相同。"
            )

        # 配对并验证
        samples = []
        for idx, (rgb_path, x_path) in enumerate(zip(rgb_list, x_list)):
            samples.append(self._create_sample_spec(
                rgb_path=Path(rgb_path),
                x_path=Path(x_path),
                sample_idx=idx
            ))

        if self.verbose:
            LOGGER.info(f"双模态配对完成: {len(samples)} 对有效样本")

        return samples

    def resolve_n_modal(
        self,
        sources: Dict[str, Union[str, Path, List[Union[str, Path]], None]],
        contract=None,
        strict_match: bool = True,
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """N 模态槽位配对解析（方案一：sources 字典 → 槽位样本列表）。

        Args:
            sources: 槽位 → 图像源。键为槽位字母（'A'/'B'/'C'/...，大小写不敏感），
                值为单图/目录/列表/None（None=该槽缺失，由数据集零填充，单路消融）。
            contract: MultiModalChannelContract；提供时校验键必须 ∈ contract.slots。
            strict_match: 多源批量时按文件名 stem 严格/交集匹配。

        Returns:
            List[Dict]，每项 {'id': stem, 'A': Path|None, 'B': Path|None, ...}，
            覆盖 contract.slots 全部槽位（无契约时覆盖出现的全部键）。

        Raises:
            ValueError: 键不是单槽位字母 / 不在契约槽位 / 数量不匹配。
        """
        import re as _re
        norm: Dict[str, List[Path] | None] = {}
        for key, val in sources.items():
            k = str(key).strip().upper()
            if len(k) != 1 or not k.isalpha():
                raise ValueError(f"sources 键 {key!r} 必须是单个槽位字母（A-Z），不支持组合槽/Fuse")
            if contract is not None and k not in contract.slots:
                raise ValueError(f"sources 键 {k!r} 不在契约槽位 {list(contract.slots)} 中")
            if val is None:
                norm[k] = None
                continue
            lst = self._normalize_to_list(val, f"sources[{k!r}]")
            # 目录 → 展开为目录内图像文件
            if len(lst) == 1 and lst[0].is_dir():
                _exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.npy', '.npz'}
                expanded = sorted(pp for pp in lst[0].rglob('*') if pp.is_file() and pp.suffix.lower() in _exts)
                if not expanded:
                    raise FileNotFoundError(f"sources[{k!r}] 目录内无图像: {lst[0]}")
                norm[k] = expanded
            else:
                norm[k] = lst

        slots = list(contract.slots) if contract is not None else sorted(norm.keys())
        # 未提供源的槽位 → None（零填充）
        for s in slots:
            norm.setdefault(s, None)

        # 以提供源最多的槽位为基准
        base_slot = max((s for s in slots if norm[s] is not None),
                        key=lambda s: len(norm[s]), default=None)
        if base_slot is None:
            raise ValueError("sources 全部为 None，至少提供一个槽位的图像源")

        base_stems = [p.stem for p in norm[base_slot]]

        # 每槽单文件 → 按位置直接配对（旧 resolve 单图对语义，不看文件名；
        # strict_match=False 且 stem 无交集时放行到交集逻辑以报无共同 stem）
        _all_single = all(norm[s] is None or len(norm[s]) == 1 for s in slots if norm[s] is not None)
        _stems_ok = all(
            norm[s] is None or norm[s][0].stem == norm[base_slot][0].stem or strict_match
            for s in slots
        )
        if _all_single and _stems_ok and all(
                len(norm[s]) == len(norm[base_slot]) for s in slots if norm[s] is not None):
            spec: Dict[str, Union[str, Path, None]] = {"id": base_stems[0]}
            for s in slots:
                pth = norm[s][0] if norm[s] is not None else None
                if pth is not None and not pth.exists():
                    raise FileNotFoundError(f"sources[{s!r}] 文件不存在: {pth}")
                spec[s] = pth
            samples = [spec]
            if self.verbose:
                LOGGER.info(f"N 模态槽位配对完成（单图对按位置配对）: 槽位 {slots}")
            return samples

        # 按 stem 建索引；数量/缺失校验收进 strict（修 #3）
        stem_maps: Dict[str, Dict[str, Path]] = {}
        for s in slots:
            if norm[s] is None:
                continue
            sm: Dict[str, Path] = {}
            for pth in norm[s]:
                if pth.stem in sm and strict_match:
                    raise ValueError(f"sources[{s!r}] 存在重名 stem: {pth.stem}")
                sm[pth.stem] = pth
            stem_maps[s] = sm

        provided = [s for s in slots if norm[s] is not None]
        if strict_match:
            for s in provided:
                if len(norm[s]) != len(norm[base_slot]):
                    raise ValueError(
                        f"sources[{s!r}] 数量 {len(norm[s])} 与基准槽位 {base_slot!r} 数量 "
                        f"{len(norm[base_slot])} 不一致；strict 模式要求完全匹配（或 strict_match=False 取交集）"
                    )
                missing = [st for st in base_stems if st not in stem_maps[s]]
                if missing:
                    raise ValueError(
                        f"sources[{s!r}] 与基准槽位 {base_slot!r} 无法按文件名配对（数量相同但 stem 不齐），"
                        f"缺失 stem 示例: {missing[:5]}；如确需按位置配对请逐张调用"
                    )
        else:
            # 交集模式：已提供槽位全交集，保持基准槽原顺序；提供图但缺该张的样本被跳过（非 None 消融）
            common = set.intersection(*(set(stem_maps[s]) for s in provided))
            skipped = len(base_stems) - sum(1 for st in base_stems if st in common)
            base_stems = [st for st in base_stems if st in common]
            if skipped:
                LOGGER.warning(f"strict_match=False：跳过 {skipped} 个无法在全部已提供槽位配对的样本")
            if not base_stems:
                raise ValueError("strict_match=False：已提供槽位之间无共同 stem")

        samples = []
        for idx, stem in enumerate(base_stems):
            spec = {"id": stem}
            for s in slots:
                pth = stem_maps.get(s, {}).get(stem)
                if pth is not None and not pth.exists():
                    raise FileNotFoundError(f"sources[{s!r}] 文件不存在: {pth}")
                spec[s] = pth
            samples.append(spec)

        if self.verbose:
            LOGGER.info(f"N 模态槽位配对完成: {len(samples)} 个样本, 槽位 {slots}")
        return samples

    def _resolve_single_modality(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        modality: str
    ) -> List[Dict[str, Union[str, Path, None]]]:
        """
        解析单模态输入（支持目录扫描）

        Args:
            source: 图像源（单个、目录或列表）
            modality: 'rgb' 或 'x'
        """
        # 目录输入：扫描目录内所有图像文件
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            src_dir = Path(source)
            files = sorted([
                p for p in src_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MultiModalSourceMatcher.SUPPORTED_FORMATS
            ])
            if not files:
                modality_name = "RGB" if modality == "rgb" else self.x_modality
                raise ValueError(f"{modality_name}目录中未找到受支持的图像文件: {src_dir}")
        else:
            files = self._normalize_to_list(source, f"{modality}_source")

        samples = []
        for idx, path in enumerate(files):
            if modality == "rgb":
                samples.append(self._create_sample_spec(rgb_path=Path(path), x_path=None, sample_idx=idx))
            else:
                samples.append(self._create_sample_spec(rgb_path=None, x_path=Path(path), sample_idx=idx))

        if self.verbose:
            if modality == "rgb":
                LOGGER.info(f"单RGB模态推理: {len(samples)} 个样本（{self.x_modality}将使用零填充）")
            else:
                LOGGER.info(f"单{self.x_modality}模态推理: {len(samples)} 个样本（RGB将使用零填充）")

        return samples

    def _normalize_to_list(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        param_name: str
    ) -> List[Path]:
        """
        统一输入为列表格式

        Args:
            source: 输入源（单个或列表）
            param_name: 参数名称（用于错误提示）

        Returns:
            Path对象列表
        """
        if isinstance(source, (str, Path)):
            return [Path(source)]
        elif isinstance(source, list):
            if not source:
                raise ValueError(f"{param_name} 不能为空列表")
            return [Path(item) for item in source]
        else:
            raise ValueError(
                f"{param_name} 类型不支持: {type(source)}\n"
                f"支持类型: str, Path, List[str], List[Path]"
            )

    def _create_sample_spec(
        self,
        rgb_path: Union[Path, None],
        x_path: Union[Path, None],
        sample_idx: int
    ) -> Dict[str, Union[str, Path, None]]:
        """
        创建配对样本规格并验证文件存在性（支持None占位）

        Args:
            rgb_path: RGB图像路径（可为None表示缺失）
            x_path: X模态图像路径（可为None表示缺失）
            sample_idx: 样本索引（用于生成ID）

        Returns:
            样本规格字典

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 路径不是文件
        """
        # 验证RGB文件（如果提供）
        if rgb_path is not None:
            if not rgb_path.exists():
                raise FileNotFoundError(f"RGB文件不存在: {rgb_path}")
            if not rgb_path.is_file():
                raise ValueError(f"RGB路径不是文件: {rgb_path}")

        # 验证X模态文件（如果提供）
        if x_path is not None:
            if not x_path.exists():
                raise FileNotFoundError(f"X模态文件不存在: {x_path}")
            if not x_path.is_file():
                raise ValueError(f"X模态路径不是文件: {x_path}")

        # 生成样本ID（优先使用RGB文件名，否则使用X模态文件名）
        if rgb_path is not None:
            sample_id = rgb_path.stem
        elif x_path is not None:
            sample_id = x_path.stem
        else:
            sample_id = f"sample_{sample_idx:03d}"

        return {
            "id": sample_id,
            "rgb_path": rgb_path,
            "x_path": x_path,
            "x_modality": self.x_modality
        }
