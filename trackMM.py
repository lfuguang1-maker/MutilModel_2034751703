# 这是提供参考示例的 TRACK 追踪器使用方法
# 必须按照你的需求来更改而不是盲目去使用
#
# 用途：对已训练的 YOLOMM / RTDETRMM 权重（detect / segment / pose / obb）
#       做离线多模态多目标跟踪，输出带编号的结果、标注视频和轨迹文件。
#
# 输入支持（各槽位形态必须一致）：
#   1. 一对视频        sources={'A': 'seq01_rgb.mp4',  'B': 'seq01_ir.mp4'}
#   2. 一对帧目录      sources={'A': 'seq01_rgb/',     'B': 'seq01_ir/'}
#   3. 多序列根目录    sources={'A': 'root/rgb/',      'B': 'root/ir/'}   # 里面每个子目录是一段序列
#   4. 单模态          sources={'A': 'seq01_rgb/'}                        # B 槽按契约零填充
#
# 槽位字母（A/B/C...）来自权重训练时的 data.yaml modality_used 顺序。

from ultralytics import TRACK

# ========== 1. 必填参数 ==========
WEIGHTS = "/path/to/weights/best.pt"  # 已训练的多模态 .pt 权重

# 各槽位的输入路径（视频文件、帧目录或序列根目录均可）
SOURCES = {
    "A": "/path/to/seq01_rgb",   # RGB（锚定槽位，坐标以它为基准）
    "B": "/path/to/seq01_ir",    # 红外/事件等副模态；单模态就删掉这一行
}

# ========== 2. 常用可调参数 ==========
OUT = dict(
    project="runs/track",  # 输出根目录
    name="exp",            # 本次实验名（输出在 runs/track/exp）
    exist_ok=True,         # 目录已存在时复用
    save=True,             # 保存标注视频（编号 + 尾迹，各模态横向拼接）
    save_txt=True,         # 保存轨迹文件 labels/<seq>.txt（MOT 格式）
    stream=True,           # 流式处理，长序列必开，省内存
)
TRACKER = "botsort.yaml"   # 跟踪器配置：botsort.yaml / bytetrack.yaml
CONF = 0.1                 # 检测置信度阈值（ByteTrack 建议调低，如 0.05~0.1）

# ========== 3. 运行 ==========
model = TRACK(WEIGHTS)
print(f"已加载: {model.family} {model.task}, 类别: {model.names}")

results = model.track(
    sources=SOURCES,
    tracker=TRACKER,
    conf=CONF,
    verbose=True,
    **OUT,
    # --- 以下按需打开 ---
    # classes=[0],          # 只跟踪指定类别（按权重 names 的编号）
    # fps=30,               # 帧目录序列的帧率（默认 30；视频自动读）
    # imgsz=640,            # 推理尺寸（默认取权重的训练尺寸）
    # iou=0.7,              # NMS IoU 阈值
    # max_det=300,          # 每帧最多检测数
    # trail=30,             # 视频里的轨迹尾迹长度（帧），0 关闭
    # device="0",           # GPU 编号，省略则自动
    # strict_match=False,   # 帧不齐时取交集（默认严格报错）
)

# ========== 4. 逐帧结果（stream=True 时边跑边出） ==========
for r in results:
    print(
        f"{r.meta['seq']} 第{r.meta['frame']:>4}帧 | "
        f"轨迹编号: {None if r.ids is None else r.ids.tolist()} | "
        f"检测数: {len(r.boxes)}"
    )
    # segment 任务: r.masks；obb 任务: r.obb；pose 任务: r.keypoints
    # 需要逐帧自定义处理时在这里写

# ========== 5. 输出位置 ==========
# runs/track/exp/seq01.mp4          <- 标注视频（save=True）
# runs/track/exp/labels/seq01.txt   <- 轨迹文件（save_txt=True），可直接用于 TRACK.val 评估
