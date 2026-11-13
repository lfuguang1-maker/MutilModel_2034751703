# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# TRACK 跟踪评估示例：按需修改权重与 YOLOMMtrack 数据集路径。

from ultralytics import TRACK

model = TRACK("/path/to/weights/best.pt")
metrics = model.val(
    data="/path/to/mmtrack_ds/data.yaml",
    split="val",
    tracker="bytetrack.yaml",
    project="runs/track",
    name="track_val",
    exist_ok=True,
)
print(f"HOTA={metrics.hota:.3f} MOTA={metrics.mota:.3f} IDF1={metrics.idf1:.3f}")
