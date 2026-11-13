#这是提供 参考示例的 YOLOMM推理器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM
from ultralytics.tools import MultiModalSampler

# 模型权重：与 trainMM.py 同源（三模态 rgb+event+thermal，9ch 契约随 ckpt 携带）
model = YOLOMM('/home/zhizi/work/ultralyticsmm/ResTest/3mmm2/weights/best.pt')

# 三模态槽位采样（A=rgb, B=event, C=thermal，顺序来自 data.yaml modality_used）
sampler = MultiModalSampler('/home/zhizi/work/datasets/dsert_tri/data.yaml', split='val')
sources = sampler.sample_sources(n=10)
# for slot, paths in sources.items():
#     print(f"  槽位 {slot}: {len(paths)} 张, 例: {paths[0]}")

# 测试三模态预测（sources 槽位字典 API）
print("=== 测试三模态预测 (A+B+C) ===")
model.predict(sources=sources,
              project='ResTest',
              name='predict3mm',
              save=True,
              exist_ok=True,
              # conf=0.001,
              # iou=0.01,
              )

# 测试单路 RGB 消融（B/C 槽零填充）
print("\n=== 测试单路 RGB 消融（仅槽位 A）===")
model.predict(sources={'A': sources['A']},
              project='ResTest',
              name='predict3mm-rgbA',
              save=True,
              exist_ok=True,
              # conf=0.1,
              )

# 旧双输入 API 仍可用（内部自动转 {'A': rgb, 'B': x}）
# print("\n=== 旧双输入兼容 ===")
# model.predict(rgb_source=rgb_list, x_source=x_list, project='ResTest', name='predict-dual', save=True)
