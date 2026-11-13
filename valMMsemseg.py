#这是提供 参考示例的 YOLOMM语义分割验证器使用方法
#按必须按照你的需求来更改而不是盲目去使用

from ultralytics import YOLOMM

model = YOLOMM('ResTest/semseg-test/weights/best.pt')
model.val(data='datasets/tree_semseg/data.yaml',
          split='test', device='0',
          # modality='x',  # 模态消融参数 非必要不得开启
          project='ResTest',
          name='semseg-val')
