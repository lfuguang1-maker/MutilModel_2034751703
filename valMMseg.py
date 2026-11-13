import os
from ultralytics import YOLOMM

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model = YOLOMM('ResTest/semseg-test/weights/best.pt')
data = 'datasets/tree_semseg/data.yaml'
model.val(data=data,
          split='test',device='0',
        #   modality='x',
          project='ResTest',
          name='Val')