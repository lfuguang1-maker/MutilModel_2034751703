"""从 讲稿1.ipynb 生成 讲稿2.ipynb：同一套讲稿骨架，内容换成无人机 RGBT / RGBD 双模态检测。
原始文字尽量保留，只做领域替换和必要的少量补充。
"""
import json
from pathlib import Path

ROOT = Path('/root/ultralyticsmm')
SRC = ROOT / '讲稿1.ipynb'
DST = ROOT / '讲稿2.ipynb'

TITLE = """\
# 怎么快速的上手做一篇无人机双模态（RGBT / RGBD）
"""

C1 = """\
## ① 基本概念

### 1.1 传统目标检测

一个检测器对外只有一个约定：进来一张 H×W×3 的图，出去 N 个框，每个框五个数，左上角、右下角各两个，加一个类别。

这个约定从 R-CNN 那一代定下来，到今天没怎么变。变的是中间那部分怎么实现，五花八门。

YOLO 这一路的做法是把图切成网格，每个网格直接回归框和类别，一次前向出结果。它快，代价是密集预测容易在重叠目标上打架，所以后面要 NMS 收拾。

检测器要做的事可以拆成两件。第一件是把像素变成有语义的特征，这是骨干的活。第二件是在特征上判断哪里有什么，这是检测头的活。

这两件事，单模态都已经做得不错了。COCO 上七十几的 mAP，航拍画面里认一辆车、一条船，也够用。
"""

C2 = """\
### 1.2 它解决了什么，又卡在哪

#### 先说做到的。纹理、颜色、边缘、形状，只要在图上留下了像素差异，骨干就能学到。绝大多数检测任务靠这一套都跑得通。

#### 但是在无人机航拍中有三种很常见的问题，会影响到单纯依赖于可见光的检测

### 一是遮挡。
被树冠挡掉一半的车，剩下的像素和树冠、阴影混在一起，成像上已经分不出前后。人眼能看出那是个车，靠的是生活经验补的，模型没有这份经验。

### 二 是成像条件本身失效。夜里的路灯、逆光、水面反光、同色的路面和车身，这几种情况下像素分布直接塌掉，再强的骨干也提不出有用特征。

### 三是目标本身太小。无人机飞得高，一个行人在画面里可能就十几个像素，框的宽高比和周围纹理都分不开。这一类不是信息不够，是分辨率不够。

<img src="assets/uav_rgb_case.png" width="30%">

上图 26 个绿框是人工标的车和行人，最小的框只有 4 个像素宽。看那些压在人行横道阴影里的、还有几乎和路面一个色的，这就是上面说的第一类情况。

这说明还有**一类可见光根本没有记录的信息**。
"""

C3 = """\
### 1.3 加一路模态

#### 从视觉的角度来说，我们一直认为单模态检测假设「手上的这一路信号，包含了判断所需的全部信息」。

#### 但是实际上并不能完全得到充分的信息

#### 因此我们需要补充多模态的信息，用于提供更多信息，从从而当模型从不同角度和方面学会到的互补的信息，增强对于任务的理解

#### 我们在本内容中之所以使用 RGBT 是因为 1. 无人机上的双光吊舱本来就同时出可见光和红外，两路数据是现成的 2. 红外记录的是热辐射，夜里、逆光、雾里这些可见光失效的场景，人和车身上还带着温度差，正好补上这一路

#### 如果手上只有可见光，那就走 RGBD：深度项目能自动生成，第二路换成深度，数据格式、模型、训练流程一行都不用改
"""

C4 = """\
### 1.4 两路在哪合

加了一路，接着要决定在哪把两路并起来。这有三种放法，

**早期融合**，进来就拼成 6 通道，共用一个骨干。
好处很直接，模型改动最小，骨干只跑一遍，算力最省。
代价是两路信号从第一层起就被强制同分布处理。红外图和可见光的统计特性差得远，共享的卷积核要同时拟合两种分布，早期那些本该各自学的浅层特征就被搅在一起。

**晚期融合**，两路各跑一套完整网络，最后在框和分数上合并。
好处是最干净，两路互不干扰，各自都能训到最好。
代价是要养两份骨干加两份检测头，参数量和算力基本翻倍，部署时也更重。

**中层融合**，两路骨干各自先提特征，到中间某几层再把特征合起来，后面共用一个检测头。
好处是浅层各自学各自的成像特点，不被强制同分布；深层已经足够语义化，这时再合，互补的信息才用得上。而且只有骨干是双份，检测头还是单份，代价比晚期融合小得多。
代价是融合点得选，选在哪是个超参，选错了效果不如另外两个方案。

三个方案摆在一起，就逼出了我们的选择选择。早期融合省算力但浅层被搅浑，晚期融合干净但贵一倍。想要「浅层分开、深层合并、头部只算一次」，就只剩中层融合这一条。

<img src="assets/structure_rgbt.png" width="78%">
"""

C5 = """\
## ② 多模态数据长什么样

### 为了高效统一的管理多模态数据，我定义了一个新的数据格式即 YOLOMM 格式，他是 YOLO 格式的超集，你会 YOLO 格式就可以很快的上手

```yaml
path: /root/datasets/rgbt_uav_demo
train: images/train
val:   images/val
modality:
  rgb:     images          # 可见光
  thermal: images_thermal  # 红外（灰度复制成 3 通道）
modality_used: [rgb, thermal]
Xch: 3
nc: 7
names: ['ship', 'car', 'cyclist', 'pedestrian', 'bus', 'drone', 'plane']  #类别名字
```

目录是两份并排：

```
/root/datasets/rgbt_uav_demo/
├── images/          train 1200  val 300   可见光
├── images_thermal/  train 1200  val 300   红外（3 通道）
└── labels/          train 1200  val 300   标注（只有一份）
```

全量是 rgbt_tiny（DJI 双光吊舱航拍）的 3.4 万张训练图 / 1.2 万张验证图，这里为了课堂演示按固定种子抽了 1200 / 300 的子集，把 yaml 里的 path 换回全量目录就是完整实验。
"""

C5B = """\
### 同一套数据，第二路换成深度就是 RGBD

```yaml
modality:
  rgb:   images          # 可见光
  depth: images_depth    # 深度（项目自动生成）
modality_used: [rgb, depth]
```

深度不用另外采，项目里带着深度生成器 `depthGen.py`，权重从网盘下好放到 `weights/`，对原来的 `images/` 跑一遍就落到 `images_depth/`。

后面的模型、训练、验证、推理全都不动，通道数也都是 3 + 3 = 6，区别只是第二路换成了生成出来的深度图。
"""

C6 = """\
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt

DATA_YAML = '/root/datasets/rgbt_uav_demo/uav_rgbt.yaml'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CFG = yaml.safe_load(open(DATA_YAML, encoding='utf-8'))
ROOT = Path(CFG['path'])
NAMES = {int(k): v for k, v in (CFG['names'].items() if isinstance(CFG['names'], dict)
                                else enumerate(CFG['names']))}
MODALITIES = CFG['modality']          # {模态名: 目录}，几个模态就有几路输入

print(f'类别 {len(NAMES)} 个 -> {NAMES}')
print(f'模态 {list(MODALITIES)}   融合通道数 = 3 x 2 = 6')
for split in ('train', 'val'):
    imgs = [p for p in (ROOT / CFG[split]).iterdir() if p.suffix.lower() in IMG_EXTS]
    labs = list((ROOT / 'labels' / split).glob('*.txt'))
    print(f'  {split:<5} 可见光 {len(imgs):>4}   标注 {len(labs):>4}')
"""

C7 = """\
def load_mm_pair(split='val', index=0):
    '按序号取同一时刻的各模态图，返回 {模态: 图}, xywh 框, 类别, 文件名。'
    files = sorted(p for p in (ROOT / CFG[split]).iterdir() if p.suffix.lower() in IMG_EXTS)
    stem = files[int(index)].stem                      # 两路靠同名主干配对

    imgs = {}
    for name, sub in MODALITIES.items():
        folder = ROOT / sub / split
        hit = next((p for p in folder.iterdir() if p.stem == stem and p.suffix.lower() in IMG_EXTS), None)
        if hit is None:
            raise FileNotFoundError(f'{name} 缺少配对文件: {folder / stem}.*')
        img = cv2.imread(str(hit), cv2.IMREAD_UNCHANGED)   # UNCHANGED：红外也按原样读，避免被转格式
        imgs[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    lab = ROOT / 'labels' / split / f'{stem}.txt'
    rows = [ln.split() for ln in lab.read_text().strip().splitlines()] if lab.exists() else []
    arr = np.array(rows, dtype=np.float32).reshape(-1, 5)    # cls cx cy w h
    return imgs, arr[:, 1:5], arr[:, 0].astype(int), files[int(index)].name


def draw_mm(axs, split, index, show_labels=False):
    '在给定的一行子图上画一对多模态样本。'
    imgs, boxes, cls, fname = load_mm_pair(split, index)
    for ax, (name, img) in zip(axs, imgs.items()):
        h, w = img.shape[:2]
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(f'{name}   {w}x{h}', fontsize=10)
        ax.axis('off')
        if not show_labels:
            continue
        for (cx, cy, bw, bh), c in zip(boxes, cls):
            x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h   # 归一化 xywh -> 像素 xyxy
            ax.add_patch(plt.Rectangle((x0, y0), bw * w, bh * h,
                                       fill=False, edgecolor='#2E8B57', lw=2))
    tag = f'   标注 {len(boxes)} 框' if show_labels else ''
    return f'{split} 第 {index} 张  {fname}{tag}'


# 刻意挑标注多的样本：空图讲不了"两路对齐"
SHOWCASE = ('val', 123)                      # val 里第 123 张，夜里的路口，26 个真值框
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
title = draw_mm(axs, *SHOWCASE, show_labels=False)
fig.suptitle(title, fontsize=12)
plt.show()
"""

C8 = """\
### 同一架无人机的同一时刻，左边是可见光，右边是红外
"""

C9 = """\
fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
title = draw_mm(axs, *SHOWCASE, show_labels=True)
fig.suptitle(title, fontsize=12)
plt.show()
"""

C10 = """\
# 换几张看：从稀疏到密集，对齐与标注共用都不是个例
PICKS = [('val', 149), ('train', 243), ('train', 281)]   # 6 / 32 / 6 个真值框
fig, axs = plt.subplots(3, 2, figsize=(10, 14), constrained_layout=True)
for row, (split, idx) in enumerate(PICKS):
    draw_mm(axs[row], split, idx, show_labels=True)
plt.show()
"""

C11 = """\
## ③ 模型结构：40 层里融合发生在哪

单模态 YOLOv11 是 3 通道进去，骨干出 P3、P4、P5 三个尺度，检测头在这三处解码。

多模态要做的只是把一路骨干变成两路，然后决定在哪把特征合回来。

红外是灰度图，按项目约定复制成 3 通道再进骨干，所以两路的卷积核、融合点、检测头，和单模态那套写法完全一样。
"""

C12 = """\
<img src="assets/structure_rgbt.png" width="62%">

看图的下半部分，从上往下读：

| 层号 | 作用 |
|---|---|
| L0–L10 | 可见光骨干，出 P3(L4) / P4(L6) / P5(L10) |
| L11–L21 | 红外骨干，出 P3(L14) / P4(L17) / P5(L21) |
| L22–L23 | 融合点一：P4 级 `Concat(L6, L17)` → C3k2 |
| L24–L26 | 融合点二：P5 级 `Concat(L10, L21)` → C3k2 → C2PSA |
| L27–L39 | 检测头，与单模态 YOLOv11 完全一样 |
"""

C13 = """\
from ultralytics import YOLOMM

model = YOLOMM('yolo11n-mm-mid.yaml')     

net = model.model
n_par = sum(p.numel() for p in net.parameters())
print(f'总层数 {len(list(net.model))}    参数 {n_par / 1e6:.2f}M')

FOCUS = {0: '可见光入口', 6: '可见光 P4', 10: '可见光 P5',
         11: '红外入口', 17: '红外 P4', 21: '红外 P5',
         22: 'P4 融合 Concat', 23: 'P4 融合 C3k2',
         24: 'P5 融合 Concat', 25: 'P5 融合 C3k2', 26: 'P5 融合 C2PSA',
         39: '检测头 Detect'}
for k, m in net.model.named_children():
    name = type(m).__name__
    if int(k) in FOCUS:
        print(f'  L{int(k):<3} {name:<10} {FOCUS[int(k)]}')
"""

C14 = """\
import torch
CHECK = {0: '可见光入口', 6: '可见光 P4', 10: '可见光 P5',
         11: '红外入口', 17: '红外 P4', 21: '红外 P5',
         22: 'P4 融合 Concat', 23: 'P4 融合输出',
         24: 'P5 融合 Concat', 26: 'P5 融合输出'}

seen = {}
hooks = [m.register_forward_hook(lambda mod, i, o, k=int(k): seen.__setitem__(k, tuple(o.shape)))
         for k, m in model.model.model.named_children() if int(k) in CHECK]

model.model.eval()
with torch.no_grad():
    model.model(torch.zeros(1, 6, 640, 640))   # 6 通道 = 可见光 3 + 红外 3
for h in hooks:
    h.remove()

for k, label in CHECK.items():
    print(f'  L{k:<3} {label:<14} 输出 {seen.get(k)}')
"""

C15 = """\
#### 所以大家可以看到，我们多模态的整个模型，就是在单模态的基础上扩充变化得来的。当然，从模型的结构来说，它从单模态变成多模态的那一刻，就已经和单模态没有关系了。但是这种“藕断丝连”的用法，可以让我们很好地从单模态上手。只要你会 YOLO，会 RT-DETR，就可以很快地上手这个模型，去跑多模态的对应训练。

#### 同时，我也提供了大量的相关配置，你可以很好地去咨询并进行对应的实验，或者让 AI 去给你跑对应的实验都可以，我们做良好的支持
"""

C16 = """\
## ④ 训练 / 验证 / 推理
"""

C17 = """\
model = YOLOMM('yolo11n-mm-mid.yaml')

results = model.train(
    data='/root/datasets/rgbt_uav_demo/uav_rgbt.yaml',
    epochs=2, batch=16, imgsz=640, close_mosaic=0,
    scale='n',
    device='0',
    cache=True,    
    workers=8,
    project='ResTest', name='uav_rgbt_demo',
    exist_ok=True,
    plots=True,
)
"""

C18 = """\
训练日志里先看损失有没有降：

```
epoch 1   box {e1_box}   cls {e1_cls}   dfl {e1_dfl}
epoch 2   box {e2_box}   cls {e2_cls}   dfl {e2_dfl}
```

2 个 epoch 之后的验证结果，在 `ResTest/uav_rgbt_demo/results.csv` 末行：

| epoch | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | {p1} | {r1} | {m1} | {m951} |
| 2 | {p2} | {r2} | {m2} | {m952} |

同样是这份数据、这套配置，跑满 30 epoch 的结果（`ResTest/uav_rgbt30`）：

| epoch | mAP50 | mAP50-95 |
|---|---|---|
| {c1_ep} | {c1_m} | {c1_m95} |
| {c2_ep} | {c2_m} | {c2_m95} |
| {c3_ep} | {c3_m} | {c3_m95} |

2 个 epoch 基本还什么都没学到，验证全是 0 很正常；子集只有 1200 张、目标又小，跑满 30 个 epoch 也才到 0.1 出头，加数据量、上更大的 scale，曲线还会继续涨。
"""

C19 = """\
metrics = model.val(
    data='/root/datasets/rgbt_uav_demo/uav_rgbt.yaml',
    split='val',
    device='0',
    project='ResTest', name='uav_rgbt_demo_val',
    exist_ok=True,
)

print(f'mAP50     {metrics.box.map50:.4f}')
print(f'mAP50-95  {metrics.box.map:.4f}')
print(f'P / R     {metrics.box.mp:.4f} / {metrics.box.mr:.4f}')
"""

C20 = """\
STEMS = ['DJI_0181_3_00060',      # 26 个真值 / 夜里路口
         'DJI_0237_2_00089',      # 6 个真值 / 稀疏
         'DJI_0073_3_00269']      # 46 个真值 / 密集
picks = [ROOT / 'images' / 'val' / f'{s}.jpg' for s in STEMS]
sources = {
    'A': [str(p) for p in picks],
    'B': [str(ROOT / 'images_thermal' / 'val' / (p.stem + '.jpg')) for p in picks],
}

# 刚训 2 epoch 的 model 几乎不出框，演示改用已经跑满的权重
demo = YOLOMM('/root/ultralyticsmm/ResTest/uav_rgbt30/weights/best.pt')
preds = demo.predict(sources=sources, device='0', conf=0.25, verbose=False)
for p, r in zip(picks, preds):
    print(f'{p.stem[:34]}   {len(r.boxes)} 框')
"""

C21 = """\
# plot() 返回 {槽位: 叠加图} 的字典；plot_merged() 把两路并排成一张
fig, axs = plt.subplots(3, 2, figsize=(13, 15), constrained_layout=True)
for i, (p, r) in enumerate(zip(picks, preds)):
    merged = r.plot_merged()                       # (H, 2W, 3)：左可见光 右红外
    axs[i][0].imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
    axs[i][0].set_title(f'可见光 输入   {p.name[:28]}', fontsize=9)
    axs[i][0].axis('off')
    axs[i][1].imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
    axs[i][1].set_title(f'两路预测并排   {len(r.boxes)} 框', fontsize=9)
    axs[i][1].axis('off')
plt.show()
"""

C22 = """\
#### 最后我们收个尾，如果是说想做无人机检测，包括本文提及我们讲的一个航拍小目标检测，或者说你想做夜间巡检、海上搜救，都可以考虑用多模态的方法来做。尤其是你现在已经做了无人机，你觉得这个无人机啊，现在单模态很不好发，确实也不太好发，那你想把它去做进一步的扩展，或者说把它变得很好发，那你完全可以用来我们的多模态方法，用可见光加红外的这种结合形式。

#### 只需要你给一个可见光加红外的数据集（手头没有红外，就退一步做 RGBD，深度我们可以通过项目来自动生成），那你就可以得到整个的一个多模态数据集，然后用多模态的方法来做，那你的论文的创新会比单模态你单纯去甩一个文章要更好。

#### 而且你这样子其实本质上他们大差不差，在多模态里面做的你也是一个缝合叠加的工作，但是从收益上来说，你就得到了不一样的角度。
"""

CELLS = [
    ('markdown', TITLE),
    ('markdown', C1),
    ('markdown', C2),
    ('markdown', C3),
    ('markdown', C4),
    ('markdown', C5),
    ('markdown', C5B),
    ('code', C6),
    ('code', C7),
    ('markdown', C8),
    ('code', C9),
    ('code', C10),
    ('markdown', C11),
    ('markdown', C12),
    ('code', C13),
    ('code', C14),
    ('markdown', C15),
    ('markdown', C16),
    ('code', C17),
    ('markdown', C18),
    ('code', C19),
    ('code', C20),
    ('code', C21),
    ('markdown', C22),
]

nb = json.loads(SRC.read_text(encoding='utf-8'))
assert nb['nbformat'] == 4
nb['cells'] = []
for ctype, src in CELLS:
    cell = {'cell_type': ctype, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if ctype == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    nb['cells'].append(cell)

nb['metadata']['kernelspec'] = {'display_name': 'yolon_bak', 'language': 'python', 'name': 'python3'}
DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote', DST, len(nb['cells']), 'cells')
