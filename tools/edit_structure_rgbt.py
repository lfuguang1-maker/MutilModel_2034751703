"""把 assets/structure.png 里的「深度」相关标注改成「红外」，产出 assets/structure_rgbt.png。
只做像素级替换：擦掉原文字，按原字体、原字号、原位置重画。
通道数不动（RGBT 的红外按项目约定复制成 3 通道，和可见光拼成 6 通道）。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path('/root/ultralyticsmm/assets/structure.png')
DST = Path('/root/ultralyticsmm/assets/structure_rgbt.png')
FONT_R = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

im = Image.open(SRC).convert('RGB')
px = np.asarray(im).copy()
print('image', im.size)


def ink_bbox(box, bg, thr=40):
    x0, y0, x1, y1 = box
    sub = px[y0:y1, x0:x1].astype(int)
    d = np.abs(sub - np.array(bg, dtype=int)).max(2)
    ys, xs = np.where(d > thr)
    if len(xs) == 0:
        return None
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def stroke_color(bb):
    x0, y0, x1, y1 = bb
    sub = px[y0:y1, x0:x1].astype(int)
    lum = sub.sum(2)
    iy, ix = np.unravel_index(np.argmin(lum), lum.shape)
    return tuple(int(v) for v in sub[iy, ix])


def fit_size(text, target_w, target_h, bold=False):
    best = None
    for size in range(14, 60):
        f = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)
        b = f.getbbox(text)
        w, h = b[2] - b[0], b[3] - b[1]
        err = abs(w - target_w) + abs(h - target_h)
        if best is None or err < best[0]:
            best = (err, size, w, h)
    return best[1], best[2], best[3]


def draw_ink(draw, text, center, font, color):
    b = font.getbbox(text)
    w, h = b[2] - b[0], b[3] - b[1]
    draw.text((center[0] - w / 2 - b[0], center[1] - h / 2 - b[1]), text, font=font, fill=color)


def rewrite(label_box, bg, new_text, bold=False, size_ref=None, pad_x=12, pad_top=6, pad_bot=6):
    bb = ink_bbox(label_box, bg)
    color = stroke_color(bb)
    d.rectangle((bb[0] - pad_x, bb[1] - pad_top, bb[2] + pad_x, bb[3] + pad_bot), fill=bg)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    size = fit_size(*size_ref, bold=bold)[0] if size_ref else fit_size(new_text, tw, th, bold=bold)[0]
    font = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=0)
    center = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    draw_ink(d, new_text, center, font, color)
    b = font.getbbox(new_text)
    print(f'  {new_text!r} size={size} ink={b[2]-b[0]}x{b[3]-b[1]} 旧ink={tw}x{th} center={center}')


d = ImageDraw.Draw(im)

# ---------- 1. 三个面板顶部的「深度」-> 「红外」 ----------
for bx in (513, 1467, 2420):
    bg = tuple(int(v) for v in px[252, bx + 25])
    print('top label, bg', bg)
    rewrite((bx + 20, 236, bx + 350, 274), bg, '红外', bold=True,
            size_ref=('深度', 41, 21), pad_top=8, pad_bot=8)

# ---------- 2. 右上槽位框第一行「深度 images_depth/」-> 「红外 images_thermal/」 ----------
# 下面一行（3 ch · 槽位 B）原样保留，所以纵向只擦 4px，别碰下一行
bg = tuple(int(v) for v in px[1000, 1955])
print('slot box, bg', bg)
rewrite((1950, 916, 2740, 946), bg, '红外 images_thermal/', bold=False,
        size_ref=('深度', 41, 21), pad_x=14, pad_top=6, pad_bot=4)

im.save(DST)
print('saved', DST, im.size)
