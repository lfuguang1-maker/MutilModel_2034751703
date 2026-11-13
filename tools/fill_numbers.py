"""把真实训练/验证数字填进 讲稿2.ipynb 的 cell 19，并重排执行序号。
数字来源：ResTest/uav_rgbt_demo（notebook 里跑的 2 epoch）与 ResTest/uav_rgbt30（后台跑满 30 epoch）。
"""
import csv
import json
from pathlib import Path

ROOT = Path('/root/ultralyticsmm')
NB = ROOT / '讲稿2.ipynb'


def load(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    return rows


def num(r, key):
    return float(r[key])


demo = load(ROOT / 'ResTest/uav_rgbt_demo/results.csv')
full = load(ROOT / 'ResTest/uav_rgbt30/results.csv')
print('demo cols', list(demo[0].keys()))
print('demo rows', len(demo), 'full rows', len(full))


def get(row, name):
    for k in row:
        if k.strip() == name or k.strip().startswith(name):
            return row[k]
    raise KeyError(name)


def cell_demo(r):
    return dict(box=get(r, 'train/box_loss'), cls=get(r, 'train/cls_loss'),
                dfl=get(r, 'train/dfl_loss'), p=get(r, 'metrics/precision(B)'),
                r=get(r, 'metrics/recall(B)'), m=get(r, 'metrics/mAP50(B)'),
                m95=get(r, 'metrics/mAP50-95(B)'))


d = {int(get(r, 'epoch')): cell_demo(r) for r in demo}
print('demo epochs', sorted(d))


def f3(x):
    return f'{float(x):.3f}'


def f4(x):
    return f'{float(x):.4f}'


nb = json.loads(NB.read_text(encoding='utf-8'))
cell = nb['cells'][19]
src = ''.join(cell['source'])
assert '{e1_box}' in src, 'placeholder cell not found'

repl = {
    '{e1_box}': f3(d[1]['box']), '{e1_cls}': f3(d[1]['cls']), '{e1_dfl}': f3(d[1]['dfl']),
    '{e2_box}': f3(d[2]['box']), '{e2_cls}': f3(d[2]['cls']), '{e2_dfl}': f3(d[2]['dfl']),
    '{p1}': f4(d[1]['p']), '{r1}': f4(d[1]['r']), '{m1}': f4(d[1]['m']), '{m951}': f4(d[1]['m95']),
    '{p2}': f4(d[2]['p']), '{r2}': f4(d[2]['r']), '{m2}': f4(d[2]['m']), '{m952}': f4(d[2]['m95']),
}
# 30 epoch 曲线上取 10 / 20 / 30
picks = {int(get(r, 'epoch')): r for r in full}
curve = [(e, f4(get(picks[e], 'metrics/mAP50(B)')), f4(get(picks[e], 'metrics/mAP50-95(B)')))
         for e in (10, 20, 30)]
repl.update({
    '{c1_ep}': str(curve[0][0]), '{c1_m}': curve[0][1], '{c1_m95}': curve[0][2],
    '{c2_ep}': str(curve[1][0]), '{c2_m}': curve[1][1], '{c2_m95}': curve[1][2],
    '{c3_ep}': str(curve[2][0]), '{c3_m}': curve[2][1], '{c3_m95}': curve[2][2],
})
for k, v in repl.items():
    src = src.replace(k, v)
assert '{' not in src, f'unfilled placeholder: {src}'
cell['source'] = src.splitlines(keepends=True)

# 重排执行序号
n = 0
for c in nb['cells']:
    if c['cell_type'] == 'code':
        n += 1
        c['execution_count'] = n
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(src)
print('saved', NB)
