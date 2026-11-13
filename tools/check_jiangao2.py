"""讲稿2.ipynb 的体检：结构、输出、错误、残留旧领域词。"""
import json
import re
from pathlib import Path

NB = Path('/root/ultralyticsmm/讲稿2.ipynb')
ASSETS = Path('/root/ultralyticsmm/assets')
nb = json.loads(NB.read_text(encoding='utf-8'))
cells = nb['cells']
code = [c for c in cells if c['cell_type'] == 'code']
md = [c for c in cells if c['cell_type'] == 'markdown']
print('cells', len(cells), 'code', len(code), 'markdown', len(md))

bad = 0
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    outs = c.get('outputs', [])
    errs = [o['ename'] for o in outs if o['output_type'] == 'error']
    imgs = [o for o in outs if o['output_type'] in ('display_data', 'execute_result')
            and 'image/png' in o.get('data', {})]
    txt = sum(len(o.get('text', '')) if isinstance(o.get('text'), str) else 0 for o in outs)
    print(f'  cell {i:2d}  outputs={len(outs):2d} img={len(imgs)} err={errs} exec={c.get("execution_count")}')
    if errs or not outs:
        bad += 1

alltext = '\n'.join(''.join(c['source']) for c in md)
print('\n-- 残留旧领域词 --')
for kw in ['pfruitlet', 'fruitlet', '果园', '幼果', '果树', '农业', 'jiagao', 'depth[B]']:
    hits = [m.start() for m in re.finditer(kw, alltext)]
    print(f'  {kw}: {len(hits)}')
print('\n-- 图片引用 --')
for m in re.finditer(r'assets/([\w.]+\.png)', alltext):
    p = ASSETS / m.group(1)
    print(f'  {m.group(1)}: exists={p.exists()} size={p.stat().st_size if p.exists() else 0}')

print('\n-- 训练/验证数字 --')
code_txt = '\n'.join(''.join(c['source']) for c in code)
for kw in ['uav_rgbt_demo', 'uav_rgbt30', 'images_thermal', 'RGBT']:
    print(f'  {kw}: 出现在本次讲稿 {alltext.count(kw)+code_txt.count(kw)} 次')

print('\nBAD CELLS', bad)
