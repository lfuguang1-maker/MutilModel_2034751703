"""用 jupyter_client 驱动 kernel 执行 notebook 的 code cell，把真实输出写回 ipynb。
（环境里没有 nbclient / nbconvert，这里手写一个最小执行器。）
"""
import json
import sys
import time
from pathlib import Path
from queue import Empty

from jupyter_client import KernelManager

NB = Path(sys.argv[1])
CWD = sys.argv[2] if len(sys.argv) > 2 else '/root/ultralyticsmm'
CELL_TIMEOUT = int(sys.argv[3]) if len(sys.argv) > 3 else 5400

nb = json.loads(NB.read_text(encoding='utf-8'))
km = KernelManager(kernel_name='python3')
km.start_kernel(cwd=CWD)
kc = km.client()
kc.start_channels()
kc.wait_for_ready(timeout=180)


def drain_ready():
    while True:
        try:
            kc.get_iopub_msg(timeout=0.2)
        except Empty:
            return


def run(code, timeout=CELL_TIMEOUT):
    msg_id = kc.execute(code)
    outs = []
    deadline = time.time() + timeout
    while True:
        left = max(1, deadline - time.time())
        try:
            msg = kc.get_iopub_msg(timeout=min(left, 60))
        except Empty:
            if time.time() > deadline:
                return outs, None, 'timeout'
            continue
        if msg['parent_header'].get('msg_id') != msg_id:
            continue
        mt, c = msg['msg_type'], msg['content']
        if mt == 'status' and c.get('execution_state') == 'idle':
            break
        if mt == 'stream':
            outs.append({'name': c['name'], 'output_type': 'stream', 'text': c['text']})
        elif mt in ('display_data', 'execute_result'):
            outs.append({'output_type': mt, 'data': c['data'], 'metadata': c.get('metadata', {})})
        elif mt == 'error':
            outs.append({'output_type': 'error', 'ename': c['ename'], 'evalue': c['evalue'],
                         'traceback': c['traceback']})
        elif mt == 'clear_output':
            outs = []
    try:
        rep = kc.get_shell_msg(timeout=120)
    except Empty:
        return outs, None, 'no-reply'
    cnt = rep['content'].get('execution_count')
    status = rep['content'].get('status')
    return outs, cnt, status


# 让 matplotlib 走 inline 后端（notebook 里没写 %matplotlib inline）
run('%matplotlib inline')
run('import matplotlib; print("backend:", matplotlib.get_backend())')
drain_ready()

status_log = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    t0 = time.time()
    outs, cnt, status = run(src)
    cell['outputs'] = outs
    cell['execution_count'] = cnt
    dt = time.time() - t0
    err = [o['ename'] for o in outs if o['output_type'] == 'error']
    print(f'[cell {i}] {status} {dt:.1f}s outputs={len(outs)} err={err}', flush=True)
    status_log.append((i, status, err))
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    if err:
        print('  !! error, stop here', flush=True)
        break

print('DONE', status_log)
kc.stop_channels()
km.shutdown_kernel(now=True)
