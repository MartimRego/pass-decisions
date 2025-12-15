import json, subprocess, tempfile, os, sys
nb='pass_decision_analysis.ipynb'
with open(nb) as f:
    nb_json = json.load(f)
cells = nb_json['cells']
root = os.getcwd()
py_exe = '/home/macaco3001/Projectos/Twelve/twelve-deep-learning/.venv/bin/python'
print('Using python:', py_exe)
for i,cell in enumerate(cells[:120]):
    if cell.get('cell_type')!='code':
        continue
    src=''.join(cell.get('source',[]))
    if not src.strip():
        continue
    # prepare wrapper
    wrapper = f"""
import sys, os
sys.path.insert(0, r'{root}')
import matplotlib
matplotlib.use('Agg')
# cell {i}
{src}
print('CELL_{i}_OK')
"""
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tf:
        tf.write(wrapper)
        tmpfn = tf.name
    print(f'Running cell {i} (len={len(src)} chars) -> {tmpfn}')
    try:
        p = subprocess.run([py_exe, tmpfn], capture_output=True, text=True, timeout=120)
    except Exception as e:
        print('Cell',i,'execution error:',e)
        raise
    print('Returncode', p.returncode)
    if p.stdout:
        print('STDOUT:\n', p.stdout[:1000])
    if p.stderr:
        print('STDERR:\n', p.stderr[:2000])
    if p.returncode!=0:
        print('Cell failed at index',i)
        print('Stopping further checks')
        sys.exit(2)
    os.unlink(tmpfn)
print('Checked first 120 code cells, none failed in isolated runs')
