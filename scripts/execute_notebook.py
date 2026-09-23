"""Execute plain-Python notebook cells sequentially without network sockets.

This lightweight runner supports this notebook's Python cells and matplotlib
figures. It is not a general replacement for Jupyter (no magics or widgets).
The notebook also runs normally in Jupyter with the project's Python kernel.
"""
from pathlib import Path
import os,sys,ast,io,contextlib,base64,time
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'work/cache'))
os.chdir(ROOT)
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.core.formatters import DisplayFormatter
path=ROOT/'notebooks/competition-pipeline.ipynb'
n=nbformat.read(path,as_version=4);scope={'__name__':'__main__'};formatter=DisplayFormatter();count=0
for idx,cell in enumerate(n.cells):
    if cell.cell_type!='code':continue
    count+=1;outputs=[];stdout=io.StringIO();stderr=io.StringIO()
    print(f'Executing cell {idx+1}/{len(n.cells)}',flush=True)
    def show(*args,**kwargs):
        for num in plt.get_fignums():
            fig=plt.figure(num);b=io.BytesIO();fig.savefig(b,format='png',dpi=100,bbox_inches='tight')
            outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'Matplotlib figure'},metadata={}))
        plt.close('all')
    plt.show=show
    tree=ast.parse(cell.source)
    expr=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
    with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
        exec(compile(tree,f'cell-{idx+1}','exec'),scope)
        if expr:
            result=eval(compile(ast.Expression(expr.value),f'cell-{idx+1}','eval'),scope)
            if result is not None:
                data,metadata=formatter.format(result)
                outputs.append(nbformat.v4.new_output('execute_result',data=data,metadata=metadata,execution_count=count))
    if stdout.getvalue():outputs.insert(0,nbformat.v4.new_output('stream',name='stdout',text=stdout.getvalue()))
    if stderr.getvalue():outputs.append(nbformat.v4.new_output('stream',name='stderr',text=stderr.getvalue()))
    cell.outputs=outputs;cell.execution_count=count
    nbformat.write(n,path)
n.metadata['execution_note']='Executed sequentially by scripts/execute_notebook.py without a socket-based Jupyter kernel.'
nbformat.validate(n);nbformat.write(n,path)
print(f'Notebook execution passed: {count} code cells',flush=True)
