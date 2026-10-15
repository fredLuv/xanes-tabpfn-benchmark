"""Execute the report with this interpreter; no global Jupyter kernel required."""
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as temp:
    import json
    kernel=Path(temp)/'kernels/reproduction';kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps(dict(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],display_name='Reproduction',language='python')))
    os.environ['JUPYTER_PATH']=temp+os.pathsep+os.environ.get('JUPYTER_PATH','')
    path=ROOT/'notebooks/01_benchmark_results.ipynb'
    notebook=nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=600,kernel_name='reproduction',resources={'metadata':{'path':str(ROOT)}}).execute()
    notebook.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(notebook,path)
    print('Executed all report cells, rebuilding figures from saved predictions.')
