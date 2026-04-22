$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$notebookPath = Join-Path $projectRoot "notebooks\experiments.ipynb"

$env:IPYTHONDIR = Join-Path $projectRoot ".ipython"
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot ".jupyter"
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot ".jupyter-data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot ".jupyter-runtime"
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"

& $pythonExe -m notebook $notebookPath
