$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$notebookPath = Join-Path $projectRoot "notebooks\experiments.ipynb"

$env:IPYTHONDIR = Join-Path $projectRoot ".local-ipython"
$env:JUPYTER_CONFIG_DIR = Join-Path $projectRoot ".local-jupyter-config"
$env:JUPYTER_DATA_DIR = Join-Path $projectRoot ".local-jupyter-data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $projectRoot ".local-jupyter-runtime"
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"

New-Item -ItemType Directory -Force -Path $env:IPYTHONDIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:JUPYTER_CONFIG_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:JUPYTER_DATA_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $env:JUPYTER_RUNTIME_DIR | Out-Null

# Register a project-local kernel so the notebook uses this .venv on any machine.
$kernelDir = Join-Path $env:JUPYTER_DATA_DIR "kernels\social-media-sentiment-analysis"
New-Item -ItemType Directory -Force -Path $kernelDir | Out-Null

$kernelSpec = @{
    argv = @(
        $pythonExe,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}"
    )
    display_name = "Python (.venv) - Social Media Sentiment Analysis"
    language = "python"
    metadata = @{
        debugger = $true
    }
}

$kernelSpec | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $kernelDir "kernel.json")

& $pythonExe -m notebook $notebookPath
