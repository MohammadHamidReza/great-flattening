$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = "python"
$env:ARXIV_PARQUET = "data\arxiv_metadata.parquet"
$env:RESULTS_DIR = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path (Join-Path $root "logs") | Out-Null

& $py (Join-Path $root "analysis\01_inspect.py") 2>&1 | Tee-Object (Join-Path $root "logs\01_inspect.log")
& $py (Join-Path $root "analysis\02_build_sample.py") 2>&1 | Tee-Object (Join-Path $root "logs\02_build_sample.log")
& $py (Join-Path $root "analysis\03_metrics.py") 2>&1 | Tee-Object (Join-Path $root "logs\03_metrics.log")
& $py (Join-Path $root "analysis\04_models.py") 2>&1 | Tee-Object (Join-Path $root "logs\04_models.log")
& $py (Join-Path $root "analysis\05_figures.py") 2>&1 | Tee-Object (Join-Path $root "logs\05_figures.log")
& $py (Join-Path $root "analysis\06_truncated_semantic.py") 2>&1 | Tee-Object (Join-Path $root "logs\06_truncated_semantic.log")
