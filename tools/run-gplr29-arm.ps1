# Relocated from build/run-gplr29-arm.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 arm body: the probe invocation itself. Invoked by file path from
# run-gplr29-pinprobe.ps1 under one GPU-mutex hold.
$exe   = (Join-Path $root 'build/gplr29-pinprobe.exe')
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$log   = (Join-Path $root 'build/llama-cuda/gplr29-pinprobe.log')

& $exe $model 1073741824 453000000 5 > $log 2>&1
"PROBE EXIT: $LASTEXITCODE"
