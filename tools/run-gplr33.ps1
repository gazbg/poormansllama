# T-GPLR-33 outer wrapper: acquire the shared-GPU mutex once, then run the
# two arms (batch0 baseline vs batch1 one-H2D-per-bank) plus strict
# check-telemetry on both traces. ~20 min total (3 passes x 4k x 2 arms on
# the MoE vehicle). Analyzer: tools/analyze-gplr33.py batch0 batch1.
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr33-arms.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 1800 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
