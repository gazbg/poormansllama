# T-GPLR-32 outer wrapper: acquire the shared-GPU mutex once, then run the
# two arms (ser0 baseline vs ser1 one-DMA-in-flight) plus strict
# check-telemetry on both traces. ~20 min total (3 passes x 4k x 2 arms on
# the MoE vehicle). Analyzer: tools/analyze-gplr32.py ser0 ser1.
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr32-arms.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 1800 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
