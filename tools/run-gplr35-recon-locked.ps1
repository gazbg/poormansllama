# T-GPLR-35 recon outer wrapper: acquire the shared-GPU mutex once, then run the
# two recon arms (ov0 = dedicated copy backend's own stream, ov1 = aux stream
# inside the compute context) plus strict check-telemetry on both traces.
# Short by design: --pp-repeat 2 (cold + warm), ~6-8 min total on the MoE vehicle.
# Analyzer: tools/analyze-gplr35-recon.py
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr35-recon.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 1800 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
