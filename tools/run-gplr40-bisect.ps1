# T-GPLR-40 wrapper: one mutex hold for the two bisect arms (~25 min worst case:
# each arm is pass 1 (~80 s) + pass 2 (crashes ~37% in, or completes at ~160 s)).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr40-bisect-arms.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 2400 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
