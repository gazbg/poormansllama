# T-GPLR-37 addendum 3 wrapper: one mutex hold for the four 200k arms
# (~10 min per pass x 2 passes x 4 arms = ~80-90 min measured budget; the lock
# timeout covers the worst case plus the external co-tenant's hold).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr37c-arms.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 7200 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
