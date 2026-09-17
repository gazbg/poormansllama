# T-GPLR-41 wrapper: one mutex hold for both arms (~15 min total).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr41-mctx-arms.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 2400 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
