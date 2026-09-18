# T-GPLR-36 follow-up wrapper: one mutex hold for the 200k host-memory arm
# (~10 min of run, up to 60 min of wait for the shared GPU -- raised from 900 s
# on 2026-09-18 because the DeepSeek V4 baseline project was mid-run under the
# same mutex and a 15 min window would have expired before it released).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr36-hostmem.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 3600 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
