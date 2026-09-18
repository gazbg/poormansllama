# T-GPLR-37 stage 1 wrapper: one mutex hold for the mmap-registration probe.
# The mutex wrapper's -Command is executed by cmd.exe, so the inner arm is
# launched as a PowerShell FILE (the established pattern in this repo).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools/run-gplr37-probe.ps1'
$lock  = Join-Path $root 'tools/with-gpu-lock.ps1'
& $lock -TimeoutSeconds 600 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
