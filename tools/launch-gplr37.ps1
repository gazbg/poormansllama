# T-GPLR-37 launcher: the four 4k arms run ~12 min (four process loads), so start
# the mutex wrapper detached and poll build/llama-cuda/gplr37-wrapper.log.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr37-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr37-wrapper.err.log'
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools/run-gplr37.ps1') `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -WindowStyle Hidden -PassThru
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
