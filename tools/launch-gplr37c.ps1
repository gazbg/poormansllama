# T-GPLR-37 addendum 3 launcher: the four 200k arms take ~80-90 min total, so
# start the wrapper detached and poll build/llama-cuda/gplr37c-wrapper.log.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr37c-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr37c-wrapper.err.log'
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools/run-gplr37c.ps1') `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -WindowStyle Hidden -PassThru
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
