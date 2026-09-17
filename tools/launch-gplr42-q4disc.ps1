# T-GPLR-42 launcher: start the mutex wrapper detached; poll
# build/llama-cuda/gplr42-wrapper.log for progress.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr42-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr42-wrapper.err.log'
Remove-Item $out, $err -ErrorAction SilentlyContinue
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools\run-gplr42-q4disc.ps1') `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err `
    -WindowStyle Hidden
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
