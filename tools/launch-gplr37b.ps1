# T-GPLR-37 churn-free launcher: ~15 min (four process loads + the candidates'
# pre-timed registration of ~17 GiB of file pages), so start it detached and poll
# build/llama-cuda/gplr37b-wrapper.log.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr37b-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr37b-wrapper.err.log'
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools/run-gplr37b.ps1') `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -WindowStyle Hidden -PassThru
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
