# T-GPLR-36 launcher: the ledger arm runs ~10 min (two 146 s passes + load), so
# start the mutex wrapper detached and poll build/llama-cuda/gplr36-wrapper.log.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr36-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr36-wrapper.err.log'
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools/run-gplr36-ledger-locked.ps1') `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -WindowStyle Hidden -PassThru
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
