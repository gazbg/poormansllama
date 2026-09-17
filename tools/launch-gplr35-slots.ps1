$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root 'build/llama-cuda/gplr35-slots-wrapper.log'
$err  = Join-Path $root 'build/llama-cuda/gplr35-slots-wrapper.err.log'
$p = Start-Process powershell `
    -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $root 'tools/run-gplr35-slots-locked.ps1') `
    -RedirectStandardOutput $out -RedirectStandardError $err `
    -WindowStyle Hidden -PassThru
Write-Host ("LAUNCHED PID {0}" -f $p.Id)
