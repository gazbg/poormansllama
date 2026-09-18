# T-GPLR-37 stage 1 (inner arm, no lock): the mmap-registration probe. Tiny GPU
# footprint (one 453 MiB staging buffer, ~10 s). Bank-sized window mapped from
# the real MoE GGUF at offset 1 GiB. Run it through run-gplr37-probe-locked.ps1
# so it holds the shared-GPU mutex.
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe   = Join-Path $root 'build/gplr37-mmapreg-probe.exe'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$log   = Join-Path $root 'build/llama-cuda/gplr37-probe.log'
$bytes = '474480640'   # 453 MiB, the measured bank size
$off   = '1073741824'  # 1 GiB into the file

& $exe $model $bytes $off *> $log
Write-Host ("PROBE EXIT: {0}" -f $LASTEXITCODE)
Get-Content $log
