# T-GPLR-61 step-C evidence: actually invoke `--mtp --experts-host 2` so the
# experts_host_with_mtp guard (main.cpp:5138) fires for real. The guard sits
# AFTER llama_model_load_from_file (main.cpp:4474) despite its comment, so the
# run goes through the GPU lock with the wrapper's nvidia-smi probes.
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$anchor = Join-Path $root 'build/llama-cuda/gplr61-ac3-anchor-512.txt'
$tr = Join-Path $root 'build/llama-cuda/gplr61-mtp-refusal.jsonl'
$tx = Join-Path $root 'build/llama-cuda/gplr61-mtp-refusal-tg.txt'
$errlog = Join-Path $root 'build/llama-cuda/gplr61-mtp-refusal.err'
$outlog = Join-Path $root 'build/llama-cuda/gplr61-mtp-refusal.out'
$cmd = ('"' + $exe + '" --model ' + $model +
        ' --prompt-file ' + $anchor +
        ' --ubatch 512 --tg-tokens 96 --pp-no-logits --pp-schedule batch-major' +
        ' --residency moe-expert-bank --kv-cache q8 --allow-full-residency' +
        ' --no-handoff --residency-tg --experts-host 2 --experts-host-budget inf' +
        ' --mtp --telemetry ' + $tr + ' --tg-text ' + $tx +
        ' 1> ' + $outlog + ' 2> ' + $errlog)
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'with-gpu-lock.ps1') -TimeoutSeconds 1800 -Command $cmd -VerdictTag 'gplr61-mtp-refusal'
exit $LASTEXITCODE
