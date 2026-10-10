# T-GPLR-61 step-C evidence: exercise the host_finite non-finite abort for real
# by injecting one NaN at the first scanned element via LLAMA_LS_TEST_NONFINITE
# (test-only fault injection in the scan; unset = inert). Arm-C shaped flags,
# no --mtp. Under the GPU lock: the run loads the model and computes for real.
# -Suffix appends to every output/verdict tag so post-refactor evidence can be
# written to fresh files; default '' reproduces the recorded commands exactly.
param([switch] $Control, [string] $Suffix = '')
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$anchor = Join-Path $root 'build/llama-cuda/gplr61-ac3-anchor-512.txt'
$tag = if ($Control) { 'gplr61-nonfinite-control' } else { 'gplr61-nonfinite' }
$tag = $tag + $Suffix
$tr = Join-Path $root ('build/llama-cuda/' + $tag + '.jsonl')
$tx = Join-Path $root ('build/llama-cuda/' + $tag + '-tg.txt')
$errlog = Join-Path $root ('build/llama-cuda/' + $tag + '.err')
$outlog = Join-Path $root ('build/llama-cuda/' + $tag + '.out')
$inject = if ($Control) { '' } else { 'set LLAMA_LS_TEST_NONFINITE=1&& ' }
$cmd = ($inject + '"' + $exe + '" --model ' + $model +
        ' --prompt-file ' + $anchor +
        ' --ubatch 512 --tg-tokens 96 --pp-no-logits --pp-schedule batch-major' +
        ' --residency moe-expert-bank --kv-cache q8 --allow-full-residency' +
        ' --no-handoff --residency-tg --experts-host 2 --experts-host-budget inf' +
        ' --telemetry ' + $tr + ' --tg-text ' + $tx +
        ' 1> ' + $outlog + ' 2> ' + $errlog)
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'with-gpu-lock.ps1') -TimeoutSeconds 1800 -Command $cmd -VerdictTag $tag
exit $LASTEXITCODE
