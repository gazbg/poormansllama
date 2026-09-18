# T-GPLR-43 qa-smoke closure arm: the recorded 512 smoke shape + --pp-repeat 2
# so the default per-pass rebuild (default ON only for repeat > 1) is actually
# exercised. Same model/anchor/flags as run-tests.ps1's qa-smoke except the
# repeat count. Graded with the same three steps: strict --check-telemetry,
# answer checker PASS, negative control FAIL.
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe   = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
$chk   = Join-Path $root 'tools\semantic-qa\check-answer.py'
$base  = Join-Path $root 'build\llama-cuda'
$prompt = Join-Path $base 'gplr07-qa-prompts\anchor-512.txt'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$trace = Join-Path $base 'gplr07-qa-512-repeat2.jsonl'
$tgtext = Join-Path $base 'gplr07-qa-512-repeat2.txt'
$log   = Join-Path $base 'gplr07-qa-512-repeat2.log'
Remove-Item $trace, $tgtext, $log -ErrorAction SilentlyContinue

$runCmd = "`"$exe`" --model `"$model`" --prompt-file `"$prompt`" --ubatch 512 --tg-tokens 96 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --no-handoff --allow-full-residency --pp-repeat 2 --telemetry `"$trace`" --tg-text `"$tgtext`" > `"$log`" 2>&1"
& $lock -TimeoutSeconds 1800 -Command $runCmd
Write-Host ("RUNNER EXIT: {0}" -f $LASTEXITCODE)

& $exe --check-telemetry $trace
Write-Host ("CHECK EXIT: {0}" -f $LASTEXITCODE)

python $chk check --tg-text $tgtext --items N1,N2,N3,N4,K1
Write-Host ("ANSWER EXIT: {0}" -f $LASTEXITCODE)

$neg = & python $chk check --tg-text $tgtext --items N1,N2,N3,N4,K1 --negative 2>&1
Write-Host $neg
Write-Host ("NEGATIVE EXIT: {0} (must be nonzero, aggregate=FAIL)" -f $LASTEXITCODE)
