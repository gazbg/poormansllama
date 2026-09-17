# T-GPLR-43 diagnostic inner: tiny 4k x --pp-repeat 2 q8 run; stderr carries
# the per-set CREATED/reused decisions from the instrumented begin_set.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'
$trace  = Join-Path $base 'gplr43-diag.jsonl'
$log    = Join-Path $base 'gplr43-diag.log'
Remove-Item $trace, $log -ErrorAction SilentlyContinue
$a = @('--model', $model, '--prompt-file', $prompt,
       '--pp-tokens', '4096', '--ctx-size', '4608',
       '--ubatch', '512', '--tg-tokens', '0', '--pp-no-logits',
       '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '2', '--allow-full-residency',
       '--pp-repeat', '2', '--copy-source', 'mmap-registered',
       '--pp-async-feed', '1',
       '--telemetry', $trace)
& $exe @a > $log 2>&1
Write-Host ("INNER EXIT: {0}" -f $LASTEXITCODE)
$created = (Select-String -Path $log -Pattern 'CREATED' -Encoding Unicode).Count
$reused  = (Select-String -Path $log -Pattern 'reused' -Encoding Unicode).Count
Write-Host ("SET decisions: CREATED={0} reused={1}" -f $created, $reused)
