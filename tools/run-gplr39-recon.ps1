# T-GPLR-39 recon arm: one instrumented run at the regressed point of the
# T-GPLR-38 sweep (u2048, 3 slots, ron mode, 4k anchor) to attribute which part
# of the `other` ledger bucket grows with ubatch size. The ledger now emits
# route/tel/loop sub-buckets (recon-only, checker-agnostic fields).
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

$trace = Join-Path $base 'gplr39-u2048-recon.jsonl'
$log   = Join-Path $base 'gplr39-u2048-recon.log'
Remove-Item $trace, $log -ErrorAction SilentlyContinue
$a = @('--model', $model, '--prompt-file', $prompt,
       '--ubatch', '2048', '--tg-tokens', '96', '--pp-no-logits',
       '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--pp-repeat', '2', '--copy-source', 'mmap-registered',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)
Write-Host '[gplr39] arm u2048-recon starting...'
& $exe @a > $log 2>&1
Write-Host ("ARM EXIT: {0}" -f $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
  Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
  exit 1
}
Write-Host '[gplr39] strict check-telemetry...'
& $exe --check-telemetry $trace
Write-Host ("CHECK EXIT: {0}" -f $LASTEXITCODE)
# Ledger sub-bucket report (closure: other == route + tel + loop)
python (Join-Path $root 'tools/analyze-gplr39.py') $trace
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)
Write-Host 'DONE'
