# Relocated from build/run-gplr29-af1direct.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 attribution control: af=1 with DIRECT copy source (no pinned), so
# the af1-pinned gain can be split into "overlap itself" vs "pinned chain".
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'
$trace  = Join-Path $base 'gplr29-afab-af1direct.jsonl'
$log    = Join-Path $base 'gplr29-afab-af1direct.log'
$tg     = Join-Path $base 'gplr29-afab-af1direct.tgtext'

Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
$a = @('--model', $model, '--prompt-file', $prompt,
       '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
       '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
       '--layer-slots', '2', '--allow-full-residency',
       '--pp-repeat', '3', '--pp-async-feed', '1',
       '--telemetry', $trace, '--tg-text', $tg)
& $exe @a > $log 2>&1
"ARM af1direct EXIT: $LASTEXITCODE"
if (Test-Path $log) {
  Select-String -Path $log -Pattern 'phase pp|phase tg|check-telemetry: OK|error' | ForEach-Object { $_.Line } | Select-Object -First 8
}
