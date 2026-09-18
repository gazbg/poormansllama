# Relocated from build/run-gplr29-af1-3slot.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# Counterfactual test for the wait-signature investigation: af=1 + pinned +
# --layer-slots 3. Prediction: per-layer wait -> ~0 (the next bank's DMA now
# overlaps the previous compute window as well), warm walls -> ~2.0-2.4 s.
# TG reduced to 3 tokens: handoff auto declines at 3 slots at this shape, so
# the 96-token leg is T-GPLR-10-illegal; PP is the measurement.
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'
$trace  = Join-Path $base 'gplr29-af1-pinned-3slot.jsonl'
$log    = Join-Path $base 'gplr29-af1-pinned-3slot.log'
$tg     = Join-Path $base 'gplr29-af1-pinned-3slot.tgtext'

Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
$a = @('--model', $model, '--prompt-file', $prompt,
       '--ubatch', '512', '--tg-tokens', '3', '--pp-no-logits',
       '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
       '--layer-slots', '3', '--allow-full-residency',
       '--pp-repeat', '3', '--copy-source', 'pinned',
       '--pp-async-feed', '1',
       '--telemetry', $trace, '--tg-text', $tg)
& $exe @a > $log 2>&1
"ARM EXIT: $LASTEXITCODE"
if (Test-Path $log) {
  Select-String -Path $log -Pattern 'phase pp|phase tg|check-telemetry: OK|error' | ForEach-Object { $_.Line } | Select-Object -First 8
}
