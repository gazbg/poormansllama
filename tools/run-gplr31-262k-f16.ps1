# Relocated from build/run-gplr31-262k-f16.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# Charter task 1: the crowded-VRAM scenario at the new optimum.
# 262,144-token prefill, F16 KV (the widest standard KV), layer-major +
# af=1 + pinned + 2 slots (the measured af=1+pinned optimum, T-GPLR-29
# addendum). bf16 slabs per the approved T-Q38PW-02 decision. No --pp-repeat
# (single-pass prefill; this rung is about crowded-VRAM PP, not repeat warm
# state). No TG leg (PP is the measurement; TG tokens 0 keeps the run clean
# and the VRAM budget entirely on PP+KV). Q8 reference already recorded:
# 262.62 tok/s @ 19,468 MiB peak (af=0, 4 slots).
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
# anchor-200000.txt is the longest on-disk prompt; --pp-tokens 262144 slices
# the prefill to the full 262k width (the T-Q38PW-02 262K runs used the same
# mechanism: prompt text + explicit pp-tokens).
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'
$trace  = Join-Path $base 'gplr31-262k-f16-af1pinned.jsonl'
$log    = Join-Path $base 'gplr31-262k-f16-af1pinned.log'

Remove-Item $trace, $log -ErrorAction SilentlyContinue
$a = @('--model', $model, '--prompt-file', $prompt,
       '--pp-tokens', '262144', '--ubatch', '512',
       '--pp-no-logits',
       '--kv-cache', 'f16',
       '--residency', 'moe-expert-bank',
       '--layer-slots', '2',
       '--allow-full-residency',
       '--copy-source', 'pinned',
       '--pp-async-feed', '1',
       '--act-slab-dtype', 'bf16',
       '--telemetry', $trace)
& $exe @a > $log 2>&1
Write-Host ("ARM EXIT: {0}" -f $LASTEXITCODE)
if (Test-Path $log) {
    Select-String -Path $log -Pattern 'phase pp|check-telemetry: OK|error|refus|run_refused' |
        ForEach-Object { $_.Line } | Select-Object -First 10
}
