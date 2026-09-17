# T-GPLR-37 churn-free variant: does registering every executed layer's mapped
# source span ONCE, before the timed PP, remove the per-launch setup inflation
# the churn implementation measured?
#
# What the first A/B established (measured, see docs/currentstate.md):
#   warm pin-s2 3123.9 ms / 1311.2 tok/s -> reg-s2 3692.9 / 1109.2 (-15.4%)
#   warm pin-s3 2811.9 ms / 1456.6 tok/s -> reg-s3 3689.0 / 1110.3 (-23.8%)
#   ledger: `slot` fell (the hop did shrink: host submit 1587.8 -> 1222.3 ms at
#   2 slots) but `setup` (sched reset + alloc_graph) exploded 11.2 -> 227.8 ms
#   (2 slots) / 11.9 -> 134.7 ms (3 slots) and `compute` 1329.5 -> 1845.1 /
#   1581.4 -> 2441.7 ms. Bank DMA windows stayed pure (31.48 ms) and nothing
#   starved. Indicated cause: 80 register/unregister cycles over ~111k pages
#   each (34.46 GiB of churn per run) landing inside the per-launch windows.
#
# PRE-REGISTERED PREDICTION
#  mechanism: `mmap_register` fires 40 times with mode "init-once", ALL before the
#    first `layer_compute` of pass 1, plus one `mmap_register_init` carrying the
#    total bytes and the pre-timed cost. No registration happens during a pass.
#  ledger: `setup` returns to the pinned control's ~11-12 ms, and `compute`
#    returns to the control's band (1329 / 1581 ms). Host submit per bank stays
#    reduced (no registration inside the window).
#  wall: with no churn but the same DMA, warm wall within +-3% of the pinned
#    control at BOTH depths; a gain is only expected if the hop is exposed at
#    2 slots (<= 5%).
#  FALSIFIER: if `setup` still reads 100-230 ms without churn, the cost is
#    inherent to file-backed registered pages, not to churn -> the mmap-registered
#    family closes. A >= 5% warm regression at either depth rejects the variant.
#  cost to report: the pre-timed registration phase (expected seconds; ~17 GiB of
#    file pages pinned and held for the run) - the RAM price of removing churn.
#
# Order note: arms run pin-s2, regonce-s2, pin-s3, regonce-s3 (adjacent pairs).
# The candidate arms pin (and therefore page in) the whole model before their
# timed pass, so any COLD-pass advantage must be read as confounded in the
# candidate's favour; the warm comparison is the verdict target.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

# RAM price of the variant: the candidate arms hold ~17 GiB of the model's file
# pages pinned (and page them in). Sample host RSS + GPU counters for the whole
# A/B so the pinning cost is measured, not asserted. CPU-only sampling, so it
# needs no mutex of its own.
$mon = Start-Job -ScriptBlock {
  param($root)
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools/monitor-mem-gplr36.ps1') `
    -Minutes 40 -IntervalSeconds 5 -Out 'build/llama-cuda/gplr37b-mem-monitor.csv'
} -ArgumentList $root

foreach ($arm in @(
  @{ name = 'pin-s2';  slots = '2'; src = 'pinned' },
  @{ name = 'ron-s2';  slots = '2'; src = 'mmap-registered' },
  @{ name = 'pin-s3';  slots = '3'; src = 'pinned' },
  @{ name = 'ron-s3';  slots = '3'; src = 'mmap-registered' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr37b-$name.jsonl"
  $log   = Join-Path $base "gplr37b-$name.log"
  $tg    = Join-Path $base "gplr37b-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', $arm.slots, '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', $arm.src,
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr37b] arm $name (slots=$($arm.slots) copy=$($arm.src)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr37b] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'pinned in|registering mapped|phase pp|peak VRAM|refus|error' |
    Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
}

$analyzer = Join-Path $root 'tools/analyze-gplr37.py'
python $analyzer (Join-Path $base 'gplr37b-pin-s2.jsonl') (Join-Path $base 'gplr37b-ron-s2.jsonl') `
                 (Join-Path $base 'gplr37b-pin-s3.jsonl') (Join-Path $base 'gplr37b-ron-s3.jsonl')
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)

$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
python $gate (Join-Path $base 'gplr37b-ron-s3.jsonl') --baseline (Join-Path $base 'gplr37b-pin-s3.jsonl')
Write-Host ("GATE s3 EXIT: {0}" -f $LASTEXITCODE)
python $gate (Join-Path $base 'gplr37b-ron-s2.jsonl') --baseline (Join-Path $base 'gplr37b-pin-s2.jsonl')
Write-Host ("GATE s2 EXIT: {0}" -f $LASTEXITCODE)

Stop-Job $mon -ErrorAction SilentlyContinue
Remove-Job $mon -Force -ErrorAction SilentlyContinue
Write-Host 'DONE'
