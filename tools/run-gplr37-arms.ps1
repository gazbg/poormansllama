# T-GPLR-37: does direct mmap-sourced staging remove the host hop from the warm
# 4k layer-major pass?
#
# Reconnaissance that motivates it (all measured):
#  * today's staging path hops every member into a pinned staging buffer with a
#    CPU memcpy, then DMAs from the pinned copy: the recon attributed 1559 ms
#    (2 slots) / 1607 ms (3 slots) per pass to that hop, and the T-GPLR-37 probe
#    measured the pure hop at 20.22 ms per 452.5 MiB bank (~809 ms per 40-bank
#    pass) against 32.86 ms of DMA.
#  * `--copy-source direct` skips the hop but DMAs from pageable memory: 49.47 ms.
#  * registering the mapped window (cudaHostRegister, 0.16 ms) gives 32.86 ms at
#    14.44 GB/s - the pinned ceiling - with no CPU copy at all.
#
# PRE-REGISTERED PREDICTION
#  mechanism: with --copy-source mmap-registered the per-bank host submit window
#    (copy_complete.host_submit_us) collapses from ~39 ms to ~0.2 ms, one
#    `mmap_register` event is emitted per bank (40/pass, ~452 MB each), and the
#    bank DMA windows stay pure (medians ~31.4 ms, unchanged).
#  wall at 3 slots: NO measurable change (+-3%). The recorded 3-slot account
#    already closes as wall 2843 ms ~ compute 1611 + pure DMA 1255 with consumer
#    wait 225 ms, i.e. the hop is already hidden behind compute at that depth.
#  wall at 2 slots: the staging chain (hop + DMA) is longer than the per-bank
#    compute there (consumer wait was 734 ms), so the candidate may win some
#    wall; predicted <= 5%.
#  FALSIFIER: a >= 0.3 s (>=10%) warm-wall drop at 3 slots means the hop was on
#    the critical path and the "already hidden" reading is wrong; a >= 5%
#    REGRESSION at either depth means registration churn (register/unregister per
#    bank) costs more than the hop it removes.
# All four arms run in ONE mutex hold with adjacent control/candidate pairs, and
# every arm gets the strict checker. The 2-slot pair is re-measured here rather
# than compared against the older recorded numbers (session spread ~2-3%).
#
# NOTE (2026-09-17, T-GPLR-37 churn-free variant): the registered arms below now
# spell the mode 'mmap-registered-churn'. The value 'mmap-registered' was
# redefined to the churn-free design (register every executed layer's span once,
# before the timed PP) when that variant landed, so this script - which records
# the falsified per-bank-churn A/B - must keep naming the churn behavior to stay
# reproducible.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'pin-s2';  slots = '2'; src = 'pinned' },
  @{ name = 'reg-s2';  slots = '2'; src = 'mmap-registered-churn' },
  @{ name = 'pin-s3';  slots = '3'; src = 'pinned' },
  @{ name = 'reg-s3';  slots = '3'; src = 'mmap-registered-churn' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr37-$name.jsonl"
  $log   = Join-Path $base "gplr37-$name.log"
  $tg    = Join-Path $base "gplr37-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', $arm.slots, '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', $arm.src,
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr37] arm $name (slots=$($arm.slots) copy=$($arm.src)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr37] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  $nreg = (Select-String -Path $trace -Pattern 'mmap_register' -ErrorAction SilentlyContinue).Count
  Write-Host ("  mmap_register events: {0}" -f $nreg)
  Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|refus|error' | Select-Object -Last 4 | ForEach-Object { Write-Host $_ }
}

# Compact comparison table (warm pass per arm + hop accounting).
$analyzer = Join-Path $root 'tools/analyze-gplr37.py'
python $analyzer (Join-Path $base 'gplr37-pin-s2.jsonl') (Join-Path $base 'gplr37-reg-s2.jsonl') `
                 (Join-Path $base 'gplr37-pin-s3.jsonl') (Join-Path $base 'gplr37-reg-s3.jsonl')
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)

# Pipeline gate: each candidate against its adjacent control from this session.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
python $gate (Join-Path $base 'gplr37-reg-s3.jsonl') --baseline (Join-Path $base 'gplr37-pin-s3.jsonl')
Write-Host ("GATE s3 EXIT: {0}" -f $LASTEXITCODE)
python $gate (Join-Path $base 'gplr37-reg-s2.jsonl') --baseline (Join-Path $base 'gplr37-pin-s2.jsonl')
Write-Host ("GATE s2 EXIT: {0}" -f $LASTEXITCODE)
