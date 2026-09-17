# T-GPLR-37 addendum 3: the charter's crowded case -- does the churn-free
# mmap-registered staging win survive at the 200k-token / 262k-F16-KV shape,
# where 17.23 GiB of pinned file pages meet a ~5.0 GiB host-spilled KV?
#
# Shape facts this starts from (all measured, see the plan addendum 3):
#   * 200k reference (2 slots, pinned): 1363.2 tok/s, wall 146,713 ms per pass,
#     compute-dominated (91%+), wait 36.5 ms at 3 slots, peak VRAM 3265.56 MiB
#     (KV host-spilled, NOT resident).
#   * 4k result being tested for transfer: ron -23.3% warm at 2 slots, -15.4% at
#     3 slots, for 17.23 GiB of pinned file pages.
#   * pre-registered prediction: warm walls within noise (<= ~1%) at BOTH depths;
#     FALSIFIER: > 3% win or > 3% regression at either depth.
#
# Order note: pin-s2, ron-s2, pin-s3, ron-s3 (adjacent pairs, one mutex hold).
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

# Whole-run RAM monitoring (fixed instrument: waits for the first arm, samples
# until 60 s after the last one exits; adds the sys_free pressure column).
$mon = Start-Job -ScriptBlock {
  param($root)
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools/monitor-mem-gplr36.ps1') `
    -Out 'build/llama-cuda/gplr37c-mem-monitor.csv' -IntervalSeconds 5
} -ArgumentList $root

foreach ($arm in @(
  @{ name = 'pin-s2'; slots = '2'; src = 'pinned' },
  @{ name = 'ron-s2'; slots = '2'; src = 'mmap-registered' },
  @{ name = 'pin-s3'; slots = '3'; src = 'pinned' },
  @{ name = 'ron-s3'; slots = '3'; src = 'mmap-registered' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr37c-$name.jsonl"
  $log   = Join-Path $base "gplr37c-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', '262144', '--ubatch', '512', '--pp-no-logits',
         '--tg-tokens', '0',
         '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
         '--act-slab-dtype', 'bf16',
         '--pp-schedule', 'layer-major',
         '--layer-slots', $arm.slots, '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', $arm.src,
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace)
  Write-Host "[gplr37c] arm $name (slots=$($arm.slots) copy=$($arm.src)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr37c] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'phase pp|pinned in|peak VRAM|refus|error' |
    Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
}

Stop-Job $mon -ErrorAction SilentlyContinue
Remove-Job $mon -Force -ErrorAction SilentlyContinue

python (Join-Path $root 'tools/analyze-gplr37.py') `
  (Join-Path $base 'gplr37c-pin-s2.jsonl') (Join-Path $base 'gplr37c-ron-s2.jsonl') `
  (Join-Path $base 'gplr37c-pin-s3.jsonl') (Join-Path $base 'gplr37c-ron-s3.jsonl')
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)

# Paired gates: same-depth pairs with a different copy source are now legal arm
# comparisons (T-GPLR-37b gate fix), and the 200k shape skips absolute floors.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
python $gate (Join-Path $base 'gplr37c-ron-s2.jsonl') --baseline (Join-Path $base 'gplr37c-pin-s2.jsonl')
Write-Host ("GATE s2 EXIT: {0}" -f $LASTEXITCODE)
python $gate (Join-Path $base 'gplr37c-ron-s3.jsonl') --baseline (Join-Path $base 'gplr37c-pin-s3.jsonl')
Write-Host ("GATE s3 EXIT: {0}" -f $LASTEXITCODE)

# RAM pressure summary (the pre-registered RAM claim lives or dies here).
if (Test-Path (Join-Path $base 'gplr37c-mem-monitor.csv')) {
  Write-Host '[gplr37c] RAM monitor (first 3 / last 3 rows, min free RAM):'
  Get-Content (Join-Path $base 'gplr37c-mem-monitor.csv') | Select-Object -First 3 | ForEach-Object { Write-Host $_ }
  Get-Content (Join-Path $base 'gplr37c-mem-monitor.csv') | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
  $rows = Import-Csv (Join-Path $base 'gplr37c-mem-monitor.csv')
  $minFree = ($rows | Measure-Object -Property sys_free_mib -Minimum).Minimum
  $maxWs   = ($rows | Measure-Object -Property proc_ws_mib -Maximum).Maximum
  $maxPriv = ($rows | Measure-Object -Property proc_private_mib -Maximum).Maximum
  Write-Host ("RAM SUMMARY: min sys_free {0} MiB | max proc WS {1} MiB | max private {2} MiB" -f $minFree, $maxWs, $maxPriv)
}
Write-Host 'DONE'
