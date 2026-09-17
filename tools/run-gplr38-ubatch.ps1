# T-GPLR-38: ubatch sweep on the 4k anchor -- does fewer, fatter launches buy
# wall time under the current best mode (churn-free mmap-registered, 3 slots)?
#
# Pre-registered prediction (plan T-GPLR-38, written before the run):
#   warm wall 2290 -> ~2050-2200 ms at ubatch 2048, sub-linear (host overhead
#   is only ~0.19 -> ~0.02 s of the pass); ubatch 4096 flat or worse than 2048
#   (ONE chunk per bank -- granularity concern). FALSIFIER: 4096 beats 2048 ->
#   bigger is strictly better; wall WORSENS at 1024+ -> overhead model wrong.
#   Memory: peak VRAM rises modestly (buf_act 24->192 MiB + workspace term).
# Order: u512 (fresh control), u1024, u2048, u4096 -- adjacent, one mutex hold.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'u512';  ub = '512' },
  @{ name = 'u1024'; ub = '1024' },
  @{ name = 'u2048'; ub = '2048' },
  @{ name = 'u4096'; ub = '4096' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr38-$name.jsonl"
  $log   = Join-Path $base "gplr38-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', $arm.ub, '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace, '--tg-text', (Join-Path $base "gplr38-$name.tgtext"))
  Write-Host "[gplr38] arm $name (ubatch=$($arm.ub)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr38] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'phase pp|pinned in|peak VRAM|refus|error' |
    Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
}

python (Join-Path $root 'tools/analyze-gplr37.py') `
  (Join-Path $base 'gplr38-u512.jsonl') (Join-Path $base 'gplr38-u1024.jsonl') `
  (Join-Path $base 'gplr38-u2048.jsonl') (Join-Path $base 'gplr38-u4096.jsonl')
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)

# Paired gates vs the fresh u512 control (same depth, same copy source is NOT a
# comparison here -- the changing axis is ubatch, which the gate cannot see, so
# the u512 control arm is the baseline and the shape/identity checks must be
# expected to complain; the paired throughput criterion is still meaningful).
# We run the gate for the record and print, not enforce, its depth warning.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
foreach ($n in @('u1024','u2048','u4096')) {
  python $gate (Join-Path $base "gplr38-$n.jsonl") --baseline (Join-Path $base 'gplr38-u512.jsonl') 2>&1 |
    Select-String -Pattern 'paired vs|RESULT|FAIL: paired|FAIL: bank' | ForEach-Object { Write-Host $_ }
  Write-Host ("GATE {0} EXIT: {1}" -f $n, $LASTEXITCODE)
}

# Output sanity across ubatches (greedy TG should be identical -- same tokens, same math).
Write-Host '[gplr38] tg-text identity:'
$ref = Get-Content (Join-Path $base 'gplr38-u512.tgtext') -Raw
foreach ($n in @('u1024','u2048','u4096')) {
  $t = Get-Content (Join-Path $base "gplr38-$n.tgtext") -Raw -ErrorAction SilentlyContinue
  Write-Host ("  {0}: {1}" -f $n, $(if ($t -ceq $ref) { 'IDENTICAL' } else { 'DIFFER' }))
}
Write-Host 'DONE'
