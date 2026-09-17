# T-GPLR-35 addendum 1: does deeper slot lookahead convert the measured engine
# queue wait into wall time?
#
# The recon (2 slots) put the warm 4k layer-major wall at 3089 ms, of which
# compute = 1350 ms and pure DMA = 1238 ms, while the host transfer window
# (3988 ms) split into a 1559 ms host hop + 2429 ms ready latency whose
# non-DMA part is 1191 ms of ENGINE QUEUE WAIT (submitted, engine not free),
# plus a 747 ms consumer wait -- i.e. the pipeline holds only ~one bank.
#
# PRE-REGISTERED PREDICTION: if the depth is the limiter, 3 slots lets the
# worker keep a second bank in flight while one is consumed, so queue wait and
# consumer wait shrink and the warm wall falls to ~2.2-2.6 s (transfer windows
# stay pure 31.4 ms at 3 slots, as at 2).
# FALSIFIER: warm wall stays >= ~2.9 s with engine queue wait and consumer wait
# essentially unchanged -> slot count does not convert; the depth lever is dead
# at this shape and the remaining lever is the host hop.
# The 2-slot control is re-run ADJACENTLY in the same mutex hold (not compared
# against the older recon numbers) so file-cache/driver state cannot confound.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'slot2'; slots = '2' },
  @{ name = 'slot3'; slots = '3' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr35-$name.jsonl"
  $log   = Join-Path $base "gplr35-$name.log"
  $tg    = Join-Path $base "gplr35-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', $arm.slots, '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr35-slots] arm $name (slots=$($arm.slots)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 20 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr35-slots] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'PP|tok/s|slot_vram|refus|error' | Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
}

# T-GPLR-35 regression gate: the protocol checker above proves the trace is
# VALID, not that the pipeline stayed pipelined. This compares the candidate
# against the adjacent control from the same session (paired, the trustworthy
# form) plus absolute floors for each depth, and fails loudly if staging has
# silently re-serialized. Its exit code becomes this script's exit code.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
Write-Host '[gplr35-slots] pipeline regression gate: 2-slot control (absolute)'
python $gate (Join-Path $base 'gplr35-slot2.jsonl')
$gate_ctrl = $LASTEXITCODE
Write-Host ("GATE control EXIT: {0}" -f $gate_ctrl)
Write-Host '[gplr35-slots] pipeline regression gate: 3-slot candidate (absolute + paired)'
python $gate (Join-Path $base 'gplr35-slot3.jsonl') --baseline (Join-Path $base 'gplr35-slot2.jsonl')
$gate_cand = $LASTEXITCODE
Write-Host ("GATE candidate EXIT: {0}" -f $gate_cand)
if ($gate_ctrl -ne 0 -or $gate_cand -ne 0) {
  Write-Host 'PIPELINE REGRESSION GATE FAILED - see the gate output above'
  exit 1
}
