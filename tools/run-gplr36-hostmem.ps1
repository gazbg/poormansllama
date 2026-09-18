# T-GPLR-36 follow-up: the host-memory arm. Byte-identical flags to the recorded
# T-GPLR-36 ledger arm (tools/run-gplr36-ledger.ps1), so the new per-pass
# host_mem samples sit next to a known wall and a known byte_account:
#   200k-token anchor, 262,144-token F16 KV configuration, layer-major,
#   ubatch 512, af=1, pinned, bf16 slabs, 3 slots, --pp-repeat 2 (pass 1 cold,
#   pass 2 warm), --pp-recon 1 (emits host_ledger AND host_mem).
#
# Recorded references for the same shape: wall 146,030 ms / 1,369.6 tok/s warm,
# compute 134,592 ms, wait 36.5 ms, peak VRAM 3,265.56 MiB, kv 3.82 GiB (host
# spilled), act_slab 1.53 GiB, slot_window 1.46 GiB, static 0.89 GiB, free 20.70
# GiB (taken in a session whose GPU was otherwise idle).
#
# PRE-REGISTERED question: the instrument's first purpose is to place the
# owner's observed host-RSS climb. Expectation (stated before the run): the climb
# is dominated by FILE-BACKED pages (the mapped 21.4 GB GGUF being paged in and,
# in a pinned arm, the staging buffers), not by anonymous growth, so
# `ws - ws_private` should carry most of it and `eng_bytes` (pinned staging +
# pool + registered spans) should explain the engine's share of the anonymous
# part. FALSIFIER: the climb is mostly PRIVATE (anonymous) while `eng_bytes` is
# flat -- that would mean an unaccounted host allocation, i.e. a leak, and the
# plan needs a new task rather than a new explanation.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }

$name  = 'hostmem-200k'
$trace = Join-Path $base "gplr36-$name.jsonl"
$log   = Join-Path $base "gplr36-$name.log"
Remove-Item $trace, $log -ErrorAction SilentlyContinue

# VRAM context for the run: this arm's numbers are only comparable to the
# recorded reference if the occupancy is comparable, and the crowded case is the
# charter's target condition, so record what the GPU looked like at start.
Write-Host '[gplr36] GPU state at arm start:'
& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader |
  ForEach-Object { Write-Host "  $_" }
& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
  ForEach-Object { Write-Host "  $_(other project, resident)" }

$a = @('--model', $model, '--prompt-file', $prompt,
       '--pp-tokens', '262144', '--ubatch', '512', '--pp-no-logits',
       '--tg-tokens', '0',
       '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
       '--act-slab-dtype', 'bf16',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '2', '--copy-source', 'pinned',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)

Write-Host "[gplr36] arm $name (200k tokens, 3 slots, host-memory instrument on) starting..."
& $exe @a > $log 2>&1
Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
  Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
  exit 1
}

Write-Host '[gplr36] strict check-telemetry...'
& $exe --check-telemetry $trace
Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|refus|error' |
  Select-Object -Last 8 | ForEach-Object { Write-Host $_ }

Write-Host '[gplr36] host_mem lines:'
Get-Content $trace | Select-String -Pattern 'host_mem' | ForEach-Object { Write-Host $_ }

Write-Host '[gplr36] host-memory attribution:'
python (Join-Path $root 'tools/analyze-gplr36-hostmem.py') $trace
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)

# The ledger view for the same trace (timing buckets), so memory and time are
# read from one run.
python (Join-Path $root 'tools/analyze-gplr36-ledger.py') $trace
Write-Host ("LEDGER ANALYZER EXIT: {0}" -f $LASTEXITCODE)

$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
python $gate $trace
Write-Host ("GATE EXIT: {0}" -f $LASTEXITCODE)
