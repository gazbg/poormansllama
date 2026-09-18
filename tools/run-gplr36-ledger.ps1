# T-GPLR-36: the long-context PP residual arm. One 200k-token layer-major pass
# pair at 3 slots with the host ledger on (--pp-recon 1 emits `host_ledger`).
#
# Shape is the recorded charter shape, byte-identical flags to the T-GPLR-35
# addendum-2 s3-200k arm so the ledger's numbers sit next to a known wall:
#   200,000-token prefill on the 262,144-token F16 KV configuration, layer-major,
#   ubatch 512, af=1, pinned, bf16 slabs, 3 slots, --pp-repeat 2 (pass 1 cold,
#   pass 2 warm). Recorded warm reference (previous session): wall 146,030 ms,
#   compute 134,592 ms, wait 36.5 ms, 1,369.6 tok/s, checker OK.
#
# PRE-REGISTERED (see the T-GPLR-36 block in the plan): the ledger closes on
# sweep_us within ~1%; patch + setup + compute + graph_call + slot + other are
# the whole sweep. FALSIFIER: graph_call_us and other_us both under ~1 s/pass
# while closure holds -> the residual is GPU idle inside the compute window's
# sync wait, not host bookkeeping.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

$name  = 'ledger-200k'
$trace = Join-Path $base "gplr36-$name.jsonl"
$log   = Join-Path $base "gplr36-$name.log"
Remove-Item $trace, $log -ErrorAction SilentlyContinue

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

Write-Host "[gplr36] arm $name (200k tokens, 3 slots, ledger on) starting..."
& $exe @a > $log 2>&1
Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
if ($LASTEXITCODE -ne 0) {
  Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
  exit 1
}

Write-Host "[gplr36] strict check-telemetry..."
& $exe --check-telemetry $trace
Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|refus|error' |
  Select-Object -Last 8 | ForEach-Object { Write-Host $_ }

Write-Host '[gplr36] host ledger lines:'
Get-Content $trace | Select-String -Pattern 'host_ledger' | ForEach-Object { Write-Host $_ }

# Absolute gate profile for 3 slots (auto-selected from run_start.n_slots). The
# control is a different session, so this is a coarse alarm, not the paired form.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
python $gate $trace
Write-Host ("GATE EXIT: {0}" -f $LASTEXITCODE)
