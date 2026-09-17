# T-GPLR-35 addendum 2: does a FOURTH slot still fit the budget and still pay
# off at the charter's crowded-VRAM shape (200k-token prefill on the 262k F16 KV
# configuration)?
#
# Established facts this shape starts from (measured):
#   * 2-slot 200k F16 reference (tools/run-gplr31-262k-f16.ps1): 1363.2 tok/s,
#     wall 14,671 ms, compute 13,303 ms, wait 1,182 ms, peak VRAM 3,265.56 MiB,
#     checker OK -- i.e. the KV is HOST-SPILLED here, not resident (the 3.27 GB
#     peak is slot window + 1.64 GB activation slab + compute buffers).
#   * 4k layer-major: 3 slots beat 2 by -15.8% warm wall, for +498 MiB/slot.
#   * model: qwen35moe, 41 blocks, KV heads 2, key/value length 256, full
#     attention only every 4th layer.
#
# PRE-REGISTERED PREDICTION
#   fit:   4 slots fit -- predicted peak ~3.8 GB (3.27 GB + one 498 MiB slot),
#          far under the ~20.7 GB observed free; no derived-budget refusal, and
#          a deliberately tight explicit budget DOES refuse loudly.
#   payoff: little to none at this shape (<= ~5% wall), because a bank is
#          consumed across ~391 chunks, which already hides the next bank's
#          staging -- staging is not on the critical path here the way it was at
#          4k (where compute was only 44% of the wall vs 91% here).
#   FALSIFIER: if 3 or 4 slots still cut the 200k wall by > 10%, the
#   amortization reasoning is wrong and depth keeps paying at long context.
#
# Arms (one mutex hold): slots3 (control), slots4 (candidate), slots4-budget
# (tight explicit --residency-budget-mib, probes the refusal path; its trace is
# a refusal and is never fed to the pipeline gate).
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 's3-200k';      slots = '3'; budget = '0'    },
  @{ name = 's4-200k';      slots = '4'; budget = '0'    },
  @{ name = 's4-200k-tight';slots = '4'; budget = '1500' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr35-$name.jsonl"
  $log   = Join-Path $base "gplr35-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', '262144', '--ubatch', '512', '--pp-no-logits',
         '--tg-tokens', '0',
         '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
         '--act-slab-dtype', 'bf16',
         '--pp-schedule', 'layer-major',
         '--layer-slots', $arm.slots, '--allow-full-residency',
         '--residency-budget-mib', $arm.budget,
         '--pp-repeat', '2', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace)
  Write-Host "[gplr35-200k] arm $name (slots=$($arm.slots) budget=$($arm.budget) MiB) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr35-200k] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|refus|error' | Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
}

# Pipeline gate on the two measurement arms only (the tight-budget arm is a
# refusal trace with no warm pp summary). No slots4 profile exists yet, so the
# candidate is judged by the paired comparison -- the honest form for a shape
# whose absolute floors are anchored elsewhere.
$gate = Join-Path $root 'tools/check-gplr35-pipeline.py'
Write-Host '[gplr35-200k] pipeline gate: 3-slot control (absolute, shape caveat printed)'
python $gate (Join-Path $base 'gplr35-s3-200k.jsonl')
$gate_ctrl = $LASTEXITCODE
Write-Host ("GATE control EXIT: {0}" -f $gate_ctrl)
Write-Host '[gplr35-200k] pipeline gate: 4-slot candidate (paired vs the 3-slot control)'
python $gate (Join-Path $base 'gplr35-s4-200k.jsonl') --baseline (Join-Path $base 'gplr35-s3-200k.jsonl')
$gate_cand = $LASTEXITCODE
Write-Host ("GATE candidate EXIT: {0}" -f $gate_cand)
if ($gate_ctrl -ne 0 -or $gate_cand -ne 0) {
  Write-Host 'PIPELINE REGRESSION GATE FAILED - see the gate output above'
  exit 1
}
