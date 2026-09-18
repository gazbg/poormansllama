# Relocated from build/run-lm4k-trio.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Stop'
# T-GPLR layer-major payoff measurement (owner-requested): adjacent trio at 4k,
# 2 slots, ONE mutex hold so all arms share file-cache state.
#   bm1: batch-major pinned (320-copy baseline) | lm: DEFAULT (layer-major,
#   handoff auto) | bm2: batch-major drift control.
$lock  = (Join-Path $root 'tools/with-gpu-lock.ps1')
$base  = (Join-Path $root 'build/llama-cuda')
$arms  = (Join-Path $root 'build/run-lm4k-arms.ps1')

$deadline = (Get-Date).AddSeconds(1500)
while ((Get-Date) -lt $deadline) {
  $q = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
  if ($q) {
    $parts = $q -split ','
    if ([int]$parts[0].Trim() -lt 15 -and [int]$parts[1].Trim() -lt 3000) { break }
  }
  Start-Sleep -Seconds 30
}

& $lock -Command "powershell -NoProfile -ExecutionPolicy Bypass -File $arms"
"LOCK EXIT: $LASTEXITCODE"

foreach ($name in 'bm1','lm','bm2') {
  "===== ARM $name ====="
  $log   = Join-Path $base "gplr25-lm4k-$name.log"
  $trace = Join-Path $base "gplr25-lm4k-$name.jsonl"
  if (Test-Path $log) {
    Select-String -Path $log -Pattern 'phase pp|phase tg|check-telemetry|error' | ForEach-Object { $_.Line } | Select-Object -First 8
  }
  if (Test-Path $trace) {
    $rs = Select-String -Path $trace -Pattern '"event":"run_start"' | Select-Object -First 1
    if ($rs) {
      $j = $rs.Line | ConvertFrom-Json
      "run_start: schedule=$($j.schedule) handoff_mode=$($j.handoff_mode) n_slots=$($j.n_slots)"
    }
    "$((Select-String -Path $trace -Pattern '"event":"run_end"' | Measure-Object).Count) run_end record(s)"
  } else { 'NO TRACE' }
  ""
}