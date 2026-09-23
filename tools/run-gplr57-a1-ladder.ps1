# T-GPLR-57 A1-ladder: one ladder rung per invocation, ONE mutex hold.
#
# Why this exists: every recorded `gplr13-ladder` rung predates the F3/H8 part split
# (2026-09-14), so all four rungs are NOT-TAKEN for per-part TG timings and A1's
# transport-vs-compute attribution has no input at ladder lengths. This launcher re-runs
# a rung at its RECORDED shape (same flags, same recorded anchor prompt) on today's binary
# and produces the first ladder-rung trace that carries the split.
#
#   tools/run-gplr57-a1-ladder.ps1 -Rung 512            # one hold, ~15 s
#   tools/run-gplr57-a1-ladder.ps1 -Rung 200k           # one run, ~3 min
#   tools/run-gplr57-a1-ladder.ps1 -Rung 512 -PreflightOnly   # probe the card, launch nothing
#
# -CopySource <mmap-registered|pinned|direct> forces --copy-source, for the T-GPLR-55
# re-scoped source comparison: the rung's config serializes the row fetches
# (ready_latency/transfer_us ~ 1, MEASURED on the 512 and 200k rungs), which is the only
# condition in which bytes/transfer_us is a copy RATE rather than a queue-inclusive
# latency. Default is EMPTY = the recorded rung exactly as before, so every existing
# rung keeps its tag, its shape and its evidence; a source run gets a tag suffix and its
# own per-run directory.
#
# ORDER IS THE CONTRACT (the T-GPLR-53 session launcher's lesson): the read-only busy probe
# runs BEFORE anything takes the shared mutex, so a busy card refuses in seconds (exit 125)
# instead of blocking on WaitOne. The probe never touches CUDA; GPU_BUSY_PROBE_CMD stubs it
# for host-only tests.
#
# Every arm runs under the provenance enforcement: the run's verdict is gated by
# Assert-ArmProvenance (exit 10 when it cannot attest the tree and the binary it ran), and
# the escape is explicit (-AllowUnprovenanced records a waiver line). The trace is then
# checked twice: --check-telemetry on the engine, and tools/check-tg-part-timings.py
# --require-parts, which REFUSES (exit 2) rather than letting a not-measured split read as
# a zero-cost one.
param(
    [Parameter(Mandatory = $true)][ValidateSet('512', '200k')][string] $Rung,
    [ValidateSet('', 'mmap-registered', 'pinned', 'direct')][string] $CopySource = '',
    [switch] $PreflightOnly,
    [switch] $AllowUnprovenanced,
    [string] $ModelPath = '',
    [string] $RunRoot = ''
)

$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
. (Join-Path $root 'tools\arm-verdict.ps1')

# Recorded rung spec (tools/run-gplr13-ladder.ps1 lines 62-65) plus the rung's own
# recorded anchor prompt, reused byte-for-byte so the shape matches the rung it replaces.
$spec = @{
    '512'  = @{ pp = 512;    ubatch = 512; tg = 3; lockto = 2400; anchor = 'build\llama-cuda\gplr13-ladder\512\anchor-512.txt' }
    '200k' = @{ pp = 200000; ubatch = 512; tg = 3; lockto = 7600; anchor = 'build\llama-cuda\gplr13-ladder\200k\anchor-200000.txt' }
}[$Rung]

$exe = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$moe = if ($ModelPath) { $ModelPath } elseif ($env:QWEN36_MODEL) { $env:QWEN36_MODEL } else { 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
$prompt = Join-Path $root $spec.anchor
if (-not (Test-Path -LiteralPath $exe))    { Write-Host "REFUSED: no engine binary at $exe" -ForegroundColor Red; exit 2 }
if (-not (Test-Path -LiteralPath $prompt)) { Write-Host "REFUSED: no recorded anchor prompt at $prompt" -ForegroundColor Red; exit 2 }

# --- read-only pre-launch probe: no mutex, no CUDA, no side effect -------------
$gpuState = Format-GpuBusy (Get-GpuComputeApps)
if ($gpuState -eq 'probe-unavailable') {
    Write-Warning 'GPU busy probe unavailable; launching anyway (the wrapper probes again under the mutex).'
    $gpuState = 'probe-unavailable'
}
Write-Host ("PREFLIGHT {0}Z: gpu_state={1}" -f (Get-Date).ToUniversalTime().ToString('s'), $gpuState)
if ($gpuState -ne 'clean' -and $gpuState -ne 'probe-unavailable') {
    Write-Host ('PREFLIGHT REFUSED (exit 125): GPU busy - {0}' -f $gpuState) -ForegroundColor Red
    Write-Host '  A foreign compute process is visible; this session starts no arm. Re-run when the card is free.'
    exit 125
}
if ($PreflightOnly) { Write-Host 'PREFLIGHT ONLY: clean, nothing launched.'; exit 0 }

# --- one hold, one rung --------------------------------------------------------
$stamp  = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$runDir = if ($RunRoot) { $RunRoot } else { Join-Path $root ('build\llama-cuda\runs\{0}-gplr57-a1-{1}' -f $stamp, $Rung) }
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$tag    = if ($CopySource) { 'gplr57-a1-{0}-{1}' -f $Rung, $CopySource } else { 'gplr57-a1-{0}' -f $Rung }
$trace  = Join-Path $runDir ('rowset-{0}.jsonl' -f $spec.pp)
$log    = Join-Path $runDir ('rowset-{0}.log' -f $spec.pp)
$anchorSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $prompt).Hash

$srcArg = if ($CopySource) { "--copy-source $CopySource " } else { '' }
$cmd = "`"$exe`" --model `"$moe`" --prompt-file `"$prompt`" --residency moe-expert-bank --residency-tg " +
       "--pp-tokens $($spec.pp) --ubatch $($spec.ubatch) --tg-tokens $($spec.tg) --layer-slots 4 --kv-cache q8 " +
       "$srcArg" +
       "--greedy --no-mtp --pp-no-logits --allow-full-residency --telemetry `"$trace`" > `"$log`" 2>&1"

Write-Host ("rung {0}: pp={1} ubatch={2} tg={3} copy_source={4} anchor_sha256={5}" -f $Rung, $spec.pp, $spec.ubatch, $spec.tg, $(if ($CopySource) { $CopySource } else { 'default' }), $anchorSha)
Write-Host ("cmd: {0}" -f $cmd)

$lock = Join-Path $root 'tools\with-gpu-lock.ps1'
# -AllowUnprovenanced is FORWARDED to the wrapper: the gate now lives there too (the
# choke point), so the documented escape must reach it or the flag would refuse at the
# caller and pass at the wrapper - two owners disagreeing about one run.
& powershell -NoProfile -ExecutionPolicy Bypass -File $lock -TimeoutSeconds $spec.lockto `
    -Command $cmd -VerdictDir $runDir -VerdictTag $tag -AllowUnprovenanced:$AllowUnprovenanced
$armExit = $LASTEXITCODE
Write-Host ("arm exit {0}" -f $armExit)

# --- gates ---------------------------------------------------------------------
$gates = [ordered]@{ arm_exit = $armExit; anchor_sha256 = $anchorSha; run_dir = $runDir; trace = $trace;
    copy_source_requested = $(if ($CopySource) { $CopySource } else { 'default' }) }

# Both gates run through cmd.exe, not `& exe *> file`. MEASURED TRAP (2026-09-22, this
# launcher, first rung): PowerShell 5.1 turns a native command's STDERR into a
# TERMINATING error under $ErrorActionPreference='Stop', and the engine prints its
# rowset_round summary there - so the arm passed, then the launcher died in its own gate
# step with no chain verdict. cmd.exe owns the redirection, the exit code is the only
# thing that matters, and nothing here can be turned into an exception.
if (Test-Path -LiteralPath $trace) {
    & cmd.exe /d /s /c ("`"$exe`" --check-telemetry `"$trace`" > `"{0}`" 2>&1" -f (Join-Path $runDir 'check-telemetry.txt'))
    $gates['check_telemetry_exit'] = $LASTEXITCODE
    & cmd.exe /d /s /c ("`"{0}`" `"{1}`" --trace `"$trace`" --require-parts > `"{2}`" 2>&1" -f `
        (Get-Command python).Source, (Join-Path $root 'tools\check-tg-part-timings.py'), (Join-Path $runDir 'tg-parts.txt'))
    $gates['tg_parts_exit'] = $LASTEXITCODE
    $gates['tg_parts_verdict'] = ((Get-Content -LiteralPath (Join-Path $runDir 'tg-parts.txt') -Tail 4) -match '=> ' | Select-Object -First 1)
} else {
    $gates['check_telemetry_exit'] = -1
    $gates['tg_parts_exit'] = -1
    $gates['tg_parts_verdict'] = 'no trace produced'
}

$prov = Assert-ArmProvenance -Dir $runDir -Tags @($tag) -RepoRoot $root -Label ("A1-ladder {0} arm" -f $Rung) `
        -AllowUnprovenanced:$AllowUnprovenanced -WaiverNote ("run-gplr57-a1-ladder.ps1 rung=$Rung")
if (-not $prov -and $AllowUnprovenanced) {
    $note = ("WAIVED label=A1-ladder-{0} arms={1} note=operator passed -AllowUnprovenanced" -f $Rung, $tag)
    Add-Content -LiteralPath (Join-Path $runDir 'provenance-waivers.log') -Value $note
    Write-Warning $note
}
$gates['provenance_ok'] = $prov
$gates['waiver_taken'] = [bool](-not $prov -and $AllowUnprovenanced)

$status = if ($armExit -ne 0) { 'FAIL' } elseif ($gates['check_telemetry_exit'] -ne 0) { 'FAIL' } elseif ($gates['tg_parts_exit'] -ne 0) { 'FAIL' } elseif (-not $prov) { 'UNPROVENANCED' } else { 'PASS' }
# MEASURED TRAP (2026-09-22, this launcher): `-ExitCode ([int]($status -eq 'PASS'))`
# records 1 for a PASSING run, and Write-ArmVerdict derives status FROM the exit code -
# so the chain verdict said FAIL while the arm passed and the launcher exited 0. Pass the
# number, not the truth value.
$chainExit = if ($status -eq 'PASS') { 0 } elseif ($status -eq 'UNPROVENANCED') { 10 } else { 1 }
Write-ArmVerdict -Dir $runDir -Tag ('{0}-chain' -f $tag) -ExitCode $chainExit `
    -Command $cmd -Fields $gates | Out-Null
Write-Host ("RESULT rung={0} status={1} arm={2} check-telemetry={3} tg-parts={4} ({5}) provenance_ok={6} waiver={7}" -f `
    $Rung, $status, $armExit, $gates['check_telemetry_exit'], $gates['tg_parts_exit'], $gates['tg_parts_verdict'], $prov, $gates['waiver_taken'])

if ($status -eq 'PASS') { exit 0 }
if ($status -eq 'UNPROVENANCED') { exit 10 }
exit 1
