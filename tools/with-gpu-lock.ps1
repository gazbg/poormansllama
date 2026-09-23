[CmdletBinding()]
param(
    [int] $TimeoutSeconds = 7200,
    [Parameter(Mandatory = $true)]
    [string] $Command,
    [string] $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1',
    # Verdict/sentinel output. Defaulted below to <repo>/build/llama-cuda/verdicts
    # with the tag derived from the command, so every arm run through this wrapper
    # leaves a checkable result without the caller having to name anything.
    [string] $VerdictDir = '',
    [string] $VerdictTag = '',
    [switch] $NoVerdict,
    # T-GPLR-50 contention follow-up: a non-conforming project can hold VRAM and
    # run compute DURING a mutex-held arm (measured 2026-09-21: deepseek_v4.exe,
    # ~16.5 GiB, poisoned a host-suite arm and the recorded 200k remeasure). The
    # wrapper therefore probes nvidia-smi AFTER acquiring the mutex and REFUSES
    # to start the command while any compute process is visible, unless the
    # caller explicitly accepts the risk with -AllowBusy.
    [switch] $AllowBusy,
    # The escape from the provenance gate below. Explicit AND recorded: it writes a
    # `provenance-waivers.log` line beside the verdict, so a run that passed without
    # attestation says so rather than looking clean.
    [switch] $AllowUnprovenanced,
    # Seconds to wait for a just-exiting process to disappear (a pre-existing
    # compute app often tears down between mutex holders). Two probes, no sleep
    # when the first is already clean.
    [int] $BusyGraceSeconds = 30
)

if ($TimeoutSeconds -lt 0) {
    throw 'TimeoutSeconds must be zero or greater.'
}

if ([string]::IsNullOrWhiteSpace($Command)) {
    throw 'Command must not be empty.'
}

# --- verdict wiring ---------------------------------------------------------
# Written even when the command fails or the mutex times out, and the .done is
# DELETED before the run, so a stale sentinel from an earlier run can never be
# read as this run's result.
$verdictReady = $false
$vDir = ''
$vTag = ''
$vTagSource = ''
$vStarted = ''
if (-not $NoVerdict) {
    $root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
    $helper = Join-Path $root 'tools\arm-verdict.ps1'
    if (Test-Path -LiteralPath $helper) {
        . $helper
        # How the tag came to be is provenance, not trivia: a DERIVED tag is this
        # wrapper guessing from the command string, an EXPLICIT one is the caller
        # naming the artifact to inspect. The verdict records which, so a later
        # pass reading gplr53-a4-pw0.verdict.json knows the tag was chosen by the
        # driver rather than matched out of the command line.
        if ($VerdictTag) { $vTag = $VerdictTag; $vTagSource = 'explicit' }
        else { $vTag = Get-ArmTagFromCommand -Command $Command; $vTagSource = 'derived' }
        $vDir = if ($VerdictDir) { $VerdictDir } else { Join-Path $root 'build\llama-cuda\verdicts' }
        $vStarted = (Get-Date).ToUniversalTime().ToString('o')
        Clear-ArmVerdict -Dir $vDir -Tag $vTag
        # Mark the run STARTED here -- before the mutex wait and before any model
        # load -- so "never started" is distinguishable from "still working" for
        # the whole duration of the run, not merely at the end.
        Write-ArmStart -Dir $vDir -Tag $vTag -Command $Command -MutexName $MutexName | Out-Null
        Write-Host ("Started: {0}" -f (Join-Path $vDir ($vTag + '.started')))
        $verdictReady = $true
    } else {
        Write-Warning "arm-verdict helper not found at $helper; no verdict file will be written."
    }
}

$mutex = [System.Threading.Mutex]::new($false, $MutexName)
$acquired = $false
$exitCode = 1
$gpuPre  = '__unprobed__'
$gpuPost = '__unprobed__'
# Provenance for this run: captured BEFORE the command (what the binary was built
# from, and whether it even contains the newest source edit) and again AFTER it
# (did another actor edit or rebuild the tree while this arm ran). Both are read
# in the finally block, so they live out here.
$provPre  = $null
$provPost = $null

try {
    Write-Host "Waiting for GPU mutex $MutexName"

    try {
        $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds($TimeoutSeconds))
    } catch [System.Threading.AbandonedMutexException] {
        $acquired = $true
        Write-Warning 'Recovered an abandoned GPU mutex.'
    }

    if (-not $acquired) {
        [Console]::Error.WriteLine("Timed out waiting for GPU mutex after $TimeoutSeconds seconds.")
        $exitCode = 124
    } else {
        Write-Host "Acquired GPU mutex $MutexName"

        # --- GPU busy gate (post-acquisition) ---------------------------------
        # Probe only AFTER the hold is ours: a pre-existing compute app should
        # tear down once its holder loses the mutex, so give it a grace window,
        # then refuse loudly. Unverifiable probe state (no nvidia-smi) does NOT
        # block -- it is recorded and the run proceeds, annotated.
        $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
        if ($gpuPre -eq 'probe-unavailable') {
            Write-Warning 'GPU busy probe unavailable (nvidia-smi?); proceeding UNVERIFIED.'
        } else {
            Write-Host ("GPU pre-run probe: {0}" -f $gpuPre)
            if ($gpuPre -ne 'clean') {
                if ($BusyGraceSeconds -gt 0) {
                    Write-Host ("  busy; grace {0}s for a just-exiting holder..." -f $BusyGraceSeconds)
                    $deadline = (Get-Date).AddSeconds($BusyGraceSeconds)
                    while ((Get-Date) -lt $deadline) {
                        Start-Sleep -Seconds ([Math]::Min(5, [Math]::Max(1, $BusyGraceSeconds)))
                        $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
                        if ($gpuPre -eq 'clean') { break }
                    }
                    Write-Host ("GPU pre-run probe after grace: {0}" -f $gpuPre)
                }
            }
            if ($gpuPre -ne 'clean' -and -not $AllowBusy) {
                [Console]::Error.WriteLine(("REFUSING to run: GPU busy with a non-conforming compute process ({0}). " -f $gpuPre) +
                    'Rerun with -AllowBusy to accept the contamination risk.')
                $exitCode = 125
            }
        }

        if ($exitCode -ne 125) {
            if ($AllowBusy -and $gpuPre -ne 'clean' -and $gpuPre -ne 'probe-unavailable') {
                Write-Warning ("-AllowBusy given: running DESPITE a busy GPU ({0}). Any timing from this run is suspect." -f $gpuPre)
            }

            # --- provenance, captured at launch -------------------------------
            # The binary is identified from the command itself (which .exe, and
            # the llama.dll / ggml*.dll beside it), so the verdict names what it
            # RAN rather than what happened to be in the build tree. Captured
            # here -- under the hold, immediately before the command -- because
            # this is the last moment at which these facts are guaranteed to be
            # the ones the arm will execute against.
            if ($verdictReady -and $script:ArmProvenanceAvailable) {
                $provPre = Format-ArmProvenance -Origin 'launch' -Provenance (Get-ArmProvenance `
                    -SourceRoot (Join-Path $root 'third_party\llama.cpp') -RepoRoot $root `
                    -ArtifactPaths (Get-ProvenanceArtifacts -Command $Command))
            }

            & cmd.exe /d /s /c $Command
            $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
            # The command EXECUTED. Only such a run can be attested, and only such a run
            # may be refused for want of provenance: a busy refusal (125) and a mutex
            # timeout (124) describe the WRAPPER's decision, not the arm, and their codes
            # are part of the existing taxonomy (measured 2026-09-22: without this scope
            # the gate rewrote both 125 and 124 into 10).
            $commandRan = $true

            # --- GPU busy post-run probe --------------------------------------
            # A conforming process that ARRIVED mid-run still poisons the run;
            # catch it on the way out so the verdict says so.
            $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
            if ($gpuPost -ne 'clean') {
                Write-Warning ("GPU post-run probe: {0} - a non-conforming compute process may have run alongside this arm; timing suspect." -f $gpuPost)
            }

            # --- provenance, captured again on the way out --------------------
            # A rebuild or a source edit DURING a long arm is the same defect
            # class as one after it (measured 2026-09-22: the tree was rebuilt two
            # minutes after a smoke passed), and only a second capture can see it.
            if ($provPre) {
                $provPost = Format-ArmProvenance -Origin 'post-run' -Provenance (Get-ArmProvenance `
                    -SourceRoot (Join-Path $root 'third_party\llama.cpp') -RepoRoot $root `
                    -ArtifactPaths (Get-ProvenanceArtifacts -Command $Command))
            }
        }
    }
} finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
        Write-Host "Released GPU mutex $MutexName"
    }

    $mutex.Dispose()

    if ($verdictReady) {
        # The evidence this verdict already claims, persisted rather than only
        # printed (`Verdict: PASS (0) gpu_pre=clean gpu_post=clean`). Additive:
        # `gpu_pre_run`/`gpu_post_run` keep their existing values and meaning --
        # `clean` / `probe-unavailable` / "<pid> <exe>; ..." / `__unprobed__` -- and
        # `gpu_pre_status`/`gpu_post_status` spell the same evidence as a fixed
        # vocabulary (clean | busy | unavailable | not-probed) so a consumer never
        # substring-matches a sentence, and a probe that was NOT taken reads as
        # `not-probed` rather than being defaulted to `clean`.
        # `gpu_probe_source` names what produced the reading: a probe driven
        # through GPU_BUSY_PROBE_CMD is a STUB (host-only tests, never evidence
        # about the card), anything else is a real nvidia-smi query.
        $probeSource = if ($env:GPU_BUSY_PROBE_CMD) { 'stub' } else { 'nvidia-smi' }
        $fields = @{ timeout_seconds = $TimeoutSeconds; mutex_acquired = $acquired
                     gpu_pre_run = $gpuPre; gpu_post_run = $gpuPost
                     tag = $vTag; tag_source = $vTagSource
                     gpu_pre_status = (Get-GpuProbeStatus $gpuPre)
                     gpu_post_status = (Get-GpuProbeStatus $gpuPost)
                     gpu_probe_source = $probeSource }
        if ($exitCode -eq 125) { $fields['refused_busy'] = $gpuPre }
        # `provenance` is the launch capture; `provenance_post` and the delta only
        # exist for a run that actually executed. A run that was refused or timed
        # out has no launch capture, and Write-ArmVerdict then records its own
        # (origin=verdict-write) rather than leaving the field empty.
        if ($provPre) { $fields['provenance'] = $provPre }
        if ($provPre -and $provPost) {
            $fields['provenance_post'] = $provPost
            $fields['provenance_delta'] = @{
                source_changed_during_run = ("$($provPre.source_digest)" -ne "$($provPost.source_digest)")
                binary_changed_during_run = (($provPre.modules | ConvertTo-Json -Compress) -ne ($provPost.modules | ConvertTo-Json -Compress))
            }
        }
        # -AllowUnattested is the WRITER's half of the operator's -AllowUnprovenanced: one
        # escape at the CLI, honoured by both layers, or a waived run would still be recorded
        # UNCERTIFIED by the writer and the documented escape would be half a lie.
        $v = Write-ArmVerdict -Dir $vDir -Tag $vTag -ExitCode $exitCode -StartedUtc $vStarted `
            -Command $Command -MutexName $MutexName -Fields $fields -AllowUnattested:$AllowUnprovenanced
        Write-Host ("Verdict: {0} ({1}) gpu_pre={2} gpu_post={3} -> {4}" -f `
            $v.status, $exitCode, $gpuPre, $gpuPost, (Join-Path $vDir ($vTag + '.done')))

        # --- the gate, AT THE CHOKE POINT ------------------------------------
        # Every arm run through wrappers passes through HERE, so the attestation is
        # asserted here and no longer only in the callers: capture was systematic (the
        # verdict writer always records it) while enforcement was per-caller, so a
        # driver that forgot its own gate shipped uncertifiable evidence SILENTLY
        # (measured gap, audit 2026-09-22). Same rule as the callers' - a verdict that
        # cannot name the tree and the binary it ran is refused, and -AllowUnprovenanced
        # is the only escape (waiver line written by Assert-ArmProvenance itself).
        #
        # Taxonomy preserved: 125 = GPU busy, the command's own code = the command's
        # result, and a new 10 = the run happened but cannot be attributed to a
        # revision. The verdict is REWRITTEN on refusal so the verdict and the exit
        # code AGREE - a fabricated PASS beside a refusal is a contradiction this repo
        # has already recorded once. The command's own exit code is preserved in the
        # verdict as `command_exit` rather than being lost.
        $gateOk = $true
        if (-not $commandRan) {
            Write-Host ("PROVENANCE GATE: not applied - no command executed (refusal/timeout, exit {0}); 125 and 124 keep their meaning." -f $exitCode)
        } else {
        try {
            $gateOk = Assert-ArmProvenance -Dir $vDir -Tags @($vTag) -RepoRoot $root `
                -Label 'wrapper arm' -AllowUnprovenanced:$AllowUnprovenanced `
                -WaiverNote ("with-gpu-lock.ps1 tag=$vTag command_exit=$exitCode")
        } catch {
            Write-Host ("PROVENANCE GATE: the attestation itself failed ({0}) - treating it as REFUSED." -f $_.Exception.Message) -ForegroundColor Red
            $gateOk = $false
        }
        }
        if ($commandRan -and -not $gateOk) {
            $fields['command_exit'] = $exitCode
            $fields['refused_unprovenanced'] = $true
            $exitCode = 10
            $v = Write-ArmVerdict -Dir $vDir -Tag $vTag -ExitCode $exitCode -StartedUtc $vStarted `
                -Command $Command -MutexName $MutexName -Fields $fields
            Write-Host ("Verdict REWRITTEN: {0} ({1}) - refused at the wrapper for want of provenance." -f $v.status, $exitCode) -ForegroundColor Red
            Write-Host '  The run is still on disk; its evidence is not attributable to a revision.'
        }
    }
}

exit $exitCode