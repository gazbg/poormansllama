# T-GPLR-58 A3/A4 arm driver: predictor legs at the declared short shape, all legs inside
# ONE acquisition of the shared-GPU mutex.
#
# Contract (T-GPLR-58; both sessions were PRE-AUTHORIZED by the owner 2026-09-23):
# the shape is the declared 885-token non-echoing prompt (`build/calc-prompt.txt`),
# `--tg-tokens 256` and `--rep-penalty 1.15` on EVERY leg; the prediction arms run at the
# SHALLOW depth K=2 (the depth the charter case needs) and the retention reference at K=4.
#
# A3 legs: off@K2 x2, the coact count-matched null/arm pair, then retref@K4.
# A4 legs: off@K2 x2, a SEPARATE count-matched null and arm for bar09 and for fusion,
#          then retref@K4. The null is paired by the arm that supplies its per-row count;
#          one shared fixed-B null would not satisfy the contract for the barred arm.
#
# QUANTITIES (pre-registered by the block; this driver computes NONE of them itself except
# the arithmetic cross-checks noted below -- the single owner of every band is
# `tools/check-tg-predict.py`, run here in the same session over the whole leg set):
#   M1  margin gain of the arm over the session's off baseline, against the control spread
#   M2  share(null) - share(off): how much of the merged gain is only pool growth
#   M3  same-session arm@K2 share against the K=4 retention share
#   M4  margin_share / M4n the null-corrected (structure-only) gain
#   M5  staged-volume growth arm/off (soft band <= +5%)
#   M7a the reap-ordering floor (lead_min >= 2 wherever a fill was consumed)
#   M7b the late-fill rate band matched/(consumed+matched) <= 0.5 on the ARM leg
#   fill-room  how many layer-stages ever left room for a fill AT ALL
#              (`tools/analyze-gplr58-fill-room.py`): the ceiling every M1 gain
#              must be read inside, because a saturated cache admits nothing.
#
# ONE ARITHMETIC DIFFERENCE THE DRIVER OWNS, and ONE IT CROSS-CHECKS:
#   * the CONTROL SPREAD |share(off1) - share(off2)| is the driver's own (the A3 contract
#     requires M1 to exceed it; the checker's baseline is the MEAN of the supplied off legs);
#   * M3 is CROSS-CHECKED, not worked out privately: the checker owns the band -- its
#     denominator is the session's K=4 retention reference, preferring an `off`-role leg and
#     accepting this contract's `retref@K4` (fixed 2026-09-23, when keying on the `off` role
#     alone made the checker print NOT COMPUTABLE on a session that HAD the reference and
#     forced the driver to difference the two shares by hand). The driver recomputes the
#     same difference from its own parsed shares; a disagreement is a WARNING in the fields,
#     never silently reconciled.
#
# WHAT EACH LEG MUST PROVE ABOUT ITSELF before it is scored (a missing proof is a leg FAIL,
# not a note):
#   * the run exited 0 and `--check-telemetry` accepts its own trace;
#   * the RESOLVED state matches the declaration, read from the trace and never from the
#     command line: `rowset_init.retain_k` equals the leg's declared K, and
#     `run_start.row_spec_mode` is present-and-2 exactly on the legs declared speculative
#     (the P6 direction: an `off` leg carrying speculation cannot serve as a baseline);
#   * the arm/null legs name their own arm in the trace (`run_start.row_spec_arm`) and the
#     null legs show `row_spec_null == true`;
#   * the leg is not REFUSED by A2's checker (exit 2 prints no numbers: a leg that cannot be
#     justified is not evidence), and the fill-room analyzer accepts it (endpoint exit 2 is
#     on the reactive path, which has no per-layer cache and is not a prediction leg).
#
# FAILURE CONTRACT (the block's own): a REFUSED leg or a FAIL-INVARIANT (P1/P4/M7a) HALTS
# the series at that point -- the remaining legs would be measuring a broken mechanism. The
# drive order therefore puts the M3 reference last, so a broken arm cannot spend its tail
# budget: `retref@K4` is SKIPPED and the halt is recorded in the session verdict.
#
# SESSION VERDICT SEMANTICS (the A4 precedent, deliberate): exit 0 means every leg that RAN
# is usable and the outcome is reported honestly, INCLUDING the pre-registered negative
# outcome ("predictors do not buy depth-equivalent hits at this headroom"), which the block
# states is the decision-useful result. The exit code never encodes which way the measurement
# came out; the band fields in each verdict do. Exit 1 = a leg failed its integrity gates or
# the control is unusable, 2 = a leg was refused by its own checker, 124 = mutex timeout,
# 125 = GPU busy, 10 = provenance.
#
# Dry run (no GPU, no mutex, no CUDA): -ValidateOnly prints the exact commands and resolves
# every input, so the whole session can be checked before a mutex window is spent.
param(
    [ValidateSet('a3', 'a4')][string] $Session = 'a3',
    [switch] $ValidateOnly,
    [switch] $RefreshPairReportsOnly,
    [switch] $AllowUnprovenanced,
    [switch] $ExternalMutex
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$exe  = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$dll  = Join-Path $base 'bin\Release\llama.dll'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
# The DECLARED short shape: the 885-token non-echoing prompt, taken whole.
$prompt = Join-Path $root 'build\calc-prompt.txt'
$ppTokens = 885
$tgTokens = 256
$ctxSize  = 1280
$specBudget = 8
$verdictDir = Join-Path $base 'verdicts'
$checker  = Join-Path $root 'tools\check-tg-predict.py'
$fillroom = Join-Path $root 'tools\analyze-gplr58-fill-room.py'
$sessionPrefix = "gplr58-$Session"
$summaryPath = Join-Path $base ($sessionPrefix + '-summary.json')
. (Join-Path $root 'tools\arm-verdict.ps1')

# role: 'off' | 'null' | 'arm' | 'retref' (the checker's OWN role vocabulary; `retref` is
# the contract's name for the K=4 retention leg and is NOT the same role token as 'off').
$legs = @(
    @{ tag = ($sessionPrefix + '-off1');    role = 'off';    retain = 2; spec = 0; arm = '';      null = 0; note = 'incumbent retention, no prediction (baseline)' },
    @{ tag = ($sessionPrefix + '-off2');    role = 'off';    retain = 2; spec = 0; arm = '';      null = 0; note = 'the same configuration again: the session control SPREAD' }
)
if ($Session -eq 'a3') {
    $armNames = @('coact')
    $legs += @(
        @{ tag = ($sessionPrefix + '-null');  role = 'null'; retain = 2; spec = 2; arm = 'coact'; null = 1; note = 'coact count-matched structure-free null' },
        @{ tag = ($sessionPrefix + '-coact'); role = 'arm';  retain = 2; spec = 2; arm = 'coact'; null = 0; note = 'unbarred co-activation at budget 8' }
    )
} else {
    $armNames = @('bar09', 'fusion')
    foreach ($armName in $armNames) {
        $legs += @(
            @{ tag = ($sessionPrefix + '-null-' + $armName); role = 'null'; retain = 2; spec = 2; arm = $armName; null = 1; note = ($armName + ' count-matched structure-free null') },
            @{ tag = ($sessionPrefix + '-' + $armName);        role = 'arm';  retain = 2; spec = 2; arm = $armName; null = 0; note = ($armName + ' pre-registered arm at budget 8') }
        )
    }
}
$legs += @(
    @{ tag = ($sessionPrefix + '-retref4'); role = 'retref'; retain = 4; spec = 0; arm = ''; null = 0; note = 'retention-only at K=4: the M3 denominator' }
)

$missing = @()
foreach ($p in @($exe, $dll, $prompt, $checker, $fillroom)) { if (-not (Test-Path -LiteralPath $p)) { $missing += $p } }
if (-not (Test-Path -LiteralPath $model)) { $missing += $model }
if ($missing.Count -gt 0) {
    Write-Host 'MISSING INPUTS:'
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

function Get-LegArgs {
    param($L)
    # NOTE: the accumulator must NOT be named $a - PowerShell variable names are
    # case-insensitive, so $a would REBIND the $L parameter's neighbours and silently drop
    # the mode flags (the T-GPLR-53 A4 lesson).
    $out = @('--model', $model, '--prompt-file', $prompt,
             '--pp-tokens', "$ppTokens", '--ctx-size', "$ctxSize",
             '--tg-tokens', "$tgTokens", '--rep-penalty', '1.15',
             '--residency', 'moe-expert-bank', '--residency-tg',
             '--residency-tg-retain', "$($L.retain)",
             '--pp-schedule', 'batch-major', '--ubatch', '512',
             '--layer-slots', '4', '--kv-cache', 'q8',
             '--greedy', '--no-mtp', '--pp-no-logits', '--allow-full-residency',
             '--telemetry', (Join-Path $base ($L.tag + '.jsonl')))
    if ($L.spec -gt 0) {
        $out += @('--residency-tg-spec', "$($L.spec)",
                  '--residency-tg-spec-arm', $L.arm,
                  '--residency-tg-spec-budget', "$specBudget",
                  '--residency-tg-spec-null', "$($L.null)")
    }
    return $out
}

function Get-PairLegs {
    param([string] $ArmName)
    $offLegs = @($legs | Where-Object { $_.role -eq 'off' })
    $nullLeg = @($legs | Where-Object { $_.role -eq 'null' -and $_.arm -eq $ArmName })[0]
    $armLeg  = @($legs | Where-Object { $_.role -eq 'arm'  -and $_.arm -eq $ArmName })[0]
    $retLeg  = @($legs | Where-Object { $_.role -eq 'retref' })[0]
    if ($offLegs.Count -ne 2 -or $null -eq $nullLeg -or $null -eq $armLeg -or $null -eq $retLeg) {
        return @()
    }
    return @($offLegs) + @($nullLeg) + @($armLeg) + @($retLeg)
}

function Get-PairReport {
    param([string] $ArmName)
    $pairLegs = @(Get-PairLegs -ArmName $ArmName)
    if ($pairLegs.Count -ne 5) {
        return [ordered]@{
            complete = $false
            checker_exit = 1
            session_exit = 1
            reason = 'pair selection did not produce exactly two off legs, the named null/arm, and one retref leg'
        }
    }

    $pairArgs = @()
    $missingTraces = @()
    foreach ($pairLeg in $pairLegs) {
        $pairTrace = Join-Path $base ($pairLeg.tag + '.jsonl')
        if (Test-Path -LiteralPath $pairTrace) {
            $pairArgs += @('--leg', ('{0}={1}' -f $pairLeg.role, $pairTrace))
        } else {
            $missingTraces += $pairLeg.tag
        }
    }
    if ($missingTraces.Count -gt 0) {
        return [ordered]@{
            complete = $false
            checker_exit = 1
            session_exit = 1
            reason = 'missing pair trace(s): ' + ($missingTraces -join ',')
        }
    }

    $pairJson = Join-Path $base ($sessionPrefix + '-bands-' + $ArmName + '.json')
    Remove-Item -LiteralPath $pairJson -ErrorAction SilentlyContinue
    $pairOut = [string] (& python $checker @pairArgs --session ($sessionPrefix + '-' + $ArmName) --json $pairJson 2>&1 | Out-String)
    $pairExit = $LASTEXITCODE
    if ($pairExit -ne 0) {
        return [ordered]@{
            complete = $false
            checker_exit = $pairExit
            session_exit = $pairExit
            report = $pairOut
        }
    }

    $pairPayload = Get-Content -Raw -LiteralPath $pairJson | ConvertFrom-Json
    $pairOff = @($pairPayload.legs | Where-Object { $_.role -eq 'off' })
    $pairNull = @($pairPayload.legs | Where-Object { $_.role -eq 'null' })[0]
    $pairArm = @($pairPayload.legs | Where-Object { $_.role -eq 'arm' })[0]
    $pairRet = @($pairPayload.legs | Where-Object { $_.role -eq 'retref' })[0]
    if ($pairOff.Count -ne 2 -or $null -eq $pairNull -or $null -eq $pairArm -or $null -eq $pairRet) {
        return [ordered]@{
            complete = $false
            checker_exit = $pairExit
            session_exit = 1
            report = $pairOut
            reason = 'checker output did not contain two off legs plus the named null, arm and retref leg'
        }
    }

    $pairOffShares = @($pairOff | ForEach-Object { [double] $_.share })
    $pairOffMean = ($pairOffShares | Measure-Object -Average).Average
    $pairOffStats = $pairOffShares | Measure-Object -Minimum -Maximum
    $pairSpread = [math]::Round($pairOffStats.Maximum - $pairOffStats.Minimum, 6)
    $pairArmShare = [double] $pairArm.share
    $pairNullShare = [double] $pairNull.share
    $pairRetShare = [double] $pairRet.share
    $shareGain = [math]::Round($pairArmShare - $pairNullShare, 6)
    $denom = 1.0 - $pairOffMean
    $armMargin = ($pairArmShare - $pairOffMean) / $denom
    $nullMargin = ($pairNullShare - $pairOffMean) / $denom
    $marginGain = $armMargin - $nullMargin
    $consumed = [int] $pairArm.spec.consumed
    $cancelled = [int] $pairArm.spec.cancelled
    return [ordered]@{
        complete = $true
        checker_exit = 0
        session_exit = $(if ($consumed -gt 0) { 1 } else { 0 })
        report = $pairOut
        pair_roles = @($pairLegs | ForEach-Object { $_.role })
        off_share_mean = $pairOffMean
        control_spread = $pairSpread
        null_share = $pairNullShare
        arm_share = $pairArmShare
        retref4_share = $pairRetShare
        share_gain_vs_paired_null = $shareGain
        arm_margin_share = $armMargin
        null_margin_share = $nullMargin
        margin_gain_vs_paired_null = $marginGain
        M1_scoreable = ($consumed -gt 0)
        M1_unique_margin_share = $null
        M1_exceeds_control_spread = $null
        M3_arm_vs_retref4 = [math]::Round($pairArmShare - $pairRetShare, 6)
        M5_arm_vs_off_staged = [double] $pairArm.staged_bytes / (($pairOff | Measure-Object -Property staged_bytes -Average).Average)
        consumed = $consumed
        cancelled = $cancelled
        P4_exercised = ($cancelled -gt 0)
    }
}

if ($ValidateOnly) {
    $exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
    $dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
    Write-Host "VALIDATE OK  exe $exeSha  dll $dllSha"
    Write-Host "model  $model"
    Write-Host ("prompt $prompt ({0} PP tokens, ctx {1}, TG {2}, rep-penalty 1.15 on every leg)" -f $ppTokens, $ctxSize, $tgTokens)
    Write-Host ("checker {0}`nfill-room {1}" -f $checker, $fillroom)
    foreach ($L in $legs) {
        Write-Host ("LEG {0}  role={1} retain=K{2} spec={3} arm={4} null={5}  ({6})" -f `
            $L.tag, $L.role, $L.retain, $L.spec, $(if ($L.spec -gt 0) { $L.arm } else { '-' }), $L.null, $L.note)
        Write-Host ("  " + $exe + ' ' + ((Get-LegArgs $L) -join ' '))
    }
    foreach ($armName in $armNames) {
        $pairLegs = @(Get-PairLegs -ArmName $armName)
        $pairShape = @($pairLegs | ForEach-Object { $_.role + ':' + $_.arm }) -join ','
        $expectedShape = 'off:,off:,null:' + $armName + ',arm:' + $armName + ',retref:'
        if ($pairLegs.Count -ne 5 -or $pairShape -ne $expectedShape) {
            Write-Error ("PAIR CHECK FAILED for {0}: got {1}" -f $armName, $pairShape)
            exit 1
        }
        Write-Host ("PAIR CHECK {0}: {1} (5 legs)" -f $armName, $pairShape)
    }
    Write-Host ("ONE mutex hold for {0} legs; the GPU busy gate refuses (exit 125) on a non-clean pre-run probe." -f $legs.Count)
    Write-Host "Then: one full-session checker report plus one paired checker report per arm (its own count-matched null)."
    Write-Host "      The driver also checks ordered TG route identity and greedy token identity across every leg."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("{0}: exe {1} dll {2}" -f $sessionPrefix, $exeSha, $dllSha)
Write-Host "  (a new binary pair means older controls are a DIFFERENT binary; this session's own off legs are the control)"

if ($RefreshPairReportsOnly) {
    if (-not (Test-Path -LiteralPath $summaryPath)) {
        Write-Error "REFRESH REFUSED: missing summary $summaryPath"
        exit 1
    }
    $existingSummary = Get-Content -Raw -LiteralPath $summaryPath | ConvertFrom-Json
    if ($existingSummary.session -ne $sessionPrefix -or
        $existingSummary.shape.exe_sha256_16 -ne $exeSha -or
        $existingSummary.shape.llama_dll_sha256_16 -ne $dllSha) {
        Write-Error 'REFRESH REFUSED: summary session or binary provenance does not match the current inputs'
        exit 10
    }
    $legRows = $existingSummary.legs
    $badRows = @($legs | Where-Object {
        $rowProperty = $legRows.PSObject.Properties[$_.tag]
        $null -eq $rowProperty -or $rowProperty.Value.leg_exit -ne 0
    })
    if ($badRows.Count -gt 0) {
        Write-Error ('REFRESH REFUSED: missing or failed leg(s): ' + (@($badRows | ForEach-Object { $_.tag }) -join ', '))
        exit 1
    }
    if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($legs | ForEach-Object { $_.tag }) `
            -Label ($sessionPrefix + ' recorded leg') -AllowUnprovenanced:$AllowUnprovenanced)) {
        Write-Error 'REFRESH REFUSED: recorded leg provenance is not current'
        exit 10
    }

    $refreshExit = 0
    $pairReports = [ordered]@{}
    foreach ($armName in $armNames) {
        $pairReport = Get-PairReport -ArmName $armName
        $pairReports[$armName] = $pairReport
        if ($pairReport['complete']) {
            Write-Host ("  {0} paired: off {1:F6} | null {2:F6} | arm {3:F6} | share gain {4:+0.000000;-0.000000;0.000000} | margin gain {5:+0.000000;-0.000000;0.000000} | consumed {6} | cancelled {7}" -f `
                $armName, $pairReport['off_share_mean'], $pairReport['null_share'], $pairReport['arm_share'], `
                $pairReport['share_gain_vs_paired_null'], $pairReport['margin_gain_vs_paired_null'], `
                $pairReport['consumed'], $pairReport['cancelled'])
            if ($pairReport['consumed'] -gt 0) {
                Write-Warning ("M1 for {0}: fills were consumed, but aggregate trace telemetry cannot identify UNIQUE per-row hits; M1 is not scoreable from this schema." -f $armName)
            }
        } else {
            Write-Warning ("PAIR REPORT {0} INCOMPLETE: {1}" -f $armName, $pairReport['reason'])
        }
        if ($refreshExit -eq 0) { $refreshExit = [int] $pairReport['session_exit'] }
    }

    $existingSummary.bands.pair_reports = $pairReports
    $refreshInfo = [ordered]@{
        refreshed_utc = (Get-Date).ToUniversalTime().ToString('o')
        mode = 'host-only paired checker refresh; no model, CUDA, GPU or mutex'
        source_summary = $summaryPath
        checker = $checker
    }
    $existingSummary | Add-Member -NotePropertyName pair_reports_refresh -NotePropertyValue $refreshInfo -Force
    [System.IO.File]::WriteAllText($summaryPath,
        ($existingSummary | ConvertTo-Json -Depth 10), (New-Object System.Text.UTF8Encoding($false)))
    Write-Host ("  paired reports refreshed -> {0}" -f $summaryPath)
    Write-Host ("{0} REFRESH {1}" -f $sessionPrefix, $(if ($refreshExit -eq 0) { 'PASS (recorded legs and both paired reports are usable)' } else { 'FAIL' }))
    exit $refreshExit
}

# --------------------------------------------------- arm-verdict handling FIRST
foreach ($L in $legs) { Clear-ArmVerdict -Dir $verdictDir -Tag $L.tag | Out-Null }
$sessionStart = (Get-Date).ToUniversalTime().ToString('o')
foreach ($L in $legs) {
    Write-ArmStart -Dir $verdictDir -Tag $L.tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' `
        -LogPath (Join-Path $base ($L.tag + '.log')) -Command ($exe + ' ' + ((Get-LegArgs $L) -join ' ')) | Out-Null
}

$mutex = $null
$acquired = $false
$ownsMutex = -not $ExternalMutex
$sessionExit = 1
$legRows = [ordered]@{}
$halted = ''
try {
    if ($ExternalMutex) {
        Write-Host '  mutex held by tools/with-gpu-lock.ps1 (outer wrapper)'
    } else {
        $mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
        try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
        if (-not $acquired) {
            foreach ($L in $legs) {
                Write-ArmVerdict -Dir $verdictDir -Tag $L.tag -ExitCode 124 -StartedUtc $sessionStart -ErrorText 'mutex timeout' | Out-Null
            }
            exit 124
        }
        Write-Host '  mutex held'
    }

    $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_pre_run = {0}" -f $gpuPre)
    if ($gpuPre -ne 'clean') {
        if ($gpuPre -eq 'probe-unavailable') { Write-Warning 'GPU busy probe unavailable; proceeding UNVERIFIED.' }
        else {
            foreach ($L in $legs) {
                Write-ArmVerdict -Dir $verdictDir -Tag $L.tag -ExitCode 125 -StartedUtc $sessionStart `
                    -ErrorText "GPU busy: $gpuPre" -Fields @{ gpu_pre_run = $gpuPre } | Out-Null
            }
            Write-Host ("REFUSED: GPU busy ({0})" -f $gpuPre)
            exit 125
        }
    }

    $sessionExit = 0
    foreach ($L in $legs) {
        $tag = $L.tag
        if ($halted -ne '') {
            Write-Host ("== LEG {0} SKIPPED (series halted: {1}) ==" -f $tag, $halted)
            Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 3 -StartedUtc $sessionStart `
                -Fields @{ skipped = $true; halted_because = $halted; role = $L.role } | Out-Null
            continue
        }
        $trace = Join-Path $base ($tag + '.jsonl')
        $log   = Join-Path $base ($tag + '.log')
        Remove-Item $trace, $log -ErrorAction SilentlyContinue
        $fields = [ordered]@{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
                              role = $L.role; retain_k_declared = $L.retain
                              spec_mode_declared = $L.spec; arm_declared = $L.arm; null_declared = $L.null
                              note = $L.note; trace = $trace; log = $log; gpu_pre_run = $gpuPre }
        $fail = @()
        Write-Host ("== LEG {0} (role {1}, K={2}, spec {3}) ==" -f $tag, $L.role, $L.retain, $L.spec)

        $t0 = Get-Date
        & $exe @(Get-LegArgs $L) > $log 2>&1
        $legExit = $LASTEXITCODE
        $fields['leg_exit'] = $legExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  LEG EXIT {0} ({1}s)" -f $legExit, $fields['duration_s'])
        if ($legExit -ne 0) { $fail += "leg exited $legExit" }

        if ($legExit -eq 0) {
            $gpuPostLeg = Format-GpuBusy (Get-GpuComputeApps)
            $fields['gpu_post_run'] = $gpuPostLeg
            if ($gpuPostLeg -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - the leg is suspect." -f $gpuPostLeg) }

            $lines = @(Get-Content $trace)
            $routeText = [System.Text.StringBuilder]::new()
            $tgIds = ''
            foreach ($line in $lines) {
                $eventObj = $line | ConvertFrom-Json
                if ($eventObj.event -eq 'rowset_fetch') {
                    [void]$routeText.Append(('{0},{1},{2};' -f $eventObj.ubatch, $eventObj.layer, $eventObj.union_ids))
                } elseif ($eventObj.event -eq 'tg_tokens') {
                    $tgIds = (@($eventObj.ids) -join ',')
                }
            }
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try {
                $routeDigest = ([System.BitConverter]::ToString(
                    $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($routeText.ToString()))) -replace '-', '').ToLower()
            } finally {
                $sha.Dispose()
            }
            $fields['tg_route_digest'] = $routeDigest
            $fields['tg_ids'] = $tgIds

            $rs = @($lines | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $init = @($lines | Where-Object { $_ -like '*"event":"rowset_init"*' })[0] | ConvertFrom-Json
            $fields['resolved_retain_k'] = [int] $init.retain_k
            $fields['resolved_top_k'] = [int] $init.top_k
            $fields['resolved_n_expert'] = [int] $init.n_expert
            $hasMode = $null -ne ($rs.PSObject.Properties['row_spec_mode'])
            $fields['run_spec_mode_present'] = $hasMode
            $fields['resolved_spec_mode'] = if ($hasMode) { [int] $rs.row_spec_mode } else { 0 }

            if ($fields['resolved_retain_k'] -ne $L.retain) {
                $fail += "resolved retain_k $($fields['resolved_retain_k']) != declared K$($L.retain)"
            }
            if ($L.spec -gt 0) {
                if (-not $hasMode -or $fields['resolved_spec_mode'] -ne 2) {
                    $fail += 'declared speculative but run_start carries no row_spec_mode==2: the mode did not engage'
                }
                $armField = if ($null -ne $rs.PSObject.Properties['row_spec_arm']) { [string] $rs.row_spec_arm } else { '' }
                $fields['resolved_arm'] = $armField
                if ($armField -ne $L.arm) { $fail += "resolved arm '$armField' != declared '$($L.arm)'" }
                $nullField = if ($null -ne $rs.PSObject.Properties['row_spec_null']) { [bool] $rs.row_spec_null } else { $false }
                $fields['resolved_null'] = $nullField
                if ([int]$nullField -ne [int]$L.null) { $fail += "resolved row_spec_null $nullField != declared $($L.null)" }
                $fields['spec_reserve'] = if ($null -ne $rs.PSObject.Properties['row_spec_reserve']) { [int] $rs.row_spec_reserve } else { -1 }
                $fields['spec_budget'] = if ($null -ne $rs.PSObject.Properties['row_spec_budget']) { [int] $rs.row_spec_budget } else { -1 }
            } elseif ($hasMode) {
                $fail += 'an off/retref leg carries row_spec_mode: the baseline cannot be read as unpolluted (P6)'
            }

            # --- the leg's own integrity, from A2's single owner ---------------------
            # The quantities are read from the checker's own --json payload, never from a
            # regex over its prose: the printed table is for a human, and a parser that
            # reads it would break the day the table is reformatted.
            $legJson = Join-Path $base ($tag + '-bands.json')
            Remove-Item $legJson -ErrorAction SilentlyContinue
            $legOut = (& python $checker --leg "$($L.role)=$trace" --json $legJson 2>&1 | Out-String)
            $fields['checker_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -eq 2) {
                $fail += "A2 REFUSED this leg (exit 2): a leg that cannot be justified is not evidence"
                $fields['checker_refusal'] = (($legOut.Trim() -split "`n") | Select-Object -Last 3) -join ' | '
            } elseif ($LASTEXITCODE -eq 1) {
                $fail += 'A2 reported FAIL-INVARIANT on this leg (the block halts the series there)'
                $fields['checker_invariant_fail'] = $true
            } elseif ($LASTEXITCODE -ne 0) {
                $fail += "A2 checker exited $LASTEXITCODE"
            }
            if (Test-Path -LiteralPath $legJson) {
                $lp = Get-Content -Raw -LiteralPath $legJson | ConvertFrom-Json
                if (@($lp.legs).Count -ge 1) {
                    $l0 = @($lp.legs)[0]
                    $fields['leg_share'] = [double] $l0.share
                    $fields['leg_margin'] = [double] $l0.margin
                    $fields['leg_stages'] = [int] $l0.stages
                    $fields['leg_spec_events'] = [int] $l0.spec_events
                    $fields['leg_residual'] = $l0.residual
                    if ($null -ne $l0.lead_median) { $fields['leg_lead_median'] = $l0.lead_median }
                    foreach ($k in @('submitted','consumed','matched_by_demand','cancelled','lead_min','lead_max',
                                     'refused_capacity','dup_refused','waited_by_demand','cancelled_generation',
                                     'cancelled_demand','cancelled_capacity')) {
                        if ($null -ne $l0.spec.PSObject.Properties[$k]) { $fields['spec_' + $k] = $l0.spec.$k }
                    }
                    $fields['checker_failures'] = (($lp.failures) -join ',')
                }
            }

            # --- the fill-room ceiling, per prediction leg --------------------------
            if ($L.spec -gt 0) {
                $frOut = (& python $fillroom --leg $trace 2>&1 | Out-String)
                $fields['fill_room_exit'] = $LASTEXITCODE
                if ($LASTEXITCODE -ne 0) { $fail += "fill-room analyzer refused/exited $LASTEXITCODE on a prediction leg" }
                $fm = [regex]::Match($frOut, 'TOTAL:\s*(\d+)/(\d+) layer-stages leave room beyond the reserve \(([0-9.]+)%\)')
                if ($fm.Success) {
                    $fields['fill_room_admittable'] = [int] $fm.Groups[1].Value
                    $fields['fill_room_stages'] = [int] $fm.Groups[2].Value
                    $fields['fill_room_pct'] = [double] $fm.Groups[3].Value
                } else { $fail += 'the fill-room analyzer printed no TOTAL line: the ceiling is unmeasured' }
                $fields['fill_room_summary'] = (($frOut.Trim() -split "`n") | Where-Object { $_ -match 'admittable|full by stage' } | ForEach-Object { $_.Trim() }) -join ' | '
            }

            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }                    Write-Host ("  share {0} | margin {1} | spec mode {2} | retain K{3} | checker exit {4}{5}" -f `
                        $(if ($fields.Contains('leg_share')) { $fields['leg_share'] } else { 'n/a' }), `
                        $(if ($fields.Contains('leg_margin')) { $fields['leg_margin'] } else { 'n/a' }), `
                        $fields['resolved_spec_mode'], $fields['resolved_retain_k'], $fields['checker_exit'], `
                        $(if ($fields.Contains('fill_room_pct')) { (" | fill-room {0}%" -f $fields['fill_room_pct']) } else { '' }))
        }

        $legExitCode = if ($fail.Count -gt 0) { 1 } else { 0 }
        if ($fail.Count -gt 0) {
            $fields['fail_reason'] = ($fail -join '; ')
            $sessionExit = 1
            # the block's failure contract: a refused leg or a FAIL-INVARIANT halts the series
            $halted = "$tag : " + ($fail -join '; ')
        }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $legExitCode -StartedUtc $sessionStart `
            -LogPath $log -Fields $fields | Out-Null
        $legRows[$tag] = $fields
    }

    # ---------------- the bands, from their single owner, over the whole leg set ---------
    $ran = @($legs | Where-Object { $legRows.Contains($_.tag) -and $legRows[$_.tag]['leg_exit'] -eq 0 })
    if ($ran.Count -lt 2) {
        Write-Host ("  {0} RESULT: UNUSABLE - fewer than two legs produced a trace, so no band can be read" -f $sessionPrefix)
        if ($sessionExit -eq 0) { $sessionExit = 1 }
    } else {
        $legArgs = @()
        foreach ($L in $legs) {
            $tr = Join-Path $base ($L.tag + '.jsonl')
            if (Test-Path -LiteralPath $tr) { $legArgs += @('--leg', ("{0}={1}" -f $L.role, $tr)) }
        }
        $jsonOut = Join-Path $base ($sessionPrefix + '-bands.json')
        $bandOut = (& python $checker @legArgs --session $sessionPrefix --json $jsonOut 2>&1 | Out-String)
        $sessionCheckerExit = $LASTEXITCODE
        $bandLines = @($bandOut.Trim() -split "`n")
        Write-Host '  --- A2 band report (its own lines, unedited) ---'
        $bandLines | Where-Object { $_ -match '^\s*(share|margin|M1|M2|M3|M4|M4n|M5|M7|P1|P2|P4|P6|off |arm |null |retref|VERDICT|spec|lead)' } |
            ForEach-Object { Write-Host ('  | ' + $_) }

        $bands = [ordered]@{ checker_exit = $sessionCheckerExit; report = $bandOut }
        if (Test-Path -LiteralPath $jsonOut) {
            $payload = Get-Content -Raw -LiteralPath $jsonOut | ConvertFrom-Json
            $byRole = @{}
            foreach ($lg in $payload.legs) {
                if (-not $byRole.ContainsKey($lg.role)) { $byRole[$lg.role] = @() }
                $byRole[$lg.role] += $lg
                $bands[('share_' + $lg.role + '_K' + $lg.retain_k + '_' + (Split-Path -Leaf $lg.path))] = $lg.share
            }
            if ($byRole.ContainsKey('off')) {
                $offShares = @($byRole['off'] | ForEach-Object { [double] $_.share })
                $baseShare = ($offShares | Measure-Object -Average).Average
                $fields0 = $offShares | Measure-Object -Minimum -Maximum
                $spread = [math]::Round($fields0.Maximum - $fields0.Minimum, 4)
                $bands['off_share_mean'] = [math]::Round($baseShare, 4)
                $bands['off_legs'] = $offShares.Count
                $bands['control_spread_share'] = $spread
                Write-Host ("  off baseline share (mean of {0}) = {1}; CONTROL SPREAD = {2} (the M1 hurdle)" -f `
                    $offShares.Count, [math]::Round($baseShare, 4), $spread)
            }
            if ($Session -eq 'a3' -and $byRole.ContainsKey('arm')) {
                $armShare = [double] $byRole['arm'][0].share
                if ($bands.Contains('off_share_mean')) {
                    $gain = [math]::Round($armShare - [double] $bands['off_share_mean'], 4)
                    $bands['M1_margin_gain_share'] = $gain
                    $bands['M1_exceeds_control_spread'] = ($gain -gt [double] $bands['control_spread_share'])
                    Write-Host ("  M1 (driver): arm share {0} - off {1} = {2:+0.0000;-0.0000} vs control spread {3} -> {4}" -f `
                        $armShare, $bands['off_share_mean'], $gain, $bands['control_spread_share'], `
                        $(if ($bands['M1_exceeds_control_spread']) { 'ABOVE the spread' } else { 'INSIDE the spread' }))
                }
                # M3: read the CHECKER's own line, then recompute the same difference from the
                # parsed shares. The checker is the band's single owner; the driver's copy is a
                # cross-check that can only disagree LOUDLY.
                $m3m = [regex]::Match($bandOut, 'M3\s+same-session arm@K2 vs (\w+)@K4\s*:\s*([0-9.]+) vs ([0-9.]+) = ([+-][0-9.]+)')
                if ($m3m.Success) {
                    $denRole = $m3m.Groups[1].Value
                    $denShare = [double] $m3m.Groups[3].Value
                    $m3Checker = [double] $m3m.Groups[4].Value
                    $m3Driver = [math]::Round($armShare - $denShare, 4)
                    $bands['M3_checker'] = $m3Checker
                    $bands['M3_denominator_role'] = $denRole
                    $bands['M3_driver_crosscheck'] = $m3Driver
                    $bands['M3_crosscheck_agrees'] = ([math]::Abs($m3Checker - $m3Driver) -le 0.0001)
                    Write-Host ("  M3 (checker, denominator role {0}): {1:+0.0000;-0.0000}; driver cross-check {2:+0.0000;-0.0000} -> {3}" -f `
                        $denRole, $m3Checker, $m3Driver, `
                        $(if ($bands['M3_crosscheck_agrees']) { 'AGREE' } else { 'DISAGREE - investigate before reading M3' }))
                    if (-not $bands['M3_crosscheck_agrees']) {
                        Write-Warning 'M3: the checker and the driver disagree on the same two shares; the leg set is not self-consistent.'
                    }
                } else {
                    $bands['M3_checker'] = 'NOT COMPUTABLE (the checker printed no M3 line for this leg set)'
                    Write-Host '  M3: the checker printed no M3 line for this leg set (no K=4 retention reference)'
                }
            }
        }
        if ($sessionCheckerExit -eq 2) {
            Write-Host ("  {0} RESULT: the band report was REFUSED (exit 2) - no numbers, no verdict" -f $sessionPrefix)
            $sessionExit = 2
        } elseif ($sessionCheckerExit -eq 1) {
            Write-Host ("  {0} RESULT: FAIL-INVARIANT under A2 - the numbers are reported for the record and are NOT usable as evidence" -f $sessionPrefix)
            $sessionExit = 1
        }

        # A4 has two different per-row proposal counts. Each arm is therefore paired with
        # its OWN null in a five-leg checker call; the seven-leg report above remains the
        # common integrity view. This prevents the old fixed-B null from being used for the
        # barred arm and prevents M4n from being printed as ambiguous.
        $bands['pair_reports'] = [ordered]@{}
        foreach ($armName in $armNames) {
            $pairReport = Get-PairReport -ArmName $armName
            $bands['pair_reports'][$armName] = $pairReport
            if ($pairReport['complete']) {
                Write-Host ("  {0} paired: off {1:F6} | null {2:F6} | arm {3:F6} | share gain {4:+0.000000;-0.000000;0.000000} | margin gain {5:+0.000000;-0.000000;0.000000} | consumed {6} | cancelled {7}" -f `
                    $armName, $pairReport['off_share_mean'], $pairReport['null_share'], $pairReport['arm_share'], `
                    $pairReport['share_gain_vs_paired_null'], $pairReport['margin_gain_vs_paired_null'], `
                    $pairReport['consumed'], $pairReport['cancelled'])
                if ($pairReport['consumed'] -gt 0) {
                    Write-Warning ("M1 for {0}: fills were consumed, but aggregate trace telemetry cannot identify UNIQUE per-row hits; M1 is not scoreable from this schema." -f $armName)
                }
            } else {
                Write-Warning ("PAIR REPORT {0} INCOMPLETE: {1}" -f $armName, $pairReport['reason'])
            }
            if ($sessionExit -eq 0) { $sessionExit = [int] $pairReport['session_exit'] }
        }

        $identityTags = @($ran | ForEach-Object { $_.tag })
        $routeDigests = @($identityTags | ForEach-Object { $legRows[$_]['tg_route_digest'] } | Where-Object { $_ } | Select-Object -Unique)
        $idRows = @($identityTags | ForEach-Object { $legRows[$_]['tg_ids'] } | Where-Object { $_ -ne $null } | Select-Object -Unique)
        $bands['identity'] = [ordered]@{
            route_digest = if ($routeDigests.Count -eq 1) { $routeDigests[0] } else { $null }
            route_unique_count = $routeDigests.Count
            greedy_ids_sha256 = if ($idRows.Count -eq 1) {
                $sha2 = [System.Security.Cryptography.SHA256]::Create()
                try { ([System.BitConverter]::ToString($sha2.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($idRows[0]))) -replace '-', '').ToLower() }
                finally { $sha2.Dispose() }
            } else { $null }
            greedy_unique_count = $idRows.Count
        }
        Write-Host ("  identity: route unique {0} ({1}) | greedy unique {2} ({3})" -f `
            $routeDigests.Count, $bands['identity']['route_digest'], $idRows.Count, $bands['identity']['greedy_ids_sha256'])
        if ($routeDigests.Count -ne 1 -or $idRows.Count -ne 1) {
            $sessionExit = 1
        }

        # the fill-room ceiling, session-level, over every leg that has a row cache
        $frArgs = @()
        foreach ($L in $legs) {
            $tr = Join-Path $base ($L.tag + '.jsonl')
            if (Test-Path -LiteralPath $tr) { $frArgs += @('--leg', $tr) }
        }
        if ($frArgs.Count -gt 0) {
            $frSession = (& python $fillroom @frArgs 2>&1 | Out-String)
            $bands['fill_room_session'] = $frSession
            Write-Host '  --- fill-room ceiling (the bound every M1 gain must be read inside) ---'
            @($frSession.Trim() -split "`n") | Where-Object { $_ -match 'admittable|full by stage|REFUSED|leg ' } |
                ForEach-Object { Write-Host ('  | ' + $_) }
        }

        $summary = [ordered]@{
            schema = 'gplr58-session-summary/1'
            session = $sessionPrefix
            created_utc = (Get-Date).ToUniversalTime().ToString('o')
            shape = @{ prompt = $prompt; pp_tokens = $ppTokens; ctx_size = $ctxSize; tg_tokens = $tgTokens
                       rep_penalty = 1.15; spec_budget = $specBudget; exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha }
            legs = $legRows
            bands = $bands
            halted = $halted
        }
        [System.IO.File]::WriteAllText($summaryPath,
            ($summary | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
        Write-Host ("  session summary -> {0}" -f $summaryPath)
    }
    $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_post_run = {0}" -f $gpuPost)
} finally {
    if ($ownsMutex -and $acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    if ($null -ne $mutex) { $mutex.Dispose() }
}
# --- provenance gate (re-scores nothing; see the header) ---------------------
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($legs | ForEach-Object { $_.tag }) `
        -Label ($sessionPrefix + ' leg') -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("{0} VERDICT {1}" -f $sessionPrefix, $(if ($sessionExit -eq 0) { 'PASS (every leg usable; the bands above are the result)' } else { 'FAIL' }))
exit $sessionExit
