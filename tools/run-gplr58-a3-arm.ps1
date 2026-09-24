# T-GPLR-58 A3 arm driver: the predictor legs at the declared short shape, ALL legs inside
# ONE acquisition of the shared-GPU mutex.
#
# Contract (T-GPLR-58, A3; the two sessions were PRE-AUTHORIZED by the owner 2026-09-23):
# the shape is the declared 885-token non-echoing prompt (`build/calc-prompt.txt`),
# `--tg-tokens 256` and `--rep-penalty 1.15` on EVERY leg; the prediction arms run at the
# SHALLOW depth K=2 (the depth the charter case needs) and the retention reference at K=4.
#
# LEGS (exactly the contract's A3/pilot set, in the contract's order):
#   off1  @K2 - the incumbent retention, no prediction: the baseline every band is read against
#   off2  @K2 - the SAME configuration again: the second leg IS the session control SPREAD
#   null  @K2 - the count-matched, structure-free control (mode 2, arm computed, proposals
#               drawn from the fixed-seed xorshift): M2's pool-growth reading
#   coact @K2 - the SIMPLE pre-registered arm, unbarred co-activation at budget 8
#   retref@K4 - retention-only at the deeper depth: the self-contained M3 denominator
#
# QUANTITIES (pre-registered by the block; this driver computes NONE of them itself except
# the two arithmetic differences noted below -- the single owner of every band is
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
    [switch] $ValidateOnly,
    [switch] $AllowUnprovenanced
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
$summaryPath = Join-Path $base 'gplr58-a3-summary.json'
. (Join-Path $root 'tools\arm-verdict.ps1')

# role: 'off' | 'null' | 'arm' | 'retref' (the checker's OWN role vocabulary; `retref` is
# the contract's name for the K=4 retention leg and is NOT the same role token as 'off',
# which is exactly the M3 keying mismatch the driver works around).
$legs = @(
    @{ tag = 'gplr58-a3-off1';    role = 'off';    retain = 2; spec = 0; arm = '';      null = 0; note = 'incumbent retention, no prediction (baseline)' },
    @{ tag = 'gplr58-a3-off2';    role = 'off';    retain = 2; spec = 0; arm = '';      null = 0; note = 'the same configuration again: the session control SPREAD' },
    @{ tag = 'gplr58-a3-null';    role = 'null';   retain = 2; spec = 2; arm = 'coact'; null = 1; note = 'count-matched structure-free null (mode 2, arm supplies only the count)' },
    @{ tag = 'gplr58-a3-coact';   role = 'arm';    retain = 2; spec = 2; arm = 'coact'; null = 0; note = 'the SIMPLE arm: unbarred co-activation, budget 8' },
    @{ tag = 'gplr58-a3-retref4'; role = 'retref'; retain = 4; spec = 0; arm = '';      null = 0; note = 'retention-only at K=4: the M3 denominator' }
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
    Write-Host ("ONE mutex hold for {0} legs; the GPU busy gate refuses (exit 125) on a non-clean pre-run probe." -f $legs.Count)
    Write-Host "Then: tools/check-tg-predict.py over the whole leg set (M1/M2/M3/M4/M5/M7 + P1-P6), the control spread"
    Write-Host "      differenced by this driver with M3 cross-checked against the checker, and the fill-room ceiling per leg."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("gplr58-a3: exe {0} dll {1}" -f $exeSha, $dllSha)
Write-Host "  (a new binary pair means older controls are a DIFFERENT binary; this session's own off legs are the control)"

# --------------------------------------------------- arm-verdict handling FIRST
foreach ($L in $legs) { Clear-ArmVerdict -Dir $verdictDir -Tag $L.tag | Out-Null }
$sessionStart = (Get-Date).ToUniversalTime().ToString('o')
foreach ($L in $legs) {
    Write-ArmStart -Dir $verdictDir -Tag $L.tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' `
        -LogPath (Join-Path $base ($L.tag + '.log')) -Command ($exe + ' ' + ((Get-LegArgs $L) -join ' ')) | Out-Null
}

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$sessionExit = 1
$legRows = [ordered]@{}
$halted = ''
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        foreach ($L in $legs) {
            Write-ArmVerdict -Dir $verdictDir -Tag $L.tag -ExitCode 124 -StartedUtc $sessionStart -ErrorText 'mutex timeout' | Out-Null
        }
        exit 124
    }
    Write-Host '  mutex held'

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
        Write-Host '  A3 RESULT: UNUSABLE - fewer than two legs produced a trace, so no band can be read'
        if ($sessionExit -eq 0) { $sessionExit = 1 }
    } else {
        $legArgs = @()
        foreach ($L in $legs) {
            $tr = Join-Path $base ($L.tag + '.jsonl')
            if (Test-Path -LiteralPath $tr) { $legArgs += @('--leg', ("{0}={1}" -f $L.role, $tr)) }
        }
        $jsonOut = Join-Path $base 'gplr58-a3-bands.json'
        $bandOut = (& python $checker @legArgs --session gplr58-a3 --json $jsonOut 2>&1 | Out-String)
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
            if ($byRole.ContainsKey('arm')) {
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
            Write-Host '  A3 RESULT: the band report was REFUSED (exit 2) - no numbers, no verdict'
            $sessionExit = 2
        } elseif ($sessionCheckerExit -eq 1) {
            Write-Host '  A3 RESULT: FAIL-INVARIANT under A2 - the numbers are reported for the record and are NOT usable as evidence'
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
            schema = 'gplr58-a3-summary/1'
            session = 'gplr58-a3'
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
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
# --- provenance gate (re-scores nothing; see the header) ---------------------
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($legs | ForEach-Object { $_.tag }) `
        -Label 'A3 leg' -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("gplr58-a3 VERDICT {0}" -f $(if ($sessionExit -eq 0) { 'PASS (every leg usable; the bands above are the result)' } else { 'FAIL' }))
exit $sessionExit
