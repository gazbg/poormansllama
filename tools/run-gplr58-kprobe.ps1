# T-GPLR-58 K-SCALING PROBE (owner-requested 2026-09-23): "increase the slots until
# predictions actually fit, then measure how many predictions are met."
#
# QUESTION: A3 proved the K=2 cache admits ZERO speculative fills (0.00% fill room).
# The fill-room analyzer (host-only, from the RECORDING of retention legs) says the
# admittable fraction grows with K (0.00% / 1.85% / 6.62% at K=2/4/8). Nobody has ever
# run the ARM itself at higher K to see:
#   1. how many proposals the planner actually PLACES (submitted > 0), and
#   2. how many of those placed guesses are later CONFIRMED by a demand
#      (the share(arm) - share(off) delta at the SAME K, and the consumed counter).
#
# DESIGN (the control the probe needs): bigger K alone improves retention (the recorded
# honest ladder: 0.3778 / 0.5355 / 0.6487 / 0.7283 at K=2/4/6/8). So every K gets an
# OFF TWIN in the same session, and the arm's NET effect is share(arm@K) - share(off@K),
# never share(arm@K) vs a different-K baseline. The null@K8 controls the placement
# policy at the ladder's mid point: it proposes the arm's counts structure-free, so any
# gain it shows is placement/pool growth, not prediction.
#
# LEGS (one mutex hold, ~5 min of model time):
#   off-k4  off-k8  off-k12   retention-only twins at each depth (roles: off)
#   null-k8            the count-matched structure-free control at K=8
#   arm-k4  arm-k8  arm-k12   the coact arm at each depth (mode 2, budget 8)
#
# VRAM BUDGET (linear from the MEASURED anchors 7.19 GiB peak @K2, 8.37 @K4; +1.07 GiB
# per K step): K=12 predicts ~16.9 GiB peak vs 22.3 GiB free at session start. The
# engine's own byte-account preflight refuses loudly if a placement would not fit, so
# an over-budget leg is a leg FAIL that halts the series, not a silent mis-measurement.
#
# DISCIPLINE (the A3 skeleton, kept): the shape is the declared 885-token prompt,
# --tg-tokens 256, --rep-penalty 1.15 on EVERY leg; share is owned by
# tools/check-tg-predict.py per leg (never recomputed here); resolved state is read
# from the trace, not the command line; --check-telemetry gates every leg; the
# fill-room ceiling is measured per arm leg by tools/analyze-gplr58-fill-room.py; a
# leg that fails its integrity gates HALTS the series (the remaining legs would measure
# a broken mechanism); arm-verdict provenance is written for every leg.
#
# THIS IS A PROBE, NOT A CONTRACT SESSION: it pre-registers NO M-band verdict. The M
# bands are defined at fixed K=2 by the block; what this probe measures is the
# placement/hit funnel (proposals -> placed -> consumed -> extra hits) and whether the
# tok/s reacts, as the input to the owner's reserve/policy decision.
#
# Dry run (no GPU, no mutex, no CUDA): -ValidateOnly prints the exact commands.
param(
    [switch] $ValidateOnly,
    [switch] $AllowUnprovenanced,
    # HISTORICAL PROBE-ONLY escape hatch (2026-09-23 recordings only). Before
    # T-GPLR-58 A2, `spec` carried no `fill_ids`, so the runner's demand-only cache
    # model could not replay fills and reported false retention mismatches. Current
    # traces carry both `submitted_total` and `fill_ids`; this switch now refuses to
    # downgrade any trace with either key. Never use it on a contract session.
    [switch] $AllowSpecCheckerGap
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$exe  = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$dll  = Join-Path $base 'bin\Release\llama.dll'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $root 'build\calc-prompt.txt'
$ppTokens = 885
$tgTokens = 256
$ctxSize  = 1280
$specBudget = 8
$verdictDir = Join-Path $base 'verdicts'
$checker  = Join-Path $root 'tools\check-tg-predict.py'
$fillroom = Join-Path $root 'tools\analyze-gplr58-fill-room.py'
$summaryPath = Join-Path $base 'gplr58-kprobe-summary.json'
. (Join-Path $root 'tools\arm-verdict.ps1')

# role: 'off' | 'null' | 'arm' (A2's role vocabulary; NO 'retref' anywhere - this probe
# never differences across K, only within a K pair).
$legs = @(
    @{ tag = 'gplr58-kprobe-off-k4';  role = 'off'; retain = 4;  spec = 0; arm = '';      null = 0; note = 'retention-only twin at K=4' },
    @{ tag = 'gplr58-kprobe-off-k8';  role = 'off'; retain = 8;  spec = 0; arm = '';      null = 0; note = 'retention-only twin at K=8' },
    @{ tag = 'gplr58-kprobe-off-k12'; role = 'off'; retain = 12; spec = 0; arm = '';      null = 0; note = 'retention-only twin at K=12' },
    @{ tag = 'gplr58-kprobe-null-k8'; role = 'null'; retain = 8; spec = 2; arm = 'coact'; null = 1; note = 'count-matched structure-free control at K=8' },
    @{ tag = 'gplr58-kprobe-arm-k4';  role = 'arm'; retain = 4;  spec = 2; arm = 'coact'; null = 0; note = 'coact arm at K=4 (189-stage warm-up window recorded)' },
    @{ tag = 'gplr58-kprobe-arm-k8';  role = 'arm'; retain = 8;  spec = 2; arm = 'coact'; null = 0; note = 'coact arm at K=8 (678-stage warm-up window recorded)' },
    @{ tag = 'gplr58-kprobe-arm-k12'; role = 'arm'; retain = 12; spec = 2; arm = 'coact'; null = 0; note = 'coact arm at K=12 (largest window that fits VRAM)' }
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
    Write-Host ("VRAM model: peak(K) ~ 7.19 GiB + (K-2) x 1.07 GiB -> K=4 ~ 9.4, K=8 ~ 13.7, K=12 ~ 17.9 GiB (22.3 free at a clean start)")
    foreach ($L in $legs) {
        Write-Host ("LEG {0}  role={1} retain=K{2} spec={3} arm={4} null={5}  ({6})" -f `
            $L.tag, $L.role, $L.retain, $L.spec, $(if ($L.spec -gt 0) { $L.arm } else { '-' }), $L.null, $L.note)
        Write-Host ("  " + $exe + ' ' + ((Get-LegArgs $L) -join ' '))
    }
    Write-Host ("ONE mutex hold for {0} legs; the GPU busy gate refuses (exit 125) on a non-clean pre-run probe." -f $legs.Count)
    Write-Host "Read: for each K pair, share(arm@K) - share(off@K), the placed/consumed funnel, and TG tok/s."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("gplr58-kprobe: exe {0} dll {1}" -f $exeSha, $dllSha)

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
        if ($legExit -ne 0) {
            $gapDowngraded = $false
            if ($AllowSpecCheckerGap -and $L.spec -gt 0 -and $legExit -eq 1) {
                # the runner's own end-of-run check-telemetry is the only known exit-1
                # source on a healthy spec run; verify the log before downgrading
                $logTxt = [System.Text.Encoding]::Unicode.GetString([System.IO.File]::ReadAllBytes($log))
                # per-violation reports always carry 'line N:'; the bare
                # 'check-telemetry: N invariant violation(s)' summary line and the
                # teardown chatter do not (measured: the summary line made a
                # class-counting gate refuse the downgrade it was written for)
                $viol = @($logTxt.Split("`n") | Where-Object { $_ -match 'check-telemetry: line \d+:.*(retention accounting mismatch|row-capacity policy)' })
                $other = @($logTxt.Split("`n") | Where-Object { $_ -match 'check-telemetry: line \d+:' -and $_ -notmatch 'retention accounting mismatch|row-capacity policy' })
                $hasSubmittedTotal = [bool] (Select-String -LiteralPath $trace -SimpleMatch '"submitted_total":' -Quiet)
                $hasFillIds = [bool] (Select-String -LiteralPath $trace -SimpleMatch '"fill_ids":' -Quiet)
                $legacySpecSchema = -not $hasSubmittedTotal -and -not $hasFillIds
                if ($legacySpecSchema -and $viol.Count -gt 0 -and $other.Count -eq 0) {
                    $fields['telemetry_gap_allowed'] = $true
                    $fields['telemetry_gap_schema'] = 'legacy-cumulative-without-fill-ids'
                    $fields['telemetry_gap_violations'] = $viol.Count
                    Write-Warning ("  check-telemetry: {0} known spec-model-gap violations DOWNgraded to a warning (-AllowSpecCheckerGap); the trace stays suspect for contract use" -f $viol.Count)
                    $gapDowngraded = $true
                }
            }
            if (-not $gapDowngraded) { $fail += "leg exited $legExit" }
        }

        if ($legExit -eq 0) {
            $gpuPostLeg = Format-GpuBusy (Get-GpuComputeApps)
            $fields['gpu_post_run'] = $gpuPostLeg
            if ($gpuPostLeg -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - the leg is suspect." -f $gpuPostLeg) }

            # resolved state, from the trace
            $lines = @(Get-Content $trace)
            $rs = @($lines | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $init = @($lines | Where-Object { $_ -like '*"event":"rowset_init"*' })[0] | ConvertFrom-Json
            $fields['resolved_retain_k'] = [int] $init.retain_k
            if ($fields['resolved_retain_k'] -ne $L.retain) { $fail += "resolved retain_k $($fields['resolved_retain_k']) != declared K$($L.retain)" }
            $hasMode = $null -ne ($rs.PSObject.Properties['row_spec_mode'])
            $fields['resolved_spec_mode'] = if ($hasMode) { [int] $rs.row_spec_mode } else { 0 }
            if ($L.spec -gt 0) {
                if (-not $hasMode -or $fields['resolved_spec_mode'] -ne 2) { $fail += 'declared speculative but run_start carries no row_spec_mode==2' }
            } elseif ($hasMode) {
                $fail += 'an off leg carries row_spec_mode: the baseline is polluted (P6)'
            }

            # timing + placement funnel, from the trace's own events (python: one pass)
            $fields['leg_tg_toks'] = (& python -c @"
import json,sys
tg=None; sub=0; cons=0; prop=0; placed_stages=0; ready_spec=0; peak=0
for line in open(sys.argv[1],encoding='utf-8'):
    try: e=json.loads(line)
    except: continue
    ev=e.get('event')
    if ev=='phase_summary' and e.get('phase')=='tg': tg=e
    if ev=='run_end': peak=e.get('peak_vram_bytes',0)
    if ev=='spec':
        sub+=e.get('submitted',0); cons+=e.get('consumed',0); prop+=e.get('prop',0); ready_spec+=e.get('ready_spec',0)
        if e.get('submitted',0)>0: placed_stages+=1
out={'tg_toks': tg.get('tokens_per_s') if tg else None,
     'tg_wall_s': (tg.get('wall_us',0)/1e6) if tg else None,
     'tg_bytes_gib': (tg.get('bytes_transferred',0)/2**30) if tg else None,
     'tg_xfer_s': (tg.get('total_transfer_us',0)/1e6) if tg else None,
     'peak_vram_gib': peak/2**30,
     'proposals': prop, 'submitted': sub, 'consumed': cons, 'ready_spec': ready_spec, 'placed_stages': placed_stages}
print(json.dumps(out))
"@ $trace | ConvertFrom-Json)
            Write-Host ("  TG {0} tok/s | staged {1} GiB | peak {2} GiB | proposals {3} placed {4} (stages {5}) consumed {6}" -f `
                $fields['leg_tg_toks'].tg_toks, [math]::Round($fields['leg_tg_toks'].tg_bytes_gib,2), `
                [math]::Round($fields['leg_tg_toks'].peak_vram_gib,2), $fields['leg_tg_toks'].proposals, `
                $fields['leg_tg_toks'].submitted, $fields['leg_tg_toks'].placed_stages, $fields['leg_tg_toks'].consumed)

            # share: the checker is its single owner, per leg
            $legJson = Join-Path $base ($tag + '-bands.json')
            Remove-Item $legJson -ErrorAction SilentlyContinue
            $legOut = (& python $checker --leg "$($L.role)=$trace" --json $legJson 2>&1 | Out-String)
            $fields['checker_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -eq 2) {
                $fail += "A2 REFUSED this leg (exit 2): a leg that cannot be justified is not evidence"
                $fields['checker_refusal'] = (($legOut.Trim() -split "`n") | Select-Object -Last 3) -join ' | '
            } elseif ($LASTEXITCODE -eq 1) {
                $fail += 'A2 reported FAIL-INVARIANT on this leg (the series halts there)'
            } elseif ($LASTEXITCODE -ne 0) {
                $fail += "A2 checker exited $LASTEXITCODE"
            }
            if (Test-Path -LiteralPath $legJson) {
                $lp = Get-Content -Raw -LiteralPath $legJson | ConvertFrom-Json
                if (@($lp.legs).Count -ge 1) {
                    $l0 = @($lp.legs)[0]
                    $fields['leg_share'] = [double] $l0.share
                    $fields['leg_margin'] = [double] $l0.margin
                }
            }

            # fill-room ceiling, per arm/null leg
            if ($L.spec -gt 0) {
                $frOut = (& python $fillroom --leg $trace 2>&1 | Out-String)
                if ($LASTEXITCODE -ne 0) { $fail += "fill-room analyzer exited $LASTEXITCODE on a prediction leg" }
                $fm = [regex]::Match($frOut, 'TOTAL:\s*(\d+)/(\d+) layer-stages leave room beyond the reserve \(([0-9.]+)%\)')
                if ($fm.Success) {
                    $fields['fill_room_admittable'] = [int] $fm.Groups[1].Value
                    $fields['fill_room_stages'] = [int] $fm.Groups[2].Value
                    $fields['fill_room_pct'] = [double] $fm.Groups[3].Value
                } else { $fail += 'the fill-room analyzer printed no TOTAL line: the ceiling is unmeasured' }
            }

            & $exe --check-telemetry $trace
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }
        }

        $legExitCode = if ($fail.Count -gt 0) { 1 } else { 0 }
        if ($fail.Count -gt 0) {
            $fields['fail_reason'] = ($fail -join '; ')
            $sessionExit = 1
            $halted = "$tag : " + ($fail -join '; ')
        }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $legExitCode -StartedUtc $sessionStart `
            -LogPath $log -Fields $fields | Out-Null
        $legRows[$tag] = $fields
    }

    # ---------------- the probe read: within-K pairs -------------------------------
    if ($legRows.Count -gt 0) {
        Write-Host '  --- K-PAIR READ (the probe question) ---'
        Write-Host ("  {0} {1} {2} {3} {4} {5}" -f 'K', 'off_share', 'arm_share', 'delta', 'placed', 'consumed')
        $pairs = [ordered]@{}
        foreach ($k in @(4, 8, 12)) {
            $o = $legRows["gplr58-kprobe-off-k$k"]; $a = $legRows["gplr58-kprobe-arm-k$k"]
            $oShare = if ($o -and $o.Contains('leg_share')) { [double] $o['leg_share'] } else { $null }
            $aShare = if ($a -and $a.Contains('leg_share')) { [double] $a['leg_share'] } else { $null }
            $placed = if ($a -and $a.Contains('leg_tg_toks')) { [int] $a['leg_tg_toks'].submitted } else { 0 }
            $cons   = if ($a -and $a.Contains('leg_tg_toks')) { [int] $a['leg_tg_toks'].consumed } else { 0 }
            $delta  = if ($null -ne $oShare -and $null -ne $aShare) { [math]::Round($aShare - $oShare, 4) } else { $null }
            $pairs["$k"] = @{ off_share = $oShare; arm_share = $aShare; delta = $delta; placed = $placed; consumed = $cons }
            Write-Host ("  K{0,-2} {1} {2} {3} {4} {5}" -f $k, `
                $(if ($null -ne $oShare) { $oShare } else { 'n/a' }), `
                $(if ($null -ne $aShare) { $aShare } else { 'n/a' }), `
                $(if ($null -ne $delta) { '{0:+0.0000;-0.0000;0.0000}' -f $delta } else { 'n/a' }), $placed, $cons)
        }
        $n = $legRows["gplr58-kprobe-null-k8"]
        if ($n -and $n.Contains('leg_share')) {
            Write-Host ("  null@K8 share {0} vs off@K8 {1} (the placement-policy control at the mid point)" -f `
                $n['leg_share'], $(if ($pairs['8'].off_share -ne $null) { $pairs['8'].off_share } else { 'n/a' }))
        }
        Write-Host '  Note: a delta at or inside the off1/off2-class noise (A3 measured spread 0) is a ZERO, not a small win.'
    }

    $summary = [ordered]@{
        schema = 'gplr58-kprobe-summary/1'
        session = 'gplr58-kprobe'
        created_utc = (Get-Date).ToUniversalTime().ToString('o')
        shape = @{ prompt = $prompt; pp_tokens = $ppTokens; ctx_size = $ctxSize; tg_tokens = $tgTokens
                   rep_penalty = 1.15; spec_budget = $specBudget; exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha }
        legs = $legRows
        pairs = $pairs
        halted = $halted
    }
    [System.IO.File]::WriteAllText($summaryPath,
        ($summary | ConvertTo-Json -Depth 8), (New-Object System.Text.UTF8Encoding($false)))
    Write-Host ("  session summary -> {0}" -f $summaryPath)

    $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_post_run = {0}" -f $gpuPost)
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($legs | ForEach-Object { $_.tag }) `
        -Label 'kprobe leg' -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("gplr58-kprobe VERDICT {0}" -f $(if ($sessionExit -eq 0) { 'PASS (every leg usable; the K-pair table is the result)' } else { 'FAIL' }))
exit $sessionExit
