# T-GPLR-53 A4 arm driver: the modality discriminator at 4k, ALL arms inside ONE
# acquisition of the shared-GPU mutex.
#
# Contract (T-GPLR-53 block, A4): pre-timed pre-warm modes behind the one new
# experimental flag --pp-prewarm (0 = off default): (i) = 1, CPU read-touch of the
# registered package spans; (ii) = 2, a dummy GPU launch + sync; (iii) = 3, both.
# The same-session pinned control, a repeat-2 route-parity arm, and - added
# 2026-09-22 after the audit - the two CONTROLS that make the (i) reading
# falsifiable: 4 = a duration-matched no-touch delay, 8 = a duration-matched
# wrong-region page touch (64 MiB scratch). 4k is the shape where pass 1 IS a real
# single-prompt prefill (the excess is ~1.77 s of a ~3.55 s pass-1 wall).
#
# PRE-REGISTERED reading (recorded here BEFORE any arm runs). The BAND's QUANTITY was
# SETTLED 2026-09-23 by owner directive (T-GPLR-53 block, "SETTLED - the band's quantity"
# section): it is the PER-CHUNK EMBED EXCESS from analyze-gplr53-decay.py against the
# same-session off control (0) -
#     excess_removal = (ctrl_excess - arm_excess) / ctrl_excess * 100
# The pass-1 WALL is NOT a removal band and cannot be one: pass 1 is pass 2 plus the
# one-off, so its removable share IS the excess and its ceiling is `100*(1 - pass2/pass1)`
# - measured 56.2% at 4k, 10.5% at 50k, 0.86% at 200k. A wall band is unreachable by
# construction at every shape, so the wall is read as the RELOCATION instrument (pre-pay
# plus pass-1 wall as a SUM) and its removal % is printed only beside that ceiling.
#   (i)  removes >= 70% of the EXCESS -> page-side completion (H1)
#   (ii) removes it                   -> submission-side (H3)
#   neither                           -> the hypothesis set is incomplete: ESCALATE, do not tune
# DOSE SERIES (2026-09-22, added after the falsifier fired): five arms take the mode-4
# duration from --pp-prewarm-delay-ms (0/200/600/1200/1700 ms) so the curve of absorption
# against the wait can be read, with arm 'delay' kept flag-ABSENT as the compiled-default
# equivalence check. Reads: absorbed excess, residual, pre-pay, and the conserved sum
# (pre-pay + pass-1 wall) which is what decides whether a shorter pre-pay can win.
#
# FALSIFIER (not a branch - a check on (i)): if arm 'delay' (mode 4, no page touched)
# ALSO clears >= 70% of the control's excess (the settled band's quantity - the wall is
# not the test), then elapsed time alone accounts for
# the removal and the page-residency reading of (i) COLLAPSES. If it does not, (i)
# survives its falsifier. Arm 'wrongtouch' (mode 8) grades whether page touching in
# ANY region suffices, or only the staged source's own pages. This driver records
# those deltas; it names no mechanism and claims no speedup.
#
# Dry run (no GPU, no mutex, no CUDA): -ValidateOnly prints the exact commands.
param(
    [switch] $ValidateOnly,
    # The provenance gate (2026-09-22) refuses this run when an arm's verdict
    # cannot attest which source tree and which binary it describes. It re-scores
    # NOTHING: the pre-registered band and every arm field above are untouched.
    # The escape is explicit and RECORDED (verdicts/provenance-waivers.log).
    [switch] $AllowUnprovenanced
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$exe  = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$dll  = Join-Path $base 'bin\Release\llama.dll'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts\anchor-4096.txt'
$verdictDir = Join-Path $base 'verdicts'
$checker = Join-Path $root 'tools\check-gplr53-embed-chunks.py'
$decay   = Join-Path $root 'tools\analyze-gplr53-decay.py'
$parity  = Join-Path $root 'tools\check-route-parity.py'
. (Join-Path $root 'tools\arm-verdict.ps1')

# prewarm: 0 = off (control; the default path, byte-identical), 1/2/3 = the variants,
# 4 = no-touch delay (falsifier), 8 = wrong-region touch (control).
# delayms: sets --pp-prewarm-delay-ms for the mode-4 dose arms only (absent = the
# compiled default 1700, which arm 'delay' exercises without the flag).
$arms = @(
    @{ tag = 'gplr53-a4-pw0'; prewarm = 0; repeat = 3; source = 'mmap-registered'; route = $false; note = 'off control' },
    @{ tag = 'gplr53-a4-pw1'; prewarm = 1; repeat = 3; source = 'mmap-registered'; route = $false; note = 'i: CPU read-touch' },
    @{ tag = 'gplr53-a4-pw2'; prewarm = 2; repeat = 3; source = 'mmap-registered'; route = $false; note = 'ii: dummy GPU launch + sync' },
    @{ tag = 'gplr53-a4-pw3'; prewarm = 3; repeat = 3; source = 'mmap-registered'; route = $false; note = 'iii: both' },
    @{ tag = 'gplr53-a4-delay'; prewarm = 4; repeat = 3; source = 'mmap-registered'; route = $false; note = 'FALSIFIER: duration-matched no-touch delay (flag ABSENT = compiled default)' },
    @{ tag = 'gplr53-a4-wrongtouch'; prewarm = 8; repeat = 3; source = 'mmap-registered'; route = $false; note = 'control: duration-matched wrong-region touch (64 MiB scratch)' },
    @{ tag = 'gplr53-a4-d000'; prewarm = 4; delayms = 0;    repeat = 3; source = 'mmap-registered'; route = $false; note = 'dose 0 ms (mode 4 engaged, sleeps nothing)' },
    @{ tag = 'gplr53-a4-d200'; prewarm = 4; delayms = 200;  repeat = 3; source = 'mmap-registered'; route = $false; note = 'dose 200 ms' },
    @{ tag = 'gplr53-a4-d600'; prewarm = 4; delayms = 600;  repeat = 3; source = 'mmap-registered'; route = $false; note = 'dose 600 ms' },
    @{ tag = 'gplr53-a4-d1200'; prewarm = 4; delayms = 1200; repeat = 3; source = 'mmap-registered'; route = $false; note = 'dose 1200 ms' },
    @{ tag = 'gplr53-a4-d1700'; prewarm = 4; delayms = 1700; repeat = 3; source = 'mmap-registered'; route = $false; note = 'dose 1700 ms (explicit = arm delay without the flag)' },
    @{ tag = 'gplr53-a4-pin'; prewarm = 0; repeat = 3; source = 'pinned';          route = $false; note = 'same-session pinned control' },
    @{ tag = 'gplr53-a4-r2';  prewarm = 0; repeat = 2; source = 'mmap-registered'; route = $true;  note = 'repeat-2 route-parity arm' }
)

$missing = @()
foreach ($p in @($exe, $dll, $checker, $decay, $parity, $prompt)) { if (-not (Test-Path -LiteralPath $p)) { $missing += $p } }
if (-not (Test-Path -LiteralPath $model)) { $missing += $model }
if ($missing.Count -gt 0) {
    Write-Host 'MISSING INPUTS:'
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

function Get-ArmArgs {
    param($A)
    # NOTE: the accumulator must NOT be named $a -- PowerShell variable names are
    # case-insensitive, so $a would REBIND the $A parameter to this array. Every
    # read of $A after the assignment (the prewarm/route conditionals below) would
    # then test the array and come out empty, silently dropping --pp-prewarm and
    # --route-telemetry from every command (measured 2026-09-22 by dry-run).
    $out = @('--model', $model, '--prompt-file', $prompt,
           # T-GPLR-53 A4 shape fix (2026-09-22, measured): the anchor file holds
           # 4096 tokens, and the runner's own check requires ctx > pp+tg, so the
           # first A4 attempt died at arg validation in every arm ("--ctx-size 4096
           # is smaller than pp+tg (4099 tokens)", arm_exit 2, no trace written).
           # 4096/4352 is the RECORDED 4k convention for this anchor (gplr50-ctl-4k-f16
           # and gplr54-a3-run1-u512-on both ran pp_tokens=4096 ctx_size=4352).
           '--pp-tokens', '4096', '--ctx-size', '4352',
           '--ubatch', '512', '--pp-no-logits', '--tg-tokens', '3',
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency', '--residency-budget-mib', '0',
           '--pp-repeat', "$($A.repeat)",
           '--copy-source', $A.source,
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', (Join-Path $base ($A.tag + '.jsonl')))
    if ($A.prewarm -gt 0) { $out += @('--pp-prewarm', "$($A.prewarm)") }
    if ($A.ContainsKey('delayms')) { $out += @('--pp-prewarm-delay-ms', "$($A.delayms)") }
    if ($A.route) { $out += @('--route-telemetry', '1') }
    return $out
}

if ($ValidateOnly) {
    $exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
    $dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
    Write-Host "VALIDATE OK  exe $exeSha  dll $dllSha"
    Write-Host "model  $model"
    Write-Host "prompt $prompt (4096 tokens, ctx 4352)"
    foreach ($A in $arms) {
        Write-Host ("ARM {0}  prewarm={1} repeat={2} source={3}  ({4})" -f $A.tag, $A.prewarm, $A.repeat, $A.source, $A.note)
        Write-Host ("  " + $exe + ' ' + ((Get-ArmArgs $A) -join ' '))
    }
    Write-Host "ONE mutex hold for $($arms.Count) arms; GPU busy gate refuses (exit 125) on a non-clean pre-run probe."
    Write-Host "Then: per-chunk embed EXCESS removal vs gplr53-a4-pw0 (>=70% of the EXCESS = the settled band), and the pass-1 wall printed beside its ceiling as the relocation instrument."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("gplr53-a4: exe {0} dll {1}" -f $exeSha, $dllSha)

# --------------------------------------------------- arm-verdict handling FIRST
foreach ($A in $arms) { Clear-ArmVerdict -Dir $verdictDir -Tag $A.tag | Out-Null }
$sessionStart = (Get-Date).ToUniversalTime().ToString('o')
foreach ($A in $arms) {
    Write-ArmStart -Dir $verdictDir -Tag $A.tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' `
        -LogPath (Join-Path $base ($A.tag + '.log')) -Command ($exe + ' ' + ((Get-ArmArgs $A) -join ' ')) | Out-Null
}

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$sessionExit = 1
$pass1Wall = @{}
$armRows = @{}
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        foreach ($A in $arms) {
            Write-ArmVerdict -Dir $verdictDir -Tag $A.tag -ExitCode 124 -StartedUtc $sessionStart -ErrorText 'mutex timeout' | Out-Null
        }
        exit 124
    }
    Write-Host '  mutex held'

    $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_pre_run = {0}" -f $gpuPre)
    if ($gpuPre -ne 'clean') {
        if ($gpuPre -eq 'probe-unavailable') { Write-Warning 'GPU busy probe unavailable; proceeding UNVERIFIED.' }
        else {
            foreach ($A in $arms) {
                Write-ArmVerdict -Dir $verdictDir -Tag $A.tag -ExitCode 125 -StartedUtc $sessionStart `
                    -ErrorText "GPU busy: $gpuPre" -Fields @{ gpu_pre_run = $gpuPre } | Out-Null
            }
            Write-Host ("REFUSED: GPU busy ({0})" -f $gpuPre)
            exit 125
        }
    }

    $sessionExit = 0
    foreach ($A in $arms) {
        $tag = $A.tag
        $trace = Join-Path $base ($tag + '.jsonl')
        $log   = Join-Path $base ($tag + '.log')
        Remove-Item $trace, $log -ErrorAction SilentlyContinue
        $fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha; arm = $tag
                     prewarm_mode = $A.prewarm; pp_repeat = $A.repeat; copy_source_requested = $A.source
                     note = $A.note; trace = $trace; log = $log; gpu_pre_run = $gpuPre }
        if ($A.ContainsKey('delayms')) { $fields['prewarm_delay_ms'] = $A.delayms }
        $fail = @()
        Write-Host ("== ARM {0} (prewarm {1}) ==" -f $tag, $A.prewarm)

        $t0 = Get-Date
        & $exe @(Get-ArmArgs $A) > $log 2>&1
        $armExit = $LASTEXITCODE
        $fields['arm_exit'] = $armExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])
        if ($armExit -ne 0) { $fail += "arm exited $armExit" }

        if ($armExit -eq 0) {
            # Per-arm post-run probe, persisted (parity with the gplr51/gplr52 drivers:
            # a printed-only post-run probe leaves the verdict unable to support the
            # "card clean after" claim it is cited for).
            $gpuPostArm = Format-GpuBusy (Get-GpuComputeApps)
            $fields['gpu_post_run'] = $gpuPostArm
            if ($gpuPostArm -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - timing suspect." -f $gpuPostArm) }

            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $fields['resolved_copy_source'] = $rs.copy_source
            if ($rs.copy_source -ne $A.source) { $fail += "copy_source $($rs.copy_source), want $($A.source)" }

            # the off control must prove the flag was NOT engaged; a non-zero mode must prove it was.
            $pwLines = @(Get-Content $log | Where-Object { $_ -like '*pp-prewarm(*' })
            $fields['prewarm_engaged'] = ($pwLines.Count -gt 0)
            # The mode's own pre-pay duration (the final 'total ... ms' line of the mode).
            $mAll = [regex]::Matches(($pwLines -join ' '), 'total ([0-9.]+) ms')
            if ($mAll.Count -gt 0) { $fields['prewarm_total_ms'] = [double]$mAll[$mAll.Count - 1].Groups[1].Value }
            if ($A.prewarm -eq 0 -and $pwLines.Count -gt 0) { $fail += 'prewarm chatter in an off arm (default path not clean)' }
            if ($A.prewarm -gt 0 -and $pwLines.Count -eq 0) { $fail += 'prewarm mode set but no pre-warm line: the flag did not engage' }

            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

            $dout = (& python $decay $trace 2>&1 | Out-String)
            $fields['decay_exit'] = $LASTEXITCODE
            $fields['decay_line'] = ($dout.Trim() -split "`n" | Select-Object -Last 1)
            if ($LASTEXITCODE -eq 2) { $fail += 'decay: NOT EVALUABLE (no per-chunk excess)' }
            elseif ($LASTEXITCODE -ne 0) { $fail += "decay exit $LASTEXITCODE" }
            if ($dout -match '(FRONT_LOADED|FLAT|CHUNK0_SPIKE|MIXED|NO_EXCESS)') { $fields['decay_verdict'] = $Matches[1] }
            $m = [regex]::Match($dout, '"excess_total_ms":\s*(-?[0-9.]+)')
            if ($m.Success) { $fields['embed_excess_total_ms'] = [double]$m.Groups[1].Value }

            $i = 0
            foreach ($p in @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } | ForEach-Object { $_ | ConvertFrom-Json })) {
                $i++
                $fields[("pass{0}_wall_s" -f $i)] = [math]::Round($p.wall_us / 1e6, 2)
            }
            $fields['pp_pass_count'] = $i
            if ($fields.ContainsKey('pass1_wall_s')) { $pass1Wall[$tag] = $fields['pass1_wall_s'] }

            if ($A.route) {
                $par = (& python $parity $trace 2>&1 | Out-String)
                $fields['route_parity_exit'] = $LASTEXITCODE
                $fields['route_parity_tail'] = ($par.Trim() -split "`n" | Select-Object -Last 1)
                if ($LASTEXITCODE -ne 0) { $fail += "route-parity exit $LASTEXITCODE" }
            }
            Write-Host ("  prewarm={0} engaged={1} | pass1 wall {2}s | decay {3} excess {4} ms" -f `
                $A.prewarm, $fields['prewarm_engaged'], $fields['pass1_wall_s'], $fields['decay_verdict'], $fields['embed_excess_total_ms'])
        }

        $armExitCode = if ($fail.Count -gt 0) { 1 } else { 0 }
        if ($armExitCode -ne 0) { $fields['fail_reason'] = ($fail -join '; '); $sessionExit = 1 }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $armExitCode -StartedUtc $sessionStart `
            -LogPath $log -Fields $fields | Out-Null
        $armRows[$tag] = $fields
    }

    # pre-registered reading, computed once the arms exist. The BAND's quantity is the
    # per-chunk embed EXCESS against the off control (settled 2026-09-23; see the header):
    # a wall band is unreachable at every shape, so the wall is printed WITH its ceiling
    # and only ever as the relocation instrument.
    if ($pass1Wall.ContainsKey('gplr53-a4-pw0') -and $pass1Wall['gplr53-a4-pw0'] -gt 0) {
        $ctrl = [double] $pass1Wall['gplr53-a4-pw0']
        $ctrlExcess = if ($armRows['gplr53-a4-pw0'].ContainsKey('embed_excess_total_ms')) { [double] $armRows['gplr53-a4-pw0']['embed_excess_total_ms'] } else { 0.0 }
        $ctrlPass2 = if ($armRows['gplr53-a4-pw0'].ContainsKey('pass2_wall_s')) { [double] $armRows['gplr53-a4-pw0']['pass2_wall_s'] } else { 0.0 }
        $ceiling = if ($ctrlPass2 -gt 0.0 -and $ctrl -gt 0.0) { [math]::Round((1.0 - ($ctrlPass2 / $ctrl)) * 100.0, 1) } else { $null }
        $removal = @{}
        $excessRemoval = @{}
        foreach ($k in @('gplr53-a4-pw1', 'gplr53-a4-pw2', 'gplr53-a4-pw3', 'gplr53-a4-delay', 'gplr53-a4-wrongtouch',
                         'gplr53-a4-d000', 'gplr53-a4-d200', 'gplr53-a4-d600', 'gplr53-a4-d1200', 'gplr53-a4-d1700', 'gplr53-a4-pin')) {
            if ($pass1Wall.ContainsKey($k)) {
                $removed = [math]::Round((($ctrl - [double] $pass1Wall[$k]) / $ctrl) * 100.0, 1)
                $removal[$k] = $removed
                $ex = $null
                if ($ctrlExcess -gt 0.0 -and $armRows.ContainsKey($k) -and $armRows[$k].ContainsKey('embed_excess_total_ms')) {
                    $ex = [math]::Round((($ctrlExcess - [double] $armRows[$k]['embed_excess_total_ms']) / $ctrlExcess) * 100.0, 1)
                    $excessRemoval[$k] = $ex
                }
                # The band's quantity FIRST, the relocation instrument second, one line each.
                # One decimal always, so a whole number (e.g. 93) still prints as 93.0 - these
                # lines are compared against the record verbatim.
                $exLabel = if ($null -eq $ex) { 'n/a' } else { '{0:0.0}' -f $ex }
                Write-Host ("  {0}: excess removal {1}% (BAND) | pass-1 wall removal {2:0.0}% (relocation instrument)" -f `
                    $k, $exLabel, $removed)
            }
        }
        # The ceiling, printed ONCE and BEFORE any branch reads a wall number.
        if ($null -ne $ceiling) {
            $excessShare = if ($ctrl -gt 0.0) { [math]::Round((($ctrlExcess / 1000.0) / $ctrl) * 100.0, 1) } else { 0.0 }
            Write-Host ("  WALL CEILING: the excess is {0:0.0}% of the control's pass-1 wall and the pass-2 wall is {1:0.0}% of it, so no arm can remove more than {2:0.0}% of the wall at this shape - the 70% wall band is unreachable HERE by construction." -f `
                $excessShare, [math]::Round(($ctrlPass2 / $ctrl) * 100.0, 1), $ceiling)
        }
        # The falsifier check, stated as a computed line rather than a judgement: this is a
        # RESULT, not an arm gate - a firing falsifier does not fail the run. It is evaluated
        # on the BAND's quantity (the excess); the wall number is shown only with its ceiling.
        if ($excessRemoval.ContainsKey('gplr53-a4-delay')) {
            $d = $excessRemoval['gplr53-a4-delay']
            $dWall = if ($removal.ContainsKey('gplr53-a4-delay')) { $removal['gplr53-a4-delay'] } else { $null }
            if ($d -ge 70.0) {
                Write-Host ("  FALSIFIER FIRED: the no-touch delay removes {0:0.0}% of the EXCESS (>= 70%) - elapsed time alone accounts for the removal, so the page-residency reading of mode 1 COLLAPSES." -f $d)
            } else {
                Write-Host ("  FALSIFIER SURVIVED: the no-touch delay removes only {0:0.0}% of the EXCESS (< 70%) - the page-residency reading of mode 1 stands against this control." -f $d)
            }
            if ($null -ne $dWall -and $null -ne $ceiling -and $ceiling -lt 70.0) {
                Write-Host ("    (its pass-1 wall removal is {0:0.0}%, but the wall band is NOT EVALUABLE at this shape: ceiling {1:0.0}% < 70%. This line replaces a wall number that used to read as a failed test.)" -f $dWall, $ceiling)
            }
        }
        if ($excessRemoval.ContainsKey('gplr53-a4-wrongtouch')) {
            Write-Host ("  wrong-region touch removes {0:0.0}% of the excess (page touching in ANY region vs the staged source's own pages)." -f $excessRemoval['gplr53-a4-wrongtouch'])
        }
    }

    # The dose curve (falsifier follow-up): does absorption saturate, and is the wall conserved?
    $doseOrder = @('gplr53-a4-d000', 'gplr53-a4-d200', 'gplr53-a4-d600', 'gplr53-a4-d1200', 'gplr53-a4-d1700')
    Write-Host '  DOSE CURVE (mode 4 no-touch delay; the flag-absent arm delay = the compiled default):'
    Write-Host ("  {0,-24} {1,8} {2,11} {3,11} {4,9} {5,9} {6,10} {7,10}" -f 'arm', 'delay ms', 'prepay ms', 'excess ms', 'pass1 s', 'total s', 'prepay+p1 s', 'residual ms')
    $conserved = @()
    $cw = $pass1Wall['gplr53-a4-pw0']
    Write-Host ("  {0,-24} {1,8} {2,11} {3,11} {4,9} {5,9} {6,10} {7,10}" -f 'gplr53-a4-pw0', '0 (off)', '-', [math]::Round([double] $armRows['gplr53-a4-pw0']['embed_excess_total_ms'], 2), $cw, $armRows['gplr53-a4-pw0']['duration_s'], [math]::Round([double] $cw, 2), [math]::Round([double] $armRows['gplr53-a4-pw0']['embed_excess_total_ms'], 2))
    foreach ($k in $doseOrder) {
        if (-not $armRows.ContainsKey($k)) { continue }
        $r = $armRows[$k]
        $pp = if ($r.ContainsKey('prewarm_total_ms')) { [double] $r['prewarm_total_ms'] } else { 0.0 }
        $p1 = [double] $r['pass1_wall_s']
        $ex = [double] $r['embed_excess_total_ms']
        $conserved += ($pp / 1000.0) + $p1
        Write-Host ("  {0,-24} {1,8} {2,11} {3,11} {4,9} {5,9} {6,10} {7,10}" -f $k, $r['prewarm_delay_ms'], [math]::Round($pp, 1), [math]::Round($ex, 2), $p1, $r['duration_s'], [math]::Round(($pp / 1000.0) + $p1, 2), [math]::Round($ex, 2))
    }
    if ($conserved.Count -gt 0) {
        $mn = ($conserved | Measure-Object -Minimum).Minimum
        $mx = ($conserved | Measure-Object -Maximum).Maximum
        Write-Host ("  pre-pay + pass-1 wall across the doses: min {0} s, max {1} s, spread {2} s (a conserved sum means the time is RELOCATED, not shortened)." -f `
            [math]::Round($mn, 2), [math]::Round($mx, 2), [math]::Round($mx - $mn, 2))
    }
    $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_post_run = {0}" -f $gpuPost)
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
# --- provenance gate (re-scores nothing; see the header) ---------------------
# Exit code 10 is distinct from an arm FAIL (1) and from 124/125.
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($arms | ForEach-Object { $_.tag }) `
        -Label 'A4 arm' -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("gplr53-a4 VERDICT {0}" -f $(if ($sessionExit -eq 0) { 'PASS' } else { 'FAIL' }))
exit $sessionExit
