# T-GPLR-56 A3 arm driver: the registration-TIMING arms at 4k, ALL arms inside ONE
# acquisition of the shared-GPU mutex.
#
# Contract (T-GPLR-56 block, A3; owner calls DECIDED 2026-09-22): the pre-registered
# acceptance quantity is
#     Q_arm = pass1_end_ts - proc_start_ts   (ms)
# - the SUM from process start to the end of the first prefill pass, chosen so a
# RELOCATION cannot satisfy it. The rule (tools/check-gplr56-q.py, the single owner):
#     PASS iff (control_Q - arm_Q) > 250 ms  AND  (control_pass1_wall - arm_pass1_wall) > 0
# An arm whose wall falls while Q does not is a FAIL-RELOCATION and is reported as one.
#
# ARMS (A2-revised set; `p1-overlap` is the pre-authorized fallback and is NOT here -
# it is only for a `p1-early` contention result):
#   off    - registration at its current site (after the prebuild), nothing else changed
#            -> the same-session control.
#   early  - registration moved to the earliest legal point (after exec_layers, before
#            run_start and the prebuild). A2 measured that head start at 66.4 ms, so it
#            is expected to FAIL the margin: it is the negative control for "earlier is
#            enough". Its registration events precede run_start by design.
#   pace   - T-GPLR-56 A3's mechanism: the first group is page-locked before the pass and
#            every later group is issued from stage_members() one group (--pp-span-pace-
#            ahead 1) before the layer that needs it, so pending unlocked bytes stay near
#            one group instead of 17.23 GiB. Reported additionally: how many groups were
#            issued before each layer's first touch (tools/check-gplr56-pace.py) and the
#            copy_complete transfer medians against the control (contention).
#
# WHAT EACH ARM MUST PROVE about itself before it is scored (a missing proof is an arm
# FAIL, not a note):
#   * its `quantity` is declared in its own verdict (`q_procstart_pass1end`);
#   * `--pp-recon 1` was on (Q reads the recon-gated pass-1 `pass_end`) - the trace's
#     `run_start` must show the resolved recon flag and the pass-1 `host_mem` bracket
#     must exist, else check-gplr56-q.py refuses (exit 2) and the arm is unusable;
#   * the mode it claims is the mode that ran: `off`/`early` must carry NO `span_pace_init`
#     and `mmap_register_init` (the one-shot call), `early` must carry the pre-sink
#     `register_issue` marker with detail `early`, and `pace` must carry `span_pace_init`
#     with groups > 0 and NO `mmap_register_init`;
#   * the off control must show no pacing chatter on stderr (default path not perturbed).
#
# SESSION VERDICT SEMANTICS (deliberate, and the A4 precedent): exit 0 means EVERY arm ran
# and passed its own integrity gates and the outcome is reported honestly - including the
# pre-registered NEGATIVE outcome ("no arm clears the 250 ms margin"), which the block
# states is the decision-useful result for the charter's long-context priority. The exit
# code never encodes which way the measurement came out; `a3_acceptance_met` in each
# verdict does. Exit 1 = an arm failed its integrity gates or the control was unusable
# (no Q), 124 = mutex timeout, 125 = GPU busy, 10 = provenance.
#
# Dry run (no GPU, no mutex, no CUDA): -ValidateOnly prints the exact commands.
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
$prompt = Join-Path $base 'gplr07-qa-prompts\anchor-4096.txt'
$verdictDir = Join-Path $base 'verdicts'
$qcheck  = Join-Path $root 'tools\check-gplr56-q.py'
$pace    = Join-Path $root 'tools\check-gplr56-pace.py'
# The A1 real-trace control (same markers, same anchor, an earlier binary). The block's
# acceptance criterion asks that THIS session's control be stable against the previously
# archived control run within 250 ms; this is that run. Its absence is reported as
# "not available", never as a pass.
$archivedControl = Join-Path $base 'gplr56-p1off-4k.jsonl'
. (Join-Path $root 'tools\arm-verdict.ps1')

# pace: 0 = off (control and the early arm), 1 = paced registration.
# site: 'post-prebuild' (default, today's site) or 'early'.
$arms = @(
    @{ tag = 'gplr56-a3-off';   pace = 0; site = 'post-prebuild'; note = 'control: registration at the current site' },
    @{ tag = 'gplr56-a3-early'; pace = 0; site = 'early';         note = 'negative control: earliest legal site (A2: worth 66.4 ms)' },
    @{ tag = 'gplr56-a3-pace';  pace = 1; site = 'post-prebuild'; note = 'p1-pace: registration issued one group ahead of each layer' }
)

$missing = @()
foreach ($p in @($exe, $dll, $qcheck, $pace, $prompt)) { if (-not (Test-Path -LiteralPath $p)) { $missing += $p } }
if (-not (Test-Path -LiteralPath $model)) { $missing += $model }
if ($missing.Count -gt 0) {
    Write-Host 'MISSING INPUTS:'
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}
$haveArchivedControl = Test-Path -LiteralPath $archivedControl

function Get-ArmArgs {
    param($A)
    # NOTE: the accumulator must NOT be named $a - PowerShell variable names are
    # case-insensitive, so $a would REBIND the $A parameter and every later read of $A
    # would test the array, silently dropping the mode flags (the T-GPLR-53 A4 lesson).
    $out = @('--model', $model, '--prompt-file', $prompt,
           '--pp-tokens', '4096', '--ctx-size', '4352',
           '--ubatch', '512', '--pp-no-logits', '--tg-tokens', '3',
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency', '--residency-budget-mib', '0',
           '--pp-repeat', '3',
           '--copy-source', 'mmap-registered',
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', (Join-Path $base ($A.tag + '.jsonl')))
    if ($A.pace -gt 0) { $out += @('--pp-span-pace', "$($A.pace)", '--pp-span-pace-ahead', '1') }
    if ($A.site -ne 'post-prebuild') { $out += @('--pp-register-site', $A.site) }
    return $out
}

if ($ValidateOnly) {
    $exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
    $dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
    Write-Host "VALIDATE OK  exe $exeSha  dll $dllSha"
    Write-Host "model  $model"
    Write-Host "prompt $prompt (4096 tokens, ctx 4352)"
    Write-Host ("archived A1 control: {0} ({1})" -f $archivedControl, $(if ($haveArchivedControl) { 'present' } else { 'ABSENT - control stability will read n/a' }))
    foreach ($A in $arms) {
        Write-Host ("ARM {0}  pace={1} site={2}  ({3})" -f $A.tag, $A.pace, $A.site, $A.note)
        Write-Host ("  " + $exe + ' ' + ((Get-ArmArgs $A) -join ' '))
    }
    Write-Host "ONE mutex hold for $($arms.Count) arms; the GPU busy gate refuses (exit 125) on a non-clean pre-run probe."
    Write-Host "Then: Q per arm (check-gplr56-q.py) vs the off control, the 250 ms margin rule, and the relocation falsifier."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("gplr56-a3: exe {0} dll {1}" -f $exeSha, $dllSha)
Write-Host "  (a new binary pair means older controls are a DIFFERENT binary; this session's own off arm is the control)"

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
        # Declared BEFORE anything is measured: the quantity this arm is scored on. An
        # arm that does not name it in its own verdict cannot be scored at all.
        $fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha; arm = $tag
                     quantity = 'q_procstart_pass1end'
                     span_pace = $A.pace; register_site = $A.site; note = $A.note
                     trace = $trace; log = $log; gpu_pre_run = $gpuPre }
        $fail = @()
        Write-Host ("== ARM {0} (pace {1}, site {2}) ==" -f $tag, $A.pace, $A.site)

        $t0 = Get-Date
        & $exe @(Get-ArmArgs $A) > $log 2>&1
        $armExit = $LASTEXITCODE
        $fields['arm_exit'] = $armExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])
        if ($armExit -ne 0) { $fail += "arm exited $armExit" }

        if ($armExit -eq 0) {
            $gpuPostArm = Format-GpuBusy (Get-GpuComputeApps)
            $fields['gpu_post_run'] = $gpuPostArm
            if ($gpuPostArm -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - timing suspect." -f $gpuPostArm) }

            $lines = @(Get-Content $trace)
            $rs = @($lines | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $fields['resolved_copy_source'] = $rs.copy_source
            if ($rs.copy_source -ne 'mmap-registered') { $fail += "copy_source $($rs.copy_source), want mmap-registered" }
            # The resolved rebuild decision is read from the trace, never from the command
            # history (the T-GPLR-48/T-50 lesson: the default is KV-conditional).
            $fields['pp_rebuild_per_pass_resolved'] = $rs.pp_prebuild_per_pass
            $fields['pp_rebuild_per_pass_source'] = $rs.pp_prebuild_per_pass_source
            # Recon is not a run_start key; the only honest proof that it was ON is the
            # recon-gated pass-1 host_mem bracket existing in the trace (Q refuses without
            # it, so this field and Q's exit agree by construction).
            $fields['pass1_host_mem_bracket'] = (@($lines | Where-Object { $_ -like '*"event":"host_mem"*' -and $_ -like '*"label":"pass_begin"*' -and $_ -like '*"pass":1*' }).Count -gt 0)
            if (-not $fields['pass1_host_mem_bracket']) { $fail += 'no pass-1 host_mem bracket: --pp-recon did not engage, so Q is not readable' }

            # --- mode proof: the arm must be the mode it claims ---------------------
            $hasPaceInit  = @($lines | Where-Object { $_ -like '*"event":"span_pace_init"*' }).Count -gt 0
            $hasOneShot   = @($lines | Where-Object { $_ -like '*"event":"mmap_register_init"*' }).Count -gt 0
            $earlyMarks   = @($lines | Where-Object { $_ -like '*"event":"register_issue"*' -and $_ -like '*"early"*' })
            $fields['span_pace_init_present'] = $hasPaceInit
            $fields['mmap_register_init_present'] = $hasOneShot
            $fields['register_issue_early_marker'] = ($earlyMarks.Count -gt 0)
            if ($A.pace -eq 1) {
                if (-not $hasPaceInit) { $fail += 'pace arm set but no span_pace_init: the mode did not engage' }
                if ($hasOneShot)      { $fail += 'pace arm also ran the one-shot registration (both modes engaged)' }
                $paceChatter = @(Get-Content $log | Where-Object { $_ -like '*PACED registration armed*' })
                $fields['pace_armed_line'] = ($paceChatter.Count -gt 0)
                if ($paceChatter.Count -eq 0) { $fail += 'pace mode set but no PACED arming line on stderr' }
                if ($paceChatter.Count -gt 0 -and $paceChatter[0] -match '(\d+) of (\d+) B in (\d+) groups') {
                    $fields['pace_bytes_issued_head_start'] = [int64] $Matches[1]
                    $fields['pace_bytes_total'] = [int64] $Matches[2]
                    $fields['pace_groups'] = [int]$Matches[3]
                }
            } else {
                if ($hasPaceInit) { $fail += 'non-pace arm emitted span_pace_init (the flag was engaged)' }
                if (-not $hasOneShot) { $fail += 'non-pace arm has no mmap_register_init: the one-shot call did not run' }
                $paceChatter = @(Get-Content $log | Where-Object { $_ -like '*PACED registration armed*' })
                if ($paceChatter.Count -gt 0) { $fail += 'pacing chatter in a non-paced arm' }
                if ($A.site -eq 'early' -and -not $fields['register_issue_early_marker']) {
                    $fail += 'early arm has no register_issue marker with detail "early"'
                }
                if ($A.site -eq 'post-prebuild' -and $fields['register_issue_early_marker']) {
                    $fail += 'post-prebuild arm carries the early-site marker'
                }
            }

            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

            # --- Q: the quantity, from its single owner ----------------------------
            $qout = (& python $qcheck --trace $trace 2>&1 | Out-String)
            $fields['q_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "Q refused (exit $LASTEXITCODE): the arm cannot be scored" }
            $qm = [regex]::Match($qout, 'Q =\s*([0-9.]+) ms')
            if ($qm.Success) { $fields['q_ms'] = [double] $qm.Groups[1].Value }
            else { $fail += 'Q value not printed by check-gplr56-q.py' }
            $wm = [regex]::Match($qout, 'pass-1 wall =\s*([0-9.]+) ms')
            if ($wm.Success) { $fields['q_pass1_wall_ms'] = [double] $wm.Groups[1].Value }
            $lm = [regex]::Match($qout, 'pre-load = ([0-9.]+) ms, load = ([0-9.]+) ms')
            if ($lm.Success) {
                $fields['pre_load_ms'] = [double] $lm.Groups[1].Value
                $fields['load_ms'] = [double] $lm.Groups[2].Value
            }

            # --- pass walls (reported, and one leg of the pre-registered rule) ------
            $i = 0
            foreach ($p in @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } | ForEach-Object { $_ | ConvertFrom-Json })) {
                $i++
                $fields[("pass{0}_wall_s" -f $i)] = [math]::Round($p.wall_us / 1e6, 2)
            }
            $fields['pp_pass_count'] = $i
            if ($i -eq 0) { $fail += 'no phase_summary for pp: the pass walls are missing' }

            # --- the paced mechanism, from its own analyzer ------------------------
            if ($A.pace -eq 1) {
                $pout = (& python $pace --trace $trace --control (Join-Path $base 'gplr56-a3-off.jsonl') 2>&1 | Out-String)
                $fields['pace_analyzer_exit'] = $LASTEXITCODE
                if ($LASTEXITCODE -ne 0) { $fail += "pace analyzer exit $LASTEXITCODE (see its lines in the driver stdout)" }
                $fields['pace_summary'] = (($pout.Trim() -split "`n") | Where-Object { $_ -match 'COVERAGE|pacing LEAD|pass-1 stall|control ' } | ForEach-Object { $_.Trim() }) -join ' | '
                $cm = [regex]::Match($pout, '(\d+) covered, (\d+) NOT covered')
                if ($cm.Success) {
                    $fields['pace_layers_covered'] = [int] $cm.Groups[1].Value
                    $fields['pace_layers_uncovered'] = [int] $cm.Groups[2].Value
                }
            }

            Write-Host ("  Q {0} ms | pass1 wall {1} ms | pace init {2} | one-shot init {3}" -f `
                $fields['q_ms'], $fields['q_pass1_wall_ms'], $fields['span_pace_init_present'], $fields['mmap_register_init_present'])
        }

        $armExitCode = if ($fail.Count -gt 0) { 1 } else { 0 }
        if ($armExitCode -ne 0) { $fields['fail_reason'] = ($fail -join '; '); $sessionExit = 1 }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $armExitCode -StartedUtc $sessionStart `
            -LogPath $log -Fields $fields | Out-Null
        $armRows[$tag] = $fields
    }

    # ---------------- the pre-registered rule, applied by its single owner ---------
    $ctrlTag = 'gplr56-a3-off'
    $ctrlOk = $armRows.ContainsKey($ctrlTag) -and $armRows[$ctrlTag].ContainsKey('q_ms')
    if (-not $ctrlOk) {
        Write-Host '  A3 RESULT: UNUSABLE - the same-session control has no Q, so no arm can be scored'
        $sessionExit = 1
    } else {
        $ctrlQ = [double] $armRows[$ctrlTag]['q_ms']
        $ctrlWall = [double] $armRows[$ctrlTag]['q_pass1_wall_ms']
        Write-Host ("  control Q {0} ms | control pass-1 wall {1} ms" -f $ctrlQ, $ctrlWall)

        # Stability of the control against the archived A1 control run (pre-registered).
        if ($haveArchivedControl) {
            $aout = (& python $qcheck --trace $archivedControl 2>&1 | Out-String)
            if ($LASTEXITCODE -eq 0) {
                $am = [regex]::Match($aout, 'Q =\s*([0-9.]+) ms')
                if ($am.Success) {
                    $archQ = [double] $am.Groups[1].Value
                    $dq = [math]::Round([math]::Abs($ctrlQ - $archQ), 1)
                    $fieldsCtrl = @{ archived_q_ms = $archQ; session_q_ms = $ctrlQ; control_stability_ms = $dq
                                     archived_control = $archivedControl }
                    $stable = $dq -le 250.0
                    Write-Host ("  CONTROL STABILITY: archived Q {0} ms vs session Q {1} ms -> {2} ms ({3}; the criterion is within 250 ms)" -f `
                        $archQ, $ctrlQ, $dq, $(if ($stable) { 'STABLE' } else { 'NOT STABLE' }))
                    $fieldsCtrl['control_stable_within_250ms'] = $stable
                    Write-ArmVerdict -Dir $verdictDir -Tag 'gplr56-a3-control-stability' -ExitCode $(if ($stable) { 0 } else { 1 }) `
                        -StartedUtc $sessionStart -Fields $fieldsCtrl | Out-Null
                }
            } else {
                Write-Host '  CONTROL STABILITY: n/a - the archived control cannot yield Q (refused)'
            }
        } else {
            Write-Host ("  CONTROL STABILITY: n/a - no archived control at {0}" -f $archivedControl)
        }

        $met = $false
        foreach ($A in $arms) {
            if ($A.tag -eq $ctrlTag -or -not $armRows.ContainsKey($A.tag)) { continue }
            $r = $armRows[$A.tag]
            if (-not $r.ContainsKey('q_ms') -or $r['arm_exit'] -ne 0) {
                Write-Host ("  {0}: NOT SCORABLE (no Q or the arm did not exit 0)" -f $A.tag)
                continue
            }
            $armQ = [double] $r['q_ms']
            $armWall = if ($r.ContainsKey('q_pass1_wall_ms')) { [double] $r['q_pass1_wall_ms'] } else { 0.0 }
            $dQ = [math]::Round($ctrlQ - $armQ, 1)
            $dW = [math]::Round($ctrlWall - $armWall, 1)
            $relocation = ($dW -gt 0.0 -and $dQ -le 250.0)
            $pass = ($dQ -gt 250.0 -and $dW -gt 0.0)
            $extra = @{ q_delta_ms = $dQ; pass1_wall_delta_ms = $dW; q_rule_pass = $pass
                        q_rule_relocation = $relocation; a3_acceptance_met = $pass }
            Write-ArmVerdict -Dir $verdictDir -Tag ($A.tag + '-qrule') -ExitCode $(if ($pass) { 0 } elseif ($relocation) { 1 } else { 1 }) `
                -StartedUtc $sessionStart -Fields $extra | Out-Null
            if ($pass) { $met = $true }
            $verdict = if ($pass) { 'PASS (Q fell > 250 ms AND the wall fell)' }
                       elseif ($relocation) { 'FAIL - RELOCATION (the wall fell while Q did not: the pre-registered rule is not satisfied)' }
                       else { 'FAIL (Q did not fall by more than the margin)' }
            Write-Host ("  {0}: Q delta {1:+0.0;-0.0} ms, pass-1 wall delta {2:+0.0;-0.0} ms -> {3}" -f $A.tag, $dQ, $dW, $verdict)
        }
        if ($met) {
            Write-Host '  A3 ACCEPTANCE: MET - at least one arm cleared the margin AND the wall in one session'
        } else {
            Write-Host '  A3 ACCEPTANCE: NOT MET - no arm cleared the 250 ms margin with a falling wall.'
            Write-Host '    This is the outcome the block pre-registered for the case: the commit is overlappable only'
            Write-Host '    at short shapes, and the block closes NEGATIVE for the charter long-context priority.'
        }
    }
    $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_post_run = {0}" -f $gpuPost)
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
# --- provenance gate (re-scores nothing; see the header) ---------------------
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($arms | ForEach-Object { $_.tag }) `
        -Label 'A3 arm' -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("gplr56-a3 VERDICT {0}" -f $(if ($sessionExit -eq 0) { 'PASS' } else { 'FAIL' }))
exit $sessionExit
