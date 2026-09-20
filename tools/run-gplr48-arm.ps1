# T-GPLR-48 arm runner: one arm, one binary, one verdict + sentinel.
#
# The measured object is the RESOLVED --pp-prebuild-per-pass default, which is
# now KV-conditional (ON iff --pp-repeat > 1 AND --kv-cache != f16). Each arm
# records, from the trace and the log itself:
#   * the resolved value and its SOURCE (explicit | default)   [run_start keys]
#   * the stderr announcement (must appear IFF the default was used)
#   * whatever the contract's ACs need: pass ledgers, tg ids, route parity
#
# Shapes are cloned VERBATIM from the recorded arms the ACs compare against:
#   4k   <- gplr46-4k-off / gplr49b-4k-*     (anchor-4096, ubatch 512, tg 96)
#   50k  <- gplr46-50k-{on,off}              (anchor-50000, ubatch 512, tg 3)
#   b1   <- gplr43-fix-B1-q8                 (anchor-100000, pp 143360, -ctx 144128,
#                                             ubatch 1024, tg 0, mmap-registered; q8)
#
# Binaries: 'changed' = build/llama-cuda/bin/Release (the T-GPLR-48 tree),
#           'control' = build/llama-cuda/bin/gplr48-control (a pre-T-GPLR-48
#           binary whose llama.dll is byte-identical to the gplr49-changed
#           binary the 4k/50k echo arms ran on). Both hashes are recorded in
#           every verdict, because a stale side-copy is indistinguishable from a
#           correct one by name alone.
#
# Exit 0 = PASS. A missing announcement, a resolved value that contradicts the
# KV type (or the pinned flag), a strict-checker failure, a broken route parity
# in a rebuild-OFF run, or a failed decode all fail the arm. A crash fails it
# too -- UNLESS -ExpectCrash pins that exact code, which is how the measured
# "quantized KV + rebuild OFF crashes in pass 2" arm is graded.
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed',
    [ValidateSet('4k', '50k', 'b1')]    [string] $Shape = '50k',
    [ValidateSet('f16', 'q8')]          [string] $Kv = 'f16',
    # 'default' leaves the flag UNSET (the arm that measures the new default);
    # '0' / '1' pin it, which is what AC4's explicit pairs compare.
    [ValidateSet('default', '0', '1')]  [string] $Rebuild = 'default',
    # Optional: an existing verdict tag whose behavior this arm must reproduce
    # (AC4's pair comparison), e.g. the same command on the control binary.
    [string] $CompareTag = '',
    # Optional: an existing verdict tag whose 50k excess is the noise-floor
    # reference for this arm (AC1 uses the explicit-OFF arm).
    [string] $RefTag = '',
    # Pre-registered EXPECTED crash: a pass-2 crash is the measured T-GPLR-43
    # behavior of quantized KV with the rebuild OFF, so that arm PASSES by
    # crashing with exactly this code and its pair comparison uses the exit code
    # as the identity field (a crashed run produces no ids or ledgers).
    [switch] $ExpectCrash,
    [int] $ExpectCrashCode = -1073740791   # 0xC0000409 (measured 2026-09-19)
)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$verdictDir = Join-Path $base 'verdicts'
$parity = Join-Path $root 'tools\check-route-parity.py'
$decode = Join-Path $root 'tools\decode-tg-ids.py'

if ($Shape -eq 'b1') { $Kv = 'q8' }   # the B1 crash shape IS the q8 shape
# The control binary predates this task, so it carries NEITHER the new run_start
# keys NOR the announcement. For a control arm those absences are ASSERTED, not
# tolerated: they are the INV1 evidence that the change is confined to the
# default-resolution path.
$isControl = ($Bin -eq 'control')

$exe = Join-Path $base (("bin\{0}\llama-layer-streaming.exe") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr48-control' }))
$dll = Join-Path $base (("bin\{0}\llama.dll") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr48-control' }))
if (-not (Test-Path -LiteralPath $exe)) { Write-Host "MISSING exe: $exe"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')

# ---------------------------------------------------------------- shape flags
$anchor = $null
$a = @('--model', $model)
switch ($Shape) {
    '4k' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-4096.txt'
        $a += @('--prompt-file', $anchor, '--pp-tokens', '4096', '--ubatch', '512',
                '--pp-no-logits', '--tg-tokens', '96',
                '--residency', 'moe-expert-bank', '--kv-cache', $Kv,
                '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
                '--layer-slots', '3', '--allow-full-residency',
                '--residency-budget-mib', '0', '--pp-repeat', '2',
                '--copy-source', 'pinned', '--pp-async-feed', '1', '--pp-recon', '1')
    }
    '50k' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-50000.txt'
        $a += @('--prompt-file', $anchor, '--pp-tokens', '50000', '--ubatch', '512',
                '--pp-no-logits', '--tg-tokens', '3',
                '--residency', 'moe-expert-bank', '--kv-cache', $Kv,
                '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
                '--layer-slots', '3', '--allow-full-residency',
                '--residency-budget-mib', '0', '--pp-repeat', '2',
                '--copy-source', 'pinned', '--pp-async-feed', '1', '--pp-recon', '1')
    }
    'b1' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-100000.txt'
        $a += @('--prompt-file', $anchor, '--pp-tokens', '143360', '--ctx-size', '144128',
                '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
                '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
                '--pp-schedule', 'layer-major', '--layer-slots', '3',
                '--allow-full-residency', '--pp-repeat', '2',
                '--copy-source', 'mmap-registered', '--pp-async-feed', '1')
    }
}
if (-not (Test-Path -LiteralPath $anchor)) { Write-Host "MISSING anchor: $anchor"; exit 1 }

# The explicit flag is appended LAST and only when the arm pins it, so a
# 'default' arm's command line provably does not carry it.
if ($Rebuild -ne 'default') { $a += @('--pp-prebuild-per-pass', $Rebuild) }

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
$a += @('--telemetry', $trace)

# The resolved value this arm should end up with: a PINNED flag overrides the KV
# type (that IS INV1), so only a 'default' arm's value is derived from the KV.
$expectResolved = if ($Rebuild -eq 'default') { $(if ($Kv -eq 'f16') { 0 } else { 1 }) } else { [int]$Rebuild }
$wantSrc        = if ($Rebuild -eq 'default') { 'default' } else { 'explicit' }
$announceFrag   = if ($expectResolved -eq 1) { 'pp-prebuild-per-pass = 1 (default:' } `
                  else { 'pp-prebuild-per-pass = 0 (default:' }
# Whether the PER-PASS REBUILD actually runs. Route parity is a HARD gate only
# when it does NOT: measured 2026-09-19, pw + rebuild ON breaks route parity at
# every length (4k chunk 7; 50k chunks 10..97; the 143k B1 shape chunks 10..130,
# always 40 layers per sampled chunk). That is T-GPLR-47's defect, not this
# task's -- with the rebuild OFF the same shapes are parity-clean. So the f16
# default arm flipping parity DIVERGED -> clean IS the behavioral proof that the
# new default took effect (a telemetry key alone would prove nothing).
$effectiveRebuildOn = if ($isControl) {
    if ($Rebuild -eq 'default') { $true } else { $Rebuild -eq '1' }
} else {
    if ($Rebuild -eq 'default') { $expectResolved -eq 1 } else { $Rebuild -eq '1' }
}
$parityMustBeClean = -not $effectiveRebuildOn

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("[gplr48] {0}  bin={1} exe {2} dll {3}" -f $Tag, $Bin, $exeSha, $dllSha)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$shapeTokens = switch ($Shape) { '4k' { 4096 } '50k' { 50000 } default { 143360 } }
$fields = @{ shape = $Shape; prompt_tokens = $shapeTokens
             kv_cache = $Kv; rebuild_requested = $Rebuild; bin = $Bin
             expect_resolved_pp_prebuild_per_pass = $expectResolved
             effective_rebuild_on = $effectiveRebuildOn
             exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log }

Remove-Item $trace, $log -ErrorAction SilentlyContinue
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
$failedEarly = $false
try {
    Write-Host '  waiting for the GPU mutex'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode 124 -StartedUtc $started `
            -LogPath $log -ErrorText 'timed out waiting for the GPU mutex' -Fields $fields | Out-Null
        $failedEarly = $true
    } else {
        Write-Host '  mutex held'
        $fields['command'] = ($exe + ' ' + ($a -join ' '))

        $t0 = Get-Date
        & $exe @a > $log 2>&1
        $armExit = $LASTEXITCODE
        $fields['arm_exit'] = $armExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])
        Get-Content $log | Select-String -Pattern 'PP:|PP pass 2/2|graphs rebuilt|pp-prebuild-per-pass|error|CUDA error' |
            Select-Object -Last 8 | ForEach-Object { Write-Host "  log: $_" }

        $crashed = ($armExit -ne 0)
        $fields['crash_as_expected'] = $false
        if ($crashed -and $ExpectCrash -and $armExit -eq $ExpectCrashCode) {
            # The pre-registered crash: that IS this arm's result, not a harness
            # failure. The pair comparison below then uses the exit code as its
            # identity field, since a crashed run has no ids or ledgers.
            $fields['crash_as_expected'] = $true
            $fields['expect_crash_code'] = $ExpectCrashCode
            Write-Host '  crash AS EXPECTED (rebuild OFF + quantized KV)'
        } elseif ($crashed) {
            $fields['fail_reason'] = ("arm exited {0}" -f $armExit)
        }

        if (-not $crashed) {
            # ---- (a) resolved default + its source, from the trace's run_start
            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })
            if ($rs.Count -eq 0) {
                $fields['fail_reason'] = 'no run_start event'
            } else {
                $j = $rs[0] | ConvertFrom-Json
                $fields['run_start_has_resolved_key'] = [bool]($j.PSObject.Properties.Name -contains 'pp_prebuild_per_pass')
                $fields['run_start_has_source_key']   = [bool]($j.PSObject.Properties.Name -contains 'pp_prebuild_per_pass_source')
                if ($fields['run_start_has_resolved_key']) {
                    $fields['resolved_pp_prebuild_per_pass'] = $j.pp_prebuild_per_pass
                    $fields['pp_prebuild_per_pass_source'] = $j.pp_prebuild_per_pass_source
                }
                $fields['run_start_kv_cache'] = $j.kv_cache
                Write-Host ("  run_start: resolved={0} source={1} kv={2}" -f `
                    $j.pp_prebuild_per_pass, $j.pp_prebuild_per_pass_source, $j.kv_cache)
            }

            # ---- (b) the announcement: present IFF the value came from the default
            $ann = @(Get-Content $log | Select-String -Pattern 'pp-prebuild-per-pass = ')
            $fields['announcement_present'] = [bool]($ann.Count -gt 0)
            $fields['announcement_matches_expectation'] = [bool](@($ann | Select-String -Pattern ([regex]::Escape($announceFrag))).Count -gt 0)
            $fields['announcement_line'] = if ($ann.Count -gt 0) { ([string]$ann[0]).Trim() } else { '' }
            Write-Host ("  announcement present={0} matches={1}" -f $fields['announcement_present'], $fields['announcement_matches_expectation'])

            # ---- (c) strict checker
            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

            # ---- (d) pass ledgers (host_ledger; traced only under --pp-recon)
            $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                         ForEach-Object { $_ | ConvertFrom-Json })
            $fields['pass_count'] = $ledgers.Count
            $compute = @()
            for ($i = 0; $i -lt $ledgers.Count; $i++) {
                $l = $ledgers[$i]
                $compute += [double]$l.compute_us
                $fields[("pass{0}_compute_us" -f ($i + 1))] = [math]::Round($l.compute_us, 1)
                $fields[("pass{0}_wall_ms"    -f ($i + 1))] = [math]::Round($l.sweep_us / 1000.0, 1)
                $fields[("pass{0}_patch_us"   -f ($i + 1))] = [math]::Round($l.patch_us, 1)
                $fields[("pass{0}_launches"   -f ($i + 1))] = $l.launches
                Write-Host ("  pass {0}: sweep {1:N0} ms  compute {2:N0} ms  patch {3:N0} ms  launches {4}" -f `
                    ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0), ($l.patch_us / 1000.0), $l.launches)
            }
            if ($compute.Count -ge 2) {
                $fields['excess_2v1_s'] = [math]::Round(($compute[1] - $compute[0]) / 1e6, 2)
                Write-Host ("  excess pass2-pass1 compute: {0:N2} s" -f $fields['excess_2v1_s'])
            }

            # ---- (e) tg ids + graded decode (b1 has tg 0 by design)
            $idsLine = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
            if ($idsLine.Count -gt 0) {
                $ids = @(($idsLine[-1] | ConvertFrom-Json).ids)
                $fields['tg_id_count'] = $ids.Count
                $fields['tg_ids'] = ($ids -join ',')
                $fields['first_token'] = $ids[0]
                Write-Host ("  {0} tg ids, first {1}" -f $ids.Count, $ids[0])
                if ($Shape -eq '4k') {
                    $grade = Join-Path $base "$Tag.grade.json"
                    & python $decode --trace $trace --model $model --json $grade --quiet 2>&1 |
                        ForEach-Object { Write-Host "  decode: $_" }
                    # decode-tg-ids.py exits 0 IFF its aggregate verdict is PASS,
                    # so the exit code IS the grade; the JSON's nested 'verdict'
                    # carries the detail and 'text' is the decoded stream (AC2).
                    $fields['decode_exit'] = $LASTEXITCODE
                    if (Test-Path -LiteralPath $grade) {
                        $g = Get-Content $grade -Raw | ConvertFrom-Json
                        $fields['decode_aggregate'] = $g.verdict.aggregate
                        $fields['decode_n_ids'] = $g.n_ids
                        $text = [string]$g.text
                        $fields['answer_head'] = $text.Substring(0, [Math]::Min(90, $text.Length))
                    }
                    Write-Host ("  decode aggregate {0} (exit {1})" -f $fields['decode_aggregate'], $fields['decode_exit'])
                }
            } else {
                Write-Host '  no tg_tokens event (tg-tokens 0 by design for this shape)'
            }

            # ---- (f) route parity: the pass-2 damage signal that works at every
            #          length and needs no decode
            $parityJson = Join-Path $base "$Tag.parity.json"
            & python $parity $trace --json $parityJson --quiet 2>&1 | ForEach-Object { Write-Host "  parity: $_" }
            $fields['route_parity_exit'] = $LASTEXITCODE
            $fields['parity_expected'] = $(if ($parityMustBeClean) { 'clean' } else { 'diverged (pw + rebuild ON, T-GPLR-47)' })
            if (Test-Path -LiteralPath $parityJson) {
                $p = (Get-Content $parityJson -Raw | ConvertFrom-Json).reports[0]
                $fields['route_parity'] = $p.parity
                $fields['route_events'] = $p.route_events
                $fields['route_singletons'] = $p.singletons
                $fields['route_diverging_chunks'] = (@($p.diverging_pairs | ForEach-Object { $_.chunk }) | Sort-Object -Unique) -join ','
            }
        }

        # ---- (g) optional reference: the noise floor AC1 compares against
        if ($RefTag) {
            $rv = Read-ArmVerdict -Dir $verdictDir -Tag $RefTag
            if ($null -ne $rv -and $rv.fields.PSObject.Properties.Name -contains 'excess_2v1_s') {
                $fields['ref_tag'] = $RefTag
                $fields['ref_excess_2v1_s'] = [double]$rv.fields.excess_2v1_s
                if ($fields.ContainsKey('excess_2v1_s')) {
                    $fields['excess_delta_vs_ref_s'] = [math]::Round($fields['excess_2v1_s'] - $fields['ref_excess_2v1_s'], 2)
                    Write-Host ("  vs {0}: excess delta {1:N2} s" -f $RefTag, $fields['excess_delta_vs_ref_s'])
                }
            }
        }

        # ---- (h) optional pair comparison (AC4): identical BEHAVIOR expected.
        if ($CompareTag) {
            $cv = Read-ArmVerdict -Dir $verdictDir -Tag $CompareTag
            if ($null -eq $cv) {
                $fields['compare_tag_missing'] = $CompareTag
                $fields['fail_reason'] = ("compare tag {0} has no verdict" -f $CompareTag)
            } else {
                $fields['compare_tag'] = $CompareTag
                if ($fields['crash_as_expected'] -or $cv.fields.crash_as_expected) {
                    # A crashed run has no ids or ledgers; the crash CODE is the
                    # identity field (same defect, same signature, both binaries).
                    $fields['cmp_arm_exit_identical'] = ([int]$fields['arm_exit'] -eq [int]$cv.fields.arm_exit)
                    $fields['compare_identity_ok'] = $fields['cmp_arm_exit_identical']
                    Write-Host ("  pair crash codes: {0} vs {1} -> identical {2}" -f `
                        $fields['arm_exit'], $cv.fields.arm_exit, $fields['cmp_arm_exit_identical'])
                } else {
                    # IDENTITY: must agree exactly (same work, same output).
                    $idFields = @('tg_ids', 'pass_count', 'pass1_launches', 'pass2_launches',
                                  'route_parity', 'route_events')
                    if (($cv.fields.PSObject.Properties.Name -contains 'resolved_pp_prebuild_per_pass') -and
                        $fields.ContainsKey('resolved_pp_prebuild_per_pass')) {
                        $idFields += 'resolved_pp_prebuild_per_pass'
                    }
                    foreach ($f in $idFields) {
                        $fields[("cmp_{0}_identical" -f $f)] = ("{0}" -f $fields[$f]) -eq ("{0}" -f $cv.fields.$f)
                    }
                    $bad = @($fields.Keys | Where-Object { $_ -like 'cmp_*_identical' -and -not $fields[$_] })
                    $fields['compare_identity_ok'] = ($bad.Count -eq 0)
                    if ($bad.Count -gt 0) { Write-Host ("  PAIR IDENTITY MISMATCH on: {0}" -f (($bad -join ', '))) }
                    else { Write-Host '  pair identity fields identical' }

                    # TIMING: NOT identity. Compared inside the measured session
                    # spread (AC6 uses ~3%); the delta is recorded either way.
                    $timingBad = @()
                    foreach ($f in @('pass1_compute_us', 'pass2_compute_us')) {
                        $mine = [double]$fields[$f]
                        $theirs = [double]$cv.fields.$f
                        if ($theirs -gt 0) {
                            $pct = [math]::Round(100.0 * ($mine - $theirs) / $theirs, 2)
                            $fields[("cmp_{0}_delta_pct" -f $f)] = $pct
                            if ([math]::Abs($pct) -gt 3.0) { $timingBad += $f }
                        }
                    }
                    $fields['compare_timing_within_3pct'] = ($timingBad.Count -eq 0)
                    if ($timingBad.Count -gt 0) { Write-Host ("  PAIR TIMING OUTSIDE 3% on: {0}" -f (($timingBad -join ', '))) }
                }
            }
        }

        # ---- verdict
        $fail = @()
        if ($fields.ContainsKey('fail_reason')) { $fail += $fields['fail_reason'] }
        if ($fields['crash_as_expected']) {
            # Nothing else to assert: the crash code IS the recorded outcome.
        } else {
            if ($isControl) {
                if ($fields['run_start_has_resolved_key']) { $fail += 'CONTROL carried the new resolved key' }
                if ($fields['run_start_has_source_key'])   { $fail += 'CONTROL carried the new source key' }
                if ($fields['announcement_present'])       { $fail += 'CONTROL emitted the announcement' }
            } else {
                if ($fields.ContainsKey('resolved_pp_prebuild_per_pass')) {
                    if ([int]$fields['resolved_pp_prebuild_per_pass'] -ne $expectResolved) {
                        $fail += ("resolved {0} != expected {1} (kv {2}, requested {3})" -f $fields['resolved_pp_prebuild_per_pass'], $expectResolved, $Kv, $Rebuild)
                    }
                    if ($fields['pp_prebuild_per_pass_source'] -ne $wantSrc) {
                        $fail += ("source '{0}' != expected '{1}'" -f $fields['pp_prebuild_per_pass_source'], $wantSrc)
                    }
                } else { $fail += 'run_start carried no resolved value' }
                if ($Rebuild -eq 'default') {
                    if (-not $fields['announcement_present']) { $fail += 'no default announcement' }
                    elseif (-not $fields['announcement_matches_expectation']) { $fail += 'announcement names the wrong default' }
                } elseif ($fields['announcement_present']) {
                    $fail += 'announcement emitted for an EXPLICIT flag'
                }
            }
            if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
            if ($parityMustBeClean) {
                if ($fields['route_parity_exit'] -ne 0) {
                    $fail += ("route parity exit {0} ({1}) but rebuild is OFF" -f $fields['route_parity_exit'], $fields['route_parity'])
                }
            } elseif ($fields['route_parity_exit'] -eq 2) {
                # 'no route events' is never acceptable: the gate did not run.
                $fail += 'route parity had no route events to evaluate'
            }
            if ($fields.ContainsKey('decode_exit') -and $fields['decode_exit'] -ne 0) {
                $fail += ("decode exit {0} aggregate {1}" -f $fields['decode_exit'], $fields['decode_aggregate'])
            }
            if ($fields.ContainsKey('compare_identity_ok') -and -not $fields['compare_identity_ok']) {
                $fail += 'pair IDENTITY fields differ'
            }
            if ($fields.ContainsKey('compare_timing_within_3pct') -and -not $fields['compare_timing_within_3pct']) {
                $fail += 'pair timing outside the 3% session spread'
            }
        }
        if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}

if (-not $failedEarly) {
    Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode $exit -StartedUtc $started `
        -LogPath $log -Fields $fields | Out-Null
    Write-Host ("VERDICT {0} ({1}) resolved={2} source={3} parity={4} -> {5}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit,
        $fields['resolved_pp_prebuild_per_pass'], $fields['pp_prebuild_per_pass_source'],
        $fields['route_parity'], (Join-Path $verdictDir ($Tag + '.done')))
}
exit $exit
