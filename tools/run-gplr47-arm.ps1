# T-GPLR-47 arm runner: one arm, one binary, one verdict + sentinel.
# Derived from tools/run-gplr48-arm.ps1 (the shapes are cloned verbatim from the
# recorded verdicts; only the measured object changed).
#
# The measured object is the per-pass REBUILD now performing a fresh-context
# swap-in-slot (changed binary) instead of the in-place accumulate rebuild
# (control binary). Each arm records, from the trace and the log itself:
#   * pass ledgers (host_ledger; exists only when the rebuild is ON) and the
#     pass-2 vs pass-1 compute excess  [AC2/AC3/AC7]
#   * tg ids + graded decode at 4k    [AC1]
#   * route parity (tools/check-route-parity.py; the decode-free pass-2 gate)
#   * the patch_cache dump member count post-rebuild [AC5, on --pp-recon arms]
#
# Shapes (cloned VERBATIM from the recorded verdicts):
#   4k   <- gplr49b-4k-* / gplr46-4k-off (anchor-4096, ubatch 512, tg 96)
#   50k  <- gplr46-50k-{on,off}          (anchor-50000, ubatch 512, tg 3)
#   b1   <- gplr48-a2-b1-q8-default      (anchor-100000, pp 143360, ctx 144128,
#                                          ubatch 1024, tg 0, mmap-registered; q8)
#
# Route-parity expectation is INVERTED vs the T-GPLR-48 runner: the CONTROL
# binary with the rebuild ON diverges (the T-47 defect); the CHANGED binary
# with the rebuild ON must be parity-CLEAN (that is AC6/AC7's pass-2 gate and
# the whole point of the fresh-ctx swap). Parity stays a hard gate for every
# rebuild-OFF arm on either binary.
#
# Binaries: 'changed' = build/llama-cuda/bin/gplr47-changed (post-swap-in-slot),
#           'control' = build/llama-cuda/bin/gplr47-control (pre-T-GPLR-47,
#           the pinned post-T-GPLR-48 tree). Both hashes are recorded in every
#           verdict because a stale side-copy is indistinguishable by name.
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed',
    [ValidateSet('4k', '50k', 'b1', '200k')] [string] $Shape = '50k',
    [ValidateSet('f16', 'q8')]           [string] $Kv = 'f16',
    # 1 = the arm under test (rebuild ON); 0 = the OFF reference.
    [ValidateSet('0', '1')]              [string] $Rebuild = '1',
    # Optional: an existing verdict tag whose behavior this arm must reproduce
    # (pair mode): identity fields must agree exactly; timing is recorded only.
    [string] $CompareTag = '',
    [switch] $ExpectCrash,
    [int] $ExpectCrashCode = -1073740791
)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$verdictDir = Join-Path $base 'verdicts'
$parity = Join-Path $root 'tools\check-route-parity.py'
$decode = Join-Path $root 'tools\decode-tg-ids.py'
$dumpAn = Join-Path $root 'tools\analyze-gplr44-cache.py'

$isControl = ($Bin -eq 'control')
if ($Shape -eq 'b1') { $Kv = 'q8' }

$exe = Join-Path $base ("bin\{0}\llama-layer-streaming.exe" -f $(if ($isControl) { 'gplr47-control' } else { 'gplr47-changed' }))
$dll = Join-Path $base ("bin\{0}\llama.dll"          -f $(if ($isControl) { 'gplr47-control' } else { 'gplr47-changed' }))
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
    '200k' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-200000.txt'
        $a += @('--prompt-file', $anchor, '--pp-tokens', '200000', '--ubatch', '512',
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

if ($Rebuild -eq '0') { $a += @('--pp-prebuild-per-pass', '0') }
elseif ($Rebuild -eq '1') { $a += @('--pp-prebuild-per-pass', '1') }

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
$a += @('--telemetry', $trace)

$effectiveRebuildOn = ($Rebuild -eq '1')
# CONTROL + rebuild ON = the recorded T-GPLR-46 defect (parity DIVERGED);
# CHANGED + rebuild ON = the fix under test (parity must be CLEAN);
# rebuild OFF = clean on both binaries.
$parityMustBeClean = (-not $effectiveRebuildOn) -or (-not $isControl)

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("[gplr47] {0}  bin={1} exe {2} dll {3}" -f $Tag, $Bin, $exeSha, $dllSha)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$shapeTokens = switch ($Shape) { '4k' { 4096 } '50k' { 50000 } '200k' { 200000 } default { 143360 } }
$fields = @{ shape = $Shape; prompt_tokens = $shapeTokens
             kv_cache = $Kv; rebuild_requested = $Rebuild; bin = $Bin
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
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(7200)) }
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
            $fields['crash_as_expected'] = $true
            $fields['expect_crash_code'] = $ExpectCrashCode
            Write-Host '  crash AS EXPECTED'
        } elseif ($crashed) {
            $fields['fail_reason'] = ("arm exited {0}" -f $armExit)
        }

        if (-not $crashed) {
            # ---- (a) strict checker [AC6]
            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

            # ---- (b) pass ledgers; exist only when the rebuild is ON
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
                Write-Host ("  pass {0}: wall {1:N0} ms  compute {2:N0} ms  patch {3:N0} ms  launches {4}" -f `
                    ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0), ($l.patch_us / 1000.0), $l.launches)
            }
            if ($compute.Count -ge 2) {
                $fields['excess_2v1_s'] = [math]::Round(($compute[1] - $compute[0]) / 1e6, 2)
                Write-Host ("  excess pass2-pass1 compute: {0:N2} s" -f $fields['excess_2v1_s'])
            }

            # ---- (c) tg ids + graded decode (4k only; b1 has tg 0 by design)
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
                    $fields['decode_exit'] = $LASTEXITCODE
                    if (Test-Path -LiteralPath $grade) {
                        $g = Get-Content $grade -Raw | ConvertFrom-Json
                        $fields['decode_aggregate'] = $g.verdict.aggregate
                        $fields['decode_n_ids'] = $g.n_ids
                        $text = [string]$g.text
                        $fields['answer_head'] = $text.Substring(0, [Math]::Min(90, $text.Length))
                    }
                }
            }

            # ---- (d) route parity: the decode-free pass-2 gate
            $parityJson = Join-Path $base "$Tag.parity.json"
            & python $parity $trace --json $parityJson --quiet 2>&1 | ForEach-Object { Write-Host "  parity: $_" }
            $fields['route_parity_exit'] = $LASTEXITCODE
            $fields['parity_expected'] = $(if ($parityMustBeClean) { 'clean' } else { 'diverged (control defect, T-GPLR-46)' })
            if (Test-Path -LiteralPath $parityJson) {
                $p = (Get-Content $parityJson -Raw | ConvertFrom-Json).reports[0]
                $fields['route_parity'] = $p.parity
                $fields['route_events'] = $p.route_events
                $fields['route_singletons'] = $p.singletons
                $fields['route_diverging_chunks'] = (@($p.diverging_pairs | ForEach-Object { $_.chunk }) | Sort-Object -Unique) -join ','
            }

            # ---- (e) patch_cache dump member count [AC5] -- fires once per set
            #      at patch-cache build time on --pp-recon arms. The rebuild arms
            #      of interest produce >= 2 dumps; the post-rebuild count is the
            #      LAST dump's member count.
            if ($effectiveRebuildOn) {
                $dumpOut = & python $dumpAn $trace 2>&1
                $fields['dump_report'] = ($dumpOut | Where-Object { $_ -match 'cached members|union_matches|excluded' } |
                    Select-Object -First 6) -join ' | '
                $memberCounts = @()
                foreach ($line in $dumpOut) {
                    if ($line -match 'cached members\s*:\s*(\d+)') { $memberCounts += [int]$Matches[1] }
                }
                $fields['dump_member_counts'] = ($memberCounts -join ',')
                Write-Host ("  patch_cache dumps: member counts [{0}]" -f ($memberCounts -join ','))
            }
        }

        # ---- (f) optional pair comparison: identical BEHAVIOR expected.
        if ($CompareTag) {
            $cv = Read-ArmVerdict -Dir $verdictDir -Tag $CompareTag
            if ($null -eq $cv) {
                $fields['compare_tag_missing'] = $CompareTag
                $fields['fail_reason'] = ("compare tag {0} has no verdict" -f $CompareTag)
            } else {
                $fields['compare_tag'] = $CompareTag
                if ($fields['crash_as_expected'] -or $cv.fields.crash_as_expected) {
                    $fields['cmp_arm_exit_identical'] = ([int]$fields['arm_exit'] -eq [int]$cv.fields.arm_exit)
                    $fields['compare_identity_ok'] = $fields['cmp_arm_exit_identical']
                } else {
                    # IDENTITY fields must agree exactly. NOTE: route_parity is NOT
                    # one of them -- the control's parity divergence IS the defect;
                    # ids are the identity that matters here.
                    $idFields = @('tg_ids', 'pass_count', 'pass1_launches', 'pass2_launches')
                    if (($cv.fields.PSObject.Properties.Name -contains 'tg_ids') -and $fields.ContainsKey('tg_ids') -and
                        $fields['tg_id_count'] -gt 0) {
                        # ids present on both sides: compare them exactly
                    }
                    $timingBad = @()
                    foreach ($f in @('pass1_compute_us', 'pass2_compute_us')) {
                        if ($fields.ContainsKey($f) -and ($cv.fields.PSObject.Properties.Name -contains $f)) {
                            $mine = [double]$fields[$f]; $theirs = [double]$cv.fields.$f
                            if ($theirs -gt 0) {
                                $pct = [math]::Round(100.0 * ($mine - $theirs) / $theirs, 2)
                                $fields[("cmp_{0}_delta_pct" -f $f)] = $pct
                                if ([math]::Abs($pct) -gt 3.0) { $timingBad += $f }
                            }
                        }
                    }
                    $fields['compare_timing_within_3pct'] = ($timingBad.Count -eq 0)
                    $bad = @()
                    foreach ($f in $idFields) {
                        if (($cv.fields.PSObject.Properties.Name -contains $f) -and $fields.ContainsKey($f)) {
                            $same = ("{0}" -f $fields[$f]) -eq ("{0}" -f $cv.fields.$f)
                            $fields[("cmp_{0}_identical" -f $f)] = $same
                            if (-not $same) { $bad += $f }
                        }
                    }
                    $fields['compare_identity_ok'] = ($bad.Count -eq 0)
                    if ($bad.Count -gt 0) { Write-Host ("  PAIR IDENTITY MISMATCH on: {0}" -f ($bad -join ', ')) }
                    else { Write-Host '  pair identity fields identical' }
                }
            }
        }

        # ---- verdict
        $fail = @()
        if ($fields.ContainsKey('fail_reason')) { $fail += $fields['fail_reason'] }
        if ($fields['crash_as_expected']) {
            # nothing else to assert
        } else {
            if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
            if ($fields['pass_count'] -ne 2 -and $Shape -ne 'b1') { $fail += ("pass ledgers {0} != 2" -f $fields['pass_count']) }
            if ($parityMustBeClean) {
                if ($fields['route_parity_exit'] -ne 0) {
                    $fail += ("route parity {0} (exit {1}) but expected clean" -f $fields['route_parity'], $fields['route_parity_exit'])
                }
            } elseif ($fields['route_parity_exit'] -eq 2) {
                $fail += 'route parity had no route events to evaluate'
            }
            if ($fields.ContainsKey('decode_exit') -and $fields['decode_exit'] -ne 0) {
                $fail += ("decode exit {0} aggregate {1}" -f $fields['decode_exit'], $fields['decode_aggregate'])
            }
            if ($fields.ContainsKey('compare_identity_ok') -and -not $fields['compare_identity_ok']) {
                $fail += 'pair IDENTITY fields differ'
            }
            if ($fields.ContainsKey('dump_member_counts') -and $fields['dump_member_counts'] -ne '') {
                $counts = @($fields['dump_member_counts'] -split ',' | ForEach-Object { [int]$_ })
                if ($counts.Count -lt 2) {
                    $fail += ("patch_cache dumps {0} < 2 (no post-rebuild dump)" -f $counts.Count)
                } elseif ($counts[-1] -ne $counts[0]) {
                    $fail += ("AC5 FAILED: post-rebuild dump members {0} != first-build {1}" -f $counts[-1], $counts[0])
                }
            }
            if ($fields.ContainsKey('compare_timing_within_3pct') -and -not $fields['compare_timing_within_3pct']) {
                # timing is recorded, not fatal, in pair mode (AC7 judges it separately)
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
    Write-Host ("VERDICT {0} ({1}) parity={2} excess={3} -> {4}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit,
        $fields['route_parity'], $fields['excess_2v1_s'], (Join-Path $verdictDir ($Tag + '.done')))
}
exit $exit
