# T-GPLR-50 arm runner: derived from run-gplr48-arm.ps1 (shapes, verdict
# plumbing, and pair mechanics verbatim); the resolution expectations and
# parity policy change per the T-GPLR-50 contract:
#   * a 'default' arm on the CHANGED binary must resolve rebuild ON for EVERY
#     KV type (f16 included) whenever repeat > 1 - the flip under test;
#   * route parity must be CLEAN in every non-crash arm on the current tree:
#     the pre-T-47 "pw + rebuild ON diverges" signature is T-GPLR-47-fixed
#     (fresh-context swap-in-slot rebuild), and a clean parity at pw+rebuild-ON
#     is itself the AC2-adjacent behavioral proof that the fixed rebuild runs.
# Control (INV1) = a pre-T-50 binary: at bare default with f16 it must resolve
# rebuild OFF and emit the T-48 f16-off announcement; with q8 it resolves ON
# (the T-48 quantized branch) and announces the T-48 wording.
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed',
    [ValidateSet('4k', '50k', 'b1', '200k')][string] $Shape = '4k',
    [ValidateSet('f16', 'q8')]           [string] $Kv = 'f16',
    [ValidateSet('default', '0', '1')]   [string] $Rebuild = 'default',
    [string] $CompareTag = ''
)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$verdictDir = Join-Path $base 'verdicts'
$parity = Join-Path $root 'tools\check-route-parity.py'
$decode = Join-Path $root 'tools\decode-tg-ids.py'

if ($Shape -eq 'b1') { $Kv = 'q8' }
$isControl = ($Bin -eq 'control')

$exeDir = if ($Bin -eq 'changed') { 'Release' } else { 'gplr50-control' }
$exe = Join-Path $base ("bin\{0}\llama-layer-streaming.exe" -f $exeDir)
$dll = Join-Path $base ("bin\{0}\llama.dll" -f $exeDir)
if (-not (Test-Path -LiteralPath $exe)) { Write-Host "MISSING exe: $exe"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')

# ---------------------------------------------------------------- shape flags
# Shapes cloned verbatim from run-gplr48-arm.ps1; 200k = the recorded
# gplr36-remeasure-200k hostmem shape (the AC3 pair's measured object).
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
    '200k' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-200000.txt'
        if (-not (Test-Path -LiteralPath $anchor)) { $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-262144.txt' }
        $a += @('--prompt-file', $anchor, '--pp-tokens', '200000', '--ctx-size', '200704',
                '--ubatch', '512', '--tg-tokens', '3', '--pp-no-logits',
                '--residency', 'moe-expert-bank', '--kv-cache', $Kv,
                '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
                '--layer-slots', '3', '--allow-full-residency',
                '--residency-budget-mib', '0', '--pp-repeat', '2',
                '--copy-source', 'mmap-registered', '--pp-async-feed', '1', '--pp-recon', '1')
    }
}
if (-not (Test-Path -LiteralPath $anchor)) { Write-Host "MISSING anchor: $anchor"; exit 1 }

if ($Rebuild -ne 'default') { $a += @('--pp-prebuild-per-pass', $Rebuild) }

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
$a += @('--telemetry', $trace)

# Resolved-value expectations per binary era:
#   changed, default -> 1 for every KV (T-50 flip), source 'default'
#   changed, pinned  -> the pinned value, source 'explicit'
#   control, default -> T-48 semantics: ON iff repeat>1 AND kv != f16
#   control, pinned  -> the pinned value, source 'explicit'
if ($Rebuild -ne 'default') {
    $expectResolved = [int]$Rebuild
    $wantSrc        = 'explicit'
    $announceMust   = $false
} elseif ($isControl) {
    $expectResolved = $(if ($Kv -eq 'f16') { 0 } else { 1 })
    $wantSrc        = 'default'
    $announceMust   = $true
} else {
    $expectResolved = 1
    $wantSrc        = 'default'
    $announceMust   = $true
}
$announceFrag = if ($isControl) {
    # control announcements use the T-48 wording
    if ($expectResolved -eq 1) { 'pp-prebuild-per-pass = 1 (default:' } else { 'pp-prebuild-per-pass = 0 (default:' }
} else {
    # changed announcements carry the T-50 marker
    'default since T-GPLR-50'
}

# Parity policy on the current tree: clean REQUIRED for every non-crash arm
# that produces route events. Under --pp-recon 1 + route telemetry the 4k arms
# carry route events only with --route-telemetry 1, so arms that need the
# parity gate add it (the runner appends nothing by default; the set script
# passes -ExtraArgs for that arm).
$parityMustRun = $false

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("[gplr50] {0}  bin={1} exe {2} dll {3}" -f $Tag, $Bin, $exeSha, $dllSha)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$shapeTokens = switch ($Shape) { '4k' { 4096 } '50k' { 50000 } '200k' { 200000 } default { 143360 } }
$fields = @{ shape = $Shape; prompt_tokens = $shapeTokens
             kv_cache = $Kv; rebuild_requested = $Rebuild; bin = $Bin
             expect_resolved_pp_prebuild_per_pass = $expectResolved
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
        Get-Content $log | Select-String -Pattern 'pp-prebuild-per-pass|error|CUDA error' |
            Select-Object -Last 6 | ForEach-Object { Write-Host "  log: $_" }

        $crashed = ($armExit -ne 0)
        if ($crashed) { $fields['fail_reason'] = ("arm exited {0}" -f $armExit) }

        if (-not $crashed) {
            # ---- (a) resolved default + source from run_start
            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })
            if ($rs.Count -eq 0) {
                $fields['fail_reason'] = 'no run_start event'
            } else {
                $j = $rs[0] | ConvertFrom-Json
                $fields['resolved_pp_prebuild_per_pass'] = $j.pp_prebuild_per_pass
                $fields['pp_prebuild_per_pass_source']   = $j.pp_prebuild_per_pass_source
                $fields['run_start_kv_cache'] = $j.kv_cache
                Write-Host ("  run_start: resolved={0} source={1} kv={2}" -f `
                    $j.pp_prebuild_per_pass, $j.pp_prebuild_per_pass_source, $j.kv_cache)
            }

            # ---- (b) announcement: changed-default arms must carry the T-50
            #          wording; control-default arms must carry T-48's wording
            #          and must NOT carry the T-50 marker.
            $ann = @(Get-Content $log | Select-String -Pattern 'pp-prebuild-per-pass = ')
            $fields['announcement_present'] = [bool]($ann.Count -gt 0)
            if ($isControl) {
                $fields['control_has_T50_marker'] = [bool](@($ann | Select-String -Pattern 'T-GPLR-50').Count -gt 0)
            } else {
                $fields['announcement_matches_expectation'] = [bool](@($ann | Select-String -SimpleMatch $announceFrag).Count -gt 0)
            }
            $fields['announcement_line'] = if ($ann.Count -gt 0) { ([string]$ann[0]).Trim() } else { '' }
            Write-Host ("  announcement: {0}" -f $fields['announcement_line'])

            # ---- (c) strict checker
            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

            # ---- (d) pass ledgers + the AC3 excess metric
            $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                         ForEach-Object { $_ | ConvertFrom-Json })
            $fields['pass_count'] = $ledgers.Count
            $compute = @()
            for ($i = 0; $i -lt $ledgers.Count; $i++) {
                $l = $ledgers[$i]
                $compute += [double]$l.compute_us
                $fields[("pass{0}_compute_us" -f ($i + 1))] = [math]::Round($l.compute_us, 1)
                $fields[("pass{0}_wall_ms"    -f ($i + 1))] = [math]::Round($l.sweep_us / 1000.0, 1)
                Write-Host ("  pass {0}: sweep {1:N0} ms  compute {2:N0} ms" -f ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0))
            }
            if ($compute.Count -ge 2) {
                $fields['excess_2v1_s'] = [math]::Round(($compute[1] - $compute[0]) / 1e6, 2)
                Write-Host ("  excess pass2-pass1 compute: {0:N2} s" -f $fields['excess_2v1_s'])
            }

            # ---- (e) tg ids + graded decode (4k/200k shapes carry tg ids)
            $idsLine = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
            if ($idsLine.Count -gt 0) {
                $ids = @(($idsLine[-1] | ConvertFrom-Json).ids)
                $fields['tg_id_count'] = $ids.Count
                $fields['tg_ids'] = ($ids -join ',')
                Write-Host ("  {0} tg ids, first {1}" -f $ids.Count, $ids[0])
                if ($Shape -eq '4k') {
                    $grade = Join-Path $base "$Tag.grade.json"
                    & python $decode --trace $trace --model $model --json $grade --quiet 2>&1 |
                        ForEach-Object { Write-Host "  decode: $_" }
                    $fields['decode_exit'] = $LASTEXITCODE
                    if (Test-Path -LiteralPath $grade) {
                        $g = Get-Content $grade -Raw | ConvertFrom-Json
                        $fields['decode_aggregate'] = $g.verdict.aggregate
                    }
                }
            }

            # ---- (f) route parity whenever route events exist
            $parityJson = Join-Path $base "$Tag.parity.json"
            & python $parity $trace --json $parityJson --quiet 2>&1 | ForEach-Object { Write-Host "  parity: $_" }
            $fields['route_parity_exit'] = $LASTEXITCODE
            if (Test-Path -LiteralPath $parityJson) {
                $p = (Get-Content $parityJson -Raw | ConvertFrom-Json).reports[0]
                $fields['route_parity'] = $p.parity
                $fields['route_events'] = $p.route_events
                $fields['route_singletons'] = $p.singletons
            }
        }

        # ---- (g) optional pair comparison
        if ($CompareTag) {
            $cv = Read-ArmVerdict -Dir $verdictDir -Tag $CompareTag
            if ($null -eq $cv) {
                $fields['fail_reason'] = ("compare tag {0} has no verdict" -f $CompareTag)
            } else {
                $fields['compare_tag'] = $CompareTag
                $idFields = @('tg_ids', 'pass_count', 'route_parity', 'route_events')
                foreach ($f in $idFields) {
                    if ($fields.PSObject.Properties.Name -contains $f) {
                        $fields[("cmp_{0}_identical" -f $f)] = ("{0}" -f $fields[$f]) -eq ("{0}" -f $cv.fields.$f)
                    }
                }
                $bad = @($fields.Keys | Where-Object { $_ -like 'cmp_*_identical' -and -not $fields[$_] })
                $fields['compare_identity_ok'] = ($bad.Count -eq 0)
                if ($bad.Count -gt 0) { Write-Host ("  PAIR IDENTITY MISMATCH on: {0}" -f (($bad -join ', '))) }
                else { Write-Host '  pair identity fields identical' }
            }
        }

        # ---- verdict
        $fail = @()
        if ($fields.ContainsKey('fail_reason')) { $fail += $fields['fail_reason'] }
        if (-not $crashed) {
            if ($fields.ContainsKey('resolved_pp_prebuild_per_pass')) {
                if ([int]$fields['resolved_pp_prebuild_per_pass'] -ne $expectResolved) {
                    $fail += ("resolved {0} != expected {1}" -f $fields['resolved_pp_prebuild_per_pass'], $expectResolved)
                }
                if ("$($fields['pp_prebuild_per_pass_source'])" -ne $wantSrc) {
                    $fail += ("source '{0}' != expected '{1}'" -f $fields['pp_prebuild_per_pass_source'], $wantSrc)
                }
            } else { $fail += 'run_start carried no resolved value' }
            if ($isControl) {
                if ($fields['control_has_T50_marker']) { $fail += 'CONTROL emitted the T-50 announcement marker' }
            } elseif ($announceMust) {
                if (-not $fields['announcement_present']) { $fail += 'no default announcement' }
                elseif (-not $fields['announcement_matches_expectation']) { $fail += 'announcement missing the T-50 marker' }
            }
            if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
            if ($fields['route_parity_exit'] -eq 0) {
                if ("$($fields['route_parity'])" -ne 'clean') { $fail += ("route parity {0}" -f $fields['route_parity']) }
            } elseif ($fields['route_parity_exit'] -eq 2) {
                # no route events: fine UNLESS the arm was expected to carry them
            } else {
                $fail += ("route parity exit {0} (parity {1})" -f $fields['route_parity_exit'], $fields['route_parity'])
            }
            if ($fields.ContainsKey('decode_exit') -and $fields['decode_exit'] -ne 0) {
                $fail += ("decode exit {0} aggregate {1}" -f $fields['decode_exit'], $fields['decode_aggregate'])
            }
            if ($fields.ContainsKey('compare_identity_ok') -and -not $fields['compare_identity_ok']) {
                $fail += 'pair IDENTITY fields differ'
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
    Write-Host ("VERDICT {0} ({1}) resolved={2} source={3} -> {4}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit,
        $fields['resolved_pp_prebuild_per_pass'], $fields['pp_prebuild_per_pass_source'],
        (Join-Path $verdictDir ($Tag + '.done')))
}
exit $exit
