# T-GPLR-39 arm runner: one arm, one binary, one verdict + sentinel.
#
# The measured object is the RESOLVED --route-telemetry default, which is now
# 0 (owner decision 2026-09-20: the route event is a DIAGNOSTIC whose device
# readback + host sort cost 1161.5 ms/pass at u2048 - 100.8% of the `other`
# ledger bucket - so it runs only when explicitly needed). Each arm records,
# from the trace and the log itself:
#   * the resolved route_telemetry value and its SOURCE (explicit | default)
#     [run_start]
#   * the stderr announcement (must appear IFF the default decided)
#   * route-event presence: a defaulted (telemetry-off) arm must carry ZERO
#     route events (the parity tool's exit-2 "no route events - NOT a pass"
#     is the DESIGNED outcome there, not a failure); an explicit-on arm must
#     carry them and pass parity clean
#   * strict checker, tg ids, graded decode (4k)
#
# Shape is cloned VERBATIM from the recorded gplr37b arms (anchor-4096,
# ubatch 512, tg 96, layer-major, q8, 3 slots, af 1, recon 1, repeat 2) so the
# tg-id identity comparison against the T-GPLR-37 verdicts is like-for-like.
#
# Binaries: 'changed' = bin\Release (the T-GPLR-39 tree),
#           'control' = bin\gplr37-changed (the T-GPLR-37 RESULT binary, which
#           predates the T-39 flip and must show no new key / no announcement).
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed',
    # 'default' leaves --route-telemetry UNSET (the arm that measures the new
    # default); 'on' pins --route-telemetry 1 (the explicit-identity arm).
    [ValidateSet('default', 'on')] [string] $Mode = 'default',
    # Optional: an existing verdict tag whose tg ids / pass structure this arm
    # must reproduce (the identity pair).
    [string] $CompareTag = ''
)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$verdictDir = Join-Path $base 'verdicts'
$parity = Join-Path $root 'tools\check-route-parity.py'
$decode = Join-Path $root 'tools\decode-tg-ids.py'

$isControl = ($Bin -eq 'control')
$exe = Join-Path $base ((("bin\{0}\llama-layer-streaming.exe") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr37-changed' })))
$dll = Join-Path $base ((("bin\{0}\llama.dll") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr37-changed' })))
if (-not (Test-Path -LiteralPath $exe)) { Write-Host "MISSING exe: $exe"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')

# ---------------------------------------------------------------- shape flags
$anchor = Join-Path $base 'gplr07-qa-prompts\anchor-4096.txt'
if (-not (Test-Path -LiteralPath $anchor)) { Write-Host "MISSING anchor: $anchor"; exit 1 }
$a = @('--model', $model,
       '--prompt-file', $anchor, '--pp-tokens', '4096', '--ubatch', '512',
       '--pp-no-logits', '--tg-tokens', '96',
       '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
       '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0', '--pp-repeat', '2',
       '--pp-async-feed', '1', '--pp-recon', '1')

# The explicit flag is appended LAST and only when the arm pins it, so a
# 'default' arm's command line provably does not carry it.
if ($Mode -eq 'on') { $a += @('--route-telemetry', '1') }

$wantVal  = if ($Mode -eq 'default') { 0 } else { 1 }
$wantProv = if ($Mode -eq 'default') { 'default' } else { 'explicit' }
$routeWanted = ($Mode -eq 'on')

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("[gplr39] {0}  bin={1} mode={2} exe {3} dll {4}" -f $Tag, $Bin, $Mode, $exeSha, $dllSha)

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
$a += @('--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$fields = @{ shape = '4k'; mode = $Mode; bin = $Bin
             expect_route_telemetry = $wantVal; expect_source = $wantProv
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
        Get-Content $log | Select-String -Pattern 'PP:|PP pass 2/2|route-telemetry|phase pp|peak VRAM|refus|error' |
            Select-Object -Last 8 | ForEach-Object { Write-Host "  log: $_" }

        if ($armExit -ne 0) {
            $fields['fail_reason'] = ("arm exited {0}" -f $armExit)
        } else {
            # ---- (a) resolved value + provenance, from the trace's run_start
            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })
            if ($rs.Count -eq 0) {
                $fields['fail_reason'] = 'no run_start event'
            } else {
                $j = $rs[0] | ConvertFrom-Json
                $fields['run_start_has_rt_key'] = [bool]($j.PSObject.Properties.Name -contains 'route_telemetry')
                if ($fields['run_start_has_rt_key']) {
                    $fields['run_start_route_telemetry'] = $j.route_telemetry
                    $fields['route_telemetry_source'] = $j.route_telemetry_source
                }
                Write-Host ("  run_start: route_telemetry={0} source={1}" -f $fields['run_start_route_telemetry'], $fields['route_telemetry_source'])

                # ---- (b) route-event presence, from the trace itself
                $fields['route_events_in_trace'] = @(Get-Content $trace |
                    Where-Object { $_ -like '*"event":"route"*' }).Count
                Write-Host ("  route events in trace: {0}" -f $fields['route_events_in_trace'])

                # ---- (c) the announcement: present IFF the default decided
                $ann = @(Get-Content $log | Select-String -Pattern 'preflight: route-telemetry = ')
                $fields['announcement_present'] = [bool]($ann.Count -gt 0)
                if ($ann.Count -gt 0) { $fields['announcement_line'] = ([string]$ann[0]).Trim() }
                Write-Host ("  announcement present={0}" -f $fields['announcement_present'])

                # ---- (d) strict checker
                & $exe --check-telemetry $trace
                $fields['check_telemetry_exit'] = $LASTEXITCODE
                Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

                # ---- (e) pass ledgers
                $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                             ForEach-Object { $_ | ConvertFrom-Json })
                $fields['pass_count'] = $ledgers.Count
                for ($i = 0; $i -lt $ledgers.Count; $i++) {
                    $l = $ledgers[$i]
                    $fields[("pass{0}_compute_us" -f ($i + 1))] = [math]::Round($l.compute_us, 1)
                    $fields[("pass{0}_wall_ms"    -f ($i + 1))] = [math]::Round($l.sweep_us / 1000.0, 1)
                    Write-Host ("  pass {0}: sweep {1:N0} ms  compute {2:N0} ms" -f `
                        ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0))
                }

                # ---- (f) tg ids + graded decode
                $idsLine = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
                if ($idsLine.Count -gt 0) {
                    $ids = @(($idsLine[-1] | ConvertFrom-Json).ids)
                    $fields['tg_id_count'] = $ids.Count
                    $fields['tg_ids'] = ($ids -join ',')
                    Write-Host ("  {0} tg ids, first {1}" -f $ids.Count, $ids[0])
                    $grade = Join-Path $base "$Tag.grade.json"
                    & python $decode --trace $trace --model $model --json $grade --quiet 2>&1 |
                        ForEach-Object { Write-Host "  decode: $_" }
                    $fields['decode_exit'] = $LASTEXITCODE
                    if (Test-Path -LiteralPath $grade) {
                        $g = Get-Content $grade -Raw | ConvertFrom-Json
                        $fields['decode_aggregate'] = $g.verdict.aggregate
                        $text = [string]$g.text
                        $fields['answer_head'] = $text.Substring(0, [Math]::Min(90, $text.Length))
                    }
                } else {
                    Write-Host '  no tg_tokens event'
                }

                # ---- (g) route parity: meaningful ONLY for an explicit-on arm
                # (a defaulted arm has no route events by design; the tool's
                # exit 2 there is the DESIGNED "nothing to evaluate", not a pass
                # or a failure - recorded, never asserted).
                if ($routeWanted) {
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

                # ---- (h) optional pair comparison: identical BEHAVIOR expected.
                if ($CompareTag) {
                    $cv = Read-ArmVerdict -Dir $verdictDir -Tag $CompareTag
                    if ($null -eq $cv) {
                        $fields['fail_reason'] = ("compare tag {0} has no verdict" -f $CompareTag)
                    } else {
                        $fields['compare_tag'] = $CompareTag
                        $idFields = @('tg_ids', 'pass_count')
                        foreach ($f in $idFields) {
                            if (($cv.fields.PSObject.Properties.Name -contains $f) -and $fields.ContainsKey($f)) {
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
                if ($isControl) {
                    # The control predates the task: assert the ABSENCE of the
                    # new provenance keys and the announcement (INV1).
                    if ($fields['run_start_has_rt_key']) { $fail += 'CONTROL carried the route_telemetry key' }
                    if ($fields['announcement_present']) { $fail += 'CONTROL emitted the announcement' }
                } else {
                    if (-not $fields['run_start_has_rt_key']) { $fail += 'no route_telemetry key in run_start' }
                    if ("{0}" -f $fields['run_start_route_telemetry'] -ne "{0}" -f $wantVal) {
                        $fail += ("resolved route_telemetry '{0}' != expected '{1}'" -f $fields['run_start_route_telemetry'], $wantVal)
                    }
                    if ($fields['route_telemetry_source'] -ne $wantProv) {
                        $fail += ("provenance '{0}' != expected '{1}'" -f $fields['route_telemetry_source'], $wantProv)
                    }
                    if ($Mode -eq 'default') {
                        if (-not $fields['announcement_present']) { $fail += 'no default announcement' }
                        if ($fields['route_events_in_trace'] -ne 0) { $fail += ("route events present with telemetry OFF: {0}" -f $fields['route_events_in_trace']) }
                    } else {
                        if ($fields['announcement_present']) { $fail += 'announcement emitted for an EXPLICIT flag' }
                        if ($fields['route_events_in_trace'] -eq 0) { $fail += 'no route events with telemetry ON' }
                        if ($fields.ContainsKey('route_parity_exit') -and $fields['route_parity_exit'] -ne 0) {
                            $fail += ("route parity exit {0} ({1})" -f $fields['route_parity_exit'], $fields['route_parity'])
                        }
                    }
                }
                if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
                if ($fields.ContainsKey('decode_exit') -and $fields['decode_exit'] -ne 0) {
                    $fail += ("decode exit {0} aggregate {1}" -f $fields['decode_exit'], $fields['decode_aggregate'])
                }
                if ($fields.ContainsKey('compare_identity_ok') -and -not $fields['compare_identity_ok']) {
                    $fail += 'pair IDENTITY fields differ'
                }
                if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
            }
        }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}

if (-not $failedEarly) {
    Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode $exit -StartedUtc $started `
        -LogPath $log -Fields $fields | Out-Null
    Write-Host ("VERDICT {0} ({1}) route_telemetry={2} source={3} route_events={4} -> {5}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit,
        $fields['run_start_route_telemetry'], $fields['route_telemetry_source'],
        $fields['route_events_in_trace'], (Join-Path $verdictDir ($Tag + '.done')))
}
exit $exit
