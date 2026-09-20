# T-GPLR-37 arm runner: one arm, one binary, one verdict + sentinel.
#
# The measured object is the RESOLVED --copy-source default, which is now
# 'mmap-registered' (owner decision 2026-09-20: register every package source
# span ONCE at init and DMA straight from the pinned file pages). Each arm
# records, from the trace and the log itself:
#   * the resolved copy_source and its SOURCE (explicit | default)  [run_start]
#   * the stderr announcement (must appear IFF the default was used)
#   * the register structure for any mmap-resolved run: 40 init-once
#     mmap_register events + one mmap_register_init (~17.23 GiB), ALL before
#     the first layer_compute (the addendum-2 mechanism signature)
#   * strict checker, tg ids, graded decode (4k), route parity
#
# Shapes are cloned VERBATIM from the recorded arms the ACs compare against:
#   4k <- gplr37b-{pin,ron}-s3 (anchor-4096, ubatch 512, tg 96, layer-major,
#        q8, 3 slots, af 1, recon 1, repeat 2)
#   b1 <- gplr43-fix-B1-q8 (anchor-100000, pp 143360, ctx 144128, ubatch 1024,
#        tg 0, mmap-registered, q8, repeat 2) -- the T-47 clean q8 reference
#
# Binaries: 'changed' = bin\Release (the T-GPLR-37 tree),
#           'control' = bin\gplr37-control (pristine nested HEAD, rebuilt after
#           the control-preserving lesson: the live Release build is NEVER a
#           valid control once the task's code has compiled into it).
param(
    [Parameter(Mandatory = $true)][string] $Tag,
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed',
    [ValidateSet('4k', 'b1')]           [string] $Shape = '4k',
    # 'default' leaves --copy-source UNSET (the arm that measures the new
    # default); 'pinned' / 'mmap-registered' pin it (the explicit-identity arms).
    [ValidateSet('default', 'pinned', 'mmap-registered')] [string] $Src = 'default',
    # Optional: an existing verdict tag whose behavior this arm must reproduce
    # (the explicit-pinned pair compares control vs changed).
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
$exe = Join-Path $base (("bin\{0}\llama-layer-streaming.exe") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr37-control' }))
$dll = Join-Path $base (("bin\{0}\llama.dll") -f $(if ($Bin -eq 'changed') { 'Release' } else { 'gplr37-control' }))
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
                '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
                '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
                '--layer-slots', '3', '--allow-full-residency',
                '--residency-budget-mib', '0', '--pp-repeat', '2',
                '--pp-async-feed', '1', '--pp-recon', '1')
    }
    'b1' {
        $anchor = Join-Path $base 'gplr07-qa-prompts\anchor-100000.txt'
        $a += @('--prompt-file', $anchor, '--pp-tokens', '143360', '--ctx-size', '144128',
                '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
                '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
                '--pp-schedule', 'layer-major', '--layer-slots', '3',
                '--allow-full-residency', '--pp-repeat', '2',
                '--pp-async-feed', '1')
    }
}
if (-not (Test-Path -LiteralPath $anchor)) { Write-Host "MISSING anchor: $anchor"; exit 1 }

# The explicit flag is appended LAST and only when the arm pins it, so a
# 'default' arm's command line provably does not carry it.
if ($Src -ne 'default') { $a += @('--copy-source', $Src) }

# The resolved source this arm should end up with.
$expectSrc  = if ($Src -eq 'default') { 'mmap-registered' } else { $Src }
$wantProv   = if ($Src -eq 'default') { 'default' } else { 'explicit' }
# Whether the mmap-registered mechanism must actually run (drives the register
# structure asserts): any arm whose RESOLVED source is mmap-registered.
$mmapActive = ($expectSrc -eq 'mmap-registered')

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("[gplr37] {0}  bin={1} exe {2} dll {3}" -f $Tag, $Bin, $exeSha, $dllSha)

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
$a += @('--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$shapeTokens = switch ($Shape) { '4k' { 4096 } default { 143360 } }
$fields = @{ shape = $Shape; prompt_tokens = $shapeTokens
             copy_source_requested = $Src; bin = $Bin
             expect_resolved_copy_source = $expectSrc
             mmap_active = $mmapActive
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
        Get-Content $log | Select-String -Pattern 'PP:|PP pass 2/2|copy-source|pinned in|phase pp|peak VRAM|refus|error' |
            Select-Object -Last 8 | ForEach-Object { Write-Host "  log: $_" }

        if ($armExit -ne 0) {
            $fields['fail_reason'] = ("arm exited {0}" -f $armExit)
        } else {
            # ---- (a) resolved source + provenance, from the trace's run_start
            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })
            if ($rs.Count -eq 0) {
                $fields['fail_reason'] = 'no run_start event'
            } else {
                $j = $rs[0] | ConvertFrom-Json
                $fields['run_start_copy_source'] = $j.copy_source
                $fields['run_start_has_source_key'] = [bool]($j.PSObject.Properties.Name -contains 'copy_source_source')
                if ($fields['run_start_has_source_key']) {
                    $fields['copy_source_source'] = $j.copy_source_source
                }
                Write-Host ("  run_start: copy_source={0} source={1}" -f $j.copy_source, $fields['copy_source_source'])

                # ---- (b) the register structure for any mmap-resolved run
                if ($mmapActive) {
                    $regs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"mmap_register"*' } |
                              ForEach-Object { $_ | ConvertFrom-Json })
                    $inits = @($regs | Where-Object { $_.mode -eq 'init-once' })
                    $fields['mmap_register_events'] = $regs.Count
                    $fields['mmap_register_init_once'] = $inits.Count
                    $initMeta = @(Get-Content $trace | Where-Object { $_ -like '*"event":"mmap_register_init"*' } |
                                  ForEach-Object { $_ | ConvertFrom-Json })
                    if ($initMeta.Count -eq 1) {
                        $fields['mmap_init_bytes'] = $initMeta[0].bytes
                        $fields['mmap_init_us'] = $initMeta[0].us
                        $fields['mmap_init_layers'] = $initMeta[0].layers
                        Write-Host ("  registers: {0} init-once + init {1:N0} B in {2} us ({3} layers)" -f `
                            $inits.Count, $initMeta[0].bytes, $initMeta[0].us, $initMeta[0].layers)
                    } else {
                        $fields['mmap_init_events'] = $initMeta.Count
                    }
                    # ordering: every init-once registration BEFORE the first layer_compute
                    $lc = @(Get-Content $trace | Where-Object { $_ -like '*"event":"layer_compute"*' } |
                            ForEach-Object { $_ | ConvertFrom-Json })
                    if ($lc.Count -gt 0 -and $inits.Count -gt 0) {
                        $firstLC = [long]$lc[0].t_us
                        $lastReg = [long](($inits | ForEach-Object { [long]$_.t_us } | Measure-Object -Maximum).Maximum)
                        $fields['all_registers_before_first_compute'] = ($lastReg -le $firstLC)
                        Write-Host ("  last register t={0} vs first layer_compute t={1} -> before={2}" -f `
                            $lastReg, $firstLC, $fields['all_registers_before_first_compute'])
                    }
                }
            }

            # ---- (c) the announcement: present IFF the source came from the default
            $ann = @(Get-Content $log | Select-String -Pattern 'preflight: copy-source = ')
            $fields['announcement_present'] = [bool]($ann.Count -gt 0)
            if ($ann.Count -gt 0) { $fields['announcement_line'] = ([string]$ann[0]).Trim() }
            Write-Host ("  announcement present={0}" -f $fields['announcement_present'])

            # ---- (d) strict checker
            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

            # ---- (e) pass ledgers (4k only: the recorded b1 shape carries no --pp-recon)
            if ($Shape -eq '4k') {
                $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                             ForEach-Object { $_ | ConvertFrom-Json })
                $fields['pass_count'] = $ledgers.Count
                $compute = @()
                for ($i = 0; $i -lt $ledgers.Count; $i++) {
                    $l = $ledgers[$i]
                    $compute += [double]$l.compute_us
                    $fields[("pass{0}_compute_us" -f ($i + 1))] = [math]::Round($l.compute_us, 1)
                    $fields[("pass{0}_wall_ms"    -f ($i + 1))] = [math]::Round($l.sweep_us / 1000.0, 1)
                    $fields[("pass{0}_launches"   -f ($i + 1))] = $l.launches
                    Write-Host ("  pass {0}: sweep {1:N0} ms  compute {2:N0} ms  launches {3}" -f `
                        ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0), $l.launches)
                }
            }

            # ---- (f) tg ids + graded decode (b1 has tg 0 by design)
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
                        $text = [string]$g.text
                        $fields['answer_head'] = $text.Substring(0, [Math]::Min(90, $text.Length))
                    }
                }
            } else {
                Write-Host '  no tg_tokens event (tg-tokens 0 by design for this shape)'
            }

            # ---- (g) route parity: clean everywhere post-T-47 (all these shapes)
            $parityJson = Join-Path $base "$Tag.parity.json"
            & python $parity $trace --json $parityJson --quiet 2>&1 | ForEach-Object { Write-Host "  parity: $_" }
            $fields['route_parity_exit'] = $LASTEXITCODE
            if (Test-Path -LiteralPath $parityJson) {
                $p = (Get-Content $parityJson -Raw | ConvertFrom-Json).reports[0]
                $fields['route_parity'] = $p.parity
                $fields['route_events'] = $p.route_events
                $fields['route_singletons'] = $p.singletons
            }

            # ---- (h) optional pair comparison: identical BEHAVIOR expected.
            if ($CompareTag) {
                $cv = Read-ArmVerdict -Dir $verdictDir -Tag $CompareTag
                if ($null -eq $cv) {
                    $fields['fail_reason'] = ("compare tag {0} has no verdict" -f $CompareTag)
                } else {
                    $fields['compare_tag'] = $CompareTag
                    $idFields = @('tg_ids', 'pass_count', 'pass1_launches', 'pass2_launches',
                                  'route_parity', 'route_events', 'mmap_register_events',
                                  'mmap_register_init_once')
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
                # The control predates the task: assert the ABSENCE of the new
                # provenance key and the announcement (the INV1 evidence that
                # the change is confined to the default-resolution path).
                if ($fields['run_start_has_source_key']) { $fail += 'CONTROL carried the new source key' }
                if ($fields['announcement_present'])     { $fail += 'CONTROL emitted the announcement' }
            } else {
                if ($fields['run_start_copy_source'] -ne $expectSrc) {
                    $fail += ("resolved copy_source '{0}' != expected '{1}' (requested '{2}')" -f `
                        $fields['run_start_copy_source'], $expectSrc, $Src)
                }
                if ($fields['copy_source_source'] -ne $wantProv) {
                    $fail += ("provenance '{0}' != expected '{1}'" -f $fields['copy_source_source'], $wantProv)
                }
                if ($Src -eq 'default') {
                    if (-not $fields['announcement_present']) { $fail += 'no default announcement' }
                } elseif ($fields['announcement_present']) {
                    $fail += 'announcement emitted for an EXPLICIT flag'
                }
                if ($mmapActive) {
                    if ($fields['mmap_register_init_once'] -ne 40) { $fail += ("init-once registers {0} != 40" -f $fields['mmap_register_init_once']) }
                    if ($fields.ContainsKey('mmap_init_bytes') -and [long]$fields['mmap_init_bytes'] -lt 17000000000) {
                        $fail += ("init pinned bytes {0} below the recorded ~17.23 GiB" -f $fields['mmap_init_bytes'])
                    }
                    if ($fields.ContainsKey('all_registers_before_first_compute') -and -not $fields['all_registers_before_first_compute']) {
                        $fail += 'a registration landed INSIDE the pass (churn signature)'
                    }
                }
            }
            if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
            if ($fields.ContainsKey('route_parity_exit') -and $fields['route_parity_exit'] -ne 0) {
                $fail += ("route parity exit {0} ({1})" -f $fields['route_parity_exit'], $fields['route_parity'])
            }
            if ($fields.ContainsKey('decode_exit') -and $fields['decode_exit'] -ne 0) {
                $fail += ("decode exit {0} aggregate {1}" -f $fields['decode_exit'], $fields['decode_aggregate'])
            }
            if ($fields.ContainsKey('compare_identity_ok') -and -not $fields['compare_identity_ok']) {
                $fail += 'pair IDENTITY fields differ'
            }
            if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
        }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}

if (-not $failedEarly) {
    Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode $exit -StartedUtc $started `
        -LogPath $log -Fields $fields | Out-Null
    Write-Host ("VERDICT {0} ({1}) copy_source={2} source={3} parity={4} -> {5}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit,
        $fields['run_start_copy_source'], $fields['copy_source_source'],
        $fields['route_parity'], (Join-Path $verdictDir ($Tag + '.done')))
}
exit $exit
