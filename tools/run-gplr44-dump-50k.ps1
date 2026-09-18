# T-GPLR-44 STEP 1: the patch-cache composition dump, at 50k.
#
# WHY 50k AND NOT 200k: the task's recorded shape is 200k, but 200k costs ~10 min
# per arm and the whole point of step 1 is to iterate on the instrument and then
# on the patch. 50k keeps every flag EXCEPT the token count identical to the
# recorded 200k ledger arm (tools/run-gplr36-hostmem.ps1), so:
#   - the composition dump is shape-independent in structure (same 40 layer
#     graphs, same shared input cache) and is therefore valid at 50k; and
#   - patch_us is NOT shape-independent (it is per (layer, chunk) launch, so it
#     scales with token count: 3,402.3 ms/pass at 200k => ~850 ms/pass at 50k if
#     the per-launch cost is constant). The 50k arm therefore establishes the
#     50k baseline, and the 200k arm confirms the same share once the patch
#     works -- it does not inherit the 200k baseline.
#
# PRE-REGISTERED BEFORE THE RUN (from call structure only -- this is what the
# dump exists to test):
#   * prepare_chunk runs once per (layer, chunk) launch, but both of its caches
#     are per-SET: fa_nodes spans every layer graph, input_tensors spans the whole
#     set build context.
#   * Therefore per-launch input bytes should be ~n_layer x what the launched
#     graph needs: redundancy_input inside the 8-40x band.
#   * FALSIFIER: closure broken (some cached tensor referenced by no graph), or
#     redundancy_input below the 8x floor. Either one means the lever is not
#     there to take, and the task closes on the dump instead of patching.
#
# Exit 0 means STEP 1 IS VALID -- the dump fired and its closure holds. It does
# NOT mean the lever is worth taking: the redundancy number decides that, and it
# is recorded in the verdict fields either way.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe  = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-50000.txt'

$verdictDir = Join-Path $base 'verdicts'
$verdictTag = 'gplr44-dump-50k'
$trace = Join-Path $base 'gplr44-dump-50k.jsonl'
$log   = Join-Path $base 'gplr44-dump-50k.log'
$analyzer = Join-Path $root 'tools\analyze-gplr44-cache.py'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }
if (-not (Test-Path $analyzer)) { Write-Host "MISSING ANALYZER: $analyzer"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')
Clear-ArmVerdict -Dir $verdictDir -Tag $verdictTag
$verdictStart = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $verdictTag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0} -> {1}" -f $verdictTag, (Join-Path $verdictDir ($verdictTag + '.started')))

# An uncaught error must still leave a verdict; otherwise the absence of a result
# is indistinguishable from a run in progress.
trap {
    $msg = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
    try { Add-Content -LiteralPath $log -Value "[FATAL] $msg" } catch { }
    Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -Status 'ERROR' `
        -StartedUtc $verdictStart -LogPath $log -ErrorText $msg | Out-Null
    Write-Host ("VERDICT ERROR (1) -> {0}" -f (Join-Path $verdictDir ($verdictTag + '.done')))
    exit 1
}

Remove-Item $trace, $log -ErrorAction SilentlyContinue

$fields = @{
    shape_tokens = 50000
    ubatch = 512
    layer_slots = 3
    kv_cache = 'f16'
    copy_source = 'pinned'
    pp_repeat = 2
    step = 'step1-composition-dump'
    trace = $trace
    log = $log
}

# Same mutex name and wait/release behavior as tools/with-gpu-lock.ps1: this arm
# writes its own verdict (richer than the wrapper's), so it holds the lock itself
# rather than nesting inside the wrapper.
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
try {
    Write-Host 'Waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-Host 'MUTEX TIMEOUT'
        $fields['timeout_seconds'] = 3600
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 124 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText 'timed out waiting for the GPU mutex' -Fields $fields | Out-Null
        Write-Host ("VERDICT TIMEOUT (124) -> {0}" -f (Join-Path $verdictDir ($verdictTag + '.done')))
        exit 124
    }
    Write-Host 'Acquired GPU mutex'

    Write-Host '[gplr44] GPU state at arm start:'
    & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader |
        ForEach-Object { Write-Host "  $_" }
    & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader |
        ForEach-Object { Write-Host "  $_(other project, resident)" }

    # Byte-identical to tools/run-gplr36-hostmem.ps1 except --pp-tokens (50k) and
    # the anchor, so this arm's ledger and its later A/B sit next to a recorded
    # reference arm rather than beside a shape nobody has run.
    $a = @('--model', $model, '--prompt-file', $prompt,
           '--pp-tokens', '50000', '--ubatch', '512', '--pp-no-logits',
           '--tg-tokens', '0',
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency',
           '--residency-budget-mib', '0',
           '--pp-repeat', '2', '--copy-source', 'pinned',
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', $trace)
    $fields['command'] = ($exe + ' ' + ($a -join ' '))

    Write-Host '[gplr44] 50k composition-dump arm starting...'
    $t0 = (Get-Date)
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $fields['arm_exit'] = $armExit
    Write-Host ("ARM EXIT: {0}" -f $armExit)

    if ($armExit -ne 0) {
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host $_ }
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText ("arm exited $armExit") -Fields $fields | Out-Null
        Write-Host ("VERDICT FAIL (1) -> {0}" -f (Join-Path $verdictDir ($verdictTag + '.done')))
        exit 1
    }

    Write-Host '[gplr44] strict check-telemetry...'
    & $exe --check-telemetry $trace
    $checkExit = $LASTEXITCODE
    $fields['check_telemetry_exit'] = $checkExit
    Write-Host ("CHECK EXIT: {0}" -f $checkExit)

    Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|tokens/s|refus|error' |
        Select-Object -Last 10 | ForEach-Object { Write-Host $_ }

    # Ledger baselines: pass 1 carries the one-time dump walk inside its patch
    # bucket, so the baseline is the LAST pass, not the first.
    Write-Host '[gplr44] host ledger (patch baseline):'
    $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                 ForEach-Object { $_ | ConvertFrom-Json })
    if ($ledgers.Count -gt 0) {
        $fields['ledger_passes'] = $ledgers.Count
        $fields['patch_us_pass1'] = [math]::Round($ledgers[0].patch_us, 1)
        $fields['patch_us_baseline'] = [math]::Round($ledgers[-1].patch_us, 1)
        $fields['launches_baseline'] = $ledgers[-1].launches
        $fields['sweep_us_baseline'] = [math]::Round($ledgers[-1].sweep_us, 1)
        $fields['compute_us_baseline'] = [math]::Round($ledgers[-1].compute_us, 1)
        foreach ($l in $ledgers) {
            Write-Host ("  pass: sweep {0:N0} ms  compute {1:N0} ms  patch {2:N0} ms  launches {3}" -f `
                ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0), ($l.patch_us / 1000.0), $l.launches)
        }
    } else {
        Write-Host '  NO host_ledger EVENTS -- the run did not reach the PP sweep'
    }

    # Did the arm actually stream? A "full residency" arm would make the patch
    # baseline meaningless, so this is recorded rather than assumed from the flags.
    $copies = @(Get-Content $trace | Select-String -Pattern '"event":"copy_complete"').Count
    $rounds = @(Get-Content $trace | Select-String -Pattern '"event":"rowset_round"').Count
    $fields['copy_complete_events'] = $copies
    $fields['rowset_round_events'] = $rounds
    Write-Host ("[gplr44] staging: copy_complete={0} rowset_round={1}" -f $copies, $rounds)

    Write-Host '[gplr44] composition dump:'
    python $analyzer $trace
    $anaExit = $LASTEXITCODE
    $fields['analyzer_exit'] = $anaExit
    Write-Host ("ANALYZER EXIT: {0}" -f $anaExit)

    $dumps = @(Get-Content $trace | Where-Object { $_ -like '*"event":"patch_cache"*' } |
               ForEach-Object { $_ | ConvertFrom-Json })
    $fields['patch_cache_events'] = $dumps.Count
    if ($dumps.Count -gt 0) {
        $d = $dumps[-1]
        $fields['set_input_tensors'] = $d.set_input_tensors
        $fields['set_input_bytes'] = $d.set_input_bytes
        $fields['set_fa_nodes'] = $d.set_fa_nodes
        $fields['union_input_tensors'] = $d.union_input_tensors
        $fields['union_matches'] = [int]$d.union_matches
        $fields['redundancy_input'] = [math]::Round($d.redundancy_input, 3)
        $fields['redundancy_fa'] = [math]::Round($d.redundancy_fa, 3)
        $fields['layer_graphs'] = $d.layer_graphs
        $fields['layer_input_bytes_mean'] = [math]::Round($d.layer_input_bytes_mean, 1)
        $fields['n_graphs'] = $d.n_graphs
    }

    # STEP 1 is valid when the dump fired with closure intact. Whether the lever
    # is worth taking is the redundancy field's job, not this exit code's.
    $ok = ($checkExit -eq 0) -and ($dumps.Count -gt 0) -and ($dumps[-1].union_matches -eq 1)
    $exit = if ($ok) { 0 } else { 1 }
    if (-not $ok) {
        $fields['fail_reason'] = if ($dumps.Count -eq 0) { 'no patch_cache event emitted' } `
            elseif ($dumps[-1].union_matches -ne 1) { 'closure broken: cached inputs referenced by no graph' } `
            else { "check-telemetry exit $checkExit" }
        Write-Host ("[gplr44] STEP 1 INVALID: {0}" -f $fields['fail_reason'])
    } else {
        Write-Host '[gplr44] STEP 1 VALID: dump fired, closure intact'
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host 'Released GPU mutex' }
    $mutex.Dispose()
}

Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $exit -StartedUtc $verdictStart `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($verdictTag + '.done')))
exit $exit
