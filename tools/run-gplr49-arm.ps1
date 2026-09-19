# T-GPLR-49 single arm. Cloned from run-gplr45-arm.ps1; TWO changes:
#
#  1. -Mode pw|nopw. 'pw' is the unchanged default (identity regression pair);
#     'nopw' adds --no-pw-probe, the ONLY CLI combination that reaches the
#     pre-fix slab offset formula (the defect T-GPLR-49 repairs).
#  2. -ExpectRefusal. The pre-fix control is PREDICTED to refuse loudly at a
#     specific chunk (4 of 8 at 4k, 49 of 98 at 50k) because the doubled offset
#     runs past the slab half. For those arms PASS means "exited nonzero AND
#     printed the slab-exhausted refusal", and the observed refusal chunk index
#     is recorded so it can be compared against the prediction. If the control
#     does NOT refuse, the code model is wrong and the task says stop and
#     re-read -- so that arm fails loudly here instead of being waved through.
param(
    [Parameter(Mandatory = $true)][ValidateSet('control', 'changed')][string] $Label,
    [Parameter(Mandatory = $true)][string] $Exe,
    [ValidateSet(4096, 50000)][int] $PromptTokens = 50000,
    [int] $TgTokens = 3,
    [ValidateSet('pw', 'nopw')][string] $Mode = 'pw',
    [switch] $ExpectRefusal
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base ("gplr07-qa-prompts/anchor-{0}.txt" -f $PromptTokens)

$verdictDir = Join-Path $base 'verdicts'
$shape = if ($PromptTokens -eq 4096) { '4k' } else { '50k' }
$verdictTag = "gplr49-$shape-$Mode-$Label"
$trace = Join-Path $base "gplr49-$shape-$Mode-$Label.jsonl"
$log   = Join-Path $base "gplr49-$shape-$Mode-$Label.log"

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')
Clear-ArmVerdict -Dir $verdictDir -Tag $verdictTag
$verdictStart = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $verdictTag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0} -> {1}" -f $verdictTag, (Join-Path $verdictDir ($verdictTag + '.started')))

trap {
    $msg = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
    try { Add-Content -LiteralPath $log -Value "[FATAL] $msg" } catch { }
    Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -Status 'ERROR' `
        -StartedUtc $verdictStart -LogPath $log -ErrorText $msg | Out-Null
    exit 1
}

Remove-Item $trace, $log -ErrorAction SilentlyContinue
$fields = @{ label = $Label; shape_tokens = $PromptTokens; ubatch = 512; layer_slots = 3
             kv_cache = 'f16'; copy_source = 'pinned'; pp_repeat = 2; tg_tokens = $TgTokens
             mode = $Mode; expect_refusal = [bool]$ExpectRefusal
             step = 'gplr49-slab-offset-repair'; trace = $trace; log = $log; exe = $exe }
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path (Split-Path $exe -Parent) 'llama.dll')).Hash.Substring(0, 16)
$fields['exe_sha256_16'] = $sha
$fields['llama_dll_sha256_16'] = $dllHash
Write-Host ("[gplr49] binary {0} sha256[:16]={1} (llama.dll {2})" -f $exe, $sha, $dllHash)

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
try {
    Write-Host 'Waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 124 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText 'timed out waiting for the GPU mutex' -Fields $fields | Out-Null
        exit 124
    }
    Write-Host 'Acquired GPU mutex'
    & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  GPU: $_" }

    $a = @('--model', $model, '--prompt-file', $prompt,
           '--pp-tokens', "$PromptTokens", '--ubatch', '512', '--pp-no-logits',
           '--tg-tokens', "$TgTokens",
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency',
           '--residency-budget-mib', '0',
           '--pp-repeat', '2', '--copy-source', 'pinned',
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', $trace)
    if ($Mode -eq 'nopw') { $a += '--no-pw-probe' }
    $fields['command'] = ($exe + ' ' + ($a -join ' '))

    Write-Host ("[gplr49] arm {0} starting..." -f $Label)
    $t0 = (Get-Date)
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $fields['arm_exit'] = $armExit
    Write-Host ("ARM EXIT: {0} ({1}s)" -f $armExit, $fields['duration_s'])

    # --- the refusal path (pre-fix control, --no-pw-probe) -------------------
    if ($ExpectRefusal) {
        $logText = Get-Content $log -Raw
        $exhausted = ($logText -match 'activation slab [AB] exhausted')
        $fields['refusal_seen'] = $exhausted
        $begun = [regex]::Matches($logText, 'failed to begin graph set .*?chunk (-?\d+)')
        if ($begun.Count -gt 0) {
            $fields['refused_chunk'] = [int]$begun[0].Groups[1].Value
            Write-Host ("[gplr49] refusal names chunk {0}" -f $fields['refused_chunk'])
        }
        $exline = [regex]::Match($logText, 'activation slab [AB] exhausted \([^)]*\)')
        if ($exline.Success) {
            $fields['refusal_message'] = $exline.Value
            Write-Host ("[gplr49] {0}" -f $exline.Value)
        }
        if ($armExit -eq 0) {
            $fields['fail_reason'] = 'expected a loud refusal; the arm COMPLETED (prediction wrong)'
            Write-Host '[gplr49] WARNING: the control did NOT refuse -- the predicted symptom is WRONG'
            $exit = 1
        } elseif (-not $exhausted) {
            $fields['fail_reason'] = 'exited nonzero WITHOUT the slab-exhausted refusal'
            Get-Content $log -Tail 20 | ForEach-Object { Write-Host $_ }
            $exit = 1
        } else {
            $exit = 0
        }
        if ($exit -eq 0) {
            Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 0 -StartedUtc $verdictStart `
                -LogPath $log -Fields $fields | Out-Null
            Write-Host ("VERDICT {0} (0) -> {1}" -f 'REFUSED-AS-PREDICTED', (Join-Path $verdictDir ($verdictTag + '.done')))
            exit 0
        }
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $exit -StartedUtc $verdictStart `
            -LogPath $log -Fields $fields | Out-Null
        Write-Host ("VERDICT FAIL ({0}) -> {1}" -f $exit, (Join-Path $verdictDir ($verdictTag + '.done')))
        exit $exit
    }

    if ($armExit -ne 0) {
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host $_ }
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText ("arm exited $armExit") -Fields $fields | Out-Null
        exit 1
    }

    # the repaired binary must NOT produce the refusal in --no-pw-probe mode
    $logText = Get-Content $log -Raw
    $fields['refusal_seen'] = ($logText -match 'activation slab [AB] exhausted')
    if ($fields['refusal_seen']) {
        $fields['fail_reason'] = 'the arm completed but printed a slab-exhausted refusal'
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText $fields['fail_reason'] -Fields $fields | Out-Null
        exit 1
    }

    & $exe --check-telemetry $trace
    $checkExit = $LASTEXITCODE
    $fields['check_telemetry_exit'] = $checkExit
    Write-Host ("CHECK EXIT: {0}" -f $checkExit)

    # --- the identity payload: generated ids, verbatim ---
    $idsLine = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
    if ($idsLine.Count -gt 0) {
        $tg = $idsLine[-1] | ConvertFrom-Json
        $ids = @($tg.ids)
        $fields['tg_id_count'] = $ids.Count
        $fields['tg_ids'] = ($ids -join ',')
        Write-Host ("[gplr49] generated {0} ids" -f $ids.Count)
    } else {
        Write-Host '[gplr49] NO tg_tokens EVENT -- identity cannot be checked from this trace'
    }

    # --- the act_bind stream: the pw regression's other half ----------------
    $binds = @(Get-Content $trace | Where-Object { $_ -like '*"event":"act_bind"*' })
    $fields['act_bind_events'] = $binds.Count
    $fields['act_bind_sha256_16'] = if ($binds.Count -gt 0) {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        $bytes = [System.Text.Encoding]::UTF8.GetBytes(($binds -join "`n"))
        ([System.BitConverter]::ToString($sha256.ComputeHash($bytes)) -replace '-', '').Substring(0, 16)
    } else { '' }
    Write-Host ("[gplr49] act_bind events {0} sha256[:16]={1}" -f $fields['act_bind_events'], $fields['act_bind_sha256_16'])

    # --- the pass ledger ---
    $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                 ForEach-Object { $_ | ConvertFrom-Json })
    if ($ledgers.Count -gt 0) {
        $l = $ledgers[-1]
        $fields['patch_us_baseline'] = [math]::Round($l.patch_us, 1)
        $fields['sweep_us_baseline'] = [math]::Round($l.sweep_us, 1)
        $fields['compute_us_baseline'] = [math]::Round($l.compute_us, 1)
        $fields['launches_baseline'] = $l.launches
        $fields['patch_share_pct'] = [math]::Round(100.0 * $l.patch_us / $l.sweep_us, 3)
        foreach ($x in $ledgers) {
            Write-Host ("  pass: sweep {0:N0} ms  compute {1:N0} ms  patch {2:N0} ms ({3:N3}%)  launches {4}" -f `
                ($x.sweep_us / 1000.0), ($x.compute_us / 1000.0), ($x.patch_us / 1000.0),
                (100.0 * $x.patch_us / $x.sweep_us), $x.launches)
        }
    }

    # --- composition dump (T-GPLR-45 fields sit beside it) -------------------
    $dumps = @(Get-Content $trace | Where-Object { $_ -like '*"event":"patch_cache"*' } |
               ForEach-Object { $_ | ConvertFrom-Json })
    # The dump is a PW-ONLY artefact and its absence in --no-pw-probe mode is a
    # property of the code, not missing evidence: prepare_chunk() early-returns
    # when pw_probe is off, so there is no per-launch patch, hence no patch
    # cache to describe (each chunk owns a set already bound to its own slice).
    $dumpRequired = ($Mode -eq 'pw')
    $fields['dump_expected'] = $dumpRequired
    $fields['patch_cache_events'] = $dumps.Count
    if ($dumps.Count -gt 0) {
        $d = $dumps[-1]
        $fields['redundancy_input'] = [math]::Round($d.redundancy_input, 3)
        $fields['union_matches'] = [int]$d.union_matches
        $fields['set_input_bytes'] = $d.set_input_bytes
        $fields['excluded_count'] = [int]$d.excluded_count
        $fields['excluded_bytes'] = [long]$d.excluded_bytes
        $fields['walked_bytes'] = [long]$d.walked_bytes
        Write-Host ("  filter: kept {0} B + excluded {1} B = walked {2} B" -f `
            $d.set_input_bytes, $d.excluded_bytes, $d.walked_bytes)
    }

    $ok = ($checkExit -eq 0) -and ($idsLine.Count -gt 0) -and ($ledgers.Count -ge 2) -and ($binds.Count -gt 0) `
          -and ((-not $dumpRequired) -or ($dumps.Count -gt 0))
    $exit = if ($ok) { 0 } else { 1 }
    if (-not $ok) { $fields['fail_reason'] = 'missing identity payload, act_bind stream, ledger, or (pw mode) dump' }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host 'Released GPU mutex' }
    $mutex.Dispose()
}

Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $exit -StartedUtc $verdictStart `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($verdictTag + '.done')))
