# T-GPLR-45 single arm: identical flags across labels; only the -Exe differs
# (the control IS a different binary). Cloned from run-gplr44-ab-50k.ps1 with
# the tags renamed gplr45-*. The composition dump stays ON in both arms
# (--pp-recon 1), so the dump's one-time walk is charged to pass 1's patch
# bucket in BOTH arms and the pass-2 comparison stays like-for-like.
param(
    [Parameter(Mandatory = $true)][ValidateSet('control', 'changed')][string] $Label,
    [Parameter(Mandatory = $true)][string] $Exe,
    [ValidateSet(4096, 50000, 200000)][int] $PromptTokens = 50000,
    [int] $TgTokens = 3
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base ("gplr07-qa-prompts/anchor-{0}.txt" -f $PromptTokens)

$verdictDir = Join-Path $base 'verdicts'
$shape = if ($PromptTokens -eq 4096) { '4k' } elseif ($PromptTokens -eq 200000) { '200k' } else { '50k' }
$verdictTag = "gplr45-$shape-$Label"
$trace = Join-Path $base "gplr45-$shape-$Label.jsonl"
$log   = Join-Path $base "gplr45-$shape-$Label.log"

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
             step = 'gplr45-filter-identity-and-patch-us'; trace = $trace; log = $log; exe = $exe }
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$fields['exe_sha256_16'] = $sha
Write-Host ("[gplr45] binary {0} sha256[:16]={1}" -f $exe, $sha)

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
    $fields['command'] = ($exe + ' ' + ($a -join ' '))

    Write-Host ("[gplr45] arm {0} starting..." -f $Label)
    $t0 = (Get-Date)
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $fields['arm_exit'] = $armExit
    Write-Host ("ARM EXIT: {0} ({1}s)" -f $armExit, $fields['duration_s'])

    if ($armExit -ne 0) {
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host $_ }
        Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText ("arm exited $armExit") -Fields $fields | Out-Null
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
        Write-Host ("[gplr45] generated {0} ids" -f $ids.Count)
    } else {
        Write-Host '[gplr45] NO tg_tokens EVENT -- identity cannot be checked from this trace'
    }

    # --- the performance payload: the pass-2 ledger ---
    $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                 ForEach-Object { $_ | ConvertFrom-Json })
    if ($ledgers.Count -gt 0) {
        $l = $ledgers[-1]
        $fields['patch_us_baseline'] = [math]::Round($l.patch_us, 1)
        $fields['sweep_us_baseline'] = [math]::Round($l.sweep_us, 1)
        $fields['compute_us_baseline'] = [math]::Round($l.compute_us, 1)
        $fields['launches_baseline'] = $l.launches
        $fields['patch_us_per_launch'] = [math]::Round($l.patch_us / [double] $l.launches, 4)
        $fields['patch_share_pct'] = [math]::Round(100.0 * $l.patch_us / $l.sweep_us, 3)
        foreach ($x in $ledgers) {
            Write-Host ("  pass: sweep {0:N0} ms  compute {1:N0} ms  patch {2:N0} ms ({3:N3}%)  launches {4}" -f `
                ($x.sweep_us / 1000.0), ($x.compute_us / 1000.0), ($x.patch_us / 1000.0),
                (100.0 * $x.patch_us / $x.sweep_us), $x.launches)
        }
    }

    # Composition + the T-GPLR-45 filter audit fields, so the excluded set sits
    # beside the dump that names it.
    $dumps = @(Get-Content $trace | Where-Object { $_ -like '*"event":"patch_cache"*' } |
               ForEach-Object { $_ | ConvertFrom-Json })
    $fields['patch_cache_events'] = $dumps.Count
    if ($dumps.Count -gt 0) {
        $d = $dumps[-1]
        $fields['redundancy_input'] = [math]::Round($d.redundancy_input, 3)
        $fields['union_matches'] = [int]$d.union_matches
        $fields['set_input_bytes'] = $d.set_input_bytes
        $fields['layer_input_bytes_mean'] = [math]::Round($d.layer_input_bytes_mean, 1)
        $fields['excluded_count'] = [int]$d.excluded_count
        $fields['excluded_bytes'] = [long]$d.excluded_bytes
        $fields['walked_bytes'] = [long]$d.walked_bytes
        $fields['excluded_members'] = (@($d.excluded_members) | ForEach-Object { "$($_.name):$($_.bytes)" }) -join ','
        Write-Host ("  filter: kept {0} B + excluded {1} B = walked {2} B  (excluded: {3})" -f `
            $d.set_input_bytes, $d.excluded_bytes, $d.walked_bytes, $fields['excluded_members'])
    }
    $ok = ($checkExit -eq 0) -and ($idsLine.Count -gt 0) -and ($ledgers.Count -ge 2) -and ($dumps.Count -gt 0)
    $exit = if ($ok) { 0 } else { 1 }
    if (-not $ok) { $fields['fail_reason'] = 'missing identity payload, ledger, or dump' }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host 'Released GPU mutex' }
    $mutex.Dispose()
}

Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $exit -StartedUtc $verdictStart `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($verdictTag + '.done')))
