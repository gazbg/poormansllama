# T-GPLR-44 steps 2-3: the paired 50k arm that carries BOTH the identity checks and
# the patch_us comparison. Run once per build with -Label unscoped (the control,
# from the pre-patch binary) and once with -Label scoped (the patched binary).
#
# Flags are identical across labels except the label itself; --tg-tokens 96 is the
# one addition over the composition-dump arm, because identity here means
# "generated token ids byte-identical", and an arm with --tg-tokens 0 generates
# nothing to compare.
#
# WHY A SEPARATE CONTROL ARM AND NOT A FLAG: the task's contract rules out using
# "the same build with the scoping gated off" as the patch_us control. The control
# is therefore the REAL pre-patch binary, run in the SAME session at the SAME
# shape, which is also the only honest way to compare the wall given the recorded
# ~2.9% session-to-session spread.
#
# The composition dump stays ON in both arms (identical flags), so the dump's
# one-time walk is charged to pass 1's patch bucket in BOTH arms and the pass-2
# baseline stays comparable.
# -Exe is a parameter because the CONTROL ARM IS A DIFFERENT BINARY: the streaming
# code lives in llama.dll, so the pre-patch pair is side-copied into
# build/llama-cuda/bin/unscoped-bin/ and run from there. Same session, same flags,
# two real builds - which is what makes the paired comparison mean anything.
#
# -TgTokens is 3 rather than 96 at the 50k shape because layer-major REFUSES more
# than 3 TG tokens when handoff auto did not activate (measured: exit 2, "supports
# at most 3 --tg-tokens without an active handoff"). More ids at 50k would require
# forcing the handoff path, which changes the execution mode both arms run and
# would confound the comparison; the long-id identity check is therefore taken at
# the 4k shape, where 96 tokens are permitted.
param(
    [Parameter(Mandatory = $true)][ValidateSet('unscoped', 'scoped')][string] $Label,
    [string] $Exe = '',
    [ValidateSet(4096, 50000, 200000)][int] $PromptTokens = 50000,
    [int] $TgTokens = 3
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
if (-not $Exe) { $Exe = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe' }
$exe  = $Exe
$base = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base ("gplr07-qa-prompts/anchor-{0}.txt" -f $PromptTokens)

$verdictDir = Join-Path $base 'verdicts'
$shape = if ($PromptTokens -eq 4096) { '4k' } elseif ($PromptTokens -eq 200000) { '200k' } else { '50k' }
$verdictTag = "gplr44-ab-$shape-$Label"
$trace = Join-Path $base "gplr44-ab-$shape-$Label.jsonl"
$log   = Join-Path $base "gplr44-ab-$shape-$Label.log"


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
             step = 'step2-identity-and-patch-us'; trace = $trace; log = $log; exe = $exe }
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$fields['exe_sha256_16'] = $sha
Write-Host ("[gplr44] binary {0} sha256[:16]={1}" -f $exe, $sha)

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

    Write-Host ("[gplr44] arm {0} starting..." -f $Label)
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
        Write-Host ("[gplr44] generated {0} ids" -f $ids.Count)
    } else {
        Write-Host '[gplr44] NO tg_tokens EVENT -- identity cannot be checked from this trace'
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

    # Composition, so the scoped arm's numbers sit beside the dump that predicted them.
    $dumps = @(Get-Content $trace | Where-Object { $_ -like '*"event":"patch_cache"*' } |
               ForEach-Object { $_ | ConvertFrom-Json })
    $fields['patch_cache_events'] = $dumps.Count
    if ($dumps.Count -gt 0) {
        $fields['redundancy_input'] = [math]::Round($dumps[-1].redundancy_input, 3)
        $fields['union_matches'] = [int]$dumps[-1].union_matches
        $fields['set_input_bytes'] = $dumps[-1].set_input_bytes
        $fields['layer_input_bytes_mean'] = [math]::Round($dumps[-1].layer_input_bytes_mean, 1)
    }
    $ok = ($checkExit -eq 0) -and ($idsLine.Count -gt 0) -and ($ledgers.Count -ge 2)
    $exit = if ($ok) { 0 } else { 1 }
    if (-not $ok) { $fields['fail_reason'] = 'missing identity payload or ledger' }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host 'Released GPU mutex' }
    $mutex.Dispose()
}

Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $exit -StartedUtc $verdictStart `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($verdictTag + '.done')))
exit $exit
