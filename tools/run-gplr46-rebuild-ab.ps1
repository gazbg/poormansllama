# T-GPLR-46: isolate the pass-2 compute regression. Each arm clones the recorded
# gplr44 50k command VERBATIM (recovered from the verdict's fields.command) and
# changes ONLY what the arm tests:
#   - 50k-on    : default (rebuild ON)   -> same-session reference for the gate
#   - 50k-off   : --pp-prebuild-per-pass 0  -> the trigger test (gate: <= 0.5 s)
#   - 100k-on   : default at 100k         -> growth-law point C
#   - 100k-off  : rebuild off at 100k     -> growth-law control E
#   - 50k-3pass : default, --pp-repeat 3  -> saturation test D
# Same binary for every arm (build/llama-cuda/bin/Release), same session, so the
# pass-2 excess (pass2.compute_us - pass1.compute_us, SAME RUN) is the response
# variable and no cross-session spread can touch it.
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('50k-on', '50k-off', '100k-on', '100k-off', '50k-3pass', '4k-off', '4k-1pass')]
    [string] $Arm
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe  = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base = Join-Path $root 'build\llama-cuda'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'

$spec = @{
    '50k-on'    = @{ tokens = 50000;  repeat = 2; prebuild = $true  ; tg = 3  }
    '50k-off'   = @{ tokens = 50000;  repeat = 2; prebuild = $false; tg = 3  }
    '100k-on'   = @{ tokens = 100000; repeat = 2; prebuild = $true  ; tg = 3  }
    '100k-off'  = @{ tokens = 100000; repeat = 2; prebuild = $false; tg = 3  }
    '50k-3pass' = @{ tokens = 50000;  repeat = 3; prebuild = $true  ; tg = 3  }
    # 4k with 96 ids: the long-id identity probe. The gplr44 pair validated these
    # 96 ids byte-for-byte across binaries (rebuild ON); this arm checks them
    # against the SAME baseline with the rebuild OFF.
    '4k-off'    = @{ tokens = 4096;   repeat = 2; prebuild = $false; tg = 96 }
    # Single pass, 96 ids: no memory_clear and no rebuild ever fire, so this is
    # the pass-1 stream itself. Against the gplr44 96-id baseline (which is a
    # pass-2-after-rebuild stream) it says whether "clean pass 2" == pass 1.
    '4k-1pass'  = @{ tokens = 4096;   repeat = 1; prebuild = $true  ; tg = 96 }
}[$Arm]
$tag    = "gplr46-$Arm"
$prompt = Join-Path $base ("gplr07-qa-prompts/anchor-{0}.txt" -f $spec.tokens)
$trace  = Join-Path $base "$tag.jsonl"
$log    = Join-Path $base "$tag.log"
$verdictDir = Join-Path $base 'verdicts'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }

. (Join-Path $root 'tools\arm-verdict.ps1')
Clear-ArmVerdict -Dir $verdictDir -Tag $tag
$verdictStart = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0} -> {1}" -f $tag, (Join-Path $verdictDir ($tag + '.started')))

trap {
    $msg = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
    try { Add-Content -LiteralPath $log -Value "[FATAL] $msg" } catch { }
    Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 1 -Status 'ERROR' `
        -StartedUtc $verdictStart -LogPath $log -ErrorText $msg | Out-Null
    exit 1
}

Remove-Item $trace, $log -ErrorAction SilentlyContinue
$fields = @{ arm = $Arm; pp_tokens = $spec.tokens; pp_repeat = $spec.repeat
             pp_prebuild_per_pass = $(if ($spec.prebuild) { 1 } else { 0 })
             ubatch = 512; layer_slots = 3; kv_cache = 'f16'; copy_source = 'pinned'
             tg_tokens = $spec.tg; step = 't-gplr-46-rebuild-ab'; trace = $trace; log = $log; exe = $exe }
$sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$fields['exe_sha256_16'] = $sha
Write-Host ("[gplr46] binary {0} sha256[:16]={1}" -f $exe, $sha)

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
try {
    Write-Host 'Waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 124 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText 'timed out waiting for the GPU mutex' -Fields $fields | Out-Null
        exit 124
    }
    Write-Host 'Acquired GPU mutex'
    & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  GPU: $_" }

    $a = @('--model', $model, '--prompt-file', $prompt,
           '--pp-tokens', "$($spec.tokens)", '--ubatch', '512', '--pp-no-logits',
           '--tg-tokens', "$($spec.tg)",
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency',
           '--residency-budget-mib', '0',
           '--pp-repeat', "$($spec.repeat)", '--copy-source', 'pinned',
           '--pp-async-feed', '1', '--pp-recon', '1')
    if (-not $spec.prebuild) { $a += @('--pp-prebuild-per-pass', '0') }
    $a += @('--telemetry', $trace)
    $fields['command'] = ($exe + ' ' + ($a -join ' '))

    Write-Host ("[gplr46] arm {0} starting..." -f $Arm)
    $t0 = (Get-Date)
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    $fields['arm_exit'] = $armExit
    Write-Host ("ARM EXIT: {0} ({1}s)" -f $armExit, $fields['duration_s'])

    if ($armExit -ne 0) {
        Get-Content $log -Tail 30 | ForEach-Object { Write-Host $_ }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 1 -StartedUtc $verdictStart `
            -LogPath $log -ErrorText ("arm exited $armExit") -Fields $fields | Out-Null
        exit 1
    }

    & $exe --check-telemetry $trace
    $checkExit = $LASTEXITCODE
    $fields['check_telemetry_exit'] = $checkExit
    Write-Host ("CHECK EXIT: {0}" -f $checkExit)

    # --- identity payload: generated ids, verbatim ---
    $idsLine = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
    if ($idsLine.Count -gt 0) {
        $tg = $idsLine[-1] | ConvertFrom-Json
        $ids = @($tg.ids)
        $fields['tg_id_count'] = $ids.Count
        $fields['tg_ids'] = ($ids -join ',')
        Write-Host ("[gplr46] generated {0} ids" -f $ids.Count)
    } else {
        Write-Host '[gplr46] NO tg_tokens EVENT -- identity cannot be checked from this trace'
    }

    # --- the response variable: per-pass ledgers, then excesses ---
    $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                 ForEach-Object { $_ | ConvertFrom-Json })
    $fields['pass_count'] = $ledgers.Count
    $compute = @()
    for ($i = 0; $i -lt $ledgers.Count; $i++) {
        $l = $ledgers[$i]
        $compute += [double]$l.compute_us
        Write-Host ("  pass {0}: sweep {1:N0} ms  compute {2:N0} ms  patch {3:N0} ms  launches {4}" -f `
            ($i + 1), ($l.sweep_us / 1000.0), ($l.compute_us / 1000.0), ($l.patch_us / 1000.0), $l.launches)
        $fields[("pass{0}_compute_us" -f ($i + 1))] = [math]::Round($l.compute_us, 1)
        $fields[("pass{0}_sweep_us"   -f ($i + 1))] = [math]::Round($l.sweep_us, 1)
        $fields[("pass{0}_patch_us"   -f ($i + 1))] = [math]::Round($l.patch_us, 1)
        $fields[("pass{0}_launches"   -f ($i + 1))] = $l.launches
    }
    if ($ledgers.Count -ge 2) {
        $fields['excess_2v1_s'] = [math]::Round(($compute[1] - $compute[0]) / 1e6, 2)
    }
    if ($ledgers.Count -ge 3) {
        $fields['excess_3v2_s'] = [math]::Round(($compute[2] - $compute[1]) / 1e6, 2)
        # D's gate shape: pass3-pass2 >= pass2-pass1 means the excess GROWS again.
        $fields['saturation'] = $(if (($compute[2] - $compute[1]) -ge ($compute[1] - $compute[0])) { 'GROWING' } else { 'SATURATED' })
    }

    $ok = ($checkExit -eq 0) -and ($ledgers.Count -ge $spec.repeat) -and ($idsLine.Count -gt 0)
    $exit = if ($ok) { 0 } else { 1 }
    if (-not $ok) { $fields['fail_reason'] = 'missing identity payload or ledger' }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host 'Released GPU mutex' }
    $mutex.Dispose()
}

Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $exit -StartedUtc $verdictStart `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($tag + '.done')))
exit $exit
