# T-GPLR-49b: attribute the 4k/96 "pw echoes, non-pw answers" difference to the
# SCHEDULE, to the per-pass REBUILD, or to their interaction.
#
# The 2x2 already has three measured cells, but on TWO different binaries:
#
#   schedule  rebuild  4k/96 stream      evidence
#   pw        ON       ECHO             gplr44-ab-4k-*, gplr45-4k-*, gplr49-4k-pw-* (ids identical)
#   pw        OFF      ANSWER           gplr46-4k-off      (binary 9BA0AB55, PRE-T-GPLR-45)
#   nopw      ON       ANSWER           gplr49-4k-nopw-changed (current binary)
#   nopw      OFF      ??               --
#
# T-GPLR-45 changed the very restore path the rebuild interacts with, so the
# pw+OFF cell cannot be carried across it on trust: this script re-takes it on
# the current binary, and takes the missing nopw+OFF cell beside it. Both arms
# therefore run with --pp-prebuild-per-pass 0 on build/llama-cuda/bin/gplr49-changed,
# same 4k anchor / ubatch 512 / tg 96 as the arms they are compared against.
#
# PRE-REGISTERED predictions (falsifiable, written before the run):
#   A pw+OFF   -> needle PASS, first id 248046 (the answer stream)
#   B nopw+OFF -> needle PASS, first id 248046
# Falsifiers:
#   A echoes      -> the echo is NOT rebuild-triggered on the current binary; the
#                    pw per-chunk patch path (causal window / KV bound rewriting)
#                    becomes the suspect, and T-GPLR-47's target moves.
#   B echoes      -> rebuild OFF is itself harmful to the non-pw path (a NEW
#                    defect), which would invert the T-GPLR-48 default logic.
#
# Known stream signatures, both measured today on this binary:
#   ECHO   starts 760, 13017, 1433, ...   (document continuation, needle FAIL)
#   ANSWER starts 248046, 198, 248045, ... (needle PASS 5/5)
param(
    [ValidateSet(4096)] [int] $PromptTokens = 4096,
    [int] $TgTokens = 96
)
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base  = Join-Path $root 'build\llama-cuda'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt= Join-Path $base ("gplr07-qa-prompts/anchor-{0}.txt" -f $PromptTokens)
$exe   = Join-Path $base 'bin\gplr49-changed\llama-layer-streaming.exe'
$dll   = Join-Path $base 'bin\gplr49-changed\llama.dll'
$decode= Join-Path $root 'tools\decode-tg-ids.py'
$verdictDir = Join-Path $base 'verdicts'

foreach ($p in @($prompt, $exe, $dll, $decode)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "MISSING: $p"; exit 1 }
}
# The two arms this one is compared against (same binary, same session window).
$refEcho   = Join-Path $base 'gplr49-4k-pw-changed.jsonl'
$refAnswer = Join-Path $base 'gplr49-4k-nopw-changed.jsonl'

. (Join-Path $root 'tools\arm-verdict.ps1')

function Get-Ids([string] $trace) {
    $line = @(Get-Content $trace | Where-Object { $_ -like '*"event":"tg_tokens"*' })
    if ($line.Count -eq 0) { return $null }
    return @(($line[-1] | ConvertFrom-Json).ids)
}

function Get-StreamClass($ids) {
    if ($null -eq $ids -or $ids.Count -eq 0) { return 'none' }
    switch ($ids[0]) {
        '760'    { return 'echo'   }
        '248046' { return 'answer' }
        default  { return ('other-first-id-{0}' -f $ids[0]) }
    }
}

$arms = @(
    @{ Tag = 'gplr49b-4k-pw-rebuild0';   Mode = 'pw';   Predict = 'answer' },
    @{ Tag = 'gplr49b-4k-nopw-rebuild0'; Mode = 'nopw'; Predict = 'answer' }
)

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0,16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0,16)
Write-Host ("[gplr49b] binary exe {0} llama.dll {1}" -f $exeSha, $dllSha)

foreach ($arm in $arms) {
    $tag    = $arm.Tag
    $trace  = Join-Path $base "$tag.jsonl"
    $log    = Join-Path $base "$tag.log"
    $grade  = Join-Path $base "$tag.grade.json"
    Clear-ArmVerdict -Dir $verdictDir -Tag $tag | Out-Null
    $started = (Get-Date).ToUniversalTime().ToString('o')
    Write-ArmStart -Dir $verdictDir -Tag $tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
    Write-Host ("START {0}" -f $tag)

    $fields = @{ shape_tokens = $PromptTokens; ubatch = 512; tg_tokens = $TgTokens
                 mode = $arm.Mode; pp_prebuild_per_pass = 0; pp_repeat = 2
                 kv_cache = 'f16'; act_slab_dtype = 'bf16'; layer_slots = 3
                 predicted_stream = $arm.Predict
                 exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
                 trace = $trace; log = $log
                 ref_echo_trace = $refEcho; ref_answer_trace = $refAnswer }

    Remove-Item $trace, $log, $grade -ErrorAction SilentlyContinue
    $mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
    $acquired = $false
    $exit = 1
    try {
        Write-Host '  waiting for the GPU mutex'
        try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
        if (-not $acquired) {
            Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 124 -StartedUtc $started `
                -LogPath $log -ErrorText 'timed out waiting for the GPU mutex' -Fields $fields | Out-Null
            continue
        }
        Write-Host '  mutex held'
        $a = @('--model', $model, '--prompt-file', $prompt,
               '--pp-tokens', "$PromptTokens", '--ubatch', '512', '--pp-no-logits',
               '--tg-tokens', "$TgTokens",
               '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
               '--act-slab-dtype', 'bf16', '--pp-schedule', 'layer-major',
               '--layer-slots', '3', '--allow-full-residency',
               '--residency-budget-mib', '0', '--pp-repeat', '2',
               '--copy-source', 'pinned', '--pp-async-feed', '1',
               '--pp-prebuild-per-pass', '0',
               '--telemetry', $trace)
        if ($arm.Mode -eq 'nopw') { $a += '--no-pw-probe' }
        $fields['command'] = ($exe + ' ' + ($a -join ' '))

        $t0 = Get-Date
        & $exe @a > $log 2>&1
        $armExit = $LASTEXITCODE
        $fields['arm_exit'] = $armExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])
        if ($armExit -ne 0) {
            Get-Content $log -Tail 20 | ForEach-Object { Write-Host $_ }
            Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 1 -StartedUtc $started `
                -LogPath $log -ErrorText ("arm exited $armExit") -Fields $fields | Out-Null
            continue
        }

        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        Write-Host ("  CHECK EXIT {0}" -f $fields['check_telemetry_exit'])

        $ids = Get-Ids $trace
        if ($null -eq $ids) {
            $fields['fail_reason'] = 'no tg_tokens event'
            Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 1 -StartedUtc $started `
                -LogPath $log -ErrorText $fields['fail_reason'] -Fields $fields | Out-Null
            continue
        }
        $fields['tg_id_count'] = $ids.Count
        $fields['tg_ids'] = ($ids -join ',')
        $fields['first_token'] = $ids[0]
        $fields['stream_class'] = Get-StreamClass $ids
        Write-Host ("  {0} ids, first {1}, class {2}" -f $ids.Count, $ids[0], $fields['stream_class'])

        # Graded decode: the machine-readable answer verdict, not an eyeball.
        & python $decode --trace $trace --model $model --json $grade --quiet 2>&1 | ForEach-Object { Write-Host "  decode: $_" }
        $fields['decode_exit'] = $LASTEXITCODE
        if (Test-Path -LiteralPath $grade) {
            $g = Get-Content $grade -Raw | ConvertFrom-Json
            $fields['decode_aggregate'] = $g.aggregate
            if ($g.PSObject.Properties.Name -contains 'answer_text') {
                $fields['answer_head'] = $g.answer_text.Substring(0, [Math]::Min(70, $g.answer_text.Length))
            }
        }

        # Same-shape reference arms (same binary): the cells already measured.
        foreach ($ref in @(@{n='ref_echo'; p=$refEcho}, @{n='ref_answer'; p=$refAnswer})) {
            if (Test-Path -LiteralPath $ref.p) {
                $rids = Get-Ids $ref.p
                if ($null -ne $rids) {
                    $fields[($ref.n + '_class')] = Get-StreamClass $rids
                    $fields[($ref.n + '_ids_identical')] = (($rids -join ',') -eq ($ids -join ','))
                }
            }
        }

        $fields['prediction_held'] = ($fields['stream_class'] -eq $arm.Predict)
        $ok = ($fields['check_telemetry_exit'] -eq 0) -and ($fields['decode_exit'] -eq 0) -and $fields['prediction_held']
        if (-not $ok) {
            $fields['fail_reason'] = "prediction '$($arm.Predict)' vs measured '$($fields['stream_class'])' (check $($fields['check_telemetry_exit']), decode $($fields['decode_exit']))"
        }
        $exit = if ($ok) { 0 } else { 1 }
    } finally {
        if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
        $mutex.Dispose()
    }

    Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $exit -StartedUtc $started `
        -LogPath $log -Fields $fields | Out-Null
    Write-Host ("VERDICT {0} ({1}) stream={2} -> {3}" -f `
        $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, $fields['stream_class'],
        (Join-Path $verdictDir ($tag + '.done')))
}
Write-Host '[gplr49b] both arms finished'
