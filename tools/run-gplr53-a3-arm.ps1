# T-GPLR-53 A3 arm driver: the decay discriminator at 50k and 200k, BOTH arms
# inside ONE acquisition of the shared-GPU mutex (the T-GPLR-52 house pattern).
#
# Contract (T-GPLR-53 block, A3): 200k and 50k, explicit --copy-source
# mmap-registered (never defaulted - the resolved source must be unambiguous),
# 3 slots, af=1, kv f16, ubatch 512, --pp-repeat 3, --pp-recon 1. Per arm the
# driver enforces, in this order:
#   1. strict `--check-telemetry <trace>`   (0 required)
#   2. `check-gplr53-embed-chunks.py <tag> --join`  (A2'rider: post-fix
#      generation join must report drift 0)
#   3. `analyze-gplr53-decay.py <trace>`    (the PRE-REGISTERED decision rule:
#      FRONT_LOADED / FLAT / CHUNK0_SPIKE / MIXED, exit 2 = nothing to evaluate
#      and is NEVER a pass)
# No performance claim is made by this driver: it records the pre-registered
# bands and the raw traces. Telemetry-only task; --pp-recon stays opt-in.
#
# Dry run (no GPU, no mutex, no CUDA): -ValidateOnly checks every input and
# prints the exact commands it would run.
param(
    [ValidateSet('50k', '200k', 'both')][string] $Shape = 'both',
    [switch] $ValidateOnly,
    # The provenance gate (2026-09-22) refuses this run when an arm's verdict
    # cannot attest which source tree and which binary it describes. The escape
    # is explicit and RECORDED (build/llama-cuda/verdicts/provenance-waivers.log);
    # it is never implied by a missing check.
    [switch] $AllowUnprovenanced
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$exe  = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$dll  = Join-Path $base 'bin\Release\llama.dll'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompts = Join-Path $base 'gplr07-qa-prompts'
$verdictDir = Join-Path $base 'verdicts'
$checker = Join-Path $root 'tools\check-gplr53-embed-chunks.py'
$decay   = Join-Path $root 'tools\analyze-gplr53-decay.py'
. (Join-Path $root 'tools\arm-verdict.ps1')

# One row per arm; the ONLY differences are the prompt, the token count and the ctx size.
$shapes = @(
    @{ key = '50k';  tag = 'gplr53-a3-50k';  prompt = 'anchor-50000.txt';  pp = 50000;  ctx = 50176  },
    @{ key = '200k'; tag = 'gplr53-a3-200k'; prompt = 'anchor-200000.txt'; pp = 200000; ctx = 200704 }
)
if ($Shape -ne 'both') { $shapes = @($shapes | Where-Object { $_.key -eq $Shape }) }

# ---------------------------------------------------------------- input checks
$missing = @()
foreach ($p in @($exe, $dll, $checker, $decay)) { if (-not (Test-Path -LiteralPath $p)) { $missing += $p } }
if (-not (Test-Path -LiteralPath $model)) { $missing += $model }
foreach ($s in $shapes) {
    $pp = Join-Path $prompts $s.prompt
    if (-not (Test-Path -LiteralPath $pp)) { $missing += $pp }
}
if ($missing.Count -gt 0) {
    Write-Host 'MISSING INPUTS:'
    $missing | ForEach-Object { Write-Host "  $_" }
    exit 1
}

function Get-ArmArgs {
    param($S)
    return @('--model', $model, '--prompt-file', (Join-Path $prompts $S.prompt),
             '--pp-tokens', "$($S.pp)", '--ctx-size', "$($S.ctx)",
             '--ubatch', '512', '--pp-no-logits', '--tg-tokens', '3',
             '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
             '--act-slab-dtype', 'bf16',
             '--pp-schedule', 'layer-major',
             '--layer-slots', '3', '--allow-full-residency', '--residency-budget-mib', '0',
             '--pp-repeat', '3',
             '--copy-source', 'mmap-registered',
             '--pp-async-feed', '1', '--pp-recon', '1',
             '--telemetry', (Join-Path $base ($S.tag + '.jsonl')))
}

if ($ValidateOnly) {
    $exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
    $dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
    Write-Host "VALIDATE OK  exe $exeSha  dll $dllSha"
    Write-Host "model  $model"
    foreach ($s in $shapes) {
        Write-Host ("ARM {0}  tag {1}" -f $s.key, $s.tag)
        Write-Host ("  " + $exe + ' ' + ((Get-ArmArgs $s) -join ' '))
        Write-Host ("  THEN {0} --check-telemetry" -f $exe)
        Write-Host ("  THEN python {0} {1} --join" -f $checker, $s.tag)
        Write-Host ("  THEN python {0} {1}" -f $decay, (Join-Path $base ($s.tag + '.jsonl')))
    }
    Write-Host "ONE mutex hold for $($shapes.Count) arm(s); GPU busy gate refuses (exit 125) on a non-clean pre-run probe."
    exit 0
}

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $dll).Hash.Substring(0, 16)
Write-Host ("gplr53-a3: exe {0} dll {1}" -f $exeSha, $dllSha)

# --------------------------------------------------- arm-verdict handling FIRST
foreach ($s in $shapes) { Clear-ArmVerdict -Dir $verdictDir -Tag $s.tag | Out-Null }
$sessionStart = (Get-Date).ToUniversalTime().ToString('o')
foreach ($s in $shapes) {
    Write-ArmStart -Dir $verdictDir -Tag $s.tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' `
        -LogPath (Join-Path $base ($s.tag + '.log')) -Command ($exe + ' ' + ((Get-ArmArgs $s) -join ' ')) | Out-Null
}

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$sessionExit = 1
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        foreach ($s in $shapes) {
            Write-ArmVerdict -Dir $verdictDir -Tag $s.tag -ExitCode 124 -StartedUtc $sessionStart `
                -ErrorText 'mutex timeout' | Out-Null
        }
        exit 124
    }
    Write-Host '  mutex held'

    # ONE busy probe for the whole session: every arm ran under the same device state.
    $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_pre_run = {0}" -f $gpuPre)
    if ($gpuPre -ne 'clean') {
        if ($gpuPre -eq 'probe-unavailable') {
            Write-Warning 'GPU busy probe unavailable; proceeding UNVERIFIED.'
        } else {
            foreach ($s in $shapes) {
                Write-ArmVerdict -Dir $verdictDir -Tag $s.tag -ExitCode 125 -StartedUtc $sessionStart `
                    -ErrorText "GPU busy: $gpuPre" -Fields @{ gpu_pre_run = $gpuPre } | Out-Null
            }
            Write-Host ("REFUSED: GPU busy ({0})" -f $gpuPre)
            exit 125
        }
    }

    $sessionExit = 0
    foreach ($s in $shapes) {
        $tag = $s.tag
        $trace = Join-Path $base ($tag + '.jsonl')
        $log   = Join-Path $base ($tag + '.log')
        Remove-Item $trace, $log -ErrorAction SilentlyContinue
        $fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha; arm = $tag
                     shape = $s.key; pp_tokens = $s.pp; trace = $trace; log = $log
                     gpu_pre_run = $gpuPre; pp_repeat = 3; copy_source_requested = 'mmap-registered' }
        $fail = @()
        Write-Host ("== ARM {0} ==" -f $tag)

        $t0 = Get-Date
        & $exe @(Get-ArmArgs $s) > $log 2>&1
        $armExit = $LASTEXITCODE
        $fields['arm_exit'] = $armExit
        $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])
        if ($armExit -ne 0) { $fail += "arm exited $armExit" }

        if ($armExit -eq 0) {
            # Per-arm post-run probe, persisted (parity with the gplr51/gplr52 drivers).
            $gpuPostArm = Format-GpuBusy (Get-GpuComputeApps)
            $fields['gpu_post_run'] = $gpuPostArm
            if ($gpuPostArm -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - timing suspect." -f $gpuPostArm) }

            $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $fields['resolved_copy_source'] = $rs.copy_source
            if ($rs.copy_source -ne 'mmap-registered') { $fail += "copy_source $($rs.copy_source), want mmap-registered" }

            & $exe --check-telemetry $trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

            # A2 rider: the post-fix (layer, ubatch)-keyed generation join must be drift 0.
            $jout = (& python $checker $tag --join 2>&1 | Out-String)
            $fields['check_embed_chunks_exit'] = $LASTEXITCODE
            $fields['check_embed_chunks_tail'] = ($jout.Trim() -split "`n" | Select-Object -Last 1)
            if ($LASTEXITCODE -ne 0) { $fail += "check-gplr53-embed-chunks exit $LASTEXITCODE" }

            # The pre-registered decay discriminator (exit 2 = nothing to evaluate, NOT a pass).
            $dout = (& python $decay $trace 2>&1 | Out-String)
            $fields['decay_exit'] = $LASTEXITCODE
            $fields['decay_line'] = ($dout.Trim() -split "`n" | Select-Object -Last 1)
            if ($LASTEXITCODE -eq 2) { $fail += 'decay: NOT EVALUABLE (no per-chunk excess)' }
            elseif ($LASTEXITCODE -ne 0) { $fail += "decay exit $LASTEXITCODE" }
            if ($dout -match '(FRONT_LOADED|FLAT|CHUNK0_SPIKE|MIXED|NO_EXCESS)') {
                $fields['decay_verdict'] = $Matches[1]
            }
            $m = [regex]::Match($dout, '"rho_spearman":\s*(-?[0-9.]+)')
            if ($m.Success) { $fields['decay_rho'] = [double]$m.Groups[1].Value }
            $m = [regex]::Match($dout, '"first_decile_share":\s*([0-9.]+)')
            if ($m.Success) { $fields['decay_first_decile_share'] = [double]$m.Groups[1].Value }
            $m = [regex]::Match($dout, '"excess_total_ms":\s*(-?[0-9.]+)')
            if ($m.Success) { $fields['embed_excess_total_ms'] = [double]$m.Groups[1].Value }

            $i = 0
            foreach ($p in @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } | ForEach-Object { $_ | ConvertFrom-Json })) {
                $i++
                $fields[("pass{0}_wall_s" -f $i)] = [math]::Round($p.wall_us / 1e6, 2)
                $fields[("pass{0}_tok_s" -f $i)] = [math]::Round($p.tokens_per_s, 1)
            }
            $fields['pp_pass_count'] = $i
            Write-Host ("  copy_source={0} | decay={1} rho={2} first-decile={3} | excess {4} ms over {5} passes" -f `
                $fields['resolved_copy_source'], $fields['decay_verdict'], $fields['decay_rho'], `
                $fields['decay_first_decile_share'], $fields['embed_excess_total_ms'], $i)
        }

        $armExitCode = if ($fail.Count -gt 0) { 1 } else { 0 }
        if ($armExitCode -ne 0) { $fields['fail_reason'] = ($fail -join '; '); $sessionExit = 1 }
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $armExitCode -StartedUtc $sessionStart `
            -LogPath $log -Fields $fields | Out-Null
    }

    $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
    Write-Host ("  gpu_post_run = {0}" -f $gpuPost)
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
# --- provenance gate ---------------------------------------------------------
# Every arm above must attest the tree and the binary it ran. A refusal here does
# not re-score any arm: it says the evidence cannot be attributed to a revision,
# which is exactly the defect that let a 16:07-built smoke stand in for a tree
# edited at 16:31. Exit code 10 is distinct from an arm FAIL (1) and from a
# mutex/busy refusal (124/125) so a reader never confuses the three.
if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags @($shapes | ForEach-Object { $_.tag }) `
        -Label 'A3 arm' -AllowUnprovenanced:$AllowUnprovenanced)) {
    if ($sessionExit -eq 0) { $sessionExit = 10 }
}
Write-Host ("gplr53-a3 VERDICT {0}" -f $(if ($sessionExit -eq 0) { 'PASS' } else { 'FAIL' }))
exit $sessionExit
