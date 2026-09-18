# T-GPLR-19: paired four-arm A/B for per-layer row residency at K=1.
#
# NOTE (2026-09-14): the MECHANISM's default depth is now K=4 (bare
# `--residency-tg-retain`; SDD scope amendment 2026-09-14). This driver's own
# default arm stays K=1 so that -AnalyzeOnly still regenerates the recorded K=1
# report in place; pass -RetainK to compare another depth.
#
# NOTE (2026-09-14, second): the bare-flag DEFAULT is now fitted to the device
# (largest depth that fits free VRAM less --reserve-mib and the slot window).
# This driver always passes an EXPLICIT depth, which is never reduced, so every
# gate reading here is anchored to the depth the arm declares and cannot be
# silently shallowed by a tight device. Do not switch an arm to the bare flag.
#
# VEHICLE (2026-09-18) -- WHICH PROMPT THE A/B PAIRS USE. `-Vehicle honest` is
# the DEFAULT and is the only vehicle acceptable as the SOLE evidence behind a
# depth decision. `-Vehicle echoed` reproduces the RECORDED rungs and is
# explicitly opt-in:
#   honest  prompt build/calc-prompt.txt (885 tokens, non-echoing code prompt);
#           baseline re-derived from the recorded 2026-09-14 re-probe
#           (build/llama-cuda/gplr15-nononchoing/); root gplr19-honest-k<K>-ab.
#           Its (b) prediction is PRE-REGISTERED: computed from that stream
#           before any honest rung ran.
#   echoed  prompt build/llama-cuda/gplr13-ladder/512/anchor-512.txt (the
#           SELF-ECHOING needle document); baseline from the recorded T-GPLR-15
#           probe; root gplr19-rowcache-ab (K=1) / gplr19-k<K>-ab. Reproduces
#           recorded evidence; its gains are the OPTIMISTIC pairing, because a
#           stream that echoes its own anchor inflates routing reuse
#           (SDD section 4; docs/currentstate.md 2026-09-14 and 2026-09-18).
# Why the default flipped: the K=2/4/6/8 ladder and the K=8 operating depth
# were chosen on echoed gains, and the honest re-run measured 22-24% MORE bytes
# staged per round on the same shapes. Under this driver an echoed run cannot
# be the only reading anyway: every run ALSO logs the other vehicle's reference
# at the same K, so both appear in one driver.log.
#
# Arms are A,B,A,B where A = the reactive row-set path (row-retention off, the
# recorded baseline) and B = `--residency-tg-retain 1`, the resolved
# row-capacity LRU cache. All four arms run back to back inside ONE acquisition
# of the shared-GPU mutex so the pairs see one GPU session; tools/gpu-mem-poll.ps1
# is attached per arm. The arms differ ONLY by the retention flag; every other
# flag is the T-GPLR-15 probe's batch-major shape, which is the only row-set
# vehicle with enough TG tokens to measure a staging change (the layer-major
# schedule caps handoff-free TG at 3 tokens and --residency-tg forces handoff
# off).
#
# Same mutex name and wait/release behavior as tools/with-gpu-lock.ps1; this
# driver acquires it once because the task's contract requires one session per
# A/B/A/B.
#
# Gate criteria evaluated at the end (T-GPLR-19 acceptance criteria 1, 2, 3, 6):
#   1  four arms exit 0 with --check-telemetry OK; greedy token IDs identical
#      across arms; the ordered layer|union_ids|hash route stream identical
#      across arms; the reactive arms byte-identical to the recorded baseline.
#   2  BYTES decide the mechanism: the K=1 arms' staged bytes per round fall to
#      395,835,932 B (69.12% of baseline) within 2%, and cross-token retention
#      is non-zero on every layer. (The contract calls this counter
#      `rows_resident_cross_token`; the shipped field name is
#      `n_rows_retained` -- the driver evaluates the shipped one and says so.)
#   3  performance, only if 2 passes, against the PRE-REGISTERED upper bounds
#      15.30 tok/s (8.419 GB/s conversion) and 13.61 tok/s (13.7 GB/s ceiling),
#      and against the direct arms' within-arm spread.
#   6  reported, not claimed: cache allocation bytes and peak VRAM.
#
# Frozen surfaces are asserted, not assumed: the recorded baseline constants
# come from the preserved T-GPLR-15 probe trace and are re-derived from it, not
# hard-coded blindly.
param(
    # Regenerate the metrics, criteria and report from the traces already on
    # disk, without touching the GPU. The A/B table must be re-derivable from
    # its own artifacts; this also keeps the recorded table honest if the
    # criteria wording changes.
    [switch] $AnalyzeOnly,

    # Retention depth for the B (treatment) arms. K=1 is the task's
    # pre-registered case; K=2/4/6/8 are the rungs the contract DEFERRED, run
    # through this same arm loop so the design, flags and criteria stay one
    # implementation. Each K writes to its own output root so no rung can
    # overwrite another's evidence.
    [int] $RetainK = 1,

    # Override the output root. Empty = derive it from the vehicle and $RetainK.
    [string] $OutRoot = '',

    # Which prompt the A/B pairs use. 'honest' (default) is the non-echoing
    # vehicle; 'echoed' reproduces the recorded rungs and is opt-in ONLY because
    # its gains are the optimistic pairing. See the VEHICLE note in the header.
    [ValidateSet('honest', 'echoed')]
    [string] $Vehicle = 'honest'
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$poller = Join-Path $root 'tools\gpu-mem-poll.ps1'
$an     = Join-Path $root 'tools\route_overlap\analyze.py'
$py     = 'python'
$moe    = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
$honestPrompt = Join-Path $root 'build\calc-prompt.txt'
$echoedPrompt = Join-Path $root 'build\llama-cuda\gplr13-ladder\512\anchor-512.txt'
$honestBase   = Join-Path $root 'build\llama-cuda\gplr15-nononchoing\probe-tg256-nononchoing.jsonl'
$echoedBase   = Join-Path $root 'build\llama-cuda\gplr15-phaseA-probe\probe-tg256.jsonl'
$honestAn     = Join-Path $root 'build\llama-cuda\gplr15-nononchoing\an\nononchoing.json'
$echoedAn     = Join-Path $root 'build\llama-cuda\gplr15-phaseA-probe\analysis-wide.json'

# The vehicle table: prompt, prefill length, recorded baseline trace and the
# offline (b) report all move together, so a rung can never mix one vehicle's
# prompt with another's prediction.
if ($Vehicle -eq 'honest') {
    $prompt    = $honestPrompt
    $ppTokens  = 885          # the prompt's own length, taken whole
    $base      = $honestBase
    $baseAn    = $honestAn
    $kTag      = "gplr19-honest-k$RetainK-ab"
} else {
    $prompt    = $echoedPrompt
    $ppTokens  = 512
    $base      = $echoedBase
    $baseAn    = $echoedAn
    $kTag      = if ($RetainK -eq 1) { 'gplr19-rowcache-ab' } else { "gplr19-k$RetainK-ab" }
}

foreach ($f in @($exe, $poller, $an, $prompt, $base, $baseAn)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing tool or vehicle artifact: $f" }
}
if (-not (Test-Path -LiteralPath $moe)) { throw "missing model: $moe" }
if ($RetainK -lt 1 -or $RetainK -gt 64) { throw "--residency-tg-retain accepts 1..64; got $RetainK" }
$rungRoot = if ($OutRoot) { $OutRoot } else { Join-Path $root "build\llama-cuda\$kTag" }
New-Item -ItemType Directory -Force -Path $rungRoot | Out-Null
$drvLog = Join-Path $rungRoot 'driver.log'
if (Test-Path $drvLog) { Remove-Item $drvLog -Force }

# --- verdict + sentinel (convention in tools/arm-verdict.ps1) ---------------
# This driver acquires the mutex itself instead of going through
# tools/with-gpu-lock.ps1, so it writes its own verdict. Checking for
# <rungRoot>/<tag>.done is the completion signal, which is what lets a launch go
# unchecked: no log tailing, no polling, and no "did it finish or die?" guessing.
# The sentinel is deleted first, so a PASS left by an earlier attempt at the same
# depth and vehicle cannot be read as this run's result.
$verdictDir    = $rungRoot
$verdictTag    = "rowcache-$Vehicle-k$RetainK"
$verdictStart  = (Get-Date).ToUniversalTime().ToString('o')
$verdictFields = @{ vehicle = $Vehicle; retain_k = $RetainK; rung_root = $rungRoot }
. (Join-Path $root 'tools\arm-verdict.ps1')
Clear-ArmVerdict -Dir $verdictDir -Tag $verdictTag

# A run that dies unexpectedly still leaves a verdict. Without this, an
# uncaught error (the 2026-09-18 cross-vehicle crash) produced no verdict line
# and no exit code worth reading, which is the worst case: the absence of a
# result is indistinguishable from a run still in progress.
trap {
    $msg = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
    # Deliberately does NOT use Log(): the trap must work even if the error
    # happened before any helper function in this script was defined.
    try { Add-Content -LiteralPath $drvLog -Value "[FATAL] $msg" } catch { }
    Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode 1 -Status 'ERROR' `
        -StartedUtc $verdictStart -LogPath $drvLog -ErrorText $msg -Fields $verdictFields | Out-Null
    Write-Host ("VERDICT ERROR (1) -> {0}" -f (Join-Path $verdictDir ($verdictTag + '.done')))
    exit 1
}

function Log([string]$msg) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg)
    Write-Host $line
    Add-Content -LiteralPath $drvLog -Value $line
}

# Route digest: SHA256 over the ordered `layer|union_ids|hash` stream. One
# definition, used for the arm traces AND for the recorded baseline trace, so
# "the A arms reproduce the recorded run" can be asserted on identity rather
# than only on byte counts.
function Get-RouteDigest([System.Collections.Generic.List[string]] $parts) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    return ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($parts -join ';')))) -replace '-', '').ToLower()
}

function Invoke-Capture {
    param([string[]] $Argv)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $all = & $Argv[0] @($Argv | Select-Object -Skip 1) 2>&1 }
    finally { $ErrorActionPreference = $old }
    $code = $LASTEXITCODE
    $lines = @($all | ForEach-Object { $_.ToString() })
    return [pscustomobject]@{ code = $code; out = $lines }
}

# --- recorded-baseline constants, re-derived from the preserved probe trace ---
$baseRaw = [System.IO.File]::ReadAllText($base)
$baseFetches = [regex]::Matches($baseRaw, '"event":"rowset_fetch"').Count
$baseStaged = [uint64]0
foreach ($fm in [regex]::Matches($baseRaw, '"event":"rowset_fetch"[^\n]*')) {
    $baseStaged += [uint64][regex]::Match($fm.Value, '"bytes_staged":(\d+)').Groups[1].Value
}
$baseRounds = [regex]::Matches($baseRaw, '"event":"rowset_round"').Count
$basePerRound = [math]::Round($baseStaged / $baseRounds, 0)
# The recorded run's IDENTITY, for a criterion-1 check stronger than bytes: the
# A arms must reproduce this digest and these greedy ids exactly.
$baseParts = New-Object System.Collections.Generic.List[string]
foreach ($fm in [regex]::Matches($baseRaw, '\{"event":"rowset_fetch"[^\n]*')) {
    $bline = $fm.Value
    $baseParts.Add(('{0}|{1}|{2}' -f ([regex]::Match($bline, '"layer":(\d+)').Groups[1].Value),
                                    ([regex]::Match($bline, '"union_ids":"([^"]*)"').Groups[1].Value),
                                    ([regex]::Match($bline, '"hash":(\d+)').Groups[1].Value)))
}
$baseDigest = Get-RouteDigest $baseParts
$baseTgIds = ''
$btgm = [regex]::Match($baseRaw, '"event":"tg_tokens"[^\n]*')
if ($btgm.Success) { $baseTgIds = [regex]::Match($btgm.Value, '"ids":\[([^\]]*)\]').Groups[1].Value }
Log ("baseline (recorded probe): fetches={0} rounds={1} staged={2} per-round={3}" -f `
     $baseFetches, $baseRounds, $baseStaged, $basePerRound)

# --- criterion 2's target, derived from the offline (b) AT THIS K -------------
# At K=1 the contract froze the target at 395,835,932 B. That number is just
# baseline x (1 - (b)_1), so the same rule generalises instead of being
# re-typed per K -- and it keeps the prediction tied to the trace it came from
# rather than to a literal someone could fat-finger.
# The offline (b) report comes from the SAME vehicle as the prompt and baseline,
# so a rung can never pair one vehicle's prompt with another's prediction.
$widePath = $baseAn
$wide = Get-Content -LiteralPath $widePath -Raw | ConvertFrom-Json
$wrs  = $wide[0].rowset
$predB = [double]$wrs.verdict_statistic_b."$RetainK".value
$predOfflineHits = [uint64]$wrs.verdict_statistic_b."$RetainK".hits_total
$predOfflineReq  = [uint64]$wrs.verdict_statistic_b."$RetainK".required_total
$predByteTarget  = [math]::Round($basePerRound * (1.0 - $predB), 0)
$predTokS_842   = 15.30         # contract criterion 3, 8.419 GB/s conversion (K=1 only)
$predTokS_137   = 13.61         # contract criterion 3, 13.7 GB/s ceiling (K=1 only)
Log ("offline statistic (b) at K={0}: {1:N6} ({2}/{3} rows) -> predicted staged/round {4}" -f `
     $RetainK, $predB, $predOfflineHits, $predOfflineReq, $predByteTarget)

# --- which vehicle this rung is, stated loudly in the rung's own log --------
Log ("vehicle: {0} | prompt: {1} | pp-tokens {2} | recorded baseline: {3}" -f $Vehicle, (Split-Path $prompt -Leaf), $ppTokens, (Split-Path $base -Leaf))
if ($Vehicle -eq 'honest') {
    Log 'this is the NON-ECHOING vehicle and the default: admissible as the sole evidence behind a depth decision. The K=2/4/6/8 ladder and the K=8 operating depth were originally chosen on ECHOED gains; the honest re-run stages 22-24% more bytes per round at the same shapes, so the echoed reading is the optimistic one.'
    Log 'the staged-byte prediction in criterion 2 is PRE-REGISTERED: the offline (b) comes from this stream and was computed before any honest rung ran.'
} else {
    Log 'VEHICLE=ECHOED -- OPT-IN REPRODUCTION ONLY. This prompt echoes its own anchor, which inflates routing reuse, so the gains below are the OPTIMISTIC pairing (SDD section 4; docs/currentstate.md 2026-09-14 and 2026-09-18). This rung is NOT admissible as the sole evidence behind a depth decision: re-run with the default -Vehicle honest for that.'
}

# --- four arms, one mutex session -------------------------------------------
$arms = @(
    [ordered]@{ name = 'A1'; retain = 0 },
    [ordered]@{ name = 'B1'; retain = $RetainK },
    [ordered]@{ name = 'A2'; retain = 0 },
    [ordered]@{ name = 'B2'; retain = $RetainK }
)
foreach ($a in $arms) {
    $a.trace = Join-Path $rungRoot ('arm-{0}.jsonl' -f $a.name)
    $a.log   = Join-Path $rungRoot ('arm-{0}.log'   -f $a.name)
    $a.txt   = Join-Path $rungRoot ('arm-{0}.txt'   -f $a.name)
    $a.csv   = Join-Path $rungRoot ('mem-{0}.csv'   -f $a.name)
    if (-not $AnalyzeOnly) {
        foreach ($p in @($a.trace, $a.log, $a.txt, $a.csv)) {
            if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
        }
    }
}

# AnalyzeOnly cannot re-measure exit status or wall time, so it reuses the ones
# recorded in the previous summary.json rather than reporting blanks or zeroes.
$prior = @{}
$priorPath = Join-Path $rungRoot 'summary.json'
if ($AnalyzeOnly -and (Test-Path -LiteralPath $priorPath)) {
    foreach ($p in (Get-Content -LiteralPath $priorPath -Raw | ConvertFrom-Json)) {
        $prior[$p.arm] = $p
    }
    Log ('reusing recorded exit/wall for arms: ' + (($prior.Keys | Sort-Object) -join ', '))
}

if ($AnalyzeOnly) {
    Log 'ANALYZE-ONLY: reading the existing arm traces; the GPU is not touched'
    # -AnalyzeOnly is only meaningful for a vehicle that has already run at this
    # K. Fail loudly with the way out rather than reporting empty criteria: the
    # default vehicle is now 'honest', so the historical
    # `-AnalyzeOnly -RetainK 1` form (which regenerates the RECORDED echoed K=1
    # report in place) needs -Vehicle echoed said out loud.
    $missingTraces = @($arms | ForEach-Object { $_.trace } | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missingTraces.Count -gt 0) {
        throw ("ANALYZE-ONLY found no traces for vehicle '{0}' at K={1} (missing: {2}). No run of this vehicle exists at this depth. To regenerate the RECORDED echoed report, pass -Vehicle echoed explicitly; to create honest evidence, drop -AnalyzeOnly and run the rung." -f $Vehicle, $RetainK, ($missingTraces -join '; '))
    }
}
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
try {
    if (-not $AnalyzeOnly) {
    Log 'waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(2400)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Log 'recovered an abandoned GPU mutex' }
    if (-not $acquired) { throw 'timed out waiting for the GPU mutex after 2400 s' }
    Log 'acquired GPU mutex (one session for all four arms)'

    foreach ($a in $arms) {
        $argList = @(
            '--model', $moe, '--prompt-file', $prompt,
            '--residency', 'moe-expert-bank', '--residency-tg',
            '--pp-schedule', 'batch-major',
            '--pp-tokens', "$ppTokens", '--ubatch', '512', '--tg-tokens', '256',
            '--layer-slots', '4', '--kv-cache', 'q8', '--greedy', '--no-mtp',
            '--pp-no-logits', '--allow-full-residency',
            '--telemetry', $a.trace, '--tg-text', $a.txt
        )
        if ($a.retain -gt 0) { $argList += @('--residency-tg-retain', "$($a.retain)") }
        Log (("arm {0} (retain={1}): " -f $a.name, $a.retain) + ($argList -join ' '))

        $poll = Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $poller,
            '-Out', $a.csv, '-IntervalMs', '500', '-DurationSeconds', '900')
        Log ("poller attached (pid {0}) -> {1}" -f $poll.Id, $a.csv)

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $old = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try { $all = & $exe @argList 2>&1 }
        finally { $ErrorActionPreference = $old }
        $a.exit = $LASTEXITCODE
        $sw.Stop()
        $a.wall_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
        [System.IO.File]::WriteAllLines($a.log, @($all | ForEach-Object { $_.ToString() }))
        Log ("arm {0} exit {1} in {2} ms" -f $a.name, $a.exit, $a.wall_ms)

        for ($i = 0; $i -lt 90; $i++) {
            if ($null -eq (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 500
        }
        if ($null -ne (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $poll.Id -Force -ErrorAction Continue
            Start-Sleep -Milliseconds 500
        }
    }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Log 'released GPU mutex' }
    $mutex.Dispose()
}

# --- per-arm metrics ---------------------------------------------------------
function Get-ArmStats([string]$trace, [string]$tl) {
    $m = [ordered]@{
        n_fetch = 0; n_round = 0; staged_bytes = 0
        rows_fetched = 0; rows_resident = 0; rows_retained = 0; rows_retired = 0
        submit_sum = 0.0; transfer_sum = 0.0; ready_sum = 0.0
        transfer_min = [double]::MaxValue; transfer_max = 0.0
        route_digest = ''; tg_ids = ''; peak_vram_bytes = $null
        pp_tok_s = ''; tg_tok_s = ''; refused = 0; retain_k = $null
        layers_with_reuse = 0; cache_alloc_mb = ''
    }
    if (Test-Path -LiteralPath $trace) {
        $raw = [System.IO.File]::ReadAllText($trace)
        $m.refused = [regex]::Matches($raw, '"event":"rowset_refused"').Count
        $m.n_round = [regex]::Matches($raw, '"event":"rowset_round"').Count
        $init = [regex]::Match($raw, '"event":"rowset_init"[^\n]*')
        if ($init.Success) { $m.retain_k = [regex]::Match($init.Value, '"retain_k":(\d+)').Groups[1].Value }
        $reuseLayers = @{}
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($fm in [regex]::Matches($raw, '\{"event":"rowset_fetch"[^\n]*')) {
            $line = $fm.Value
            $m.n_fetch++
            $s = [double][regex]::Match($line, '"host_submit_us":([\d.]+)').Groups[1].Value
            $t = [double][regex]::Match($line, '"transfer_us":([\d.]+)').Groups[1].Value
            $r = [double][regex]::Match($line, '"ready_latency_us":([\d.]+)').Groups[1].Value
            $m.submit_sum += $s; $m.transfer_sum += $t; $m.ready_sum += $r
            if ($t -lt $m.transfer_min) { $m.transfer_min = $t }
            if ($t -gt $m.transfer_max) { $m.transfer_max = $t }
            $m.staged_bytes  += [uint64][regex]::Match($line, '"bytes_staged":(\d+)').Groups[1].Value
            $m.rows_fetched  += [uint64][regex]::Match($line, '"n_rows_fetched":(\d+)').Groups[1].Value
            $m.rows_resident += [uint64][regex]::Match($line, '"n_rows_resident":(\d+)').Groups[1].Value
            $ret = [regex]::Match($line, '"n_rows_retained":(\d+)')
            if ($ret.Success) {
                $rv = [uint64]$ret.Groups[1].Value
                $m.rows_retained += $rv
                if ($rv -gt 0) {
                    $reuseLayers[[int][regex]::Match($line, '"layer":(\d+)').Groups[1].Value] = $true
                }
            }
            $rtd = [regex]::Match($line, '"n_rows_retired":(\d+)')
            if ($rtd.Success) { $m.rows_retired += [uint64]$rtd.Groups[1].Value }
            $layer = [regex]::Match($line, '"layer":(\d+)').Groups[1].Value
            $uid   = [regex]::Match($line, '"union_ids":"([^"]*)"').Groups[1].Value
            $hash  = [regex]::Match($line, '"hash":(\d+)').Groups[1].Value
            $parts.Add(('{0}|{1}|{2}' -f $layer, $uid, $hash))
        }
        $m.layers_with_reuse = $reuseLayers.Count
        $m.route_digest = Get-RouteDigest $parts
        $tgm = [regex]::Match($raw, '"event":"tg_tokens"[^\n]*')
        if ($tgm.Success) { $m.tg_ids = [regex]::Match($tgm.Value, '"ids":\[([^\]]*)\]').Groups[1].Value }
        $vm = [regex]::Matches($raw, '"peak_vram_bytes":(\d+)')
        if ($vm.Count -gt 0) { $m.peak_vram_bytes = [uint64]$vm[$vm.Count - 1].Groups[1].Value }
    }
    if (Test-Path -LiteralPath $tl) {
        $raw = [System.IO.File]::ReadAllText($tl)
        $pp = [regex]::Match($raw, 'phase pp\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($pp.Success) { $m.pp_tok_s = $pp.Groups[1].Value }
        $tg = [regex]::Match($raw, 'phase tg\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($tg.Success) { $m.tg_tok_s = $tg.Groups[1].Value }
        # reported, not claimed: cache allocation as the runtime logged it
        $ca = [regex]::Match($raw, 'row-cache \(T-GPLR-19[^\n]*')
        if ($ca.Success) { $m.cache_alloc_mb = $ca.Value }
    }
    return $m
}

# The OTHER vehicle's reading at this same K, so a rung is never one reading.
# It is a FUNCTION so the caller can wrap it in try/catch: on the first honest run
# this block threw (PowerShell 5.1 does not enumerate a top-level JSON array) and
# killed the driver AFTER the criteria were logged but BEFORE its verdict line,
# leaving a run that measured correctly and reported nothing.
function Write-OtherVehicleComparison {
    param([string] $Root, [string] $Vehicle, [int] $RetainK, [bool] $Measured, [double] $Gain)

    $otherName    = if ($Vehicle -eq 'honest') { 'echoed' } else { 'honest' }
    $otherTag     = if ($Vehicle -eq 'honest') {
        if ($RetainK -eq 1) { 'gplr19-rowcache-ab' } else { "gplr19-k$RetainK-ab" }
    } else { "gplr19-honest-k$RetainK-ab" }
    $otherSummary = Join-Path $Root ("build\llama-cuda\" + $otherTag + '\summary.json')
    # The honest K=8 evidence from 2026-09-18 sits one level deeper: a PowerShell
    # case-insensitive variable collision nested rung 2 inside rung 1's directory
    # (docs/currentstate.md 2026-09-18). Look there too rather than pretend the
    # reference does not exist.
    $nestedSummary = Join-Path $Root ("build\llama-cuda\gplr19-honest-k6-ab\" + $otherTag + '\summary.json')
    if (-not (Test-Path -LiteralPath $otherSummary) -and (Test-Path -LiteralPath $nestedSummary)) {
        $otherSummary = $nestedSummary
        Log ("note: the {0} reference at this K is the nested 2026-09-18 artifact (see docs/currentstate.md for why)" -f $otherName)
    }

    if (-not (Test-Path -LiteralPath $otherSummary)) {
        Log ("OTHER VEHICLE ({0}) at K={1}: no recorded run at {2}. For the ECHOED vehicle that leaves this rung standing alone -- the exact state this driver exists to prevent." -f $otherName, $RetainK, (Split-Path $otherSummary -Parent))
        return
    }

    # NOT `@(Get-Content -Raw | ConvertFrom-Json)`: in Windows PowerShell 5.1 that
    # produces ONE object whose properties are ARRAYS, so `$_.tg_tok_s` is an
    # Object[] and casting it to [double] throws. Assign first, then pipe -- the
    # pipeline enumerates.
    $orows = Get-Content -LiteralPath $otherSummary -Raw | ConvertFrom-Json
    $oB    = @($orows | Where-Object { $_.retain -eq $RetainK })
    if ($oB.Count -eq 0) {
        Log ("OTHER VEHICLE ({0}) at K={1}: its summary has no retained arm at this depth -- nothing to compare." -f $otherName, $RetainK)
        return
    }

    $oAMean = (@($orows | Where-Object { $_.retain -eq 0 } | ForEach-Object { [double]$_.tg_tok_s }) | Measure-Object -Average).Average
    $oBMean = (@($oB | ForEach-Object { [double]$_.tg_tok_s }) | Measure-Object -Average).Average
    $oGain  = $oBMean - $oAMean
    Log ("OTHER VEHICLE ({0}) at K={1}: reactive {2:N3} tok/s | retained {3:N3} | gain {4:N3}" -f $otherName, $RetainK, $oAMean, $oBMean, $oGain)
    if ($Measured) {
        Log ("  this vehicle's gain {0:N3} vs the other vehicle's {1:N3} -> delta {2:N3} tok/s. Read that against the within-arm spread logged above before attributing anything to the prompt: a delta smaller than the spread is NOT a measured difference." -f $Gain, $oGain, ($Gain - $oGain))
    }
}

$rows = @()
foreach ($a in $arms) {
    $s = Get-ArmStats $a.trace $a.log
    $c = Invoke-Capture @($exe, '--check-telemetry', $a.trace)
    $a.checker_exit = $c.code
    $a.stats = $s
    if ($AnalyzeOnly -and $prior.ContainsKey($a.name)) {
        $a.exit    = $prior[$a.name].exit
        $a.wall_ms = $prior[$a.name].wall_ms
    }
    if (-not $a.Contains('exit'))    { $a.exit = $null }
    if (-not $a.Contains('wall_ms')) { $a.wall_ms = $null }
    $n = [math]::Max(1, $s.n_fetch)
    $rows += [pscustomobject]@{
        vehicle = $Vehicle
        arm = $a.name; retain = $a.retain; exit = $a.exit; checker = $c.code
        wall_ms = $a.wall_ms; pp_tok_s = $s.pp_tok_s; tg_tok_s = $s.tg_tok_s
        retain_k = $s.retain_k; n_fetch = $s.n_fetch; n_round = $s.n_round
        refused = $s.refused
        staged_bytes = $s.staged_bytes
        staged_per_round = [math]::Round($s.staged_bytes / [math]::Max(1, $s.n_round), 0)
        rows_fetched = $s.rows_fetched; rows_resident = $s.rows_resident
        rows_retained = $s.rows_retained; rows_retired = $s.rows_retired
        layers_with_reuse = $s.layers_with_reuse
        transfer_mean_us = [math]::Round($s.transfer_sum / $n, 1)
        transfer_min = [math]::Round($s.transfer_min, 1)
        transfer_max = [math]::Round($s.transfer_max, 1)
        peak_vram_bytes = $s.peak_vram_bytes
        route_digest = $s.route_digest; tg_ids = $s.tg_ids
        cache_alloc = $s.cache_alloc_mb
    }
}
# written without a BOM so any consumer can read it as plain JSON
[System.IO.File]::WriteAllText((Join-Path $rungRoot 'summary.json'), ($rows | ConvertTo-Json -Depth 4),
    (New-Object System.Text.UTF8Encoding($false)))

Log ''
Log '=== T-GPLR-19 A/B table ==='
Log ("{0,-4} {1,-7} {2,-5} {3,-8} {4,-9} {5,-9} {6,-15} {7,-12} {8,-10} {9,-9} {10,-9} {11,-8}" -f `
     'arm','retain','exit','checker','pp t/s','tg t/s','staged/round','transfer_us','rows_fet','rows_ret','rows_rtd','refused')
foreach ($r in $rows) {
    Log ("{0,-4} {1,-7} {2,-5} {3,-8} {4,-9} {5,-9} {6,-15} {7,-12} {8,-10} {9,-9} {10,-9} {11,-8}" -f `
         $r.arm, $r.retain, $r.exit, $r.checker, $r.pp_tok_s, $r.tg_tok_s, $r.staged_per_round,
         $r.transfer_mean_us, $r.rows_fetched, $r.rows_retained, $r.rows_retired, $r.refused)
}

$A  = @($rows | Where-Object { $_.retain -eq 0 })
$B  = @($rows | Where-Object { $_.retain -eq $RetainK })
$same_route = (($rows | Select-Object -ExpandProperty route_digest | Select-Object -Unique).Count -eq 1)
$same_ids   = (($rows | Select-Object -ExpandProperty tg_ids | Select-Object -Unique).Count -eq 1)
$all_ok     = (@($rows | Where-Object { $_.exit -ne 0 -or $_.checker -ne 0 }).Count -eq 0)
$base_match = (@($A | Where-Object { $_.n_fetch -ne $baseFetches -or $_.staged_per_round -ne $basePerRound -or $_.rows_resident -ne 0 }).Count -eq 0)

Log ''
Log ('route stream identical across arms: {0}' -f $same_route)
Log ('greedy token ids identical across arms: {0}' -f $same_ids)
Log ('reactive arms match the recorded baseline (fetches={0}, staged/round={1}, rows_reused=0): {2}' -f $baseFetches, $basePerRound, $base_match)
$ident_match = (@($A | Where-Object { $_.route_digest -ne $baseDigest -or $_.tg_ids -ne $baseTgIds }).Count -eq 0)
Log ('reactive arms reproduce the recorded run IDENTITY (route digest {0}... AND greedy ids), not just its byte counts: {1}' -f $baseDigest.Substring(0, 16), $ident_match)

# criterion 1
$crit1 = $all_ok -and $same_route -and $same_ids -and $base_match -and $ident_match
Log ("criterion 1 (feasibility + parity): {0}" -f $(if ($crit1) { 'PASS' } else { 'FAIL' }))

# criterion 2: BYTES decide
$bMean = ($B | Measure-Object staged_per_round -Average).Average
$byteDelta = ($bMean - $predByteTarget) / $predByteTarget
$byteOk = [math]::Abs($byteDelta) -le 0.02
$reuseOk = (@($B | Where-Object { $_.rows_retained -le 0 -or $_.layers_with_reuse -lt 40 }).Count -eq 0)
$crit2 = $byteOk -and $reuseOk
Log ("criterion 2 (bytes): K={5} staged/round mean {0} vs predicted {1} -> {2:P3} (target within 2%) | cross-token reuse non-zero on all 40 layers: {3} -> {4}" -f `
     [math]::Round($bMean, 0), $predByteTarget, $byteDelta, $reuseOk, $(if ($crit2) { 'PASS' } else { 'FAIL' }), $RetainK)
Log '  note: contract criterion 2 names this counter `rows_resident_cross_token`; the shipped field is `n_rows_retained` (the driver evaluates the shipped one).'

# criterion 3: performance, only if 2 passed
if ($crit2) {
    $bTg   = @($B | ForEach-Object { [double]$_.tg_tok_s })
    $aTg   = @($A | ForEach-Object { [double]$_.tg_tok_s })
    $bMeanTg = ($bTg | Measure-Object -Average).Average
    $aMean   = ($aTg | Measure-Object -Average).Average
    $aSpread = [math]::Abs($aTg[0] - $aTg[1])
    $bSpread = [math]::Abs($bTg[0] - $bTg[1])
    $within  = [math]::Max($aSpread, $bSpread)
    $gain    = $bMeanTg - $aMean
    Log ("criterion 3 (performance): reactive mean {0:N3} tok/s | K={4} mean {1:N3} tok/s | gain {2:N3} | within-arm spread {3:N3}" -f `
         $aMean, $bMeanTg, $gain, $within, $RetainK)
    if ($RetainK -eq 1 -and $Vehicle -eq 'echoed') {
        Log ("  pre-registered upper bounds: {0} tok/s (8.419 GB/s conversion) and {1} tok/s (13.7 GB/s ceiling)" -f $predTokS_842, $predTokS_137)
    } elseif ($Vehicle -eq 'honest') {
        Log "  no pre-registered TOKENS/S bound exists for the honest vehicle: what is pre-registered here is the staged-BYTE prediction in criterion 2, from the offline (b) computed on this stream on 2026-09-14, before any honest rung ran."
    } else {
        Log "  no pre-registered bound exists for K>1: the contract deferred these rungs, so the bounds below are POST-HOC and reported as context, never as a verdict."
    }
    Log ("  gain exceeds the within-arm spread: {0}" -f $(if ($gain -gt $within) { 'YES' } else { 'NO -- falsifies byte-to-time proportionality' }))
    # The conversion is what these rungs are actually for: view 2's projection
    # rests on removed bytes converting at a rate that does NOT degrade with
    # depth, and that assumption was measured at exactly one point (K=1,
    # 14.29 GB/s). Every rung re-measures it, so drift is visible as a number.
    $bBytes  = ($B | Measure-Object staged_per_round -Average).Average
    $aBytes  = ($A | Measure-Object staged_per_round -Average).Average
    $savedB  = $aBytes - $bBytes
    $savedMs = (1000.0 / $aMean) - (1000.0 / $bMeanTg)
    $convGbs = if ($savedMs -gt 0) { ($savedB / ($savedMs / 1000.0)) / 1e9 } else { 0.0 }
    Log ("  byte-to-time conversion at K={0}: {1:N0} B saved per token over {2:N2} ms saved = {3:N2} GB/s" -f `
         $RetainK, $savedB, $savedMs, $convGbs)
    if ($RetainK -eq 1 -and $Vehicle -eq 'echoed') {
        Log ("  above the 8.419 GB/s bound: {0}; above the link-ceiling bound: {1}" -f `
             $(if ($bMeanTg -gt $predTokS_842) { 'YES' } else { 'NO' }), $(if ($bMeanTg -gt $predTokS_137) { 'YES' } else { 'NO' }))
    }
    # the artifact's View-2 projections at this K, for context only
    $projNote = if ($null -ne $wrs.verdict_statistic_b."$RetainK") {
        $offB = [double]$wrs.verdict_statistic_b."$RetainK".value
        $frac = 1.0 - $offB
        ('  offline (b) at K={0} is {1:N6}, so the staged-byte fraction should be {2:N6} of the reactive arms.' -f $RetainK, $offB, $frac)
    } else { '' }
    if ($projNote) { Log $projNote }
} else {
    Log 'criterion 3 (performance): NOT EVALUATED -- criterion 2 failed, so no timing discussion follows (contract).'
}

# criterion 6: reported, not claimed
foreach ($r in $B) {
    Log ("criterion 6 (reported): arm {0} peak VRAM {1} bytes | cache alloc: {2}" -f $r.arm, $r.peak_vram_bytes, $r.cache_alloc)
}

# --- offline (b) vs runtime reuse on the SAME stream -------------------------
Log ''
Log ("=== offline (b) at K={0} vs runtime reuse accounting, same stream ===" -f $RetainK)
$analyzeFailed = 0
foreach ($a in $arms) {
    $rep = Join-Path $rungRoot ('analysis-{0}.json' -f $a.name)
    $g = Invoke-Capture @($py, $an, $a.trace, '--retention-k', "$RetainK", '--json', $rep)
    if ($g.code -ne 0) {
        $analyzeFailed++
        $reason = ($g.out -join ' ')
        if ($reason -match 'unsupported event type') {
            # Distinguish SCHEMA DRIFT from a real analysis failure: the analyzer
            # fails loudly on event types it does not know, which is how the
            # async_feed_* drift was found (it is fixed in the analyzer, but any
            # future engine event will land here first).
            Log ("analyze {0} FAILED -- ANALYZER/ENGINE SCHEMA DRIFT, not a measurement problem: {1}" -f $a.name, $reason)
        } else {
            Log ("analyze {0} FAILED: {1}" -f $a.name, $reason)
        }
        continue
    }
    $j = Get-Content -LiteralPath $rep -Raw | ConvertFrom-Json
    $rs = $j[0].rowset
    $offlineHits = $rs.verdict_statistic_b."$RetainK".hits_total
    $offlineReq  = $rs.verdict_statistic_b."$RetainK".required_total
    $rtRows = $a.stats.rows_retained
    if ($a.retain -eq 0) {
        # Offline (b) answers "what would a cache hold on this stream", so the
        # reactive arms have nothing to agree WITH: they run no cache at all.
        # The comparison is meaningful only on the retention arms, where the
        # runtime and the analyzer are describing the same objects.
        Log ("arm {0}: offline (b) hits {1}/{2} = {3:N6} | retention OFF, so there is no runtime cache to compare -- N/A (not a mismatch)" -f `
             $a.name, $offlineHits, $offlineReq, $rs.verdict_statistic_b."$RetainK".value)
        continue
    }
    # The COUNT is the assertion; the two fractions use different denominators on
    # purpose -- the runtime divides by every fetch x top_k, the offline statistic
    # divides by its 10,200 consecutive-window pairs (it has no predecessor for
    # the first window). Comparing the raw counts avoids that convention clash.
    $match = ($rtRows -eq [uint64]$offlineHits)
    Log ("arm {0}: offline (b) hits {1}/{2} = {3:N6} | runtime retained rows {4} (over {5} row requests) = {6:N6} | agree exactly: {7}" -f `
         $a.name, $offlineHits, $offlineReq, $rs.verdict_statistic_b."$RetainK".value, $rtRows,
         ($a.stats.n_fetch * 8), ($rtRows / [double]([math]::Max(1, $a.stats.n_fetch * 8))), $match)
}

# --- the OTHER vehicle at this same K, so a rung is never one reading -------
# The whole point of the honest default is that the echoed pairing cannot be the
# only evidence. If a run exists for the other vehicle at this depth, put both
# gains in THIS log, where a depth decision will actually read them.
if ($analyzeFailed -gt 0) {
    Log ("OFFLINE CROSS-CHECK UNAVAILABLE for this rung: {0} of {1} arms could not be analyzed. The runtime reuse accounting and the criteria above are unaffected -- this section only compares them against the offline (b) on the same stream, which the recorded K=1..K=8 rungs did have." -f $analyzeFailed, $arms.Count)
}

Log ''
try {
    $measured = ($crit1 -and $crit2)
    Write-OtherVehicleComparison -Root $root -Vehicle $Vehicle -RetainK $RetainK -Measured $measured -Gain $(if ($measured) { $gain } else { 0.0 })
} catch {
    # A comparison is CONTEXT, never the verdict: a failure here must not cost
    # the rung its result or its verdict line.
    Log ("WARNING: the cross-vehicle comparison FAILED and is SKIPPED (this rung's own result above still stands): " + $_.Exception.Message)
}

Log ''
Log ("GPLR19 AB DONE vehicle={2} K={1} overall={0}" -f $(if ($crit1 -and $crit2) { 'CRITERIA-1-2-PASS' } else { 'NOT-PASSED' }), $RetainK, $Vehicle)
if ($Vehicle -eq 'echoed') {
    Log 'REMINDER: this was the ECHOED vehicle (optimistic pairing). Do not use it alone as the evidence behind a depth decision; the default is -Vehicle honest.'
}

# --- verdict + sentinel -----------------------------------------------------
# The sentinel mirrors the driver.log verdict line, in a form a caller can read
# without parsing a log. Exit code is now meaningful: 0 only when criteria 1 and
# 2 both pass, 2 when they do not (a NOT-PASSED rung used to exit 0, which made a
# failed rung indistinguishable from a successful one to any caller).
$verdictExit = if ($crit1 -and $crit2) { 0 } else { 2 }
$finalFields = @{}
foreach ($k in $verdictFields.Keys) { $finalFields[$k] = $verdictFields[$k] }
$finalFields['crit1'] = [bool]$crit1
$finalFields['crit2'] = [bool]$crit2
$finalFields['gain_tok_s']            = $(if ($crit2) { [math]::Round($gain, 3) } else { $null })
$finalFields['spread_tok_s']          = $(if ($crit2) { [math]::Round($within, 3) } else { $null })
$finalFields['staged_bytes_per_round'] = $(if ($crit2) { [math]::Round($bBytes, 0) } else { $null })
$finalFields['analyze_failed']        = $analyzeFailed
Write-ArmVerdict -Dir $verdictDir -Tag $verdictTag -ExitCode $verdictExit -StartedUtc $verdictStart `
    -LogPath $drvLog -Fields $finalFields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f $(if ($verdictExit -eq 0) { 'PASS' } else { 'FAIL' }), $verdictExit, (Join-Path $verdictDir ($verdictTag + '.done')))
exit $verdictExit
