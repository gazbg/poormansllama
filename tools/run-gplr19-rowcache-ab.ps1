# T-GPLR-19: paired four-arm A/B for per-layer row residency at K=1.
#
# NOTE (2026-09-14): the MECHANISM's default depth is now K=4 (bare
# `--residency-tg-retain`; SDD scope amendment 2026-09-14). This driver's own
# default arm stays K=1 so that -AnalyzeOnly still regenerates the recorded K=1
# report in place; pass -RetainK to compare another depth.
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

    # Override the output root. Empty = derive it from $RetainK.
    [string] $OutRoot = ''
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$poller = Join-Path $root 'tools\gpu-mem-poll.ps1'
$an     = Join-Path $root 'tools\route_overlap\analyze.py'
$py     = 'python'
$moe    = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
$prompt = Join-Path $root 'build\llama-cuda\gplr13-ladder\512\anchor-512.txt'
$base   = Join-Path $root 'build\llama-cuda\gplr15-phaseA-probe\probe-tg256.jsonl'

foreach ($f in @($exe, $poller, $an, $prompt)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing tool: $f" }
}
if (-not (Test-Path -LiteralPath $moe)) { throw "missing model: $moe" }

if ($RetainK -lt 1 -or $RetainK -gt 64) { throw "--residency-tg-retain accepts 1..64; got $RetainK" }
$kTag = if ($RetainK -eq 1) { 'gplr19-rowcache-ab' } else { "gplr19-k$RetainK-ab" }
$outRoot = if ($OutRoot) { $OutRoot } else { Join-Path $root "build\llama-cuda\$kTag" }
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$drvLog = Join-Path $outRoot 'driver.log'
if (Test-Path $drvLog) { Remove-Item $drvLog -Force }

function Log([string]$msg) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg)
    Write-Host $line
    Add-Content -LiteralPath $drvLog -Value $line
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
Log ("baseline (recorded probe): fetches={0} rounds={1} staged={2} per-round={3}" -f `
     $baseFetches, $baseRounds, $baseStaged, $basePerRound)

# --- criterion 2's target, derived from the offline (b) AT THIS K -------------
# At K=1 the contract froze the target at 395,835,932 B. That number is just
# baseline x (1 - (b)_1), so the same rule generalises instead of being
# re-typed per K -- and it keeps the prediction tied to the trace it came from
# rather than to a literal someone could fat-finger.
$widePath = Join-Path $root 'build\llama-cuda\gplr15-phaseA-probe\analysis-wide.json'
if (-not (Test-Path -LiteralPath $widePath)) { throw "missing offline report: $widePath" }
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

# --- four arms, one mutex session -------------------------------------------
$arms = @(
    [ordered]@{ name = 'A1'; retain = 0 },
    [ordered]@{ name = 'B1'; retain = $RetainK },
    [ordered]@{ name = 'A2'; retain = 0 },
    [ordered]@{ name = 'B2'; retain = $RetainK }
)
foreach ($a in $arms) {
    $a.trace = Join-Path $outRoot ('arm-{0}.jsonl' -f $a.name)
    $a.log   = Join-Path $outRoot ('arm-{0}.log'   -f $a.name)
    $a.txt   = Join-Path $outRoot ('arm-{0}.txt'   -f $a.name)
    $a.csv   = Join-Path $outRoot ('mem-{0}.csv'   -f $a.name)
    if (-not $AnalyzeOnly) {
        foreach ($p in @($a.trace, $a.log, $a.txt, $a.csv)) {
            if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
        }
    }
}

# AnalyzeOnly cannot re-measure exit status or wall time, so it reuses the ones
# recorded in the previous summary.json rather than reporting blanks or zeroes.
$prior = @{}
$priorPath = Join-Path $outRoot 'summary.json'
if ($AnalyzeOnly -and (Test-Path -LiteralPath $priorPath)) {
    foreach ($p in (Get-Content -LiteralPath $priorPath -Raw | ConvertFrom-Json)) {
        $prior[$p.arm] = $p
    }
    Log ('reusing recorded exit/wall for arms: ' + (($prior.Keys | Sort-Object) -join ', '))
}

if ($AnalyzeOnly) {
    Log 'ANALYZE-ONLY: reading the existing arm traces; the GPU is not touched'
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
            '--pp-tokens', '512', '--ubatch', '512', '--tg-tokens', '256',
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
        $joined = ($parts -join ';')
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $m.route_digest = ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($joined))) -replace '-', '').ToLower()
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
[System.IO.File]::WriteAllText((Join-Path $outRoot 'summary.json'), ($rows | ConvertTo-Json -Depth 4),
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

# criterion 1
$crit1 = $all_ok -and $same_route -and $same_ids -and $base_match
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
    if ($RetainK -eq 1) {
        Log ("  pre-registered upper bounds: {0} tok/s (8.419 GB/s conversion) and {1} tok/s (13.7 GB/s ceiling)" -f $predTokS_842, $predTokS_137)
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
    if ($RetainK -eq 1) {
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
foreach ($a in $arms) {
    $rep = Join-Path $outRoot ('analysis-{0}.json' -f $a.name)
    $g = Invoke-Capture @($py, $an, $a.trace, '--retention-k', "$RetainK", '--json', $rep)
    if ($g.code -ne 0) { Log ("analyze {0} FAILED: {1}" -f $a.name, ($g.out -join ' ')); continue }
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

Log ''
Log ("GPLR19 AB DONE K={1} overall={0}" -f $(if ($crit1 -and $crit2) { 'CRITERIA-1-2-PASS' } else { 'NOT-PASSED' }), $RetainK)
