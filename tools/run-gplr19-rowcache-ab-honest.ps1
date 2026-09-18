# T-GPLR-19 HONEST-STREAM A/B: the row-cache rungs re-run on a NON-ECHOING prompt.
#
# WHY THIS EXISTS. The recorded K ladder (gplr19-k2/k4/k6/k8-ab) is an A/B whose
# A arms run the echoing needle prompt (anchor-512.txt). Its (b)off column is
# therefore the ECHOED series, and the docs record that such a pairing is the
# OPTIMISTIC one: a non-echoing stream compresses the gain by roughly the
# 0.0492-0.0929 (b) shift (docs/currentstate.md 2026-09-14, SDD section 4). The
# owner's chosen operating depth is K=8 and the recorded +12.609 tok/s at K=8 is
# an upper estimate that was never validated on an honest stream. This driver
# closes exactly that gap at K=6 and K=8.
#
# THE DECLARED CHANGE IS ONE: the prompt BODY. Everything else is held at the
# recorded T-GPLR-19 arm values, which are themselves the T-GPLR-15 probe's
# batch-major shape. The non-echoing prompt is build/calc-prompt.txt (885
# tokens) -- the recorded natural-language code prompt from the gplr06 runs, the
# same one the 2026-09-14 re-probe substituted. Records of that substitution and
# its partial certification (16-gram 1.0000, zlib criterion mis-chosen):
# docs/currentstate.md 2026-09-14.
#
# WHAT IS PRE-REGISTERED vs POST-HOC, stated because it changes how to read the
# output. The honest statistic (b) at each K was computed 2026-09-14 from this
# same stream, BEFORE this run, by the offline analyzer -- so the predicted
# staged-bytes-per-round is a genuine pre-registered prediction and criterion 2
# below is a real test of it. The reactive baseline constants are likewise
# re-derived from the recorded honest trace, not retyped. The K>1 performance
# bounds remain POST-HOC context, as they were in the recorded rungs.
#
# Arms are A,B,A,B per rung with A = reactive row-set (retention off) and
# B = `--residency-tg-retain K`. One mutex acquisition per rung, four arms
# inside it, so each rung's pairs see a single GPU session. An EXPLICIT depth is
# passed (never the bare flag): a named depth is a contract and is never
# silently reduced by the VRAM fit.
#
# This script writes only under build/ (gitignored) and touches no recorded
# artifact.
param(
    # Retention depths to measure, one mutex acquisition each. The owner's
    # operating depth is K=8; K=6 is its nearest recorded neighbour and the
    # shallowest depth that clears 0.6 on the honest curve.
    #
    # Taken as a STRING and split here on purpose: passed to `-File` as
    # `-Rungs 6,8`, PowerShell binds the literal "6,8" to an int[] and the
    # comma is consumed as a digit separator, yielding the single value 68 (the
    # range guard caught exactly that on the first attempt). `-Rungs 6` and
    # `-Rungs "6,8"` both work through this form.
    [string] $Rungs = '6,8',

    # Override the output root. Empty = gplr19-honest-k<K>-ab per rung.
    [string] $OutRoot = ''
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$poller = Join-Path $root 'tools\gpu-mem-poll.ps1'
$moe    = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
# The declared non-echoing prompt, and the recorded honest run of the SAME
# invocation (reactive, retention off) that serves as this A/B's baseline.
$prompt   = Join-Path $root 'build\calc-prompt.txt'
$baseTel  = Join-Path $root 'build\llama-cuda\gplr15-nononchoing\probe-tg256-nononchoing.jsonl'
$baseAn   = Join-Path $root 'build\llama-cuda\gplr15-nononchoing\an\nononchoing.json'
$ppTokens = 885

foreach ($f in @($exe, $poller, $prompt, $baseTel, $baseAn)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing tool/artifact: $f" }
}
if (-not (Test-Path -LiteralPath $moe)) { throw "missing model: $moe" }
$Ks = @($Rungs -split '[,\s]+' | Where-Object { $_ -ne '' } | ForEach-Object { [int]$_ })
if ($Ks.Count -eq 0) { throw "-Rungs gave no depths; got '$Rungs'" }
foreach ($k in $Ks) {
    if ($k -lt 1 -or $k -gt 64) { throw "--residency-tg-retain accepts 1..64; got $k (from -Rungs '$Rungs')" }
}

function Log([string]$msg, [string]$logPath) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg)
    Write-Host $line
    if ($logPath) { Add-Content -LiteralPath $logPath -Value $line }
}

# --- route-stream digest + greedy ids, defined exactly as the recorded driver ---
# SHA256 over the ordered `layer|union_ids|hash` stream, plus the tg_tokens ids.
# Copying the definition verbatim is what makes a digest comparable ACROSS
# drivers, which is the whole point of computing it here.
function Get-RouteIdentity([string]$trace) {
    $ids = ''; $digest = ''
    if (Test-Path -LiteralPath $trace) {
        $raw = [System.IO.File]::ReadAllText($trace)
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($fm in [regex]::Matches($raw, '\{"event":"rowset_fetch"[^\n]*')) {
            $line = $fm.Value
            $layer = [regex]::Match($line, '"layer":(\d+)').Groups[1].Value
            $uid   = [regex]::Match($line, '"union_ids":"([^"]*)"').Groups[1].Value
            $hash  = [regex]::Match($line, '"hash":(\d+)').Groups[1].Value
            $parts.Add(('{0}|{1}|{2}' -f $layer, $uid, $hash))
        }
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $digest = ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes(($parts -join ';')))) -replace '-', '').ToLower()
        $tgm = [regex]::Match($raw, '"event":"tg_tokens"[^\n]*')
        if ($tgm.Success) { $ids = [regex]::Match($tgm.Value, '"ids":\[([^\]]*)\]').Groups[1].Value }
    }
    return [pscustomobject]@{ digest = $digest; tg_ids = $ids }
}

# --- baseline constants re-derived from the recorded HONEST trace -------------
$baseRaw     = [System.IO.File]::ReadAllText($baseTel)
$baseFetches = [regex]::Matches($baseRaw, '"event":"rowset_fetch"').Count
$baseRounds  = [regex]::Matches($baseRaw, '"event":"rowset_round"').Count
$baseStaged  = [uint64]0
foreach ($fm in [regex]::Matches($baseRaw, '"event":"rowset_fetch"[^\n]*')) {
    $baseStaged += [uint64][regex]::Match($fm.Value, '"bytes_staged":(\d+)').Groups[1].Value
}
$basePerRound = [math]::Round($baseStaged / [math]::Max(1, $baseRounds), 0)
$baseIdent    = Get-RouteIdentity $baseTel

# Honest (b) per K, from the 2026-09-14 re-probe analysis (pre-registered here).
$anRaw = [System.IO.File]::ReadAllText($baseAn) | ConvertFrom-Json
$anWrs = if ($anRaw -is [array]) { $anRaw[0].rowset } else { $anRaw.rowset }

function Invoke-Capture {
    param([string[]] $Argv)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $all = & $Argv[0] @($Argv | Select-Object -Skip 1) 2>&1 }
    finally { $ErrorActionPreference = $old }
    $code = $LASTEXITCODE
    return [pscustomobject]@{ code = $code; out = @($all | ForEach-Object { $_.ToString() }) }
}

function Get-ArmStats([string]$trace, [string]$tl) {
    $m = [ordered]@{
        n_fetch = 0; n_round = 0; staged_bytes = [uint64]0
        rows_fetched = [uint64]0; rows_retained = [uint64]0; rows_retired = [uint64]0
        transfer_sum = 0.0; ready_sum = 0.0; submit_sum = 0.0
        route_digest = ''; tg_ids = ''; peak_vram_bytes = $null
        pp_tok_s = ''; tg_tok_s = ''; refused = 0; retain_k = $null
        layers_with_reuse = 0; cache_alloc_mb = ''
    }
    if (Test-Path -LiteralPath $trace) {
        $raw = [System.IO.File]::ReadAllText($trace)
        $m.refused  = [regex]::Matches($raw, '"event":"rowset_refused"').Count
        $m.n_round  = [regex]::Matches($raw, '"event":"rowset_round"').Count
        $init = [regex]::Match($raw, '"event":"rowset_init"[^\n]*')
        if ($init.Success) { $m.retain_k = [regex]::Match($init.Value, '"retain_k":(\d+)').Groups[1].Value }
        $reuseLayers = @{}
        foreach ($fm in [regex]::Matches($raw, '\{"event":"rowset_fetch"[^\n]*')) {
            $line = $fm.Value
            $m.n_fetch++
            $m.submit_sum   += [double][regex]::Match($line, '"host_submit_us":([\d.]+)').Groups[1].Value
            $m.transfer_sum += [double][regex]::Match($line, '"transfer_us":([\d.]+)').Groups[1].Value
            $m.ready_sum    += [double][regex]::Match($line, '"ready_latency_us":([\d.]+)').Groups[1].Value
            $m.staged_bytes += [uint64][regex]::Match($line, '"bytes_staged":(\d+)').Groups[1].Value
            $m.rows_fetched += [uint64][regex]::Match($line, '"n_rows_fetched":(\d+)').Groups[1].Value
            $ret = [regex]::Match($line, '"n_rows_retained":(\d+)')
            if ($ret.Success) {
                $rv = [uint64]$ret.Groups[1].Value
                $m.rows_retained += $rv
                if ($rv -gt 0) { $reuseLayers[[int][regex]::Match($line, '"layer":(\d+)').Groups[1].Value] = $true }
            }
            $rtd = [regex]::Match($line, '"n_rows_retired":(\d+)')
            if ($rtd.Success) { $m.rows_retired += [uint64]$rtd.Groups[1].Value }
        }
        $m.layers_with_reuse = $reuseLayers.Count
        $ident = Get-RouteIdentity $trace
        $m.route_digest = $ident.digest; $m.tg_ids = $ident.tg_ids
        $vm = [regex]::Matches($raw, '"peak_vram_bytes":(\d+)')
        if ($vm.Count -gt 0) { $m.peak_vram_bytes = [uint64]$vm[$vm.Count - 1].Groups[1].Value }
    }
    if (Test-Path -LiteralPath $tl) {
        $txt = [System.IO.File]::ReadAllText($tl)
        $pp = [regex]::Match($txt, 'phase pp\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($pp.Success) { $m.pp_tok_s = $pp.Groups[1].Value }
        $tg = [regex]::Match($txt, 'phase tg\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($tg.Success) { $m.tg_tok_s = $tg.Groups[1].Value }
        $ca = [regex]::Match($txt, 'row-cache \(T-GPLR-19[^\n]*')
        if ($ca.Success) { $m.cache_alloc_mb = $ca.Value }
    }
    return $m
}

$allRungs = @()
foreach ($K in $Ks) {
    $kTag    = "gplr19-honest-k$K-ab"
    # NOTE: named $rungRoot, NOT $outRoot. PowerShell variable names are
    # case-INSENSITIVE, so a local $outRoot silently overwrites the $OutRoot
    # PARAMETER -- with two rungs in one invocation that made rung 2 nest inside
    # rung 1's directory (observed 2026-09-18: gplr19-honest-k6-ab\gplr19-honest-k8-ab).
    $rungRoot = if ($OutRoot) { Join-Path $OutRoot $kTag } else { Join-Path $root "build\llama-cuda\$kTag" }
    New-Item -ItemType Directory -Force -Path $rungRoot | Out-Null
    $log = Join-Path $rungRoot 'driver.log'
    if (Test-Path -LiteralPath $log) { Remove-Item $log -Force }

    $predB          = [double]$anWrs.verdict_statistic_b."$K".value
    $predHits       = [uint64]$anWrs.verdict_statistic_b."$K".hits_total
    $predReq        = [uint64]$anWrs.verdict_statistic_b."$K".required_total
    $predByteTarget = [math]::Round($basePerRound * (1.0 - $predB), 0)

    Log '' $log
    Log ('===== HONEST-STREAM RUNG K={0} =====' -f $K) $log
    Log ('prompt: build/calc-prompt.txt ({0} tokens, non-echoing) | pp-tokens {1}' -f $ppTokens, $ppTokens) $log
    Log ('honest baseline (recorded 2026-09-14 re-probe): fetches={0} rounds={1} staged={2} per-round={3}' -f `
         $baseFetches, $baseRounds, $baseStaged, $basePerRound) $log
    Log ('honest statistic (b) at K={0}: {1:N6} ({2}/{3} rows) -> PRE-REGISTERED predicted staged/round {4}' -f `
         $K, $predB, $predHits, $predReq, $predByteTarget) $log
    Log ('recorded honest run: route digest {0}... tg ids {1}' -f $baseIdent.digest.Substring(0, 16), $baseIdent.tg_ids.Substring(0, [math]::Min(24, $baseIdent.tg_ids.Length))) $log

    $arms = @(
        [ordered]@{ name = 'A1'; retain = 0 },
        [ordered]@{ name = 'B1'; retain = $K },
        [ordered]@{ name = 'A2'; retain = 0 },
        [ordered]@{ name = 'B2'; retain = $K }
    )
    foreach ($a in $arms) {
        $a.trace = Join-Path $rungRoot ('arm-{0}.jsonl' -f $a.name)
        $a.log   = Join-Path $rungRoot ('arm-{0}.log'   -f $a.name)
        $a.txt   = Join-Path $rungRoot ('arm-{0}.txt'   -f $a.name)
        $a.csv   = Join-Path $rungRoot ('mem-{0}.csv' -f $a.name)
        foreach ($p in @($a.trace, $a.log, $a.txt, $a.csv)) {
            if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
        }
    }

    $mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
    $acquired = $false
    try {
        Log 'waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1' $log
        try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
        catch [System.Threading.AbandonedMutexException] { $acquired = $true; Log 'recovered an abandoned GPU mutex' $log }
        if (-not $acquired) { throw 'timed out waiting for the GPU mutex after 3600 s' }
        Log 'acquired GPU mutex (one session for all four arms of this rung)' $log
        $smi = (& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1) -join '; '
        Log ('GPU state at acquisition: {0}' -f $smi) $log

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
            Log (("arm {0} (retain={1}): " -f $a.name, $a.retain) + ($argList -join ' ')) $log

            $poll = Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru -ArgumentList @(
                '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $poller,
                '-Out', $a.csv, '-IntervalMs', '500', '-DurationSeconds', '900')
            Log ("poller attached (pid {0}) -> {1}" -f $poll.Id, $a.csv) $log

            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            $old = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try { $all = & $exe @argList 2>&1 }
            finally { $ErrorActionPreference = $old }
            $a.exit = $LASTEXITCODE
            $sw.Stop()
            $a.wall_ms = [math]::Round($sw.Elapsed.TotalMilliseconds, 1)
            [System.IO.File]::WriteAllLines($a.log, @($all | ForEach-Object { $_.ToString() }))
            Log ("arm {0} exit {1} in {2} ms" -f $a.name, $a.exit, $a.wall_ms) $log

            for ($i = 0; $i -lt 90; $i++) {
                if ($null -eq (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) { break }
                Start-Sleep -Milliseconds 500
            }
            if ($null -ne (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) {
                Stop-Process -Id $poll.Id -Force -ErrorAction Continue
                Start-Sleep -Milliseconds 500
            }
        }
    } finally {
        if ($acquired) { $mutex.ReleaseMutex(); Log 'released GPU mutex' $log }
        $mutex.Dispose()
    }

    # --- per-arm metrics, criteria, and the echoed-vs-honest comparison ---------
    $rows = @()
    foreach ($a in $arms) {
        $s = Get-ArmStats $a.trace $a.log
        $c = Invoke-Capture @($exe, '--check-telemetry', $a.trace)
        $a.checker_exit = $c.code
        $rows += [pscustomobject]@{
            rung = $K; arm = $a.name; retain = $a.retain; exit = $a.exit; checker = $c.code
            wall_ms = $a.wall_ms; pp_tok_s = $s.pp_tok_s; tg_tok_s = $s.tg_tok_s
            retain_k = $s.retain_k; n_fetch = $s.n_fetch; n_round = $s.n_round; refused = $s.refused
            staged_bytes = $s.staged_bytes
            staged_per_round = [math]::Round($s.staged_bytes / [math]::Max(1, $s.n_round), 0)
            rows_fetched = $s.rows_fetched; rows_retained = $s.rows_retained
            rows_retired = $s.rows_retired; layers_with_reuse = $s.layers_with_reuse
            transfer_mean_us = [math]::Round($s.transfer_sum / [math]::Max(1, $s.n_fetch), 1)
            ready_mean_us = [math]::Round($s.ready_sum / [math]::Max(1, $s.n_fetch), 1)
            peak_vram_bytes = $s.peak_vram_bytes
            route_digest = $s.route_digest; tg_ids = $s.tg_ids
            cache_alloc = $s.cache_alloc_mb
        }
    }
    [System.IO.File]::WriteAllText((Join-Path $rungRoot 'summary.json'), ($rows | ConvertTo-Json -Depth 4),
        (New-Object System.Text.UTF8Encoding($false)))

    Log '' $log
    Log '=== HONEST A/B TABLE ===' $log
    Log ("{0,-4} {1,-7} {2,-5} {3,-8} {4,-9} {5,-9} {6,-15} {7,-10} {8,-9} {9,-8}" -f `
         'arm','retain','exit','checker','pp t/s','tg t/s','staged/round','rows_fet','rows_ret','refused') $log
    foreach ($r in $rows) {
        Log ("{0,-4} {1,-7} {2,-5} {3,-8} {4,-9} {5,-9} {6,-15} {7,-10} {8,-9} {9,-8}" -f `
             $r.arm, $r.retain, $r.exit, $r.checker, $r.pp_tok_s, $r.tg_tok_s, $r.staged_per_round,
             $r.rows_fetched, $r.rows_retained, $r.refused) $log
    }

    $A = @($rows | Where-Object { $_.retain -eq 0 })
    $B = @($rows | Where-Object { $_.retain -eq $K })
    $same_route = (($rows | Select-Object -ExpandProperty route_digest | Select-Object -Unique).Count -eq 1)
    $same_ids   = (($rows | Select-Object -ExpandProperty tg_ids | Select-Object -Unique).Count -eq 1)
    $all_ok     = (@($rows | Where-Object { $_.exit -ne 0 -or $_.checker -ne 0 }).Count -eq 0)
    # The A arms must reproduce the RECORDED honest run, not just each other:
    # same route digest AND same greedy ids, which is a far harder test than
    # matching a byte count.
    $base_match = (@($A | Where-Object { $_.n_fetch -ne $baseFetches -or $_.staged_per_round -ne $basePerRound -or $_.rows_retained -ne 0 }).Count -eq 0)
    $ident_match = (@($A | Where-Object { $_.route_digest -ne $baseIdent.digest -or $_.tg_ids -ne $baseIdent.tg_ids }).Count -eq 0)

    Log '' $log
    Log ('route stream identical across the four arms: {0}' -f $same_route) $log
    Log ('greedy token ids identical across the four arms: {0}' -f $same_ids) $log
    Log ('reactive arms match the recorded honest baseline (fetches={0}, staged/round={1}, retained=0): {2}' -f `
         $baseFetches, $basePerRound, $base_match) $log
    Log ('reactive arms reproduce the recorded honest run (route digest AND greedy ids): {0}' -f $ident_match) $log
    $crit1 = $all_ok -and $same_route -and $same_ids -and $base_match -and $ident_match
    Log ("criterion 1 (four clean arms, parity, recorded-baseline reproduction): {0}" -f $(if ($crit1) { 'PASS' } else { 'FAIL' })) $log

    $bMean = ($B | Measure-Object staged_per_round -Average).Average
    $byteDelta = ($bMean - $predByteTarget) / $predByteTarget
    $byteOk = [math]::Abs($byteDelta) -le 0.02
    $reuseOk = (@($B | Where-Object { $_.rows_retained -le 0 -or $_.layers_with_reuse -lt 40 }).Count -eq 0)
    $crit2 = $byteOk -and $reuseOk
    Log ("criterion 2 (bytes vs PRE-REGISTERED honest (b) prediction): staged/round mean {0} vs predicted {1} -> {2:P3} (within 2%: {3}) | cross-token reuse on all 40 layers: {4} -> {5}" -f `
         [math]::Round($bMean, 0), $predByteTarget, $byteDelta, $byteOk, $reuseOk, $(if ($crit2) { 'PASS' } else { 'FAIL' })) $log

    $aTg = @($A | ForEach-Object { [double]$_.tg_tok_s })
    $bTg = @($B | ForEach-Object { [double]$_.tg_tok_s })
    $aMeanTg = ($aTg | Measure-Object -Average).Average
    $bMeanTg = ($bTg | Measure-Object -Average).Average
    $gain = $bMeanTg - $aMeanTg
    $within = [math]::Max([math]::Abs($aTg[0] - $aTg[1]), [math]::Abs($bTg[0] - $bTg[1]))
    $aBytes = ($A | Measure-Object staged_per_round -Average).Average
    $savedB = $aBytes - $bMean
    $savedMs = (1000.0 / $aMeanTg) - (1000.0 / $bMeanTg)
    $convGbs = if ($savedMs -gt 0) { ($savedB / ($savedMs / 1000.0)) / 1e9 } else { 0.0 }

    Log ("HONEST performance at K={0}: reactive mean {1:N3} tok/s | retained mean {2:N3} tok/s | gain {3:N3} | within-arm spread {4:N3} | gain exceeds spread: {5}" -f `
         $K, $aMeanTg, $bMeanTg, $gain, $within, $(if ($gain -gt $within) { 'YES' } else { 'NO' })) $log
    Log ("HONEST byte-to-time conversion at K={0}: {1:N0} B saved per token over {2:N2} ms = {3:N2} GB/s" -f `
         $K, $savedB, $savedMs, $convGbs) $log

    # The point of the whole exercise: this rung's gain against the RECORDED
    # ECHOED rung at the same depth. Read as a comparison of one number measured
    # two ways, never as a new gate verdict.
    $echoPath = Join-Path $root "build\llama-cuda\gplr19-k$K-ab\summary.json"
    if (Test-Path -LiteralPath $echoPath) {
        $echo = Get-Content -LiteralPath $echoPath -Raw | ConvertFrom-Json
        $eA = @($echo | Where-Object { $_.retain -eq 0 })
        $eB = @($echo | Where-Object { $_.retain -eq $K })
        $eAMean = (@($eA | ForEach-Object { [double]$_.tg_tok_s }) | Measure-Object -Average).Average
        $eBMean = (@($eB | ForEach-Object { [double]$_.tg_tok_s }) | Measure-Object -Average).Average
        $eGain  = $eBMean - $eAMean
        $eBytes = ($eB | Measure-Object staged_per_round -Average).Average
        $echoB  = [double]$anWrs.verdict_statistic_b."$K".value   # honest, for the delta
        Log ("ECHOED reference (recorded, gplr19-k{0}-ab): reactive {1:N3} | retained {2:N3} | gain {3:N3} | staged/round {4}" -f `
             $K, $eAMean, $eBMean, $eGain, $eBytes) $log
        Log ("ECHO COMPRESSION at K={0}: honest gain {1:N3} vs echoed gain {2:N3} -> delta {3:N3} tok/s ({4:P1} of the echoed gain); honest staged/round {5} vs echoed {6} ({7:P2} more bytes staged on the honest stream)" -f `
             $K, $gain, $eGain, ($gain - $eGain), (($gain - $eGain) / $eGain), [math]::Round($bMean, 0), $eBytes, (($bMean - $eBytes) / $eBytes)) $log
    } else {
        Log ("ECHOED reference missing at {0} -- no comparison recorded for this rung" -f $echoPath) $log
    }

    $allRungs += [pscustomobject]@{
        rung = $K; out_root = $rungRoot
        honest_b = $predB; predicted_staged_per_round = $predByteTarget
        honest_reactive_tg = $aMeanTg; honest_retained_tg = $bMeanTg; honest_gain = $gain
        honest_staged_per_round = [math]::Round($bMean, 0); conversion_gbs = [math]::Round($convGbs, 2)
        crit1 = $crit1; crit2 = $crit2
    }
}

Log '' $log
[System.IO.File]::WriteAllText((Join-Path $root 'build\llama-cuda\gplr19-honest-summary.json'),
    ($allRungs | ConvertTo-Json -Depth 4), (New-Object System.Text.UTF8Encoding($false)))
Write-Host ''
Write-Host '=== RUNG SUMMARY (honest stream) ==='
$allRungs | Format-Table rung, honest_b, predicted_staged_per_round, honest_staged_per_round, honest_reactive_tg, honest_retained_tg, honest_gain, conversion_gbs, crit1, crit2 | Out-String | Write-Host
