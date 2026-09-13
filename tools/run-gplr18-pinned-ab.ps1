# T-GPLR-18: paired 512 A/B for row-set staging through the pinned host path.
# Four arms (direct, pinned, direct, pinned) run back to back inside ONE
# acquisition of the shared-GPU mutex, so the pairs see one GPU session;
# tools/gpu-mem-poll.ps1 is attached per arm. The arms differ ONLY by
# --copy-source; every other flag is the T-GPLR-13 512 ladder shape.
# Same mutex name and wait/release behavior as tools/with-gpu-lock.ps1; this
# driver acquires it once because the method requires one session per A/B/A/B.
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$tok    = Join-Path $root 'build\llama-cuda\bin\Release\llama-tokenize.exe'
$poller = Join-Path $root 'tools\gpu-mem-poll.ps1'
$gen    = Join-Path $root 'tools\semantic-qa\gen-needle-prompt.py'
$chk    = Join-Path $root 'tools\semantic-qa\check-answer.py'
$py     = 'python'
$moe    = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }

foreach ($f in @($exe, $tok, $poller, $gen, $chk)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing tool: $f" }
}
if (-not (Test-Path -LiteralPath $moe)) { throw "missing model: $moe" }

$outRoot = Join-Path $root 'build\llama-cuda\gplr18-pinned-ab'
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

# --- 512 anchor prompt (host only; no GPU) -------------------------------
$prompt = Join-Path $outRoot 'anchor-512.txt'
Log 'calibrating 512-token anchor prompt...'
$g = Invoke-Capture @($py, $gen, '--target-tokens', '512', '--calibrate', $tok, '--model', $moe, '--out', $prompt)
if ($g.code -ne 0) { Log ('calibration FAILED: ' + ($g.out -join ' ')); throw 'prompt calibration failed' }
$achieved = ($g.out | Select-String 'achieved=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
$items    = ($g.out | Select-String 'items=(\S+)'   | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
Log ("prompt achieved=$achieved items=$items")

# --- four arms, one mutex session ----------------------------------------
$arms = @(
    [ordered]@{ name = 'A1'; copy_source = 'direct' },
    [ordered]@{ name = 'B1'; copy_source = 'pinned' },
    [ordered]@{ name = 'A2'; copy_source = 'direct' },
    [ordered]@{ name = 'B2'; copy_source = 'pinned' }
)
foreach ($a in $arms) {
    $a.trace = Join-Path $outRoot ('arm-{0}.jsonl' -f $a.name)
    $a.log   = Join-Path $outRoot ('arm-{0}.log'   -f $a.name)
    $a.txt   = Join-Path $outRoot ('arm-{0}.txt'   -f $a.name)
    $a.csv   = Join-Path $outRoot ('mem-{0}.csv'   -f $a.name)
    foreach ($p in @($a.trace, $a.log, $a.txt, $a.csv)) {
        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
    }
}

$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
try {
    Log 'waiting for GPU mutex Global\AI-GPU-EXCLUSIVE-v1'
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(2400)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Log 'recovered an abandoned GPU mutex' }
    if (-not $acquired) { throw 'timed out waiting for the GPU mutex after 2400 s' }
    Log 'acquired GPU mutex (one session for all four arms)'

    foreach ($a in $arms) {
        $argList = @(
            '--model', $moe, '--prompt-file', $prompt,
            '--residency', 'moe-expert-bank', '--residency-tg',
            '--pp-tokens', "$achieved", '--ubatch', '512', '--tg-tokens', '3',
            '--layer-slots', '4', '--kv-cache', 'q8', '--greedy', '--no-mtp',
            '--pp-no-logits', '--allow-full-residency',
            '--copy-source', $a.copy_source,
            '--telemetry', $a.trace, '--tg-text', $a.txt
        )
        Log ("arm {0} ({1}): {2}" -f $a.name, $a.copy_source, ($argList -join ' '))

        $poll = Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru -ArgumentList @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $poller,
            '-Out', $a.csv, '-IntervalMs', '500', '-DurationSeconds', '600')
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

        # the poller may hold the CSV briefly; wait for it to exit before reading
        for ($i = 0; $i -lt 60; $i++) {
            if ($null -eq (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 500
        }
        if ($null -ne (Get-Process -Id $poll.Id -ErrorAction SilentlyContinue)) {
            Stop-Process -Id $poll.Id -Force -ErrorAction Continue
            Start-Sleep -Milliseconds 500
        }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Log 'released GPU mutex' }
    $mutex.Dispose()
}

# --- per-arm metrics -----------------------------------------------------
function Get-ArmStats([string]$trace, [string]$log) {
    $m = [ordered]@{
        n_fetch = 0; staged_bytes = 0; rows_fetched = 0; rows_resident = 0
        submit_sum = 0.0; transfer_sum = 0.0; ready_sum = 0.0
        submit_min = [double]::MaxValue; submit_max = 0.0
        transfer_min = [double]::MaxValue; transfer_max = 0.0
        route_digest = ''; tg_ids = ''; peak_vram_bytes = $null
        pp_tok_s = ''; tg_tok_s = ''; refused = 0
    }
    if (Test-Path -LiteralPath $trace) {
        $raw = [System.IO.File]::ReadAllText($trace)
        $m.refused = [regex]::Matches($raw, '"event":"rowset_refused"').Count
        $parts = New-Object System.Collections.Generic.List[string]
        foreach ($fm in [regex]::Matches($raw, '\{"event":"rowset_fetch"[^\n]*')) {
            $line = $fm.Value
            $m.n_fetch++
            $s = [double][regex]::Match($line, '"host_submit_us":([\d.]+)').Groups[1].Value
            $t = [double][regex]::Match($line, '"transfer_us":([\d.]+)').Groups[1].Value
            $r = [double][regex]::Match($line, '"ready_latency_us":([\d.]+)').Groups[1].Value
            $m.submit_sum += $s; $m.transfer_sum += $t; $m.ready_sum += $r
            if ($s -lt $m.submit_min) { $m.submit_min = $s }
            if ($s -gt $m.submit_max) { $m.submit_max = $s }
            if ($t -lt $m.transfer_min) { $m.transfer_min = $t }
            if ($t -gt $m.transfer_max) { $m.transfer_max = $t }
            $m.staged_bytes += [uint64][regex]::Match($line, '"bytes_staged":(\d+)').Groups[1].Value
            $m.rows_fetched += [uint64][regex]::Match($line, '"n_rows_fetched":(\d+)').Groups[1].Value
            $m.rows_resident += [uint64][regex]::Match($line, '"n_rows_resident":(\d+)').Groups[1].Value
            $layer = [regex]::Match($line, '"layer":(\d+)').Groups[1].Value
            $uid   = [regex]::Match($line, '"union_ids":"([^"]*)"').Groups[1].Value
            $hash  = [regex]::Match($line, '"hash":(\d+)').Groups[1].Value
            $parts.Add(('{0}|{1}|{2}' -f $layer, $uid, $hash))
        }
        $joined = ($parts -join ';')
        $sha = [System.Security.Cryptography.SHA256]::Create()
        $m.route_digest = ([System.BitConverter]::ToString($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($joined))) -replace '-', '').ToLower()
        $tgm = [regex]::Match($raw, '"event":"tg_tokens"[^\n]*')
        if ($tgm.Success) { $m.tg_ids = [regex]::Match($tgm.Value, '"ids":\[([^\]]*)\]').Groups[1].Value }
        $vm = [regex]::Matches($raw, '"peak_vram_bytes":(\d+)')
        if ($vm.Count -gt 0) { $m.peak_vram_bytes = [uint64]$vm[$vm.Count - 1].Groups[1].Value }
    }
    if (Test-Path -LiteralPath $log) {
        $raw = [System.IO.File]::ReadAllText($log)
        $pp = [regex]::Match($raw, 'phase pp\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($pp.Success) { $m.pp_tok_s = $pp.Groups[1].Value }
        $tg = [regex]::Match($raw, 'phase tg\s*:\s*\d+ tokens,\s*([\d.]+) tok/s')
        if ($tg.Success) { $m.tg_tok_s = $tg.Groups[1].Value }
    }
    return $m
}

$rows = @()
foreach ($a in $arms) {
    $s = Get-ArmStats $a.trace $a.log
    $c = Invoke-Capture @($exe, '--check-telemetry', $a.trace)
    $a.checker_exit = $c.code
    $a.checker_line = (($c.out | Select-String 'check-telemetry' | Select-Object -First 1) -as [string])
    $a.stats = $s
    $n = [math]::Max(1, $s.n_fetch)
    $rows += [pscustomobject]@{
        arm = $a.name; copy_source = $a.copy_source; exit = $a.exit; checker = $c.code
        wall_ms = $a.wall_ms; pp_tok_s = $s.pp_tok_s; tg_tok_s = $s.tg_tok_s
        n_fetch = $s.n_fetch; refused = $s.refused
        staged_bytes = $s.staged_bytes; rows_fetched = $s.rows_fetched; rows_resident = $s.rows_resident
        submit_us = [math]::Round($s.submit_sum / $n, 1)
        ready_us  = [math]::Round($s.ready_sum  / $n, 1)
        transfer_us = [math]::Round($s.transfer_sum / $n, 1)
        transfer_min = [math]::Round($s.transfer_min, 1); transfer_max = [math]::Round($s.transfer_max, 1)
        peak_vram_bytes = $s.peak_vram_bytes
        route_digest = $s.route_digest; tg_ids = $s.tg_ids
    }
}

$rows | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $outRoot 'summary.json') -Encoding UTF8

Log ''
Log '=== A/B table (per-fetch means us; wall ms) ==='
Log ("{0,-4} {1,-8} {2,-5} {3,-8} {4,-10} {5,-10} {6,-11} {7,-11} {8,-11} {9,-12} {10,-8}" -f `
     'arm','source','exit','checker','wall_ms','pp tok/s','tg tok/s','submit_us','ready_us','transfer_us','fetches')
foreach ($r in $rows) {
    Log ("{0,-4} {1,-8} {2,-5} {3,-8} {4,-10} {5,-10} {6,-11} {7,-11} {8,-11} {9,-12} {10,-8}" -f `
         $r.arm, $r.copy_source, $r.exit, $r.checker, $r.wall_ms, $r.pp_tok_s, $r.tg_tok_s, $r.submit_us, $r.ready_us, $r.transfer_us, $r.n_fetch)
}

$direct = @($rows | Where-Object { $_.copy_source -eq 'direct' })
$pinned = @($rows | Where-Object { $_.copy_source -eq 'pinned' })
$dmean = ($direct | Measure-Object transfer_us -Average).Average
$pmean = ($pinned | Measure-Object transfer_us -Average).Average
$dspread = $direct[0].transfer_us - $direct[1].transfer_us
$pspread = $pinned[0].transfer_us - $pinned[1].transfer_us
$within = [math]::Max([math]::Abs($dspread), [math]::Abs($pspread))
$same_staged = (($rows | Select-Object -ExpandProperty staged_bytes | Select-Object -Unique).Count -eq 1)
$same_route  = (($rows | Select-Object -ExpandProperty route_digest   | Select-Object -Unique).Count -eq 1)
$same_ids    = (($rows | Select-Object -ExpandProperty tg_ids         | Select-Object -Unique).Count -eq 1)

Log ''
Log ('direct transfer mean {0} us | pinned transfer mean {1} us | delta {2} us' -f [math]::Round($dmean,1), [math]::Round($pmean,1), [math]::Round($dmean - $pmean,1))
Log ('largest within-arm transfer spread {0} us' -f [math]::Round($within,1))
Log ('parity: staged bytes identical {0} | route stream identical {1} | greedy token ids identical {2}' -f $same_staged, $same_route, $same_ids)

$feasible = (@($rows | Where-Object { $_.exit -ne 0 -or $_.checker -ne 0 }).Count -eq 0) -and $same_staged -and $same_route -and $same_ids
$gate2 = $feasible -and ($dmean - $pmean -gt $within)
Log ("criterion 1 feasibility: {0}" -f $(if ($feasible) { 'PASS' } else { 'FAIL' }))
Log ("criterion 2 instrumentation gate (pinned transfer below direct by more than the within-arm spread): {0}" -f $(if ($gate2) { 'PASS' } else { 'FAIL / falsified-by-evidence' }))
Log 'GPLR18 AB DONE'
