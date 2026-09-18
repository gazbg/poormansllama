# T-GPLR-13 ladder: row-set path at PP 50 / 512 / 4000 / 200000 + qa-smoke-512 regression.
# Every GPU run goes under tools/with-gpu-lock.ps1; tools/gpu-mem-poll.ps1 (read-only,
# host CIM only) is attached to each run for per-size VRAM/RAM evidence.
# Harness-level only: no library changes. Max 2 attempts per size; on failure capture
# evidence (first error line, exit code, last trace events) and move on.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$tok    = Join-Path $root 'build\llama-cuda\bin\Release\llama-tokenize.exe'
$lock   = Join-Path $root 'tools\with-gpu-lock.ps1'
$poller = Join-Path $root 'tools\gpu-mem-poll.ps1'
$gen    = Join-Path $root 'tools\semantic-qa\gen-needle-prompt.py'
$chk    = Join-Path $root 'tools\semantic-qa\check-answer.py'
$py     = 'python'

$moe = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }

foreach ($f in @($exe, $tok, $poller, $gen, $chk)) {
    if (-not (Test-Path -LiteralPath $f)) { throw "missing tool: $f" }
}
if (-not (Test-Path -LiteralPath $moe)) { throw "missing model: $moe" }

$Sizes = @('50', '512', 'qa-smoke-512', '4k', '200k')
if ($args.Count -gt 0) { $Sizes = $args }

$outRoot = Join-Path $root 'build\llama-cuda\gplr13-ladder'
New-Item -ItemType Directory -Force -Path $outRoot | Out-Null
$drvLog = Join-Path $outRoot 'driver.log'

function Log([string]$msg) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss'), $msg)
    Write-Host $line
    Add-Content -LiteralPath $drvLog -Value $line
}

function Get-U64([string]$s) {
    # malformed/error CSV rows must never terminate the driver
    $v = [uint64]0
    if ([string]::IsNullOrWhiteSpace($s)) { return [uint64]0 }
    if ([uint64]::TryParse($s, [ref]$v)) { return $v }
    return [uint64]0
}

function Invoke-Capture {
    # Run a native command capturing stdout+stderr WITHOUT letting Windows
    # PowerShell 5.1 turn stderr records into terminating NativeCommandError
    # under $ErrorActionPreference='Stop'. Returns code + string lines.
    # Argv-array form: positional -File/-Command cannot collide with params.
    param([string[]] $Argv)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $all = & $Argv[0] @($Argv | Select-Object -Skip 1) 2>&1 }
    finally { $ErrorActionPreference = $old }
    $code = $LASTEXITCODE
    $lines = @($all | ForEach-Object { $_.ToString() })
    return [pscustomobject]@{ code = $code; out = $lines }
}

# rowset sizes: pp token count -> poll seconds, lock timeout seconds
$spec = @{
    '50'   = @{ pp = 50;     ubatch = 0;   tg = 3; poll = 240;  lockto = 1500 }
    '512'  = @{ pp = 512;    ubatch = 512; tg = 3; poll = 480;  lockto = 2400 }
    '4k'   = @{ pp = 4000;   ubatch = 512; tg = 3; poll = 1500; lockto = 3600 }
    '200k' = @{ pp = 200000; ubatch = 512; tg = 3; poll = 7500; lockto = 7600 }
}

function Read-CsvLines([string]$path) {
    # The just-stopped poller may hold the handle briefly; retry instead of dying.
    for ($i = 0; $i -lt 60; $i++) {
        try { return [System.IO.File]::ReadAllLines($path) } catch { Start-Sleep -Milliseconds 500 }
    }
    return @()
}

function Get-Metrics([string]$trace, [string]$csv, [string]$log) {
    $m = @{
        tg_tok_s = ''; runner_peak_vram_bytes = $null; rowset_pct = ''; n_fetch = 0; n_round = 0
        ded_peak = 0; ded_name = ''; shared_peak = 0; commit_peak = 0; avail_min = $null
        checker = ''; first_error = ''; trace_tail = ''
    }
    if (Test-Path $log) {
        $t = (Select-String -LiteralPath $log -Pattern 'phase tg :\s*(\d+) tokens, ([\d.]+) tok/s' | Select-Object -Last 1)
        if ($t) { $m.tg_tok_s = $t.Matches[0].Groups[2].Value }
    }
    if (Test-Path $trace) {
        $raw = [System.IO.File]::ReadAllText($trace)
        $mm = [regex]::Matches($raw, '"peak_vram_bytes":(\d+)')
        if ($mm.Count -gt 0) { $m.runner_peak_vram_bytes = [uint64]$mm[$mm.Count - 1].Groups[1].Value }
        $m.n_fetch = [regex]::Matches($raw, '"event":"rowset_fetch"').Count
        $m.n_round = [regex]::Matches($raw, '"event":"rowset_round"').Count
        if ($m.n_round -gt 0) {
            $staged = [uint64]0; $bank = [uint64]0
            foreach ($rm in [regex]::Matches($raw, '\{"event":"rowset_round"[^\n]*')) {
                $bs = [regex]::Match($rm.Value, '"bytes_staged":(\d+)')
                $bb = [regex]::Match($rm.Value, '"bytes_full_bank":(\d+)')
                if ($bs.Success) { $staged += [uint64]$bs.Groups[1].Value }
                if ($bb.Success) { $bank = [uint64]$bb.Groups[1].Value }
            }
            if ($bank -gt 0) { $m.rowset_pct = ('{0:F2}' -f (100.0 * $staged / ($m.n_round * $bank))) }
        }
    }
    if (Test-Path $csv) {
        $sharedByAdapter = @{}
        foreach ($line in (Read-CsvLines $csv)) {
            if ($line.StartsWith('gpu_adapter,')) {
                $nm = [regex]::Match($line, 'name=([^,]+)').Groups[1].Value
                $d  = Get-U64 ([regex]::Match($line, 'dedicated_bytes=(\d+)').Groups[1].Value)
                $s  = Get-U64 ([regex]::Match($line, 'shared_bytes=(\d+)').Groups[1].Value)
                if ($d -gt $m.ded_peak) { $m.ded_peak = $d; $m.ded_name = $nm }
                if (-not $sharedByAdapter.ContainsKey($nm) -or $s -gt $sharedByAdapter[$nm]) { $sharedByAdapter[$nm] = $s }
            } elseif ($line.StartsWith('host_commit,')) {
                $c = Get-U64 ([regex]::Match($line, 'committed_bytes=(\d+)').Groups[1].Value)
                if ($c -gt $m.commit_peak) { $m.commit_peak = $c }
            } elseif ($line.StartsWith('host,')) {
                $a = Get-U64 ([regex]::Match($line, 'avail_phys_bytes=(\d+)').Groups[1].Value)
                if ($null -eq $m.avail_min -or $a -lt $m.avail_min) { $m.avail_min = $a }
            }
        }
        if ($m.ded_name -and $sharedByAdapter.ContainsKey($m.ded_name)) { $m.shared_peak = $sharedByAdapter[$m.ded_name] }
    }
    return $m
}

function Capture-Failure([string]$log, [string]$trace, [int]$exitCode) {
    $ev = @{ exit = $exitCode; first_error = ''; trace_tail = '' }
    if (Test-Path $log) {
        $hit = Select-String -LiteralPath $log -Pattern 'error|failed|refused|abort|exception' -CaseSensitive:$false | Select-Object -First 1
        if ($hit) { $ev.first_error = $hit.Line.Trim() }
        if (-not $ev.first_error) { $ev.first_error = ('(no error line in log; last line: ' + (Get-Content -LiteralPath $log -Tail 1) + ')') }
    }
    if (Test-Path $trace) {
        $lines = [System.IO.File]::ReadAllLines($trace)
        $ev.trace_tail = (($lines | Select-Object -Last 3) -join ' | ')
        if ($ev.trace_tail.Length -gt 600) { $ev.trace_tail = $ev.trace_tail.Substring(0, 600) }
    }
    return $ev
}

function Run-Polled([string]$cmd, [string]$csv, [int]$pollSec, [int]$lockto) {
    $proc = Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $poller,
        '-Out', $csv, '-IntervalMs', '500', '-DurationSeconds', "$pollSec")
    Log ("poller attached (pid {0}) -> {1}" -f $proc.Id, $csv)
    $t0 = Get-Date
    $r = Invoke-Capture @('powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $lock, '-TimeoutSeconds', "$lockto", '-Command', $cmd)
    $exit = $r.code
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Log ("locked run finished: exit {0} in {1} min" -f $exit, $mins)
    # Stop the poller and WAIT until its process object is truly gone; otherwise
    # it keeps the CSV open and every later read fails. Log what actually happened.
    $alive = $null -ne (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)
    if ($alive) {
        Stop-Process -Id $proc.Id -Force -ErrorAction Continue
        for ($i = 0; $i -lt 60; $i++) {
            if ($null -eq (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) { $alive = $false; break }
            Start-Sleep -Milliseconds 500
        }
    }
    Log ("poller stopped (was alive after run: {0}, confirmed gone: {1})" -f $alive, (-not $alive))
    return $exit
}

$results = @()

foreach ($size in $Sizes) {
    $dir = Join-Path $outRoot $size
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Log "=== size $size ==="

    if ($size -eq 'qa-smoke-512') {
        # T-GPLR-07 qa-smoke 512 rung, default-off path (mirrors run-tests.ps1 exactly).
        $prompt = Join-Path $dir 'anchor-512.txt'
        Log 'calibrating 512-token anchor prompt...'
        $g = Invoke-Capture @($py, $gen, '--target-tokens', '512', '--calibrate', $tok, '--model', $moe, '--out', $prompt)
        $genOut = $g.out
        if ($g.code -ne 0) { Log ("calibration FAILED: " + ($genOut -join ' ')); continue }
        $achieved = ($genOut | Select-String 'achieved=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
        $items    = ($genOut | Select-String 'items=(\S+)'   | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
        Log ("prompt achieved=$achieved items=$items")
        $trace  = Join-Path $dir 'qa512.jsonl'
        $tgtext = Join-Path $dir 'qa512.txt'
        $log    = Join-Path $dir 'qa512.log'
        $csv    = Join-Path $dir 'mem.csv'
        if (Test-Path $trace) { Remove-Item $trace -Force }
        $cmd = "`"$exe`" --model `"$moe`" --prompt-file `"$prompt`" --ubatch $achieved --tg-tokens 96 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --no-handoff --allow-full-residency --telemetry `"$trace`" --tg-text `"$tgtext`" > `"$log`" 2>&1"
        $verdict = 'FAIL'; $ev = $null; $mt = $null; $answer = 'SKIP'; $neg = 'SKIP'
        foreach ($attempt in 1..2) {
            Log "attempt $attempt"
            $exit = Run-Polled $cmd $csv 600 1800
            $mt = Get-Metrics $trace $csv $log
            $c = Invoke-Capture @($exe, '--check-telemetry', $trace)
            $chkOut = $c.out
            $mt.checker = ($chkOut | Where-Object { $_ -like '*check-telemetry*' } | Select-Object -First 1)
            if ($exit -eq 0 -and $c.code -eq 0) {
                $gr = Invoke-Capture @($py, $chk, 'check', '--tg-text', $tgtext, '--items', $items)
                if ($gr.code -eq 0) {
                    $answer = 'PASS'
                    $ng = Invoke-Capture @($py, $chk, 'check', '--tg-text', $tgtext, '--items', $items, '--negative')
                    $negOut = $ng.out
                    if ($ng.code -ne 0 -and (($negOut -join ' ') -match 'aggregate=FAIL')) { $neg = 'FAILS' }
                    $verdict = 'PASS'; break
                } else { $answer = 'FAIL'; $ev = @{ exit = $exit; first_error = 'answer grading failed'; trace_tail = '' } }
            } else {
                $ev = Capture-Failure $log $trace $exit
                $ev.first_error = ("exit $exit : " + $ev.first_error)
            }
            if ($attempt -lt 2) { Log "retrying after failure: $($ev.first_error)" }
        }
        $results += [pscustomobject]@{ size = $size; run_exit = if ($verdict -eq 'PASS') { 0 } else { $ev.exit }; checker = $mt.checker; tg_tok_s = $mt.tg_tok_s
            runner_peak_vram_bytes = $mt.runner_peak_vram_bytes; poller_ded_peak = $mt.ded_peak; poller_ded_name = $mt.ded_name; poller_shared_peak = $mt.shared_peak
            commit_peak = $mt.commit_peak; avail_min = $mt.avail_min; rowset_pct = $mt.rowset_pct; n_fetch = $mt.n_fetch; n_round = $mt.n_round
            answer = $answer; negative = $neg; attempts = $attempt; verdict = $verdict; first_error = $ev.first_error; trace_tail = $ev.trace_tail }
        Log "size $size verdict: $verdict (answer=$answer negative=$neg)"
        continue
    }

    # row-set ladder size
    $s = $spec[$size]
    $ub = $s.ubatch; if ($ub -eq 0) { $ub = $s.pp }
    $prompt = Join-Path $dir ("anchor-{0}.txt" -f $s.pp)
    Log ("calibrating {0}-token prompt..." -f $s.pp)
    $g = Invoke-Capture @($py, $gen, '--target-tokens', "$($s.pp)", '--calibrate', $tok, '--model', $moe, '--out', $prompt)
    $genOut = $g.out
    if ($g.code -ne 0) { Log ("calibration FAILED: " + ($genOut -join ' ')); $results += [pscustomobject]@{ size = $size; verdict = 'FAIL'; first_error = 'prompt calibration failed' }; continue }
    $achieved = ($genOut | Select-String 'achieved=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
    Log ("prompt achieved=$achieved tokens")
    $trace = Join-Path $dir ('rowset-{0}.jsonl' -f $s.pp)
    $log   = Join-Path $dir ('rowset-{0}.log' -f $s.pp)
    $csv   = Join-Path $dir 'mem.csv'
    if (Test-Path $trace) { Remove-Item $trace -Force }
    $cmd = "`"$exe`" --model `"$moe`" --prompt-file `"$prompt`" --residency moe-expert-bank --residency-tg " +
           "--pp-tokens $achieved --ubatch $ub --tg-tokens $($s.tg) --layer-slots 4 --kv-cache q8 " +
           "--greedy --no-mtp --pp-no-logits --allow-full-residency --telemetry `"$trace`" > `"$log`" 2>&1"
    Log "cmd: $cmd"
    $verdict = 'FAIL'; $ev = $null; $mt = $null
    foreach ($attempt in 1..2) {
        Log "attempt $attempt"
        $exit = Run-Polled $cmd $csv $s.poll $s.lockto
        $mt = Get-Metrics $trace $csv $log
        $c = Invoke-Capture @($exe, '--check-telemetry', $trace)
        $chkOut = $c.out
        $chkExit = $c.code
        $mt.checker = ($chkOut | Where-Object { $_ -like '*check-telemetry*' } | Select-Object -First 1)
        if ($exit -eq 0 -and $chkExit -eq 0) { $verdict = 'PASS'; break }
        $ev = Capture-Failure $log $trace $exit
        $ev.first_error = ("run exit $exit / check exit $chkExit : " + $ev.first_error)
        Log "attempt $attempt failed: $($ev.first_error)"
    }
    $results += [pscustomobject]@{ size = $size; run_exit = $exit; checker = $mt.checker; tg_tok_s = $mt.tg_tok_s
        runner_peak_vram_bytes = $mt.runner_peak_vram_bytes; poller_ded_peak = $mt.ded_peak; poller_ded_name = $mt.ded_name; poller_shared_peak = $mt.shared_peak
        commit_peak = $mt.commit_peak; avail_min = $mt.avail_min; rowset_pct = $mt.rowset_pct; n_fetch = $mt.n_fetch; n_round = $mt.n_round
        answer = ''; negative = ''; attempts = $attempt; verdict = $verdict; first_error = $ev.first_error; trace_tail = $ev.trace_tail }
    Log "size $size verdict: $verdict"
}

# summary + table
# Unit convention: commitGB/availGB columns are GiB (PowerShell 1GB = 2^30).
# The mem.csv byte values are authoritative; the Python analysis file labels
# the same bytes as decimal GB. Cross-doc comparisons must convert.
$summary = Join-Path $outRoot 'summary.json'
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summary -Encoding UTF8
Log "wrote $summary"

Log ""
Log ("{0,-12} {1,-5} {2,-28} {3,-9} {4,-12} {5,-12} {6,-10} {7,-9} {8,-9} {9,-8} {10,-6} {11,-6} {12,-8}" -f `
    'size','exit','checker','tg tok/s','runnerVRAM','pollerDED','sharedDED','commitGB','availGB','rowset%','fetch','round','verdict')
foreach ($r in $results) {
    $rv = if ($r.runner_peak_vram_bytes) { '{0:N0} MiB' -f ($r.runner_peak_vram_bytes / 1MB) } else { '-' }
    $pd = if ($r.poller_ded_peak) { '{0:N0} MiB' -f ($r.poller_ded_peak / 1MB) } else { '-' }
    $ps = if ($r.poller_shared_peak) { '{0:N0} MiB' -f ($r.poller_shared_peak / 1MB) } else { '-' }
    $cg = if ($r.commit_peak) { '{0:F1}' -f ($r.commit_peak / 1GB) } else { '-' }
    $ag = if ($r.avail_min) { '{0:F1}' -f ($r.avail_min / 1GB) } else { '-' }
    Log ("{0,-12} {1,-5} {2,-28} {3,-9} {4,-12} {5,-12} {6,-10} {7,-9} {8,-9} {9,-8} {10,-6} {11,-6} {12,-8}" -f `
        $r.size, $r.run_exit, ($r.checker -replace 'check-telemetry:\s*',''), $r.tg_tok_s, $rv, $pd, $ps, $cg, $ag, $r.rowset_pct, $r.n_fetch, $r.n_round, $r.verdict)
}
foreach ($r in $results | Where-Object { $_.verdict -ne 'PASS' }) {
    Log "FAILURE [$($r.size)] $($r.first_error)"
    if ($r.trace_tail) { Log "  trace tail: $($r.trace_tail)" }
}
Log "LADDER DONE"
