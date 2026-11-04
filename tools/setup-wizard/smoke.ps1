# tools/setup-wizard/smoke.ps1
# E2E server smoke for the setup wizard's emitted run-server.ps1 (plan AC-6).
#
# What it does, in order:
#   1. Emits a fresh run-server.ps1 in the repo via setup.ps1 (in-repo,
#      build skipped, model resolved from disk) for the given row/model.
#   2. Starts that script as a child process (server runs in the foreground
#      inside it, logging to build\llama-cuda\server-<row>.log).
#   3. Polls http://127.0.0.1:<port>/health until {"status":"ok"} or timeout.
#   4. Sends the pong chat request (USAGE.md section 5, verbatim body with
#      model qwen3.6-35b-a3b) and requires "pong" in the response content.
#   5. Kills the child tree, verifies the process is gone and the server log
#      tail has no access violation.
#
# Run it under the GPU mutex (one explicit run, tag swz-smoke):
#   & tools\with-gpu-lock.ps1 -Command 'powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\smoke.ps1 -Row R5 -Model "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" -Port 8033' -VerdictTag swz-smoke
param(
    [string]$Row = 'R5',
    [Parameter(Mandatory = $true)][string]$Model,
    [int]$Port = 8033,
    [int]$HealthTimeoutSec = 300
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$emit = Join-Path $repo 'run-server.ps1'
$exe  = Join-Path $repo 'build\llama-cuda\bin\Release\llama-server.exe'

# row -> model key (the recipe table is per-model; R5/R6/R7 are qwen36, R1/R2 qwen38, R3 qwen38, R4 flint)
$modelKey = ''
switch ($Row) {
    'R1' { $modelKey = 'qwen38'; $vram = 'full'; $ctx = '262k' }
    'R2' { $modelKey = 'qwen38'; $vram = '16g';  $ctx = '262k' }
    'R3' { $modelKey = 'qwen38'; $vram = '8g';   $ctx = '128k' }
    'R4' { $modelKey = 'flint';  $vram = 'full'; $ctx = '262k' }
    'R5' { $modelKey = 'qwen36'; $vram = 'full'; $ctx = '262k' }
    'R6' { $modelKey = 'qwen36'; $vram = '16g';  $ctx = '262k' }
    'R7' { $modelKey = 'qwen36'; $vram = '8g';   $ctx = '262k' }
    default { Write-Host "smoke: unknown row $Row" -ForegroundColor Red; exit 2 }
}

Write-Host "== smoke: emit run-server.ps1 ($Row, $modelKey, $vram, $ctx)" -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo 'setup.ps1') `
    -NonInteractive -SkipBuild -Model $modelKey -Vram $vram -Context $ctx `
    -ModelFile $Model -Port $Port -NoStart
if ($LASTEXITCODE -ne 0) { Write-Host "smoke: emit failed (exit $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
if (-not (Test-Path $emit)) { Write-Host 'smoke: run-server.ps1 was not written' -ForegroundColor Red; exit 1 }

Write-Host "== smoke: start server (child)" -ForegroundColor Cyan
$child = Start-Process -FilePath 'powershell' -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',$emit `
    -RedirectStandardOutput (Join-Path $repo 'build\llama-cuda\smoke-stdout.log') `
    -RedirectStandardError  (Join-Path $repo 'build\llama-cuda\smoke-stderr.log') -PassThru -WindowStyle Hidden
if (-not $child) { Write-Host 'smoke: failed to start the server child' -ForegroundColor Red; exit 1 }
Write-Host "smoke: server child PID $($child.Id)"

function Stop-ServerTree {
    param([int]$pid)
    & taskkill /T /F /PID $pid 2>$null | Out-Null   # stderr unmerged: EAP=Stop + 2>&1 would throw on the SUCCESS line
    # wait until the whole tree is gone
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        $left = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if (-not $left) { return $true }
    }
    return $false
}

$healthOk = $false
$pongOk = $false
$sw = [Diagnostics.Stopwatch]::StartNew()
try {
    # 3. health poll
    Write-Host "== smoke: poll /health (timeout ${HealthTimeoutSec}s)" -ForegroundColor Cyan
    while ($sw.Elapsed.TotalSeconds -lt $HealthTimeoutSec) {
        if ($child.HasExited) { break }
        try {
            $h = Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "http://127.0.0.1:$Port/health"
            if ($h.StatusCode -eq 200 -and $h.Content -match '"status"\s*:\s*"ok"') { $healthOk = $true; break }
        } catch { }
        Start-Sleep -Seconds 3
    }
    if ($healthOk) {
        Write-Host ("smoke: health ok after {0:N0}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
        # 4. pong (USAGE.md section 5, verbatim body; model name as in the doc)
        $body = '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Reply with exactly one word: pong"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
        $resp = Invoke-WebRequest -UseBasicParsing -TimeoutSec 120 "http://127.0.0.1:$Port/v1/chat/completions" `
            -Method POST -ContentType 'application/json' -Body $body
        $j = $resp.Content | ConvertFrom-Json
        $content = $j.choices[0].message.content
        Write-Host "smoke: model reply: $content"
        if ($content -match 'pong') { $pongOk = $true }
    }
}
finally {
    # 5. stop the server tree
    Write-Host "== smoke: stop server" -ForegroundColor Cyan
    $gone = Stop-ServerTree -pid $child.Id
    if (-not $gone) { Write-Host 'smoke: WARNING - server process still alive after 30s' -ForegroundColor Yellow }
}

# log tail check
$log = Join-Path $repo "build\llama-cuda\server-$Row.log"
$bad = $false
if (Test-Path $log) {
    $tail = Get-Content $log -Tail 40 -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($tail -match 'access violation|0xC0000005|0xC0000409') {
        $bad = $true
        Write-Host 'smoke: server log tail contains a fatal exception:' -ForegroundColor Red
        $tail | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" }
    }
}

Write-Host ''
if ($healthOk -and $pongOk -and $gone -and -not $bad) {
    Write-Host ("SMOKE PASS ($Row): health + pong + clean stop") -ForegroundColor Green
    exit 0
}
Write-Host ("SMOKE FAIL ($Row): health=$healthOk pong=$pongOk stopped=$gone logClean=$(-not $bad)") -ForegroundColor Red
exit 1
