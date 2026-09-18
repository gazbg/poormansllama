# T-GPLR-13: checker negative controls (schema draft section 8), DUAL PATH.
# Tamper the good row-set trace and prove each row-set rule fails loudly on
# BOTH validators: --check-telemetry and --validate-telemetry (they must stay
# identical on rowset events). Host-only (pure JSONL parsing) -- no GPU mutex.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe  = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$tel  = Join-Path $root 'build\llama-cuda\gplr13\rowset-pp64-tg3.jsonl'
$work = Join-Path $root 'build\llama-cuda\gplr13\negfix'
New-Item -ItemType Directory -Force -Path $work | Out-Null
if (-not (Test-Path $tel)) { throw "missing good trace: $tel" }

$script:fail = 0

function Invoke-Capture([string[]]$Argv) {
    # Windows PowerShell 5.1 turns native stderr into terminating errors under
    # Stop; capture instead (the checker reports verdicts on stderr).
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $all = & $Argv[0] @($Argv | Select-Object -Skip 1) 2>&1 }
    finally { $ErrorActionPreference = $old }
    $lines = @($all | ForEach-Object { $_.ToString() })
    return [pscustomobject]@{ code = $LASTEXITCODE; out = $lines }
}

function Run-Check([string]$mode, [string]$file) {
    return Invoke-Capture @($exe, "--$mode", $file)
}

function Expect([string]$name, [int]$code, [int]$wantCode, [string]$msgs, [string]$needle) {
    if ($code -ne $wantCode) {
        Write-Host "FAIL $name : exit $code (want $wantCode)"; $script:fail++
    } elseif ($needle -and ($msgs -notlike "*$needle*")) {
        Write-Host "FAIL $name : message missing '$needle'"; $script:fail++
    } else {
        Write-Host "PASS $name"
    }
}

$lines = [System.IO.File]::ReadAllLines($tel)

function Tamper([string]$name, [scriptblock]$mutate) {
    $out = Join-Path $work ("{0}.jsonl" -f $name)
    $arr = [System.IO.File]::ReadAllLines($tel)
    & $mutate $arr
    [System.IO.File]::WriteAllLines($out, $arr)
    return $out
}

# --- fixtures: name -> { file, expected-error (identical on both paths) ---
$fixtures = @{}

# positive control: the untouched good trace must pass BOTH validators
$r = Run-Check 'check-telemetry' $tel
Expect 'base-good-trace [check]' $r.code 0 (($r.out -join "`n")) ''
$r = Run-Check 'validate-telemetry' $tel
Expect 'base-good-trace [validate]' $r.code 0 (($r.out -join "`n")) ''

# 1. hash integrity (schema 4.6): corrupt one hash
$f = Tamper 'hash-mismatch' { param($a)
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_fetch"*') {
            $a[$i] = $a[$i] -replace '"hash":\d+', '"hash":12345'; break
        }
    } }
$fixtures['hash-mismatch'] = @{ file = $f; msg = 'rowset_fetch hash mismatch over union_ids' }

# 2. accounting identity (schema 4.3): fake resident rows on first window
$f = Tamper 'identity-broken' { param($a)
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_fetch"*') {
            $a[$i] = $a[$i] -replace '"n_rows_resident":0', '"n_rows_resident":1'; break
        }
    } }
$fixtures['identity-broken'] = @{ file = $f; msg = 'route_union != n_rows_fetched + n_rows_resident' }

# 3. W=1 identity (schema 4.2): route_union != top_k
$f = Tamper 'w1-identity' { param($a)
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_fetch"*') {
            $a[$i] = $a[$i] -replace '"route_union":8', '"route_union":9'; break
        }
    } }
$fixtures['w1-identity'] = @{ file = $f; msg = 'route_union != top_k at W=1' }

# 3b. window_tokens (validator-parity rule): declare W=2 in rowset_init
$f = Tamper 'window-tokens' { param($a)
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_init"*') {
            $a[$i] = $a[$i] -replace '"window_tokens":1', '"window_tokens":2'; break
        }
    } }
$fixtures['window-tokens'] = @{ file = $f; msg = 'rowset_init window_tokens != 1 (W>1 is not implemented)' }

# 4a. generation lifecycle (schema 4.5): first-window phantom resident rows.
# With 10 layers per slot the owner rotates between visits, so every
# slot-0/layer-0 fetch is a first-window claim (nr must be 0, nf == ru).
# Keep ru == 8 (W=1 identity intact) but move one row from fetched to
# resident: the first-window branch must reject the phantom resident row.
$f = Tamper 'lifecycle-firstwindow' { param($a)
    $seen = 0
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_fetch"*' -and $a[$i] -like '*"slot":0,*' -and $a[$i] -like '*"layer":0,*') {
            $seen++
            if ($seen -eq 2) {
                $a[$i] = $a[$i] -replace '"n_rows_fetched":8', '"n_rows_fetched":7'
                $a[$i] = $a[$i] -replace '"n_rows_resident":0', '"n_rows_resident":1'; break
            }
        }
    }
    if ($seen -lt 2) { throw 'fixture error: no round-2 slot-0 fetch found' } }
$fixtures['lifecycle-firstwindow'] = @{ file = $f; msg = 'first window per (slot, generation) claims resident rows' }

# 4b. generation lifecycle (schema 4.5): tracked-coverage desync.
# Relabel a slot-0 fetch as layer 36 (the previous owner of slot 0): the
# checker then takes the tracked branch and must see nr == its coverage.
$f = Tamper 'lifecycle-tracked' { param($a)
    $seen = 0
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"rowset_fetch"*' -and $a[$i] -like '*"slot":0,*' -and $a[$i] -like '*"layer":0,*') {
            $seen++
            if ($seen -eq 2) {
                $a[$i] = $a[$i] -replace '"layer":0,', '"layer":36,'; break
            }
        }
    }
    if ($seen -lt 2) { throw 'fixture error: no round-2 slot-0 fetch found' } }
$fixtures['lifecycle-tracked'] = @{ file = $f; msg = 'n_rows_resident != tracked slot coverage' }

# 5. default-off immunity (schema 4.9): marker stripped from run_start
$f = Tamper 'marker-stripped' { param($a)
    for ($i = 0; $i -lt $a.Count; $i++) {
        if ($a[$i] -like '*"event":"run_start"*') {
            $a[$i] = $a[$i] -replace '"rowset":"on",', ''; break
        }
    } }
$fixtures['marker-stripped'] = @{ file = $f; msg = 'rowset events in a run whose run_start carries no rowset marker' }

# 6. structured refusal propagates as a loud check failure (full rewrite:
# insert a structured rowset_refused right after run_start)
$refPath = Join-Path $work 'refused-injected.jsonl'
$out = New-Object System.Collections.Generic.List[string]
$ins = '{"event":"rowset_refused","run_id":1,"layer":0,"slot":0,"generation":11,"reason":"bad_shape","detail":"negative-control fixture","phase":1,"t_us":1}'
foreach ($l in $lines) {
    $out.Add($l)
    if ($l -like '*"event":"run_start"*') { $out.Add($ins) }
}
[System.IO.File]::WriteAllLines($refPath, $out)
$fixtures['refused-injected'] = @{ file = $refPath; msg = 'rowset_refused: bad_shape' }

# --- run every fixture through BOTH validators ---------------------------
foreach ($name in $fixtures.Keys) {
    $fx = $fixtures[$name]
    $rc = Run-Check 'check-telemetry' $fx.file
    Expect "$name [check]" $rc.code 1 (($rc.out -join "`n")) $fx.msg
    $rv = Run-Check 'validate-telemetry' $fx.file
    Expect "$name [validate]" $rv.code 1 (($rv.out -join "`n")) $fx.msg
}

Write-Host ''
if ($script:fail -gt 0) { Write-Host "NEGATIVE CONTROLS: $script:fail FAILED"; exit 1 }
Write-Host 'NEGATIVE CONTROLS: all pass on both validators (every tampered fixture fails loudly, good trace passes both)'
