# Does an arm's saved evidence still describe the tree it was produced from?
#
# USAGE
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/check-arm-provenance.ps1 `
#       -Dir build/llama-cuda/verdicts -Tag gplr56-p1off
#   powershell ... -File tools/check-arm-provenance.ps1 -Verdict <path to .verdict.json>
#   powershell ... -File tools/check-arm-provenance.ps1 -Selftest      # host-only proof
#
# WHY THIS EXISTS (measured 2026-09-22). A 4k control arm passed at 16:28 against
# the 16:07 build. At 16:31 another actor edited the engine sources and the tree
# was rebuilt at 16:31:58/16:32:01. The verdict named an exe hash, nothing
# compared it to anything, and the record's "same-binary with the reverted tree"
# claim silently stopped being true. Re-deriving the digest from the tree NOW and
# comparing it to the digest the arm recorded is the missing comparison.
#
# EXIT CODES -- 2 is NOT a pass:
#   0  CURRENT                  the recorded evidence still describes this tree
#   1  stale or at risk         STALE-SOURCES | SOURCES-NEWER-THAN-BINARY |
#                               SOURCES-CHANGED-DURING-RUN | BINARY-CHANGED-DURING-RUN |
#                               STALE-BINARY
#   2  cannot certify           NO-PROVENANCE | NO-VERDICT | CANNOT-EVALUATE |
#                               BINARY-UNKNOWN
# The decision rule itself lives in tools/source-revision.ps1 (Compare-ArmProvenance),
# beside the digest it uses, so this tool cannot drift from the writer.
[CmdletBinding()]
param(
    [string] $Verdict = '',
    [string] $Dir = '',
    [string] $Tag = '',
    [string] $SourceRoot = '',
    [string] $RepoRoot = '',
    [switch] $Json,
    [switch] $Quiet,
    [switch] $Selftest
)

$ErrorActionPreference = 'Stop'
$toolsDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path $MyInvocation.MyCommand.Path -Parent }
. (Join-Path $toolsDir 'source-revision.ps1')

function Resolve-RepoRoot {
    param([string] $Given = '')
    if ($Given) { return $Given }
    return (Split-Path $toolsDir -Parent)
}

# Read one verdict and decide. Split out from the CLI so the selftest exercises
# the SAME code path the CLI does, rather than a description of it.
function Test-ArmProvenance {
    param(
        [string] $VerdictPath = '',
        [string] $Dir = '',
        [string] $Tag = '',
        [string] $SourceRoot = '',
        [string] $RepoRoot = ''
    )
    $root = Resolve-RepoRoot -Given $RepoRoot
    $path = $VerdictPath
    if (-not $path -and $Dir -and $Tag) { $path = Join-Path $Dir ($Tag + '.verdict.json') }

    $recorded = $null
    $status = ''
    $arm = $Tag
    if ($path -and (Test-Path -LiteralPath $path -PathType Leaf)) {
        $v = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        $status = "$($v.status)"
        $arm = "$($v.arm)"
        if ($v.PSObject.Properties.Name -contains 'fields') {
            $recorded = $v.fields.provenance
        }
        # The pre/post delta lives beside the launch capture, so fold it in: a run
        # whose tree moved underneath it must be readable from the record alone.
        if ($null -ne $recorded -and $v.fields.PSObject.Properties.Name -contains 'provenance_delta') {
            if ($v.fields.provenance_delta.source_changed_during_run -eq $true) { $recorded | Add-Member -NotePropertyName sources_changed_during_run -NotePropertyValue $true -Force }
            if ($v.fields.provenance_delta.binary_changed_during_run -eq $true) { $recorded | Add-Member -NotePropertyName binary_changed_during_run -NotePropertyValue $true -Force }
        }
    } elseif ($Tag -or $VerdictPath) {
        $cmp = Compare-ArmProvenance -Recorded $null -Current $null -RepoRoot $root -HasVerdict $false
        return [pscustomobject]@{ verdict = $cmp.verdict; exit = $cmp.exit; reasons = $cmp.reasons
                                  path = $path; arm = $arm; status = $status; recorded = $null
                                  current = $null; module_now = $null }
    } else {
        throw 'Give either -Verdict <file> or -Dir <dir> -Tag <tag>.'
    }

    $src = if ($SourceRoot) { $SourceRoot } else { Join-Path $root 'third_party\llama.cpp' }
    $current = Format-ArmProvenance -Provenance (Get-ArmProvenance -SourceRoot $src -RepoRoot $root)
    $cmp = Compare-ArmProvenance -Recorded $recorded -Current $current -RepoRoot $root -HasVerdict $true
    $moduleNow = $null
    if ($null -ne $recorded -and "$($recorded.binary_state)" -eq 'hashed') {
        $moduleNow = Get-BinaryIdentityNow -Recorded $recorded -RepoRoot $root
    }
    return [pscustomobject]@{ verdict = $cmp.verdict; exit = $cmp.exit; reasons = $cmp.reasons
                              path = $path; arm = $arm; status = $status; recorded = $recorded
                              current = $current; module_now = $moduleNow }
}

function Write-ArmProvenanceReport {
    param($Result, [switch] $AsJson, [switch] $Quiet)
    if ($AsJson) {
        $o = [ordered]@{
            schema = 'arm-provenance-check/1'
            path = "$($Result.path)"
            arm = "$($Result.arm)"
            status = "$($Result.status)"
            verdict = $Result.verdict
            exit = $Result.exit
            certified = ($Result.exit -eq 0)
            reasons = @($Result.reasons)
            recorded_digest = if ($Result.recorded) { "$($Result.recorded.source_digest)" } else { '' }
            current_digest = if ($Result.current) { "$($Result.current.source_digest)" } else { '' }
            recorded_head = if ($Result.recorded) { "$($Result.recorded.source_head)" } else { '' }
            current_head = if ($Result.current) { "$($Result.current.source_head)" } else { '' }
            binary_exe = if ($Result.recorded) { "$($Result.recorded.binary_exe)" } else { '' }
            binary_exe_sha = if ($Result.recorded) { "$($Result.recorded.binary_exe_sha)" } else { '' }
            streaming_lib_sha = if ($Result.recorded) { "$($Result.recorded.streaming_lib_sha)" } else { '' }
        }
        Write-Host ($o | ConvertTo-Json -Depth 4)
        return
    }
    if (-not $Quiet) {
        Write-Host ("arm-provenance: {0}  arm={1}  status={2}" -f $Result.verdict, $Result.arm, $Result.status)
        Write-Host ("  verdict : {0}" -f $Result.path)
        if ($Result.recorded) {
            Write-Host ("  recorded: digest {0}  head {1}  {2} tracked / {3} untracked changed  ({4})" -f `
                $Result.recorded.source_digest, (Short-Sha $Result.recorded.source_head),
                $Result.recorded.source_changed_tracked, $Result.recorded.source_changed_untracked,
                $Result.recorded.source_state)
            Write-Host ("  binary  : {0} {1} / {2} {3}  in {4}" -f `
                $Result.recorded.binary_exe, $Result.recorded.binary_exe_sha,
                $Result.recorded.streaming_lib, $Result.recorded.streaming_lib_sha, $Result.recorded.binary_dir)
        } else {
            Write-Host '  recorded: (no provenance field in this verdict)'
        }
        if ($Result.current) {
            Write-Host ("  now     : digest {0}  head {1}  {2} tracked / {3} untracked changed" -f `
                $Result.current.source_digest, (Short-Sha $Result.current.source_head),
                $Result.current.source_changed_tracked, $Result.current.source_changed_untracked)
        }
    }
    if ($Result.exit -eq 0) {
        if (-not $Quiet) { Write-Host '  RESULT: CURRENT - this arm still describes the tree, and the binary it ran is unchanged.' }
    } else {
        $tag = if ($Result.exit -eq 2) { 'NOT A PASS' } else { 'NOT CERTIFIED' }
        Write-Host ("  RESULT: {0} (exit {1}) - {2}" -f $Result.verdict, $Result.exit, $tag)
        foreach ($r in @($Result.reasons)) { Write-Host ("    - " + $r) }
    }
}

# --------------------------------------------------------------------------
# Host-only selftest. Builds throwaway git trees and binaries; no GPU, no model,
# no CUDA. The central case is the measured one: a build, a saved verdict, and
# then a SOURCE EDIT -- which must come back STALE, not CURRENT.
function Invoke-ArmProvenanceSelftest {
    $fail = 0
    function Check([string] $name, [bool] $cond, [string] $detail = '') {
        if ($cond) { Write-Host ("  PASS  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' })) }
        else { Write-Host ("  FAIL  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' })); $script:fail++ }
    }

    $tmp = Join-Path $env:TEMP ('arm-prov-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    $engine = Join-Path $tmp 'engine'      # the "nested" engine tree (a real git repo)
    $bin    = Join-Path $tmp 'bin'         # the "build output" dir
    New-Item -ItemType Directory -Force -Path (Join-Path $engine 'src'), $bin | Out-Null

    # Deterministic fixture: autocrlf off (so a checkout does not look modified),
    # a fixed identity (so the commit cannot fail on a machine with no git
    # config), and a fixed clock offset between the "build" and the sources.
    & git -C $engine init -q 2>$null
    & git -C $engine config core.autocrlf false
    & git -C $engine config user.email 'selftest@example.invalid'
    & git -C $engine config user.name 'arm-provenance selftest'
    Set-Content -Path (Join-Path $engine 'src\engine.cpp') -Value 'int engine() { return 24; }' -Encoding Ascii
    Set-Content -Path (Join-Path $engine 'src\build.py')   -Value 'print("build")' -Encoding Ascii
    & git -C $engine add -A 2>$null | Out-Null
    & git -C $engine commit -q -m 'fixture' 2>$null | Out-Null
    $head = (& git -C $engine rev-parse HEAD).Trim()

    $exe = Join-Path $bin 'llama-layer-streaming.exe'
    $dll = Join-Path $bin 'llama.dll'
    Set-Content -Path $exe -Value 'FAKE EXE BUILD 1' -Encoding Ascii
    Set-Content -Path $dll -Value 'FAKE ENGINE BUILD 1' -Encoding Ascii
    $built = (Get-Date).AddHours(-2)
    (Get-Item $exe).LastWriteTime = $built
    (Get-Item $dll).LastWriteTime = $built
    # Sources committed an hour BEFORE the build: the healthy case.
    Get-ChildItem -Path $engine -Recurse -File | Where-Object { $_.FullName -notlike '*\.git\*' } |
        ForEach-Object { $_.LastWriteTime = $built.AddHours(-1) }

    $verdicts = Join-Path $tmp 'verdicts'
    $sessionStart = (Get-Date).ToUniversalTime().ToString('o')
    Write-ArmVerdict -Dir $verdicts -Tag 'arm-a' -ExitCode 0 -StartedUtc $sessionStart `
        -Command (('"{0}" -m fixture.gguf' -f $exe)) -SourceRoot $engine -RepoRoot $tmp | Out-Null
    $path = Join-Path $verdicts 'arm-a.verdict.json'

    Write-Host '=== 1. a verdict written through the shared writer carries provenance ==='
    $j = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    Check 'the verdict has a provenance field' ($null -ne $j.fields.provenance)
    Check 'it names the engine tree and its HEAD' ($j.fields.provenance.source_state -eq 'git' -and $j.fields.provenance.source_head -eq $head) `
        ("state={0} head={1}" -f $j.fields.provenance.source_state, $j.fields.provenance.source_head)
    Check 'it carries a source digest' ($j.fields.provenance.source_digest -and $j.fields.provenance.source_digest.Length -eq 16) $j.fields.provenance.source_digest
    Check 'it names the binary the command runs' ($j.fields.provenance.binary_exe -eq 'llama-layer-streaming.exe') $j.fields.provenance.binary_exe
    Check 'it carries the streaming library hash' ($j.fields.provenance.streaming_lib_sha -and $j.fields.provenance.streaming_lib -eq 'llama.dll')
    Check 'the build is NEWER than every changed source, so nothing is flagged' ($j.fields.provenance.sources_newer_than_binary -eq $false) `
        ("flag={0} detail={1}" -f $j.fields.provenance.sources_newer_than_binary, $j.fields.provenance.sources_newer_detail)

    Write-Host '=== 2. an untouched tree certifies ==='
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'CURRENT, exit 0' ($r.verdict -eq 'CURRENT' -and $r.exit -eq 0) ("{0} exit {1}" -f $r.verdict, $r.exit)

    Write-Host '=== 3. THE MEASURED FAILURE: a source edit after the build ==='
    # 2026-09-22: a smoke passed at 16:28 against the 16:07 build; at 16:31 the
    # engine sources were edited. Nothing in the record noticed. This must.
    Start-Sleep -Milliseconds 1100   # fs timestamps; the digest is content-based anyway
    Set-Content -Path (Join-Path $engine 'src\engine.cpp') -Value 'int engine() { return 25; }' -Encoding Ascii
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'STALE-SOURCES, exit 1' ($r.verdict -eq 'STALE-SOURCES' -and $r.exit -eq 1) ("{0} exit {1}" -f $r.verdict, $r.exit)
    Check 'the reason names both digests' (($r.reasons -join ' ') -like '*SOURCES CHANGED*recorded*digests to*')
    Check 'and it NAMES the path that changed, not just that something did' (($r.reasons -join ' ') -like '*engine.cpp*') `
        (($r.reasons | Select-Object -Last 1))
    $rq = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'a stale arm is NOT reported as certified' ($rq.exit -ne 0)

    Write-Host '=== 4. reverting the edit certifies again (content, not mtimes) ==='
    Set-Content -Path (Join-Path $engine 'src\engine.cpp') -Value 'int engine() { return 24; }' -Encoding Ascii
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'CURRENT again, exit 0' ($r.verdict -eq 'CURRENT' -and $r.exit -eq 0) ("{0} exit {1}" -f $r.verdict, $r.exit)

    Write-Host '=== 5. an uncommitted but build-relevant file is covered too ==='
    Set-Content -Path (Join-Path $engine 'src\new-kernel.cpp') -Value 'int k() { return 1; }' -Encoding Ascii
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'a NEW untracked source also reads STALE-SOURCES' ($r.verdict -eq 'STALE-SOURCES' -and $r.exit -eq 1) ("{0} exit {1}" -f $r.verdict, $r.exit)
    Remove-Item -LiteralPath (Join-Path $engine 'src\new-kernel.cpp') -Force

    Write-Host '=== 6. a rebuilt engine module is caught ==='
    Set-Content -Path $dll -Value 'FAKE ENGINE BUILD 2' -Encoding Ascii
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'STALE-BINARY, exit 1' ($r.verdict -eq 'STALE-BINARY' -and $r.exit -eq 1) ("{0} exit {1}" -f $r.verdict, $r.exit)
    Check 'the reason names the module and both hashes' (($r.reasons -join ' ') -like '*llama.dll*REBUILT*')
    Set-Content -Path $dll -Value 'FAKE ENGINE BUILD 1' -Encoding Ascii
    (Get-Item $dll).LastWriteTime = $built   # writing the file moved its mtime to NOW
    $r = Test-ArmProvenance -VerdictPath $path -SourceRoot $engine -RepoRoot $tmp
    Check 'restoring the module certifies again' ($r.verdict -eq 'CURRENT' -and $r.exit -eq 0) ("{0} exit {1}" -f $r.verdict, $r.exit)

    Write-Host '=== 7. a binary that PREDATES its sources is refused ==='
    # The mirror failure of the measured one: edit a source, do NOT rebuild, then
    # run the older binary. Content AND a newer mtime are needed -- a revert to
    # HEAD is correctly not a change (asserted in case 4).
    $lagSrc = Join-Path $engine 'src\engine.cpp'
    Set-Content -Path $lagSrc -Value 'int engine() { return 26; }' -Encoding Ascii
    (Get-Item $lagSrc).LastWriteTime = $built.AddHours(1)   # newer than the build
    $cap = Get-ArmProvenance -SourceRoot $engine -RepoRoot $tmp -ArtifactPaths @($exe, $dll)
    Check 'a source edited after the build sets the flag on capture' ($cap.sources_newer_than_binary -eq $true) `
        ("flag={0} detail={1}" -f $cap.sources_newer_than_binary, $cap.sources_newer_detail)
    $lagRec = Format-ArmProvenance -Provenance $cap
    $cur = Format-ArmProvenance -Provenance (Get-ArmProvenance -SourceRoot $engine -RepoRoot $tmp)
    $cmpl = Compare-ArmProvenance -Recorded $lagRec -Current $cur -RepoRoot $tmp
    Check 'and the rule refuses it: SOURCES-NEWER-THAN-BINARY, exit 1' `
        ($cmpl.verdict -eq 'SOURCES-NEWER-THAN-BINARY' -and $cmpl.exit -eq 1) ("{0} exit {1}" -f $cmpl.verdict, $cmpl.exit)
    Set-Content -Path $lagSrc -Value 'int engine() { return 24; }' -Encoding Ascii
    (Get-Item $lagSrc).LastWriteTime = $built.AddHours(-1)

    Write-Host '=== 7b. the staleness flag ignores NON-build inputs, and still catches real ones ==='
    # The mechanism's one judgement call, so it gets its own negative control.
    # Measured 2026-09-22 in the real tree: another actor's scratch
    # `.tmp-split/n.*.diff`, written after the last build, made every arm read
    # sources_newer_than_binary and would have refused a smoke that was fine.
    $scratch = Join-Path $engine 'notes.diff'
    Set-Content -Path $scratch -Value 'scratch output, not a build input' -Encoding Ascii
    (Get-Item $scratch).LastWriteTime = $built.AddHours(2)
    $cap = Get-ArmProvenance -SourceRoot $engine -RepoRoot $tmp -ArtifactPaths @($exe, $dll)
    Check 'a scratch file newer than the build does NOT flag the binary as stale' ($cap.sources_newer_than_binary -eq $false) `
        ("flag={0} detail={1}" -f $cap.sources_newer_than_binary, $cap.sources_newer_detail)
    Check 'but it IS still named as the newest change (the digest covers it)' ($cap.source.newest_change_path -eq 'notes.diff') `
        $cap.source.newest_change_path
    Remove-Item -LiteralPath $scratch -Force
    # ... and the real case still bites: a compile input newer than the build.
    $realSrc = Join-Path $engine 'src\engine.cpp'
    Set-Content -Path $realSrc -Value 'int engine() { return 27; }' -Encoding Ascii
    (Get-Item $realSrc).LastWriteTime = $built.AddHours(1)
    $cap2 = Get-ArmProvenance -SourceRoot $engine -RepoRoot $tmp -ArtifactPaths @($exe, $dll)
    Check 'a COMPILE INPUT newer than the build still flags' ($cap2.sources_newer_than_binary -eq $true) `
        ("flag={0} input={1}" -f $cap2.sources_newer_than_binary, $cap2.source.newest_input_path)
    Set-Content -Path $realSrc -Value 'int engine() { return 24; }' -Encoding Ascii
    (Get-Item $realSrc).LastWriteTime = $built.AddHours(-1)

    Write-Host '=== 8. a change DURING the run is refused ==='
    Write-ArmVerdict -Dir $verdicts -Tag 'arm-mid' -ExitCode 0 -StartedUtc $sessionStart `
        -Command (('"{0}" -m fixture.gguf' -f $exe)) -SourceRoot $engine -RepoRoot $tmp | Out-Null
    $mid = Join-Path $verdicts 'arm-mid.verdict.json'
    $m = Get-Content -LiteralPath $mid -Raw | ConvertFrom-Json
    $m.fields | Add-Member -NotePropertyName provenance_delta -NotePropertyValue ([pscustomobject]@{ source_changed_during_run = $true; binary_changed_during_run = $false }) -Force
    $m | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $mid -Encoding UTF8
    $r = Test-ArmProvenance -VerdictPath $mid -SourceRoot $engine -RepoRoot $tmp
    Check 'SOURCES-CHANGED-DURING-RUN, exit 1' ($r.verdict -eq 'SOURCES-CHANGED-DURING-RUN' -and $r.exit -eq 1) ("{0} exit {1}" -f $r.verdict, $r.exit)

    Write-Host '=== 9. absence is never a pass ==='
    # (a) a verdict written before the mechanism landed: every saved verdict in
    # the repo is in this class, so it must read as NOT CERTIFIABLE, not as fine.
    $legacy = Join-Path $verdicts 'legacy.verdict.json'
    [System.IO.File]::WriteAllText($legacy, (@{
        schema = 'arm-verdict/1'; arm = 'legacy'; status = 'PASS'; exit_code = 0
        fields = @{ gpu_pre_run = 'clean'; mutex_acquired = $true }
    } | ConvertTo-Json -Depth 4))
    $r = Test-ArmProvenance -VerdictPath $legacy -SourceRoot $engine -RepoRoot $tmp
    Check 'a pre-provenance verdict reads NO-PROVENANCE, exit 2' ($r.verdict -eq 'NO-PROVENANCE' -and $r.exit -eq 2) ("{0} exit {1}" -f $r.verdict, $r.exit)

    # (b) a verdict with provenance but no binary identity
    Write-ArmVerdict -Dir $verdicts -Tag 'arm-nobin' -ExitCode 0 -StartedUtc $sessionStart `
        -Command 'exit 0' -SourceRoot $engine -RepoRoot $tmp | Out-Null
    $r = Test-ArmProvenance -VerdictPath (Join-Path $verdicts 'arm-nobin.verdict.json') -SourceRoot $engine -RepoRoot $tmp
    Check 'no binary identity reads BINARY-UNKNOWN, exit 2' ($r.verdict -eq 'BINARY-UNKNOWN' -and $r.exit -eq 2) ("{0} exit {1}" -f $r.verdict, $r.exit)

    # (c) nothing at all
    $r = Test-ArmProvenance -Dir $verdicts -Tag 'never-ran' -SourceRoot $engine -RepoRoot $tmp
    Check 'a missing verdict reads NO-VERDICT, exit 2' ($r.verdict -eq 'NO-VERDICT' -and $r.exit -eq 2) ("{0} exit {1}" -f $r.verdict, $r.exit)

    # (d) a tree that is not a repository at all: cannot evaluate, never "current"
    $plain = Join-Path $tmp 'plain'
    New-Item -ItemType Directory -Force -Path $plain | Out-Null
    Set-Content -Path (Join-Path $plain 'x.cpp') -Value 'int x;' -Encoding Ascii
    Write-ArmVerdict -Dir $verdicts -Tag 'arm-plain' -ExitCode 0 -StartedUtc $sessionStart `
        -Command (('"{0}" -m fixture.gguf' -f $exe)) -SourceRoot $plain -RepoRoot $tmp | Out-Null
    $j = Get-Content -LiteralPath (Join-Path $verdicts 'arm-plain.verdict.json') -Raw | ConvertFrom-Json
    Check 'an undigestible tree is recorded as such, not as unchanged' ($j.fields.provenance.source_state -eq 'not-a-repo') $j.fields.provenance.source_state
    $r = Test-ArmProvenance -VerdictPath (Join-Path $verdicts 'arm-plain.verdict.json') -SourceRoot $plain -RepoRoot $tmp
    Check 'CANNOT-EVALUATE, exit 2' ($r.verdict -eq 'CANNOT-EVALUATE' -and $r.exit -eq 2) ("{0} exit {1}" -f $r.verdict, $r.exit)

    Write-Host '=== 10. no behaviour change: capture failure does not break a verdict ==='
    Check 'the verdict was still written with provenance absent-but-named' ($j.fields.provenance.source_error -ne '')

    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction Continue
    Write-Host ''
    if ($fail -eq 0) { Write-Host 'ALL ARM-PROVENANCE SELFTEST CASES PASS'; return 0 }
    Write-Host ("ARM-PROVENANCE SELFTEST FAILED: {0}" -f $fail); return 1
}

if ($Selftest) {
    . (Join-Path $toolsDir 'arm-verdict.ps1')
    exit (Invoke-ArmProvenanceSelftest)
}

$result = Test-ArmProvenance -VerdictPath $Verdict -Dir $Dir -Tag $Tag -SourceRoot $SourceRoot -RepoRoot $RepoRoot
Write-ArmProvenanceReport -Result $result -AsJson:$Json -Quiet:$Quiet
exit $result.exit
