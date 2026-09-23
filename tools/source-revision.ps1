# Source-revision and binary-identity provenance for arm verdicts.
# DOT-SOURCE this file; it defines functions only and has no side effects.
#
# WHY THIS EXISTS (measured 2026-09-22). A 4k control arm passed at 16:28 against
# the 16:07 build. At 16:31 another actor edited
# third_party/llama.cpp/src/llama-layer-streaming.{h,cpp}, and the tree was
# rebuilt at 16:31:58 / 16:32:01. Nothing noticed: the verdict named only an exe
# hash, no consumer compared that hash to anything, and the record's
# "same-binary with the reverted tree" claim silently stopped being true. The
# same trace shows the mirror error is possible too -- tools/layer-streaming/
# main.cpp was modified at 16:17, AFTER the 16:07 build, so the 16:28 smoke's
# binary could not have contained it and no instrument said so.
#
# The exe hash is also the wrong anchor: the streaming code lands in llama.dll,
# and a relink churns the exe hash while the engine is unchanged.
#
# So every arm verdict now carries, in `fields.provenance`:
#   * a DIGEST of the engine source tree the binary was built from -- the nested
#     HEAD plus the full content diff of tracked changes plus the content of
#     untracked files under it. Content, not mtimes: reverting an edit returns
#     the digest to its earlier value, which is the correct reading.
#   * the IDENTITY of the binary the arm actually ran: the exe, the streaming
#     library (llama.dll) and the ggml backends beside it, by SHA-256 prefix.
#   * whether a changed source was already NEWER than that binary at capture
#     time, i.e. the binary cannot contain it.
#
# tools/check-arm-provenance.ps1 re-derives the same digest from the current tree
# and refuses to certify an arm whose sources or binary have moved since. The
# digest algorithm and the decision rule live HERE only, so the writer and the
# checker cannot drift apart.

# SHA-256 prefix of a file's CONTENT, or '' when it cannot be read. Streamed, so
# hashing the 34 MB ggml-cuda.dll does not buffer it in memory. 16 hex chars
# matches the house style already used by the arm drivers (Get-FileHash ... .Hash
# .Substring(0, 16)), so recorded hashes stay comparable with earlier records.
function Get-FileSha256Prefix {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [int] $Chars = 16
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return '' }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $fs = $null
    try {
        $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $h = $sha.ComputeHash($fs)
    } catch {
        return ''
    } finally {
        if ($fs) { $fs.Dispose() }
        $sha.Dispose()
    }
    return ([System.BitConverter]::ToString($h) -replace '-', '').Substring(0, [Math]::Min($Chars, 64))
}

function Get-StringSha256Prefix {
    param([string] $Text, [int] $Chars = 16)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $h = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($Text)) }
    finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($h) -replace '-', '').Substring(0, [Math]::Min($Chars, 64))
}

# Best-effort repo-relative form of a path, with forward slashes, so a verdict
# stays readable on another checkout. A path outside the repo is kept as-is: an
# out-of-tree binary is still worth recording.
function ConvertTo-RepoRelativePath {
    param([string] $Path = '', [string] $RepoRoot = '')
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    $p = $Path -replace '\\', '/'
    if ([string]::IsNullOrWhiteSpace($RepoRoot)) { return $p }
    $r = ($RepoRoot -replace '\\', '/').TrimEnd('/')
    if ($p.StartsWith($r + '/', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $p.Substring($r.Length + 1)
    }
    return $p
}

# Which changed paths can make the BINARY stale. Two questions live here and they
# deserve two answers: the DIGEST covers the whole tree (nothing may hide), while
# the MTIME relation asks "could this edit be reflected in the executable?".
# Answering the second with the whole tree produced a false flag on measurement
# (2026-09-22): another actor's scratch `.tmp-split/n.*.diff` files, written after
# the last build, made every arm read sources_newer_than_binary and would have
# refused a real smoke that was in fact fine. The sets below are the
# compile-relevant shapes this tree uses; a Planner who disagrees with the boundary
# changes this list and nothing else.
$script:BuildInputExtensions = @('.c', '.cc', '.cpp', '.cxx', '.cu', '.cuh', '.h', '.hh',
                                 '.hpp', '.hxx', '.m', '.mm', '.s', '.asm', '.inc', '.cmake',
                                 '.def', '.rc')
$script:BuildInputNames = @('cmakelists.txt', 'makefile', 'gnumakefile')

function Test-BuildInputPath {
    param([string] $Path = '')
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    $leaf = (Split-Path -Leaf $Path).ToLower()
    if ($script:BuildInputNames -contains $leaf) { return $true }
    return ($script:BuildInputExtensions -contains [System.IO.Path]::GetExtension($leaf))
}

# Split `git status --porcelain` into tracked and untracked changed paths. Kept
# tiny on purpose: this only has to count changes and name the newest one.
function Get-GitChangedPaths {
    param([string[]] $PorcelainLines)
    $tracked = @()
    $untracked = @()
    foreach ($line in @($PorcelainLines)) {
        $s = "$line"
        if ($s.Length -lt 4) { continue }
        $code = $s.Substring(0, 2)
        $path = $s.Substring(3).Trim()
        if ($path.Length -gt 1 -and $path.StartsWith('"') -and $path.EndsWith('"')) {
            $path = $path.Substring(1, $path.Length - 2)
        }
        if ([string]::IsNullOrWhiteSpace($path)) { continue }
        if ($code -eq '??' -or $code -eq '!!') { $untracked += $path } else { $tracked += $path }
    }
    return [pscustomobject]@{ tracked = $tracked; untracked = $untracked }
}

# The engine source tree's identity. `state` names WHY a digest is absent, so an
# unevaluable tree is never mistaken for an unchanged one:
#   git          readable; digest is meaningful
#   not-a-repo   no git HEAD in that directory
#   git-missing  git is not on PATH
#   missing      the directory does not exist
#   error        an unexpected failure; `error` carries the text
function Get-SourceRevision {
    param(
        [string] $Root = '',
        [string] $Scope = 'third_party/llama.cpp',
        [string] $RepoRoot = ''
    )
    $r = [ordered]@{
        schema = 'source-revision/1'
        scope = $Scope
        root = (ConvertTo-RepoRelativePath -Path $Root -RepoRoot $RepoRoot)
        state = 'missing'
        head = ''
        changed_tracked = 0
        changed_untracked = 0
        newest_change_path = ''
        newest_change_utc = ''
        newest_input_path = ''
        newest_input_utc = ''
        changed_paths = @()
        changed_paths_truncated = $false
        digest = ''
        captured_utc = (Get-Date).ToUniversalTime().ToString('o')
        error = ''
    }
    if ([string]::IsNullOrWhiteSpace($Root) -or -not (Test-Path -LiteralPath $Root -PathType Container)) {
        $r.error = "source root not found: $Root"
        return [pscustomobject]$r
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        $r.state = 'git-missing'; $r.error = 'git is not on PATH'
        return [pscustomobject]$r
    }
    # Native stderr + ErrorActionPreference=Stop is a trap: PowerShell 5.1 raises
    # a TERMINATING error for a native command's stderr line, so `git rev-parse
    # HEAD` outside a repository aborted this function with state=error instead of
    # reporting not-a-repo (measured 2026-09-22, under the checker's own
    # ErrorActionPreference=Stop). A caller's preference is not this module's to
    # assume, so it is neutralised locally and every outcome is read from
    # $LASTEXITCODE, which is the only thing that actually means anything here.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $head = @(& git -C $Root rev-parse HEAD 2>$null) | Select-Object -First 1
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace("$head")) {
            $r.state = 'not-a-repo'; $r.error = "git could not read HEAD in $Root"
            return [pscustomobject]$r
        }
        # `rev-parse HEAD` happily climbs to an ENCLOSING repository, so a path
        # that is not itself a checkout would silently report the parent's
        # identity (and the parent's change set). Require the root to be the
        # toplevel of the repository it just read.
        $toplevel = @(& git -C $Root rev-parse --show-toplevel 2>$null) | Select-Object -First 1
        if (-not [string]::IsNullOrWhiteSpace("$toplevel")) {
            $a = ("$toplevel" -replace '\\', '/').TrimEnd('/')
            $b = ((Resolve-Path -LiteralPath $Root).Path -replace '\\', '/').TrimEnd('/')
            if ($a -and $b -and -not $a.Equals($b, [System.StringComparison]::OrdinalIgnoreCase)) {
                $r.state = 'not-a-repo'
                $r.error = "$Root is inside the repository at $a, not a repository root"
                return [pscustomobject]$r
            }
        }
        $r.head = "$head".Trim()

        $porcelain = @(& git -C $Root status --porcelain 2>$null)
        # -c on both counts: a pager or colour codes would change the digest for
        # reasons that have nothing to do with the sources.
        $diff = @(& git -C $Root -c core.pager=cat -c color.ui=false diff HEAD 2>$null)
        $changes = Get-GitChangedPaths -PorcelainLines $porcelain
        $r.changed_tracked = @($changes.tracked).Count
        $r.changed_untracked = @($changes.untracked).Count

        # The newest change is what the "binary predates its sources" test needs.
        # BOTH the newest change of any kind (informational) and the newest
        # COMPILE-RELEVANT change (the one the staleness flag is allowed to use).
        foreach ($rel in (@($changes.tracked) + @($changes.untracked))) {
            $full = Join-Path $Root $rel
            if (-not (Test-Path -LiteralPath $full)) { continue }
            $m = (Get-Item -LiteralPath $full).LastWriteTimeUtc
            if (-not $r.newest_change_utc -or $m -gt [datetime]$r.newest_change_utc) {
                $r.newest_change_utc = $m.ToString('o')
                $r.newest_change_path = $rel
            }
            if ((Test-BuildInputPath $rel) -and (-not $r.newest_input_utc -or $m -gt [datetime]$r.newest_input_utc)) {
                $r.newest_input_utc = $m.ToString('o')
                $r.newest_input_path = $rel
            }
        }

        # The untracked FILES, individually: `git status --porcelain` collapses a
        # new directory to one `?? dir/` line (measured 2026-09-22, a scratch
        # `.tmp-split/` appearing mid-pass), which names nothing about what is
        # inside it. Hashing is content work, so it needs the files themselves.
        $untrackedFiles = @(& git -C $Root ls-files --others --exclude-standard 2>$null) |
            Where-Object { -not [string]::IsNullOrWhiteSpace("$_") } | Sort-Object

        # Canonical text: HEAD, the whole tracked diff, then every untracked file
        # by content hash. Untracked files are build inputs too (a new test source
        # changes the binary), so omitting them would let a real change hide
        # behind "no diff".
        $canon = "head=$($r.head)`ntracked-diff:`n" + ($diff -join "`n") + "`nuntracked:`n"
        foreach ($rel in $untrackedFiles) {
            $full = Join-Path $Root $rel
            $sha = Get-FileSha256Prefix -Path $full -Chars 64
            $len = if (Test-Path -LiteralPath $full -PathType Leaf) { (Get-Item -LiteralPath $full).Length } else { -1 }
            $canon += "$rel`t$sha`t$len`n"
        }

        # A bounded manifest of WHAT changed, so a drill can be NAMED rather than
        # only counted. The digest is still the comparison; this is the explanation.
        $all = @(@($changes.tracked) + @($untrackedFiles) | Where-Object { $_ } | Sort-Object -Unique)
        $cap = 50
        $r.changed_paths = @($all | Select-Object -First $cap)
        $r.changed_paths_truncated = ($all.Count -gt $cap)
        $r.digest = Get-StringSha256Prefix -Text $canon
        $r.state = 'git'
    } catch {
        $r.state = 'error'
        $r.error = $_.Exception.Message
    } finally {
        $ErrorActionPreference = $previousEap
    }
    return [pscustomobject]$r
}

# Identity of the binary an arm ran: the exe, the streaming library, and every
# ggml backend module beside it. `state`:
#   hashed       paths were given and at least the exe exists
#   missing      paths were given but none exists (a dry command, a bad path)
#   not-supplied the writer had no binary to name (a stub arm, a CTest verdict)
function Get-BinaryIdentity {
    param(
        [string[]] $Paths = @(),
        [string] $StreamingLibrary = 'llama.dll',
        [string] $RepoRoot = ''
    )
    $r = [ordered]@{
        state = 'not-supplied'
        exe = ''
        exe_sha = ''
        bin_dir = ''
        built_utc = ''
        streaming_lib = $StreamingLibrary
        streaming_lib_sha = ''
        modules = [ordered]@{}
        missing = @()
    }
    $given = @($Paths | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    if ($given.Count -eq 0) { return [pscustomobject]$r }

    $existing = @($given | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
    $r.missing = @($given | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($existing.Count -eq 0) { $r.state = 'missing'; return [pscustomobject]$r }

    $exe = $existing[0]
    $r.exe = Split-Path -Leaf $exe
    $r.exe_sha = Get-FileSha256Prefix -Path $exe
    $r.bin_dir = ConvertTo-RepoRelativePath -Path (Split-Path -Parent $exe) -RepoRoot $RepoRoot
    $built = (Get-Item -LiteralPath $exe).LastWriteTimeUtc
    foreach ($p in $existing) {
        $leaf = Split-Path -Leaf $p
        $r.modules[$leaf] = Get-FileSha256Prefix -Path $p
        $m = (Get-Item -LiteralPath $p).LastWriteTimeUtc
        if ($m -gt $built) { $built = $m }
    }
    # NEWEST artifact = when this build finished, which is the only defensible
    # anchor for "was this source edited after the build". The OLDEST artifact was
    # tried first and rejected on measurement: ggml-cuda.dll has not been rebuilt
    # since 2026-09-17, so testing against it flagged every later source edit
    # (measured 2026-09-22: a false `sources_newer` on a tree whose exe and engine
    # dll were both rebuilt minutes after the newest edit).
    $r.built_utc = $built.ToString('o')
    if ($r.modules.Contains($StreamingLibrary)) { $r.streaming_lib_sha = $r.modules[$StreamingLibrary] }
    $r.state = 'hashed'
    return [pscustomobject]$r
}

# The binary paths a command line names: the .exe it runs plus the streaming
# library and ggml backends beside it. Empty when the command names no existing
# exe (a stub arm), which is recorded as binary_state=not-supplied rather than
# guessed at.
function Get-ProvenanceArtifacts {
    param(
        [string] $Command = '',
        [string] $StreamingLibrary = 'llama.dll'
    )
    $exe = ''
    foreach ($m in [regex]::Matches("$Command", '(?i)("([^"]+?\.exe)"|([^\s"'']+?\.exe))')) {
        $cand = if ($m.Groups[2].Success) { $m.Groups[2].Value } else { $m.Groups[3].Value }
        if (Test-Path -LiteralPath $cand -PathType Leaf) { $exe = $cand }
    }
    if (-not $exe) { return @() }
    $dir = Split-Path -Parent $exe
    $mods = @()
    foreach ($pat in @($StreamingLibrary, 'ggml*.dll')) {
        $mods += @(Get-ChildItem -LiteralPath $dir -Filter $pat -File -ErrorAction SilentlyContinue |
                   Sort-Object Name | ForEach-Object { $_.FullName })
    }
    return @($exe) + @($mods | Where-Object { $_ -ne $exe })
}

# One capture: the source revision, the binary identity, and the relation between
# them. Everything a verdict needs, computed in one place so writer and checker
# cannot disagree about what "the sources that produced this binary" means.
function Get-ArmProvenance {
    param(
        [string] $SourceRoot = '',
        [string] $Scope = 'third_party/llama.cpp',
        [string[]] $ArtifactPaths = @(),
        [string] $RepoRoot = '',
        [string] $StreamingLibrary = 'llama.dll'
    )
    $src = Get-SourceRevision -Root $SourceRoot -Scope $Scope -RepoRoot $RepoRoot
    $bin = Get-BinaryIdentity -Paths $ArtifactPaths -StreamingLibrary $StreamingLibrary -RepoRoot $RepoRoot
    $newer = $false
    $detail = ''
    if ($bin.state -eq 'hashed' -and $src.state -eq 'git' -and $src.newest_input_utc) {
        if ([datetime]$src.newest_input_utc -gt [datetime]$bin.built_utc) {
            $newer = $true
            $detail = ("{0} ({1}) was edited AFTER the newest artifact of the build ({2})" -f `
                $src.newest_input_path, $src.newest_input_utc, $bin.built_utc)
        }
    }
    return [pscustomobject]@{
        source = $src; binary = $bin; sources_newer_than_binary = $newer
        sources_newer_detail = $detail; captured_utc = (Get-Date).ToUniversalTime().ToString('o')
    }
}

# Flatten a capture into the JSON-friendly record that goes into a verdict. Flat
# scalars on purpose: a reader sees every field without walking a tree, and a
# partially-filled record is still readable.
function Format-ArmProvenance {
    param($Provenance, [string] $Origin = 'captured')
    $s = $Provenance.source
    $b = $Provenance.binary
    return [ordered]@{
        schema                    = 'arm-provenance/1'
        origin                    = $Origin
        captured_utc              = $Provenance.captured_utc
        source_scope              = $s.scope
        source_root               = $s.root
        source_state              = $s.state
        source_head               = $s.head
        source_changed_tracked    = $s.changed_tracked
        source_changed_untracked  = $s.changed_untracked
        source_newest_change_path = $s.newest_change_path
        source_newest_change_utc  = $s.newest_change_utc
        source_newest_input_path  = $s.newest_input_path
        source_newest_input_utc   = $s.newest_input_utc
        source_changed_paths      = @($s.changed_paths)
        source_changed_paths_truncated = $s.changed_paths_truncated
        source_digest             = $s.digest
        source_error              = $s.error
        binary_state              = $b.state
        binary_exe                = $b.exe
        binary_exe_sha            = $b.exe_sha
        binary_dir                = $b.bin_dir
        binary_built_utc          = $b.built_utc
        streaming_lib             = $b.streaming_lib
        streaming_lib_sha         = $b.streaming_lib_sha
        modules                   = $b.modules
        sources_newer_than_binary = $Provenance.sources_newer_than_binary
        sources_newer_detail      = $Provenance.sources_newer_detail
    }
}

# Field names of an object that arrives in TWO shapes: an ordered dictionary
# while the writer still holds it in memory (Format-ArmProvenance), and a
# PSCustomObject once the record has been through ConvertTo-Json/ConvertFrom-Json
# (every verdict a later pass reads). `PSObject.Properties.Name` on a dictionary
# returns the DICTIONARY's own members -- Count, Keys, Values -- as if they were
# data (measured 2026-09-22, twice: once on the modules map, once in a shape
# guard), so both readers go through this.
function Get-FieldNames {
    param($Object)
    if ($null -eq $Object) { return @() }
    if ($Object -is [System.Collections.IDictionary]) { return @($Object.Keys) }
    return @($Object.PSObject.Properties.Name)
}

# Re-read the modules a verdict recorded, by name, from the directory that
# recorded them. A missing directory reads as state=missing, never as unchanged.
function Get-BinaryIdentityNow {
    param($Recorded, [string] $RepoRoot = '')
    if ($null -eq $Recorded -or "$($Recorded.binary_state)" -eq 'not-supplied') { return $null }
    $dir = "$($Recorded.binary_dir)"
    if ($dir -and -not [System.IO.Path]::IsPathRooted($dir)) { $dir = Join-Path $RepoRoot $dir }
    if (-not $dir -or -not (Test-Path -LiteralPath $dir -PathType Container)) {
        return [pscustomobject]@{ state = 'missing'; modules = [ordered]@{} }
    }
    $mods = [ordered]@{}
    foreach ($leaf in (Get-FieldNames $Recorded.modules)) {
        $mods[$leaf] = Get-FileSha256Prefix -Path (Join-Path $dir $leaf)
    }
    return [pscustomobject]@{ state = 'hashed'; modules = $mods }
}

function Short-Sha {
    param([string] $Sha = '')
    if ([string]::IsNullOrWhiteSpace($Sha) -or $Sha.Length -le 12) { return "$Sha" }
    return $Sha.Substring(0, 12)
}

# THE DECISION RULE, in one owner. Given what a verdict RECORDED and what the
# tree and the binary say NOW, returns a verdict, the reasons that fired, and an
# exit code. 0 certifies the evidence; 2 means "cannot certify" (never a pass);
# 1 means stale, or at risk of describing different code than the record claims.
# `-Recorded` and `-Current` are both FLAT provenance records in the shape
# Format-ArmProvenance produces (the recorded one read back from the verdict JSON,
# the current one built from the tree now). A raw Get-SourceRevision object has
# different field names, and passing one used to read as "cannot digest now"
# (measured 2026-09-22) -- a silent wrong answer, so it is rejected loudly instead.
function Compare-ArmProvenance {
    param(
        $Recorded = $null,
        $Current = $null,
        [string] $RepoRoot = '',
        [bool] $HasVerdict = $true
    )
    if ($null -ne $Current -and -not ((Get-FieldNames $Current) -contains 'source_state')) {
        throw 'Compare-ArmProvenance: -Current must be a flat provenance record (Format-ArmProvenance), not a raw revision object.'
    }
    $reasons = New-Object System.Collections.ArrayList
    function Add-Reason([string] $Tag, [string] $Message) {
        $null = $reasons.Add([pscustomobject]@{ tag = $Tag; message = $Message })
    }

    if (-not $HasVerdict) { Add-Reason 'no-verdict' 'no verdict file was found to read' }
    elseif ($null -eq $Recorded) { Add-Reason 'no-provenance' 'this verdict carries no source/binary provenance (written before the mechanism landed)' }
    else {
        $curState = if ($null -ne $Current) { "$($Current.source_state)" } else { 'unavailable' }
        if ("$($Recorded.source_state)" -ne 'git') {
            Add-Reason 'source-undigestible' ("the engine tree could not be digested when the arm ran (source_state={0}{1})" -f `
                $Recorded.source_state, $(if ($Recorded.source_error) { ": $($Recorded.source_error)" } else { '' }))
        } elseif ($curState -ne 'git') {
            Add-Reason 'source-undigestible' ("the engine tree cannot be digested now (source_state={0})" -f $curState)
        } elseif ("$($Recorded.source_digest)" -ne "$($Current.source_digest)") {
            Add-Reason 'source-drift' ("SOURCES CHANGED since this arm was built: recorded digest {0} (head {1}, {2} tracked / {3} untracked changed) but the tree now digests to {4} (head {5}, {6} tracked / {7} untracked changed)" -f `
                $Recorded.source_digest, (Short-Sha $Recorded.source_head), $Recorded.source_changed_tracked,
                $Recorded.source_changed_untracked, $Current.source_digest, (Short-Sha $Current.source_head),
                $Current.source_changed_tracked, $Current.source_changed_untracked)
            # NAME the difference when the manifests are comparable: "something
            # changed" is a lead, "these paths appeared" is a decision.
            $was = @($Recorded.source_changed_paths | Where-Object { $_ })
            $now = @($Current.source_changed_paths | Where-Object { $_ })
            if ($was.Count -gt 0 -or $now.Count -gt 0) {
                $appeared = @($now | Where-Object { $was -notcontains $_ } | Select-Object -First 5)
                $vanished = @($was | Where-Object { $now -notcontains $_ } | Select-Object -First 5)
                $parts = @()
                if ($appeared.Count -gt 0) { $parts += ('now changed: ' + ($appeared -join ', ')) }
                if ($vanished.Count -gt 0) { $parts += ('no longer changed: ' + ($vanished -join ', ')) }
                if ($parts.Count -eq 0) { $parts += ('the same {0} path(s), but their CONTENT differs (e.g. {1})' -f $now.Count, ($now | Select-Object -First 3) -join ', ') }
                Add-Reason 'source-drift-detail' ($parts -join '; ')
            }
        }
        if ($Recorded.sources_newer_than_binary -eq $true) {
            Add-Reason 'binary-predates-sources' ("a changed source was ALREADY NEWER than the binary when the arm ran: {0}" -f $Recorded.sources_newer_detail)
        }
        if ($Recorded.sources_changed_during_run -eq $true) {
            Add-Reason 'source-moved-in-run' 'sources changed WHILE the arm ran (pre and post capture digests differ)'
        }
        if ($Recorded.binary_changed_during_run -eq $true) {
            Add-Reason 'binary-moved-in-run' 'the binary changed WHILE the arm ran (a module hash differs between the pre and post capture)'
        }

        $binState = "$($Recorded.binary_state)"
        if ($binState -eq 'not-supplied') {
            Add-Reason 'binary-unknown' 'no binary identity was recorded for this arm, so the engine it ran cannot be certified'
        } elseif ($binState -eq 'missing') {
            Add-Reason 'binary-missing' ("the binary named when the arm ran did not exist ({0})" -f $Recorded.binary_exe)
        } elseif ($binState -eq 'hashed') {
            $now = Get-BinaryIdentityNow -Recorded $Recorded -RepoRoot $RepoRoot
            if ($null -eq $now -or $now.state -ne 'hashed') {
                Add-Reason 'binary-unknown' ("the binary this arm ran cannot be re-read now (binary_dir={0})" -f $Recorded.binary_dir)
            } else {
                foreach ($leaf in (Get-FieldNames $Recorded.modules)) {
                    $was = "$($Recorded.modules.$leaf)"
                    $is = "$($now.modules.$leaf)"
                    if (-not $is) { Add-Reason 'module-drift' ("module {0} is GONE from {1}" -f $leaf, $Recorded.binary_dir) }
                    elseif ($is -ne $was) { Add-Reason 'module-drift' ("module {0} was REBUILT since this arm: recorded {1}, now {2}" -f $leaf, $was, $is) }
                }
            }
        }
    }

    # Precedence, strongest statement first. A drifted SOURCE is the strongest:
    # the arm's code may not be the code the record describes. An uncertifiable
    # binary is the weakest, but it still never reports a pass.
    $precedence = @(
        @{ tag = 'no-verdict';            verdict = 'NO-VERDICT';                exit = 2 },
        @{ tag = 'no-provenance';         verdict = 'NO-PROVENANCE';             exit = 2 },
        @{ tag = 'source-undigestible';   verdict = 'CANNOT-EVALUATE';           exit = 2 },
        @{ tag = 'source-drift';          verdict = 'STALE-SOURCES';             exit = 1 },
        @{ tag = 'binary-predates-sources'; verdict = 'SOURCES-NEWER-THAN-BINARY'; exit = 1 },
        @{ tag = 'source-moved-in-run';   verdict = 'SOURCES-CHANGED-DURING-RUN'; exit = 1 },
        @{ tag = 'binary-moved-in-run';   verdict = 'BINARY-CHANGED-DURING-RUN'; exit = 1 },
        @{ tag = 'module-drift';          verdict = 'STALE-BINARY';              exit = 1 },
        @{ tag = 'binary-missing';        verdict = 'STALE-BINARY';              exit = 1 },
        @{ tag = 'binary-unknown';        verdict = 'BINARY-UNKNOWN';            exit = 2 }
    )
    $verdict = 'CURRENT'
    $exit = 0
    foreach ($p in $precedence) {
        if ($reasons | Where-Object { $_.tag -eq $p.tag }) { $verdict = $p.verdict; $exit = $p.exit; break }
    }
    return [pscustomobject]@{
        verdict = $verdict
        exit = $exit
        reasons = @($reasons | ForEach-Object { $_.message })
        tags = @($reasons | ForEach-Object { $_.tag })
        recorded = $Recorded
        current = $Current
    }
}
