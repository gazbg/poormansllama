# T-GPLR-40 stock probe: locked inner script (invoked by the wrapper via -File).
param(
  [string]$Exe,
  [string]$Model,
  [string]$Prompt
)
$ErrorActionPreference = 'Continue'
# The probe links llama.lib (src/Release) but loads llama.dll/ggml.dll from
# bin/Release at runtime -- put the DLL dir on PATH for the child process.
$env:PATH = (Join-Path (Split-Path (Split-Path $PSCommandPath -Parent) -Parent) 'build\llama-cuda\bin\Release') + ';' + $env:PATH
foreach ($kv in @('q8_0','f16')) {
  Write-Host "[gplr40-stock] arm kv=$kv starting..."
  & $Exe $Model $Prompt $kv 1024 144128 2>&1 |
    Select-String -Pattern 'probe:|CUDA|illegal|error' |
    Select-Object -Last 14 | ForEach-Object { Write-Host $_ }
  Write-Host ("ARM kv={0} EXIT: {1}" -f $kv, $LASTEXITCODE)
}
Write-Host 'DONE'
