param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceId,
    [string]$Email = '',
    [string]$HfAccountHelper = (Join-Path $env:USERPROFILE 'Downloads\mainframe\hf-account.ps1'),
    [string]$SecretsPath = ''
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($SecretsPath)) {
    $SecretsPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'config\local.secrets.json'
}

if (-not (Test-Path -LiteralPath $SecretsPath)) {
    throw "Local secret file not found: $SecretsPath"
}

$secrets = Get-Content -LiteralPath $SecretsPath -Raw | ConvertFrom-Json
$secretLines = New-Object System.Collections.Generic.List[string]
foreach ($property in $secrets.PSObject.Properties) {
    if ([string]::IsNullOrWhiteSpace([string]$property.Value)) {
        continue
    }
    $secretLines.Add("$($property.Name)=$($property.Value)") | Out-Null
}

if ($secretLines.Count -eq 0) {
    throw 'No secrets found to sync.'
}

$tempPath = Join-Path ([System.IO.Path]::GetTempPath()) ("hf-litellm-secrets-{0}.env" -f ([guid]::NewGuid().ToString('N')))
try {
    Set-Content -LiteralPath $tempPath -Value $secretLines -Encoding UTF8
    attrib +h $tempPath | Out-Null

    if (-not [string]::IsNullOrWhiteSpace($Email) -and (Test-Path -LiteralPath $HfAccountHelper)) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $HfAccountHelper run $Email spaces secrets add $SpaceId --secrets-file $tempPath --format json | Out-Null
    } else {
        & hf spaces secrets add $SpaceId --secrets-file $tempPath --format json | Out-Null
    }

    if ($LASTEXITCODE -ne 0) {
        throw "hf spaces secrets add failed with exit code $LASTEXITCODE"
    }
} finally {
    Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
}

[pscustomobject]@{
    spaceId = $SpaceId
    profile = if ([string]::IsNullOrWhiteSpace($Email)) { 'hf-cli' } else { $Email }
    secretsSynced = $secretLines.Count
} | ConvertTo-Json
