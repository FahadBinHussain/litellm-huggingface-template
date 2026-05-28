param(
    [string]$SecretsPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'config\local.secrets.json')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $SecretsPath)) {
    throw "Local secret file not found: $SecretsPath"
}

$secrets = Get-Content -LiteralPath $SecretsPath -Raw | ConvertFrom-Json
foreach ($property in $secrets.PSObject.Properties) {
    if ([string]::IsNullOrWhiteSpace([string]$property.Value)) {
        continue
    }
    [Environment]::SetEnvironmentVariable($property.Name, [string]$property.Value, 'Process')
}

[pscustomobject]@{
    secretsPath = $SecretsPath
    loaded = $secrets.PSObject.Properties.Count
} | ConvertTo-Json
