param(
    [string]$HostName = '127.0.0.1',
    [int]$Port = 4000,
    [string]$LiteLlmExe = '',
    [switch]$StrictEnv
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$template = Join-Path $root 'config\config.yaml'
$rendered = Join-Path $root 'config\local.generated.yaml'
$secrets = Join-Path $root 'config\local.secrets.json'

$liteLlmCommand = @()
if ($LiteLlmExe -and (Test-Path -LiteralPath $LiteLlmExe)) {
    $liteLlmCommand = @($LiteLlmExe)
} else {
    $pathLiteLlm = Get-Command litellm -ErrorAction SilentlyContinue
    if ($pathLiteLlm) {
        $liteLlmCommand = @($pathLiteLlm.Source)
    } else {
        $uvx = Get-Command uvx -ErrorAction SilentlyContinue
        if (-not $uvx) {
            throw 'LiteLLM is not installed and uvx was not found. Install with: uv tool install litellm'
        }
        $liteLlmCommand = @($uvx.Source, '--from', 'litellm', 'litellm')
    }
}

$loadOutput = . (Join-Path $PSScriptRoot 'load-local-secrets.ps1') -SecretsPath $secrets
Write-Host $loadOutput

$renderArgs = @(
    (Join-Path $PSScriptRoot 'render-config.py'),
    $template,
    $rendered,
    '--secrets',
    $secrets
)
if ($StrictEnv) {
    $renderArgs += '--strict-env'
}

& python @renderArgs
if ($LASTEXITCODE -ne 0) {
    throw "render-config.py failed with exit code $LASTEXITCODE"
}

$liteLlmExeResolved = $liteLlmCommand[0]
$liteLlmArgs = @()
if ($liteLlmCommand.Count -gt 1) {
    $liteLlmArgs = $liteLlmCommand[1..($liteLlmCommand.Count - 1)]
}

& $liteLlmExeResolved @liteLlmArgs --config $rendered --host $HostName --port $Port
