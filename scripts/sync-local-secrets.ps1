param(
    [Parameter(Mandatory = $true)]
    [string]$Email,
    [Parameter(Mandatory = $true)]
    [string]$SpreadsheetId,
    [string]$SheetName = 'emails',
    [string]$GwsAccountHelper = (Join-Path $env:USERPROFILE 'Downloads\mainframe\gws-account.ps1'),
    [string]$OutputPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'config\local.secrets.json')
)

$ErrorActionPreference = 'Stop'

function ConvertTo-ColumnName {
    param([int]$Index)

    $name = ''
    while ($Index -gt 0) {
        $Index--
        $name = [char](65 + ($Index % 26)) + $name
        $Index = [math]::Floor($Index / 26)
    }
    return $name
}

function Get-JsonObjectFromOutput {
    param([string[]]$Output)

    $text = ($Output -join "`n")
    $start = $text.IndexOf('{')
    $end = $text.LastIndexOf('}')
    if ($start -lt 0 -or $end -le $start) {
        throw 'Could not find JSON in command output.'
    }
    return $text.Substring($start, $end - $start + 1) | ConvertFrom-Json
}

function Quote-ProcessArgument {
    param([string]$Value)

    if ($Value -match '^[A-Za-z0-9_./:+@=-]+$') {
        return $Value
    }

    return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-AiApiEntries {
    param([string]$CellText)

    $entries = New-Object System.Collections.Generic.List[string]
    $inBlock = $false
    $lines = $CellText -split "`r?`n"

    foreach ($line in $lines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^\[(.+)\]$') {
            $section = $Matches[1].Trim()
            $inBlock = ($section -ieq 'ai apis')
            continue
        }

        if ($inBlock -and $trimmed.Length -gt 0) {
            $entries.Add($trimmed) | Out-Null
        }
    }

    return $entries
}

function Set-PrivateFileAcl {
    param([string]$Path)

    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $acl = New-Object System.Security.AccessControl.FileSecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule($identity, 'FullControl', 'Allow')))
    $acl.AddAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule('SYSTEM', 'FullControl', 'Allow')))
    Set-Acl -LiteralPath $Path -AclObject $acl
    attrib +h $Path | Out-Null
}

$providerSlots = @{
    'admin.mistral.ai' = @('MISTRAL_API_KEY')
    'aimlapi.com' = @('AIMLAPI_API_KEY')
    'aistudio.google.com' = @('GEMINI_API_KEY')
    'app.edenai.run' = @('EDENAI_API_KEY')
    'app.genlabs.dev' = @('GENLABS_API_KEY')
    'app.inference.sh' = @('INFERENCE_SH_API_KEY')
    'app.tavily.com' = @('TAVILY_API_KEY')
    'assemblyai.com' = @('ASSEMBLYAI_API_KEY')
    'cloud.cerebras.ai' = @('CEREBRAS_API_KEY')
    'cloud.sambanova.ai' = @('SAMBANOVA_API_KEY')
    'cloudflare.com' = @('CLOUDFLARE_API_TOKEN')
    'console.deepgram.com' = @('DEEPGRAM_API_KEY')
    'console.groq.com' = @('GROQ_API_KEY')
    'dashboard.cohere.com' = @('COHERE_API_KEY')
    'dashboard.exa.ai' = @('EXA_API_KEY')
    'discord.com' = @('DISCORD_TOKEN')
    'elevenlabs.io' = @('ELEVENLABS_API_KEY')
    'github.com' = @('GITHUB_API_KEY')
    'huggingface.co' = @('HUGGINGFACE_API_KEY', 'HUGGINGFACE_API_KEY_1', 'HUGGINGFACE_API_KEY_2')
    'jina.ai' = @('JINA_AI_API_KEY')
    'mapleflow.io' = @('MAPLEFLOW_API_KEY')
    'modal.com' = @('MODAL_API_KEY')
    'openrouter.ai' = @(
        'OPENROUTER_API_KEY',
        'OPENROUTER_API_KEY_1',
        'OPENROUTER_API_KEY_2',
        'OPENROUTER_API_KEY_3',
        'OPENROUTER_API_KEY_4'
    )
    'platform.openai.com' = @('OPENAI_API_KEY')
    'platform.worldlabs.ai' = @('WORLDLABS_API_KEY')
    'playground.electronhub.ai' = @('ELECTRONHUB_API_KEY')
    'playground.twelvelabs.io' = @('TWELVELABS_API_KEY')
    'pollinations.ai' = @('POLLINATIONS_API_KEY', 'POLLINATIONS_API_KEY_1')
    'stablehorde.net' = @('STABLEHORDE_API_KEY')
    'vercel.com' = @('VERCEL_AI_GATEWAY_API_KEY')
    'voidai.app' = @('VOIDAI_API_KEY')
    'xiaomimimo.com' = @('XIAOMI_MIMO_API_KEY', 'XIAOMI_MIMO_PLAN_TOKEN')
    'you.com' = @('YOU_API_KEY')
}

if (-not (Test-Path -LiteralPath $GwsAccountHelper)) {
    throw "GWS account helper not found: $GwsAccountHelper"
}

$arguments = @(
    '-NoProfile',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    $GwsAccountHelper,
    'run',
    $Email,
    'sheets',
    '+read',
    '--spreadsheet',
    $SpreadsheetId,
    '--range',
    $SheetName,
    '--format',
    'json'
)

$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = 'powershell.exe'
$processInfo.Arguments = (($arguments | ForEach-Object { Quote-ProcessArgument $_ }) -join ' ')
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $processInfo
[void]$process.Start()
$stdout = $process.StandardOutput.ReadToEnd()
$stderr = $process.StandardError.ReadToEnd()
$process.WaitForExit()

if ($process.ExitCode -ne 0) {
    $safeError = (($stderr -split "`r?`n") | Where-Object { $_ -and $_ -notmatch '\S{20,}' }) -join "`n"
    throw "gws read failed with exit code $($process.ExitCode). Sanitized stderr:`n$safeError"
}

$valueRange = Get-JsonObjectFromOutput -Output @($stdout)
$headerRow = if ($valueRange.values.Count -ge 1) { @($valueRange.values[0]) } else { @() }
$inheritedHeaders = @{}
$lastHeader = ''
for ($colIndex = 0; $colIndex -lt $headerRow.Count; $colIndex++) {
    $candidate = [string]$headerRow[$colIndex]
    if ($candidate.Trim().Length -gt 0) {
        $lastHeader = $candidate.Trim()
    }
    $inheritedHeaders[$colIndex] = $lastHeader
}

$providerEntries = @{}
$cellSummary = New-Object System.Collections.Generic.List[object]
for ($rowIndex = 0; $rowIndex -lt $valueRange.values.Count; $rowIndex++) {
    $row = @($valueRange.values[$rowIndex])
    for ($colIndex = 0; $colIndex -lt $row.Count; $colIndex++) {
        $cell = [string]$row[$colIndex]
        if ($cell -notmatch '(?im)^\s*\[ai apis\]\s*$') {
            continue
        }

        $provider = if ($inheritedHeaders.ContainsKey($colIndex)) { [string]$inheritedHeaders[$colIndex] } else { '' }
        if (-not $providerSlots.ContainsKey($provider)) {
            throw "No wrapper env mapping for provider: $provider"
        }

        if (-not $providerEntries.ContainsKey($provider)) {
            $providerEntries[$provider] = New-Object System.Collections.Generic.List[string]
        }

        $entries = @(Get-AiApiEntries -CellText $cell)
        foreach ($entry in $entries) {
            $providerEntries[$provider].Add([string]$entry) | Out-Null
        }

        $coordinate = '{0}!{1}{2}' -f $SheetName, (ConvertTo-ColumnName ($colIndex + 1)), ($rowIndex + 1)
        $cellSummary.Add([pscustomobject]@{
            cell = $coordinate
            provider = $provider
            count = $entries.Count
        }) | Out-Null
    }
}

$secrets = [ordered]@{}
$assignments = New-Object System.Collections.Generic.List[object]
foreach ($provider in ($providerEntries.Keys | Sort-Object)) {
    $values = @($providerEntries[$provider])
    $slots = @($providerSlots[$provider])
    if ($values.Count -gt $slots.Count) {
        throw "Provider $provider has $($values.Count) entries but only $($slots.Count) env slots."
    }

    for ($i = 0; $i -lt $values.Count; $i++) {
        $envName = $slots[$i]
        $secrets[$envName] = $values[$i]
        $assignments.Add([pscustomobject]@{
            provider = $provider
            env = $envName
            status = 'synced'
        }) | Out-Null
    }
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

($secrets | ConvertTo-Json -Depth 4) | Set-Content -LiteralPath $OutputPath -Encoding UTF8
Set-PrivateFileAcl -Path $OutputPath

[pscustomobject]@{
    outputPath = $OutputPath
    sheetBlocks = $cellSummary.Count
    sheetEntries = ($cellSummary | Measure-Object -Property count -Sum).Sum
    secretEnvCount = $secrets.Count
    assignments = $assignments
} | ConvertTo-Json -Depth 6
