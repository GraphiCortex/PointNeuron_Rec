param(
    [ValidateSet('prepare', 'smoke', 'train', 'normalization-smoke')]
    [Parameter(Mandatory = $true)][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$Manifest,
    [string]$BaselineModels,
    [string]$OutputDirectory,
    [string]$Python,
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cuda',
    [ValidateRange(1, 2147483647)][int]$Steps = 100,
    [int[]]$EvalSteps = @(),
    [ValidateRange(1, 2147483647)][int]$LogEvery = 5,
    [Nullable[double]]$DecoderDropout
)
$ErrorActionPreference = 'Stop'
if ($null -ne $DecoderDropout -and $Action -ne 'smoke') {
    throw 'DecoderDropout is restricted to the fixed-crop smoke comparison.'
}
if ($EvalSteps.Count -gt 0 -and $Action -notin @('smoke', 'normalization-smoke')) {
    throw 'EvalSteps is restricted to the fixed-crop smoke comparison.'
}
if ($Action -eq 'normalization-smoke') {
    if (-not $BaselineModels) { throw 'Supply -BaselineModels with the completed BatchNorm models directory.' }
    $BaselineModels = (Resolve-Path -LiteralPath $BaselineModels).Path
} elseif ($BaselineModels) { throw 'BaselineModels is restricted to normalization-smoke.' }
foreach ($evaluationStep in $EvalSteps) {
    if ($evaluationStep -lt 1 -or $evaluationStep -gt $Steps) {
        throw 'Each EvalSteps value must be between 1 and Steps.'
    }
}
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply -Python with the local PyTorch environment.' }
if (-not $Manifest) {
    if ($Action -eq 'prepare') {
        $Manifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/repair_20261011_130158/cache/manifest.json'
    } else { throw 'Supply -Manifest pointing to the prepared query cache manifest.json.' }
}
$Manifest = (Resolve-Path -LiteralPath $Manifest).Path
if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $projectRoot ('tmp/pointneuron2/local_runs/query_{0}_{1}_{2}' -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID)
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a fresh output directory to preserve earlier runs.' }
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$runDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
if ($Action -eq 'prepare') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'build_query_cache.py'), '--manifest', $Manifest,
        '--data-root', (Join-Path $ArtifactRoot 'data/gold166'), '--output-dir', (Join-Path $runDirectory 'cache'))
    Write-Output 'CPU target preparation: corrected image candidates unchanged; common SWC curve targets; no training.'
} elseif ($Action -eq 'normalization-smoke') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'compare_query_normalization.py'),
        '--baseline-models', $BaselineModels, '--manifest', $Manifest,
        '--output-dir', (Join-Path $runDirectory 'models'), '--device', $Device,
        '--steps', "$Steps", '--log-every', "$LogEvery")
    if ($EvalSteps.Count -gt 0) {
        $cliArguments += '--eval-steps'
        $cliArguments += @($EvalSteps | ForEach-Object { "$_" })
    }
    Write-Output "Fresh query-only pooled LayerNorm control: $Steps updates; reuse the matched saved BatchNorm run; device=$Device."
    Write-Output 'Decoder dropout stays zero; saved inference checks include single-crop evaluation. This is not validation.'
} else {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'train_query_comparison.py'), '--manifest', $Manifest,
        '--output-dir', (Join-Path $runDirectory 'models'), '--device', $Device, '--log-every', "$LogEvery")
    if ($Action -eq 'smoke') {
        $cliArguments += @('--smoke', '--steps', "$Steps")
        if ($EvalSteps.Count -gt 0) {
            $cliArguments += '--eval-steps'
            $cliArguments += @($EvalSteps | ForEach-Object { "$_" })
        }
        $milestones = @(@($EvalSteps) + $Steps | Sort-Object -Unique)
        Write-Output "Saved inference checks at updates: $($milestones -join ', '); final update remains the primary checkpoint."
        if ($null -ne $DecoderDropout) {
            $cliArguments += @('--decoder-dropout', ([double]$DecoderDropout).ToString([System.Globalization.CultureInfo]::InvariantCulture))
            Write-Output "Query decoder dropout override: $DecoderDropout; cache and base settings unchanged."
        }
        Write-Output "Fixed two-training-crop check: both arms, $Steps updates each; device=$Device. This is not validation."
    } else {
        Write-Output "Matched pilot: 33 train / 7 validation neurons, 30 epochs per arm; device=$Device. Whole-neuron evaluation is separate."
    }
}
$launch = [ordered]@{ status = 'running'; action = $Action; manifest = $Manifest; device = $Device; arguments = $cliArguments }
$launchPath = Join-Path $runDirectory 'launch.json'
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Write-Output "Local run directory: $runDirectory"
try {
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'console.log')
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -ne 0) { throw "Run failed with exit code $runExitCode" }
    $launch.status = 'completed'
} catch {
    $launch.status = 'failed_or_interrupted'
    throw
} finally {
    $launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchPath -Encoding UTF8
}
