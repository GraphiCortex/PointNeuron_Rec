param(
    [Parameter(Mandatory=$true)][ValidateSet('prepare','preflight','train','compare')][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$ExperimentDirectory,
    [string]$SourceManifest,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_pilot_20261012' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply -Python with the local PyTorch environment.' }
if ($Action -ne 'prepare' -and $SourceManifest) { throw 'SourceManifest applies only to native cache preparation.' }
$cacheManifest = Join-Path $ExperimentDirectory 'cache/manifest.json'
$timingReport = Join-Path $ExperimentDirectory 'timing/timing.json'
$models = Join-Path $ExperimentDirectory 'model'
$config = Join-Path $projectRoot 'configs/pointneuron2/field_pilot.json'
$cliArguments = @('-u')
switch ($Action) {
    'prepare' {
        if (-not $SourceManifest) { $SourceManifest = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/query_implementation_20261011/cache/manifest.json' }
        if (-not (Test-Path -LiteralPath $SourceManifest -PathType Leaf)) { throw "Missing source cache: $SourceManifest" }
        $output = Join-Path $ExperimentDirectory 'cache'
        $cliArguments += @((Join-Path $PSScriptRoot 'prepare_field_pilot.py'), '--manifest', $SourceManifest,
            '--data-root', (Join-Path $ArtifactRoot 'data/gold166'), '--output-dir', $output, '--config', $config)
        Write-Output 'CPU native-image preparation: existing 262 train / 56 development crop boxes; full-segment weak labels; no model or optimizer.'
        Write-Output 'No-trace crops remain unsupervised. Native usable-label counts are reported at completion. Reserved test neurons are excluded.'
    }
    'preflight' {
        $output = Join-Path $ExperimentDirectory 'timing'
        $cliArguments += @((Join-Path $PSScriptRoot 'train_field_pilot.py'), '--manifest', $cacheManifest,
            '--output-dir', $output, '--config', $config, '--device', $Device, '--preflight')
        Write-Output 'Timing only: four throwaway updates, no trained checkpoint. Includes loading/transfer costs; prints the 30-epoch estimate.'
    }
    'train' {
        if (-not (Test-Path -LiteralPath $timingReport -PathType Leaf)) { throw 'Run -Action preflight before training.' }
        $output = $models
        $cliArguments += @((Join-Path $PSScriptRoot 'train_field_pilot.py'), '--manifest', $cacheManifest,
            '--output-dir', $output, '--config', $config, '--device', $Device, '--timing', $timingReport)
        Write-Output 'Visible local training: fresh U-Net field, 33 train / 7 development neurons; 30 epochs, batch 1, LR .001, float32, no augmentation.'
        Write-Output 'Saved evaluations/checkpoints: 0/5/15/30. Only usable training labels enter the optimizer. Fixed epoch 30 is used for the sampling comparison.'
    }
    'compare' {
        $output = Join-Path $ExperimentDirectory 'comparison'
        if (-not (Test-Path -LiteralPath (Join-Path $models 'pilot_report.json') -PathType Leaf)) { throw 'Complete -Action train before comparison.' }
        $cliArguments += @((Join-Path $PSScriptRoot 'compare_field_pilot.py'), '--manifest', $cacheManifest,
            '--models', $models, '--data-root', (Join-Path $ArtifactRoot 'data/gold166'),
            '--output-dir', $output, '--config', $config, '--device', $Device)
        Write-Output 'Fixed final field: seven development neurons, all 56 crop boxes, four controls, budgets 512/2048, seeds 0-4; zero optimizer updates.'
        Write-Output 'All input clouds are frozen before reference scoring. Reports annotation agreement and chain gaps; no automatic sampler promotion.'
    }
}
if ($Action -ne 'prepare' -and -not (Test-Path -LiteralPath $cacheManifest -PathType Leaf)) { throw 'Run -Action prepare first.' }
if (Test-Path -LiteralPath $output) { throw "Preserving earlier output: $output. Choose another -ExperimentDirectory for a fresh pilot." }
if (-not (Test-Path -LiteralPath $ExperimentDirectory)) { New-Item -ItemType Directory -Path $ExperimentDirectory | Out-Null }
Write-Output 'Architecture retained: Phase 1 point cloud -> PointNeXt -> Transformer -> GATv2. This runner tests only a candidate Phase 1 guide.'
Write-Output "Experiment: $ExperimentDirectory"
& $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $ExperimentDirectory "$Action.console.log")
if ($LASTEXITCODE -ne 0) { throw "Field pilot $Action failed with exit code $LASTEXITCODE. Preserve logs and choose fresh output before rerunning." }
if ($Action -eq 'prepare') {
    Write-Output 'Next: run -Action preflight with the same ExperimentDirectory. Training does not start automatically.'
} elseif ($Action -eq 'preflight') {
    Write-Output 'Review the timing estimate; -Action train is a separate visible GPU command.'
} elseif ($Action -eq 'train') {
    Write-Output 'Review pilot_report.json and development_fields overlays. The four-control comparison is a separate -Action compare command.'
}
