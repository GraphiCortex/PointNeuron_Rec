param(
    [Parameter(Mandatory=$true)][ValidateSet('preflight','train','compare')][string]$Action,
    [string]$ArtifactRoot = 'C:\Users\Seywe\Documents\GitHub\PointNeuron',
    [string]$ExperimentDirectory,
    [string]$SourceExperiment,
    [string]$Python,
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $Python) { $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe' }
if (-not $ExperimentDirectory) { $ExperimentDirectory = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_attention_20261013' }
if (-not $SourceExperiment) { $SourceExperiment = Join-Path $projectRoot 'tmp/pointneuron2/local_runs/field_pilot_20261012' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Supply the existing local PyTorch environment with -Python.' }
$manifest = Join-Path $SourceExperiment 'cache/manifest.json'
$models = Join-Path $SourceExperiment 'model'
foreach ($path in @($manifest,(Join-Path $models 'pilot_report.json'),(Join-Path $models 'checkpoints/field_epoch_000.pt'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing completed baseline input: $path" }
}
$statusFile = switch ($Action) { 'preflight' {'timing.json'} 'train' {'training_comparison.json'} 'compare' {'sampling_comparison.json'} }
if (Test-Path -LiteralPath (Join-Path $ExperimentDirectory $statusFile)) { throw 'Preserve completed/failed outputs; choose a fresh ExperimentDirectory.' }
if ($Action -ne 'preflight' -and -not (Test-Path -LiteralPath (Join-Path $ExperimentDirectory 'timing.json'))) { throw 'Run -Action preflight first.' }
if ($Action -eq 'compare' -and -not (Test-Path -LiteralPath (Join-Path $ExperimentDirectory 'training_comparison.json'))) { throw 'Complete and review -Action train first.' }
if (-not (Test-Path -LiteralPath $ExperimentDirectory)) { New-Item -ItemType Directory -Path $ExperimentDirectory | Out-Null }
$cliArguments = @('-u',(Join-Path $PSScriptRoot 'field_attention_experiment.py'),'--action',$Action,
    '--experiment-dir',$ExperimentDirectory,'--manifest',$manifest,'--baseline-models',$models,
    '--data-root',(Join-Path $ArtifactRoot 'data/gold166'),'--device',$Device)
Write-Output 'Phase 1 only: fresh baseline versus two additive U-Net skip gates; PointNeXt -> Transformer -> GATv2 remains intended.'
Write-Output 'Reuse native 33-training / 7-development cache; same loss, LR .001, 30 epochs per model, batch 1, no augmentation; reserved test stays excluded.'
if ($Action -eq 'preflight') { Write-Output 'Eight throwaway updates total, four per model; no trained checkpoint retained; estimate both fits before launch.' }
elseif ($Action -eq 'train') { Write-Output 'Two sequential visible fits; identical shared initial tensors and crop order; checkpoints 0/5/15/30. No sampling comparison starts automatically.' }
else { Write-Output 'Two frozen epoch-30 sampling comparisons; unchanged A/B/C/D, budgets 512/2048, seeds 0-4; zero optimizer updates.' }
Write-Output "Experiment: $ExperimentDirectory"
& $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $ExperimentDirectory "$Action.console.log")
if ($LASTEXITCODE -ne 0) { throw "Attention $Action failed with exit code $LASTEXITCODE. Preserve logs; no automatic restart." }
