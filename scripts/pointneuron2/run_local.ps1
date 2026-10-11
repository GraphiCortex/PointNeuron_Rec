param(
    [ValidateSet('sampling-probe', 'sampling-cache', 'sampling-train', 'sampling-smoke', 'ablation-prepare', 'ablation-train', 'ablation-evaluate', 'preprocessing-compare', 'noise-compare', 'baseline-prepare', 'baseline-smoke', 'baseline-train')]
    [string]$Action = 'sampling-probe',
    [Parameter(Mandatory = $true)]
    [string]$ArtifactRoot,
    [string]$Python,
    [string]$OutputDirectory,
    [string]$Manifest,
    [string]$Comparison,
    [string]$Experiment
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$artifactDirectory = (Resolve-Path -LiteralPath $ArtifactRoot).Path
if (-not $Python) {
    $Python = Join-Path $projectRoot 'tmp/pointneuron2/sampling_env/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python environment not found. Supply -Python with a CUDA-enabled Python executable containing the pointneuron2 dependencies.'
}
if ($Action -eq 'noise-compare' -and (-not $Comparison -or -not (Test-Path -LiteralPath $Comparison -PathType Leaf))) {
    throw 'noise-compare requires -Comparison pointing to the completed mask comparison.json whose exact patch boxes should be reused.'
}
if ($Action -in @('ablation-train', 'ablation-evaluate') -and
    (-not $Experiment -or -not (Test-Path -LiteralPath (Join-Path $Experiment 'experiment.json') -PathType Leaf))) {
    throw 'This action requires -Experiment pointing to the prepared experiment directory.'
}
if (-not $OutputDirectory) {
    $runName = '{0}_{1}_{2}' -f $Action, (Get-Date -Format 'yyyyMMdd_HHmmss'), $PID
    $OutputDirectory = Join-Path $projectRoot "tmp/pointneuron2/local_runs/$runName"
}
$runDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $runDirectory) {
    throw "Run directory already exists: $runDirectory. Choose a new directory to preserve earlier results."
}
New-Item -ItemType Directory -Path $runDirectory | Out-Null

if ($Action -eq 'sampling-train' -and (-not $Manifest -or -not (Test-Path -LiteralPath $Manifest -PathType Leaf))) {
    throw 'sampling-train requires -Manifest pointing to a completed sampling cache manifest.'
}

if ($Action -like 'ablation-*') {
    $stage = $Action.Substring('ablation-'.Length)
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'sampling_ablation.py'), '--stage', $stage)
    if ($stage -eq 'prepare') {
        $cliArguments += @('--artifact-root', $artifactDirectory, '--output-dir', (Join-Path $runDirectory 'experiment'))
        Write-Output 'CPU cache preparation only: 33 train and 7 validation neurons, up to 8 image-defined patches each; fixed threshold, full guides. Training is a separate command.'
    } else {
        $cliArguments += @('--experiment', (Resolve-Path -LiteralPath $Experiment).Path, '--device', 'cuda')
        if ($stage -eq 'train') {
            Write-Output 'Three sequential CUDA runs, five epochs each: random, learned without guide loss, learned with guide loss. Every batch is printed.'
        } else {
            Write-Output 'Whole-neuron reconstruction on the same 7 validation neurons for each arm, then spatial/topology scoring. Test neurons remain reserved.'
        }
    }
} elseif ($Action -in @('preprocessing-compare', 'noise-compare')) {
    Write-Output 'CPU preprocessing comparison: six training neurons, full guides, then candidate/selection diagnostics. No training is launched.'
    $foregroundConfig = if ($Action -eq 'noise-compare') { 'foreground_noise_comparison.json' } else { 'foreground_comparison_expanded.json' }
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'compare_foreground.py'),
        '--root', (Join-Path $artifactDirectory 'data/gold166'),
        '--config', (Join-Path $projectRoot "configs/pointneuron2/$foregroundConfig"),
        '--output-dir', (Join-Path $runDirectory 'masks'))
    if ($Action -eq 'noise-compare') {
        $cliArguments += @('--boxes-from', (Resolve-Path -LiteralPath $Comparison).Path)
        Write-Output 'Experimental noise-aware mask versus fixed/background Otsu; exact previous boxes; 4-sigma seeds and 2-sigma connected foreground.'
    }
} elseif ($Action -in @('sampling-cache', 'sampling-smoke')) {
    Write-Output 'Preparing image candidates and guides on CPU; CUDA training follows for sampling-smoke.'
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'build_sampling_cache.py'),
        '--root', (Join-Path $artifactDirectory 'data/gold166'), '--output-dir', (Join-Path $runDirectory 'cache'))
    if ($Action -eq 'sampling-smoke') {
        $cliArguments += @('--sample-index', '0', '--sample-index', '19', '--max-patches', '4')
    }
} elseif ($Action -eq 'sampling-train') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'train_sampling.py'),
        '--manifest', $Manifest, '--output-dir', (Join-Path $runDirectory 'model'), '--device', 'cuda')
} elseif ($Action -eq 'sampling-probe') {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'sampling_probe.py'),
        '--root', (Join-Path $artifactDirectory 'data/gold166'),
        '--output', (Join-Path $runDirectory 'sampling_probe.json'))
} else {
    $cliArguments = @('-u', (Join-Path $PSScriptRoot 'baseline.py'),
        '--artifact-root', $artifactDirectory, '--output-root', $runDirectory)
    if ($Action -eq 'baseline-smoke') { $cliArguments += '--smoke' }
    if ($Action -eq 'baseline-train') { $cliArguments += '--train' }
}

$launchPath = Join-Path $runDirectory 'launch.json'
$launchRecord = [ordered]@{
    status = 'running'; action = $Action; python = $Python; arguments = $cliArguments;
    started_at = (Get-Date).ToUniversalTime().ToString('o'); machine = [Environment]::MachineName
}
$launchRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchPath -Encoding UTF8
Write-Output "Local run directory: $runDirectory"
try {
    & $Python @cliArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'console.log')
    $runExitCode = $LASTEXITCODE
    if ($runExitCode -eq 0 -and $Action -eq 'sampling-smoke') {
        Write-Output 'Cache preparation completed. Starting two-epoch CUDA training with live batch output.'
        $trainArguments = @('-u', (Join-Path $PSScriptRoot 'train_sampling.py'),
            '--manifest', (Join-Path $runDirectory 'cache/manifest.json'),
            '--output-dir', (Join-Path $runDirectory 'model'), '--epochs', '2',
            '--limit-train-batches', '2', '--device', 'cuda', '--log-every', '1')
        $launchRecord['training_arguments'] = $trainArguments
        & $Python @trainArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'training.log')
        $runExitCode = $LASTEXITCODE
    }
    if ($runExitCode -eq 0 -and $Action -in @('preprocessing-compare', 'noise-compare')) {
        Write-Output 'Mask comparison completed. Comparing candidate pools and fixed point selectors on saved full guides.'
        $candidateConfig = if ($Action -eq 'noise-compare') { 'candidate_noise_comparison.json' } else { 'candidate_comparison.json' }
        $candidateArguments = @('-u', (Join-Path $PSScriptRoot 'compare_candidates.py'),
            '--comparison', (Join-Path $runDirectory 'masks/comparison.json'),
            '--config', (Join-Path $projectRoot "configs/pointneuron2/$candidateConfig"),
            '--output-dir', (Join-Path $runDirectory 'candidates'))
        $launchRecord['candidate_arguments'] = $candidateArguments
        & $Python @candidateArguments 2>&1 | Tee-Object -FilePath (Join-Path $runDirectory 'candidates.log')
        $runExitCode = $LASTEXITCODE
    }
    $launchRecord['status'] = if ($runExitCode -eq 0) { 'completed' } else { 'failed' }
    $launchRecord['exit_code'] = $runExitCode
} catch {
    $launchRecord['status'] = 'interrupted_or_failed'
    $launchRecord['error'] = $_.Exception.Message
    throw
} finally {
    $launchRecord['finished_at'] = (Get-Date).ToUniversalTime().ToString('o')
    $launchRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $launchPath -Encoding UTF8
}
exit $runExitCode
