[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Smoke', 'Train', 'All')]
    [string]$Mode = 'Smoke',
    [string]$InputRoot = '..\data',
    [string]$OutputRoot = '..\runs',
    [ValidateSet('1-4', '5-8', '9-12', '13-16')]
    [string]$ShardGroup = '1-4',
    [int]$MaxEpochs = 50,
    [int]$EpochSize = 20000000,
    [int]$ValidationSize = 1000000,
    [int]$BatchSize = 16384,
    [ValidateRange(0.0, 1.0)]
    [double]$Lambda = 1.0,
    [string]$Gpus = '1',
    [switch]$KeepLastOnly,
    [switch]$ReusePrepared,
    [switch]$Resume,
    [string]$ResumeCheckpoint,
    [switch]$AllowDatasetTransition,
    [int]$SplitSeed = 20260825,
    [int]$SplitNumerator = 1,
    [int]$SplitDenominator = 20,
    [UInt64]$ShuffleSeed = 42,
    [UInt32]$ShuffleBufferBlocks = 4096,
    [int]$HeadCoverageRecords = 1000000,
    [int]$HeadCoverageBatchSize = 2048,
    [int]$HeadCoverageWorkers = 1,
    [double]$HeadBalanceCap = 8.0,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-AbsolutePath {
    param([Parameter(Mandatory)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $PathValue))
}

function Get-FileSha256 {
    param([Parameter(Mandatory)][string]$PathValue)
    $stream = [System.IO.File]::OpenRead($PathValue)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant()
    }
    finally {
        $hasher.Dispose()
        $stream.Dispose()
    }
}

$repoRoot = Resolve-AbsolutePath (Join-Path $PSScriptRoot '..')
$inputRootFull = Resolve-AbsolutePath $InputRoot
$outputRootFull = Resolve-AbsolutePath $OutputRoot
$manifestRoot = Join-Path $outputRootFull 'manifests'
$logsRoot = Join-Path $outputRootFull 'logs'
$trainRoot = Join-Path $outputRootFull 'train'
$smokeRoot = Join-Path $outputRootFull 'smoke'
$cacheRoot = Join-Path $outputRootFull 'cache'
$helper = Join-Path $PSScriptRoot 'create_v11_jqv4_manifest.py'
$coverageScanner = Join-Path $PSScriptRoot 'scan_v11_head_coverage.py'
$trainScript = Join-Path $repoRoot 'train_v11.py'
$loaderOverride = Join-Path $outputRootFull 'build-v11\libtraining_data_loader.dll'
$alternateLoaderOverride = Join-Path $outputRootFull 'build\libtraining_data_loader.dll'
if (-not (Test-Path -LiteralPath $loaderOverride -PathType Leaf) -and
    (Test-Path -LiteralPath $alternateLoaderOverride -PathType Leaf)) {
    $loaderOverride = $alternateLoaderOverride
}

function Initialize-Layout {
    foreach ($directory in @(
            $outputRootFull, $manifestRoot, $logsRoot, $trainRoot, $smokeRoot,
            $cacheRoot,
            (Join-Path $cacheRoot 'tmp'),
            (Join-Path $cacheRoot 'torch'),
            (Join-Path $cacheRoot 'torch-extensions'),
            (Join-Path $cacheRoot 'xdg'),
            (Join-Path $cacheRoot 'cuda'),
            (Join-Path $cacheRoot 'cupy'),
            (Join-Path $cacheRoot 'pycache'),
            (Join-Path $cacheRoot 'matplotlib'),
            (Join-Path $cacheRoot 'triton'),
            (Join-Path $cacheRoot 'torch-kernels'))) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    $env:TMP = Join-Path $cacheRoot 'tmp'
    $env:TEMP = $env:TMP
    $env:TORCH_HOME = Join-Path $cacheRoot 'torch'
    $env:TORCH_EXTENSIONS_DIR = Join-Path $cacheRoot 'torch-extensions'
    $env:XDG_CACHE_HOME = Join-Path $cacheRoot 'xdg'
    $env:CUDA_CACHE_PATH = Join-Path $cacheRoot 'cuda'
    $env:CUPY_CACHE_DIR = Join-Path $cacheRoot 'cupy'
    $env:PYTHONPYCACHEPREFIX = Join-Path $cacheRoot 'pycache'
    $env:MPLCONFIGDIR = Join-Path $cacheRoot 'matplotlib'
    $env:TRITON_CACHE_DIR = Join-Path $cacheRoot 'triton'
    $env:PYTORCH_KERNEL_CACHE_PATH = Join-Path $cacheRoot 'torch-kernels'
    # nnue_dataset.py honors this explicit path before consuming any batch.
    $env:V11_TRAINING_DATA_LOADER = $loaderOverride
    $env:PYTHONUNBUFFERED = '1'
}

function Invoke-Logged {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $logPath = Join-Path $logsRoot ($Label + '.log')
    $command = (@($Python) + $Arguments) -join ' '
    Push-Location $repoRoot
    $nativeErrorActionPreference = $ErrorActionPreference
    $exitCode = 1
    [System.IO.StreamWriter]$logWriter = $null
    try {
        $logWriter = [System.IO.StreamWriter]::new(
            $logPath, $false, [System.Text.UTF8Encoding]::new($false))
        $logWriter.AutoFlush = $true
        $logWriter.WriteLine("[$([DateTime]::Now.ToString('o'))] COMMAND $command")
        # Python/Lightning writes ordinary warnings to stderr.  Windows
        # PowerShell 5.1 promotes redirected native stderr records to errors
        # when the script-wide preference is Stop, so keep the stream alive
        # and use the process exit code as the actual success signal.
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object {
            $line = [string]$_
            Write-Host $line
            $logWriter.WriteLine($line)
        }
        $exitCode = $LASTEXITCODE
        $logWriter.WriteLine("[$([DateTime]::Now.ToString('o'))] EXIT $exitCode")
    }
    finally {
        if ($null -ne $logWriter) {
            $logWriter.Dispose()
        }
        $ErrorActionPreference = $nativeErrorActionPreference
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode; see $logPath"
    }
}

function Prepare-Manifests {
    if ($ReusePrepared) {
        $trainManifest = Join-Path $manifestRoot 'train.jqv11.json'
        $validationManifest = Join-Path $manifestRoot 'validation.jqv11.json'
        if (-not (Test-Path -LiteralPath $trainManifest) -or -not (Test-Path -LiteralPath $validationManifest)) {
            throw '-ReusePrepared requires existing train and validation manifests'
        }
        return [pscustomobject]@{ train=$trainManifest; validation=$validationManifest;
            teacher_hash=((Get-Content -LiteralPath $trainManifest -Raw | ConvertFrom-Json).teacher_dataset_sha256) }
    }
    if (-not (Test-Path -LiteralPath $inputRootFull -PathType Container)) {
        throw "InputRoot does not exist: $inputRootFull"
    }
    if (-not (Test-Path -LiteralPath $helper -PathType Leaf)) {
        throw "JQv4 manifest helper is missing: $helper"
    }
    Invoke-Logged -Label 'prepare-manifest' -Arguments @(
        $helper,
        '--input-root', $inputRootFull,
        '--output-root', $manifestRoot,
        '--seed', [string]$SplitSeed,
        '--numerator', [string]$SplitNumerator,
        '--denominator', [string]$SplitDenominator,
        '--shuffle-seed', [string]$ShuffleSeed,
        '--shuffle-buffer-blocks', [string]$ShuffleBufferBlocks,
        '--feature-set', 'HalfKAv2_hm_jieqi_v11^',
        '--shard-group', $ShardGroup,
        '--allow-extra-shards'
    )
    $trainManifest = Join-Path $manifestRoot 'train.jqv11.json'
    $validationManifest = Join-Path $manifestRoot 'validation.jqv11.json'
    if (-not (Test-Path -LiteralPath $trainManifest -PathType Leaf) -or
        -not (Test-Path -LiteralPath $validationManifest -PathType Leaf)) {
        throw 'Manifest helper completed without producing train.jqv11.json and validation.jqv11.json'
    }
    return [pscustomobject]@{
        train = $trainManifest
        validation = $validationManifest
        teacher_hash = ((Get-Content -LiteralPath $trainManifest -Raw | ConvertFrom-Json).teacher_dataset_sha256)
    }
}

function Find-ResumeCheckpoint {
    param([Parameter(Mandatory)][string]$Root)
    $checkpoint = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter 'last.ckpt' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -eq $checkpoint) {
        throw "-Resume requested but no last.ckpt exists under $Root"
    }
    return $checkpoint.FullName
}

function Resolve-ResumeCheckpoint {
    param([Parameter(Mandatory)][string]$Root)
    if ($ResumeCheckpoint) {
        $checkpoint = Resolve-AbsolutePath $ResumeCheckpoint
        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "ResumeCheckpoint does not exist: $checkpoint"
        }
        $sidecar = $checkpoint + '.abjv11.json'
        if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
            throw "ResumeCheckpoint sidecar does not exist: $sidecar"
        }
        return $checkpoint
    }
    return Find-ResumeCheckpoint -Root $Root
}

function Remove-OldCheckpoints {
    param([Parameter(Mandatory)][string]$Root)
    if (-not $KeepLastOnly) { return }
    $checkpoints = @(Get-ChildItem -LiteralPath $Root -Recurse -File -Include '*.ckpt', '*.ckpt.abjv11.json' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    $last = $checkpoints | Where-Object { $_.Name -eq 'last.ckpt' } | Select-Object -First 1
    if ($null -eq $last) { return }
    foreach ($checkpoint in $checkpoints) {
        $keep = $checkpoint.FullName -eq $last.FullName -or
            $checkpoint.FullName -eq ($last.FullName + '.abjv11.json')
        if (-not $keep) {
            Remove-Item -LiteralPath $checkpoint.FullName -Force
        }
    }
}

function Prepare-HeadCoverage {
    param([Parameter(Mandatory)][pscustomobject]$Manifests)
    if (-not (Test-Path -LiteralPath $coverageScanner -PathType Leaf)) {
        throw "V11 head coverage scanner is missing: $coverageScanner"
    }
    if ($HeadCoverageRecords -le 0 -or $HeadCoverageBatchSize -le 0 -or $HeadCoverageWorkers -lt 0) {
        throw 'Head coverage records and batch size must be positive; workers must be non-negative'
    }
    if ($HeadBalanceCap -le 0.0) {
        throw 'HeadBalanceCap must be positive'
    }
    $coveragePath = Join-Path $manifestRoot 'head-coverage-v11.json'
    if ($ReusePrepared -and (Test-Path -LiteralPath $coveragePath)) {
        return [pscustomobject]@{path=$coveragePath; sha256=(Get-FileSha256 -PathValue $coveragePath)}
    }
    Invoke-Logged -Label 'head-coverage' -Arguments @(
        $coverageScanner,
        '--manifest', $Manifests.train,
        '--records', [string]$HeadCoverageRecords,
        '--batch-size', [string]$HeadCoverageBatchSize,
        '--num-workers', [string]$HeadCoverageWorkers,
        '--cap', $HeadBalanceCap.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture),
        '--feature-name', 'HalfKAv2_hm_jieqi_v11^',
        '--output', $coveragePath
    )
    if (-not (Test-Path -LiteralPath $coveragePath -PathType Leaf)) {
        throw 'Head coverage scanner completed without producing a report'
    }
    $manifestHash = Get-FileSha256 -PathValue $Manifests.train
    $coverageHash = Get-FileSha256 -PathValue $coveragePath
    $provenancePath = Join-Path $logsRoot 'head-coverage-provenance.json'
    [pscustomobject]@{
        schema = 'abjchess-v11-runner-coverage-provenance-v1'
        generated_at = [DateTime]::Now.ToString('o')
        manifest = $Manifests.train
        manifest_sha256 = $manifestHash
        report = $coveragePath
        report_sha256 = $coverageHash
        cap = [double]$HeadBalanceCap
        records = $HeadCoverageRecords
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $provenancePath -Encoding UTF8
    return [pscustomobject]@{
        path = $coveragePath
        sha256 = $coverageHash
        provenance = $provenancePath
    }
}

function Invoke-V11Training {
    param(
        [Parameter(Mandatory)][pscustomobject]$Manifests,
        [Parameter(Mandatory)][pscustomobject]$Coverage,
        [Parameter(Mandatory)][bool]$Smoke
    )
    if (-not (Test-Path -LiteralPath $trainScript -PathType Leaf)) {
        throw "V11 training entry point is missing: $trainScript"
    }
    # Resolve the final build output after Prepare/build steps have run.  This
    # also supports callers that place the Ninja tree in ``build`` instead of
    # the canonical ``build-v11`` directory.
    $runtimeLoader = $loaderOverride
    if (-not (Test-Path -LiteralPath $runtimeLoader -PathType Leaf) -and
        (Test-Path -LiteralPath $alternateLoaderOverride -PathType Leaf)) {
        $runtimeLoader = $alternateLoaderOverride
    }
    if (-not (Test-Path -LiteralPath $runtimeLoader -PathType Leaf)) {
        throw "V11 training DLL is missing; expected $loaderOverride"
    }
    $env:V11_TRAINING_DATA_LOADER = $runtimeLoader
    $root = if ($Smoke) { $smokeRoot } else { $trainRoot }
    $epochs = if ($Smoke) { 1 } else { $MaxEpochs }
    $epochSize = if ($Smoke) { 2048 } else { $EpochSize }
    $validationSize = if ($Smoke) { 512 } else { $ValidationSize }
    $batch = if ($Smoke) { 256 } else { $BatchSize }
    $label = if ($Smoke) { 'validation-training' } else { 'training' }
    $lambdaText = $Lambda.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
    $arguments = @(
        $trainScript, $Manifests.train, $Manifests.validation,
        '--features', 'HalfKAv2_hm_jieqi_v11^',
        '--gpus', $Gpus,
        '--threads', '4', '--num-workers', '1', '--seed', '42',
        '--batch-size', [string]$batch,
        '--epoch-size', [string]$epochSize,
        '--validation-size', [string]$validationSize,
        '--lambda', $lambdaText,
        '--max_epochs', [string]$epochs,
        '--default_root_dir', $root,
        '--teacher-dataset-sha256', [string]$Manifests.teacher_hash,
        '--head-coverage-report', [string]$Coverage.path,
        '--head-balance-cap', $HeadBalanceCap.ToString('R', [System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($Smoke) {
        $arguments += @('--limit_train_batches', '8', '--limit_val_batches', '2', '--log_every_n_steps', '1')
    }
    if ($KeepLastOnly) {
        $arguments += @('--keep-last-only')
    }
    if ($Resume -or $ResumeCheckpoint) {
        $checkpoint = Resolve-ResumeCheckpoint -Root $root
        $arguments += @('--resume_from_checkpoint', $checkpoint)
        if ($AllowDatasetTransition) {
            $arguments += '--allow-dataset-transition'
        }
    }
    elseif ($AllowDatasetTransition) {
        throw '-AllowDatasetTransition requires -Resume or -ResumeCheckpoint'
    }
    Invoke-Logged -Label $label -Arguments $arguments
    Remove-OldCheckpoints -Root $root
    if ($Smoke) {
        $checkpoint = Find-ResumeCheckpoint -Root $root
        Invoke-Logged -Label 'export-v11' -Arguments @(
            (Join-Path $repoRoot 'serialize_v11.py'), $checkpoint,
            (Join-Path $smokeRoot 'abjchess-v11-smoke.nnue'), '--smoke-only')
    }
}

Initialize-Layout
$manifests = $null
$coverage = $null
switch ($Mode) {
    'Prepare' { $manifests = Prepare-Manifests }
    'Smoke' { $manifests = Prepare-Manifests; $coverage = Prepare-HeadCoverage -Manifests $manifests; Invoke-V11Training -Manifests $manifests -Coverage $coverage -Smoke $true }
    'Train' { $manifests = Prepare-Manifests; $coverage = Prepare-HeadCoverage -Manifests $manifests; Invoke-V11Training -Manifests $manifests -Coverage $coverage -Smoke $false }
    'All' {
        $manifests = Prepare-Manifests
        $coverage = Prepare-HeadCoverage -Manifests $manifests
        Invoke-V11Training -Manifests $manifests -Coverage $coverage -Smoke $true
        Invoke-V11Training -Manifests $manifests -Coverage $coverage -Smoke $false
    }
}
