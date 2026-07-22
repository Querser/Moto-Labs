[CmdletBinding()]
param(
    [switch]$Reload,
    [switch]$SkipInstall,
    [switch]$Background,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
$Url = "http://127.0.0.1:8000"
$LogDirectory = Join-Path $ProjectRoot "data\logs"
New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null

function Test-MotoLapsServer {
    try {
        $Health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2
        return $Health.status -eq "ok"
    } catch {
        return $false
    }
}

# Repeated double-clicks reuse the existing process instead of starting a
# competing camera/SQLite owner.
if (Test-MotoLapsServer) {
    Write-Host "Moto Laps is already running at $Url"
    if ($OpenBrowser) { Start-Process $Url }
    exit 0
}

# Uvicorn stops accepting HTTP before its shutdown hook has fully released the
# Windows camera. Wait for the previous launcher process to exit so a new run
# does not fall back from 30 FPS Media Foundation to a 1 FPS DirectShow stream.
$PidFile = Join-Path $ProjectRoot "data\moto_laps.pid"
if (Test-Path -LiteralPath $PidFile) {
    $PreviousPidText = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($PreviousPidText -match '^\d+$') {
        $PreviousPid = [int]$PreviousPidText
        for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
            if (-not (Get-Process -Id $PreviousPid -ErrorAction SilentlyContinue)) { break }
            Start-Sleep -Milliseconds 500
        }
        if (Get-Process -Id $PreviousPid -ErrorAction SilentlyContinue) {
            throw "The previous Moto Laps process is still releasing the camera. Wait a few seconds and retry."
        }
    }
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $PythonCommand = $null
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonCommand = @("py", "-3.11") }
        } catch { Write-Verbose "Python launcher does not provide Python 3.11." }
    }
    if (-not $PythonCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
        try {
            & python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" 2>$null
            if ($LASTEXITCODE -eq 0) { $PythonCommand = @("python") }
        } catch { Write-Verbose "The python command does not provide Python 3.10+." }
    }
    if (-not $PythonCommand) {
        throw "Python 3.10+ was not found. Install Python and run this script again."
    }
    Write-Host "Creating local virtual environment..."
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv .venv
    } else {
        & $PythonCommand[0] -m venv .venv
    }
}

if (-not $SkipInstall) {
    & $VenvPython -c "import fastapi, cv2, cv2_enumerate_cameras, sqlalchemy, openpyxl, rapidocr_onnxruntime, onnxruntime, supervision, paddleocr, torch, transformers" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing project dependencies into .venv..."
        & $VenvPython -m pip install --upgrade pip
        & $VenvPython -m pip install -e .
    }
}

# PaddleX imports PyTorch indirectly, so the main Windows environment must use
# the CPU wheel and leave CUDA 12/cuDNN ownership to ONNX Runtime.
if (-not $SkipInstall) {
    & $VenvPython -c "import torch; raise SystemExit(0 if '+cpu' in torch.__version__ else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing the conflict-free CPU PyTorch runtime..."
        & $VenvPython -m pip install --force-reinstall --no-deps `
            "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" `
            --index-url "https://download.pytorch.org/whl/cpu"
    }
}

# Florence owns its CUDA 13 libraries in a second interpreter process. Separate
# site-packages and address spaces prevent its cuDNN from shadowing ORT's copy.
if (-not $SkipInstall -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $FlorencePython = Join-Path $ProjectRoot ".venv-florence\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $FlorencePython)) {
        & $VenvPython -m venv --system-site-packages ".venv-florence"
    }
    & $FlorencePython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing the isolated CUDA runtime for Florence-2..."
        & $FlorencePython -m pip install --force-reinstall --no-deps `
            "torch==2.13.0+cu130" "torchvision==0.28.0+cu130" `
            --index-url "https://download.pytorch.org/whl/cu130"
    }
    & $FlorencePython -c "import transformers, accelerate, timm, einops, cv2, PIL" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing Florence-2 worker dependencies..."
        & $FlorencePython -m pip install `
            "transformers==4.46.3" "accelerate>=1.6,<2" "timm>=1.0,<2" `
            "einops>=0.8,<1" "opencv-contrib-python==4.10.0.84" "pillow>=10,<13"
    }
}

# The application performs no network requests during a race.  On a clean
# installation only, fetch the exact official detector weight and verify it
# before the server is allowed to use it.
$ModelDirectory = Join-Path $ProjectRoot "models"
$DetectorModel = Join-Path $ModelDirectory "yolox_tiny.onnx"
$DetectorSha256 = "427CC366D34E27FF7A03E2899B5E3671425C262EA2291F88BB942BC1CC70B0F7"
if (-not (Test-Path -LiteralPath $DetectorModel)) {
    if ($SkipInstall) {
        throw "Detector model is missing: $DetectorModel"
    }
    New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null
    $TemporaryModel = "$DetectorModel.download"
    Write-Host "Downloading the official YOLOX-Tiny detector model..."
    Invoke-WebRequest `
        -Uri "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx" `
        -OutFile $TemporaryModel
    $DownloadedHash = (Get-FileHash -LiteralPath $TemporaryModel -Algorithm SHA256).Hash
    if ($DownloadedHash -ne $DetectorSha256) {
        Remove-Item -LiteralPath $TemporaryModel -Force
        throw "The downloaded detector model failed SHA-256 verification."
    }
    Move-Item -LiteralPath $TemporaryModel -Destination $DetectorModel
}
$InstalledHash = (Get-FileHash -LiteralPath $DetectorModel -Algorithm SHA256).Hash
if ($InstalledHash -ne $DetectorSha256) {
    throw "Detector model checksum is invalid: $DetectorModel"
}

$PpOcrModel = Join-Path $ProjectRoot "models\paddlex\official_models\PP-OCRv6_medium_rec_onnx\inference.onnx"
$PpOcrDetector = Join-Path $ProjectRoot "models\paddlex\official_models\PP-OCRv5_mobile_det_onnx\inference.onnx"
$FlorenceModel = Join-Path $ProjectRoot "models\florence-2-base-ft\model.safetensors"
if (-not (Test-Path -LiteralPath $PpOcrModel) -or -not (Test-Path -LiteralPath $PpOcrDetector) -or -not (Test-Path -LiteralPath $FlorenceModel)) {
    if ($SkipInstall) {
        throw "OCR/VLM weights are missing. Run: .\.venv\Scripts\python.exe scripts\setup_models.py"
    }
    Write-Host "Downloading the pinned local OCR models..."
    & $VenvPython scripts\setup_models.py
}

if (Test-Path -LiteralPath (Join-Path $ProjectRoot "alembic.ini")) {
    & $VenvPython -m alembic upgrade head
}

$Arguments = @()
if ($Reload) { $Arguments += "--reload" }
Write-Host "Opening Moto Laps at http://127.0.0.1:8000"
if (-not $Background) {
    & $VenvPython -m app.cli @Arguments
    exit $LASTEXITCODE
}

$StandardOutput = Join-Path $LogDirectory "server.stdout.log"
$StandardError = Join-Path $LogDirectory "server.stderr.log"
$Process = Start-Process -FilePath $VenvPython `
    -ArgumentList (@("-m", "app.cli") + $Arguments) `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $StandardOutput `
    -RedirectStandardError $StandardError `
    -PassThru
$Process.Id | Set-Content -LiteralPath $PidFile -Encoding ASCII

# Wait for migrations, model loading and the first HTTP response before opening
# the browser.  This avoids a blank/error page on slower first launches.
for ($Attempt = 0; $Attempt -lt 90; $Attempt++) {
    if (Test-MotoLapsServer) {
        Write-Host "Moto Laps started (PID $($Process.Id))."
        if ($OpenBrowser) { Start-Process $Url }
        exit 0
    }
    if ($Process.HasExited) {
        $Tail = if (Test-Path $StandardError) { (Get-Content $StandardError -Tail 20) -join [Environment]::NewLine } else { "No error log was created." }
        throw "Moto Laps stopped during startup.`n$Tail"
    }
    Start-Sleep -Milliseconds 500
}

throw "Moto Laps did not answer within 45 seconds. See $StandardError"
