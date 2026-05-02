$ErrorActionPreference = "Stop"

$rootDir = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $rootDir "backend"
$frontendDir = Join-Path $rootDir "frontend"
$pythonExe = Join-Path $rootDir "venv\Scripts\pythonw.exe"

function Stop-ProcessTree {
    param([int]$ProcessId)

    if ($ProcessId -le 0) {
        return
    }

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Start-ManagedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$Name
    )

    Write-Host "Starting $Name..."
    Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -NoNewWindow `
        -PassThru
}

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python not found at $pythonExe"
}

Write-Host "========================================="
Write-Host "Starting Water Irrigation System"
Write-Host "========================================="
Write-Host "Press Ctrl+C in this window to stop both servers."
Write-Host

$backend = $null
$frontend = $null

try {
    Write-Host "[1/3] Installing backend dependencies..."
    & $pythonExe -m pip install -r (Join-Path $backendDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "Backend dependency install failed."
    }

    Write-Host "[2/3] Starting Backend Server..."
    $backend = Start-ManagedProcess `
        -FilePath $pythonExe `
        -ArgumentList @("-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", "8001") `
        -WorkingDirectory $backendDir `
        -Name "backend"

    Write-Host "[3/3] Installing frontend dependencies..."
    Push-Location $frontendDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend dependency install failed."
        }
    }
    finally {
        Pop-Location
    }

    Write-Host "[3/3] Starting Frontend Server..."
    $frontend = Start-ManagedProcess `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1") `
        -WorkingDirectory $frontendDir `
        -Name "frontend"

    Write-Host
    Write-Host "Backend:  http://127.0.0.1:8001"
    Write-Host "Frontend: http://127.0.0.1:5173"
    Write-Host

    while ($true) {
        if ($backend.HasExited) {
            throw "Backend stopped unexpectedly with exit code $($backend.ExitCode)."
        }
        if ($frontend.HasExited) {
            throw "Frontend stopped unexpectedly with exit code $($frontend.ExitCode)."
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host
    Write-Host "Stopping Water Irrigation System..."
    if ($frontend -and -not $frontend.HasExited) {
        Stop-ProcessTree -ProcessId $frontend.Id
    }
    if ($backend -and -not $backend.HasExited) {
        Stop-ProcessTree -ProcessId $backend.Id
    }
    Write-Host "Stopped."
}
