import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
ML_DIR = BACKEND / "app" / "ml"
MODELS_DIR = ML_DIR / "models" / "rainfall"

REQUIRED_MODELS = [
    "lstm_1d.keras",
    "lstm_7d.keras",
    "lstm_15d.keras",
]


def models_exist():
    if not MODELS_DIR.exists():
        return False

    for model in REQUIRED_MODELS:
        if not (MODELS_DIR / model).exists():
            return False

    return True


def run_training():
    print("\n[INFO] Models not found. Starting training...\n")

    process = subprocess.Popen(
        [sys.executable, "train.py"],
        cwd=ML_DIR,
    )

    process.wait()

    if process.returncode != 0:
        print("\n[ERROR] Training failed.\n")
        sys.exit(1)

    print("\n[SUCCESS] Training completed.\n")


def start_backend():
    print("[INFO] Starting backend server...")

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ],
        cwd=BACKEND,
    )


def start_frontend():
    print("[INFO] Starting frontend server...")

    if os.name == "nt":
        return subprocess.Popen(
            ["cmd", "/c", "npm run dev"],
            cwd=FRONTEND,
            shell=True,
        )

    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND,
    )


if __name__ == "__main__":
    print("\n===================================")
    print(" AquaAI System Launcher")
    print("===================================\n")

    if not models_exist():
        run_training()
    else:
        print("[INFO] Models already exist. Skipping training.\n")

    backend_process = start_backend()

    time.sleep(5)

    frontend_process = start_frontend()

    print("\n===================================")
    print(" Backend : http://localhost:8000")
    print(" Frontend: http://localhost:5173")
    print("===================================\n")

    try:
        backend_process.wait()
        frontend_process.wait()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down services...\n")

        backend_process.terminate()
        frontend_process.terminate()