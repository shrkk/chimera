#!/usr/bin/env bash
# scripts/cloud_setup.sh
# Automated environment setup script for Cloud GPUs (RunPod, Lambda Labs, Vast.ai)
set -euo pipefail

echo "=========================================================="
echo "  Chimera Chess RLVR: Cloud GPU Environment Provisioning  "
echo "=========================================================="

# 1. System packages
echo "[1/5] Installing system packages (stockfish, tmux, git, htop)..."
if command -v apt-get &> /dev/null; then
    apt-get update -y
    apt-get install -y stockfish git tmux htop curl libgl1
elif command -v yum &> /dev/null; then
    yum install -y stockfish git tmux htop curl
fi

# Verify Stockfish engine
STOCKFISH_BIN=$(which stockfish || true)
if [ -n "$STOCKFISH_BIN" ]; then
    echo "✔ Stockfish engine found at: $STOCKFISH_BIN"
else
    echo "⚠ Warning: Stockfish not found in PATH. Checking alternative locations..."
    if [ -f "/usr/games/stockfish" ]; then
        ln -sf /usr/games/stockfish /usr/local/bin/stockfish
        echo "✔ Linked /usr/games/stockfish -> /usr/local/bin/stockfish"
    fi
fi

# 2. Check GPU
echo "[2/5] Inspecting GPU hardware..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ Warning: nvidia-smi not detected. Verify NVIDIA drivers."
fi

# 3. Python environment & dependencies
echo "[3/5] Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install torch transformers peft python-chess zstandard pyyaml pytest aiosqlite requests tqdm numpy stockfish

# 4. Ingest Tactical Dataset
echo "[4/5] Ingesting Lichess tactical puzzles dataset..."
if [ ! -f "datasets/puzzles.jsonl" ]; then
    python3 datasets/download_lichess_puzzles.py --count 1200 --output datasets/puzzles.jsonl
else
    echo "✔ Puzzles dataset already present at datasets/puzzles.jsonl"
fi

# 5. Run Verification Suite
echo "[5/5] Running invariant test suite..."
pytest tests/ -v

echo "=========================================================="
echo "✔ Setup complete! You are ready to train."
echo ""
echo "To start GRPO LoRA training inside a persistent tmux session:"
echo "    tmux new -s chimera"
echo "    python3 src/models/grpo_trainer.py --config config.yaml"
echo "=========================================================="
