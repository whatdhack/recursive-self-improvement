#!/usr/bin/env bash
# DO CPU Droplet setup — installs llama.cpp and pulls Gemma4-1B GGUF
# Tested on Ubuntu 22.04 (DO CPU-Optimized droplet, 8+ vCPUs recommended)
set -euo pipefail

echo "==> Installing system deps..."
apt-get update -qq
apt-get install -y -qq build-essential cmake git python3-pip python3-venv curl

echo "==> Cloning llama.cpp..."
git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp

echo "==> Building llama.cpp (CPU only)..."
cmake -B /opt/llama.cpp/build /opt/llama.cpp \
  -DLLAMA_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/llama.cpp/build --config Release -j "$(nproc)"

echo "==> llama-finetune binary: $(ls /opt/llama.cpp/build/bin/llama-finetune 2>/dev/null || echo 'NOT FOUND')"
echo "==> llama-export-lora binary: $(ls /opt/llama.cpp/build/bin/llama-export-lora 2>/dev/null || echo 'NOT FOUND')"

echo "==> Creating model dir..."
mkdir -p /opt/models

echo "==> Pulling Gemma4-1B-IT Q4_K_M GGUF from HuggingFace..."
# Smallest Gemma4 instruct model — good enough for LoRA student
pip3 install -q huggingface_hub
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="bartowski/google_gemma-4-1b-it-GGUF",
    filename="google_gemma-4-1b-it-Q4_K_M.gguf",
    local_dir="/opt/models",
)
print(f"Downloaded: {path}")
EOF

echo "==> Setting up Python venv..."
python3 -m venv /opt/rsi-venv
/opt/rsi-venv/bin/pip install -q openai

echo ""
echo "==> Setup complete. Next steps:"
echo "    1. Clone the repo:  git clone https://github.com/<you>/recursive-self-improvement /opt/rsi"
echo "    2. Copy .env:       cp /opt/rsi/.env.example /opt/rsi/.env && nano /opt/rsi/.env"
echo "    3. Run the loop:"
echo "       source /opt/rsi-venv/bin/activate"
echo "       cd /opt/rsi"
echo "       python -m rsi run \\"
echo "         --problem matrix-vector \\"
echo "         --base-model /opt/models/google_gemma-4-1b-it-Q4_K_M.gguf \\"
echo "         --llama-bin-dir /opt/llama.cpp/build/bin \\"
echo "         --threads $(nproc) \\"
echo "         --max-gen 10 \\"
echo "         --run-id 001"
