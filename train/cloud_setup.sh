#!/usr/bin/env bash
# Bootstrap a RunPod / Lambda GPU node for Ovata training.
# Run once after spinning up the instance.
#
# Usage:
#   bash train/cloud_setup.sh
#   INSTALL_LLAMACPP=1 bash train/cloud_setup.sh   # also builds llama.cpp

set -euo pipefail

echo "=== Ovata cloud setup ==="

apt-get update -q && apt-get install -y -q git git-lfs build-essential cmake

pip install -q --upgrade pip

pip install -q \
    "torch==2.3.0" \
    "transformers>=4.51.0" \
    "trl>=0.9.0" \
    "peft>=0.11.0" \
    "deepspeed>=0.14.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.30.0" \
    "sentencepiece" \
    "anthropic>=0.40.0" \
    "wandb" \
    "pyyaml"

if [ "${INSTALL_LLAMACPP:-0}" = "1" ]; then
    echo "=== Building llama.cpp ==="
    git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp
    cmake -B /opt/llama.cpp/build /opt/llama.cpp -DGGML_CUDA=ON
    cmake --build /opt/llama.cpp/build --config Release -j"$(nproc)"
    pip install -q -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
fi

echo "=== Setup complete ==="
