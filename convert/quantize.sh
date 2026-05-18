#!/usr/bin/env bash
# Quantize a GGUF fp16 model to Q4_K_M.
#
# Usage:
#   bash convert/quantize.sh <input_f16.gguf> <output_q4km.gguf> [llama_cpp_dir]

set -euo pipefail

INPUT="${1:?Usage: $0 <input_f16.gguf> <output_q4km.gguf> [llama_cpp_dir]}"
OUTPUT="${2:?Usage: $0 <input_f16.gguf> <output_q4km.gguf> [llama_cpp_dir]}"
LLAMACPP_DIR="${3:-/opt/llama.cpp}"

echo "Quantizing $(basename "${INPUT}") → Q4_K_M → $(basename "${OUTPUT}")"
"${LLAMACPP_DIR}/build/bin/llama-quantize" "${INPUT}" "${OUTPUT}" Q4_K_M
echo "Done: ${OUTPUT}"
