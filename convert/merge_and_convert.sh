#!/usr/bin/env bash
# Merge a LoRA adapter into the base model, then convert to GGUF fp16.
#
# Usage:
#   bash convert/merge_and_convert.sh <checkpoint_dir> <output_dir> [llama_cpp_dir]
#
# Output:
#   <output_dir>/merged_hf/          — merged HuggingFace weights
#   <output_dir>/ovata-A3B-30B-f16.gguf

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> <output_dir> [llama_cpp_dir]}"
OUTPUT_DIR="${2:?Usage: $0 <checkpoint_dir> <output_dir> [llama_cpp_dir]}"
LLAMACPP_DIR="${3:-/opt/llama.cpp}"
MERGED_DIR="${OUTPUT_DIR}/merged_hf"
GGUF_F16="${OUTPUT_DIR}/ovata-A3B-30B-f16.gguf"

mkdir -p "${OUTPUT_DIR}"

echo "[1/3] Merging LoRA adapter → base model..."
python - <<EOF
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model = AutoPeftModelForCausalLM.from_pretrained(
    "${CHECKPOINT_DIR}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
merged = model.merge_and_unload()
merged.save_pretrained("${MERGED_DIR}", safe_serialization=True)

tok = AutoTokenizer.from_pretrained("${CHECKPOINT_DIR}", trust_remote_code=True)
tok.save_pretrained("${MERGED_DIR}")
print("Saved merged model to ${MERGED_DIR}")
EOF

echo "[2/3] Converting to GGUF fp16..."
python "${LLAMACPP_DIR}/convert_hf_to_gguf.py" \
    "${MERGED_DIR}" \
    --outfile "${GGUF_F16}" \
    --outtype f16

echo "[3/3] Done."
echo "  Merged HF : ${MERGED_DIR}"
echo "  GGUF fp16 : ${GGUF_F16}"
