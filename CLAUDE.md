# ovata-A3B-30B

Distill + SFT Qwen3-30B-A3B-Instruct onto a custom Jade-compatible executor template, then export to GGUF Q4_K_M.

## Model
- Base: `Qwen/Qwen3-30B-A3B-Instruct` (MoE — ~3B active / 30B total params)
- Output: GGUF Q4_K_M for llama.cpp inference

## Pipeline

```
1. chat_template/   — design the jinja2 template
2. distill/         — generate (task → tool_call_trajectory) pairs via Claude API
3. train/           — SFT with LoRA + DeepSpeed ZeRO-3 on cloud GPUs
4. convert/         — merge LoRA → HF → GGUF fp16 → Q4_K_M
```

## Key Design Constraints
- The model is a **blind tool-calling executor** — it has no awareness of Jade lang
- Every capability is exposed as a tool; the model never chats
- The chat template must produce output that Jade's grammar enforcement layer can parse
- `<|tool_call|>` is a custom special token — must be in tokenizer vocab before training
- Loss is masked to assistant turns only (completion-only training)

## Commands

```bash
# 1. Generate distillation dataset (requires ANTHROPIC_API_KEY)
python distill/generate.py --config distill/config.yaml

# 2. Pre-process raw samples into training format (add as needed)
# python scripts/preprocess.py ...

# 3. Train — run on cloud GPU node
deepspeed --num_gpus=8 train/train.py --config train/config.yaml

# 4. Convert checkpoint → GGUF
bash convert/merge_and_convert.sh ./checkpoints/ovata-v0.1 ./output
bash convert/quantize.sh ./output/ovata-A3B-30B-f16.gguf ./output/ovata-A3B-30B-Q4_K_M.gguf

# 5. Test with llama.cpp
llama-cli -m ./output/ovata-A3B-30B-Q4_K_M.gguf \
          -p "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
```

## Template Contract
See `chat_template/spec.md` for the full format spec and open design questions.

## Cloud Setup
```bash
bash train/cloud_setup.sh
INSTALL_LLAMACPP=1 bash train/cloud_setup.sh   # also builds llama.cpp for on-node conversion
```
