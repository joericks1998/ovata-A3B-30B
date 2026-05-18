"""
SFT training — Qwen3-30B-A3B with LoRA.
Runs on multi-GPU cloud node via DeepSpeed ZeRO-3.

Usage:
    deepspeed --num_gpus=8 train/train.py --config train/config.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_chat_template(path: str) -> str:
    with open(path) as f:
        return f.read()


def apply_template(examples: dict, tokenizer) -> dict:
    return {
        "text": [
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
            )
            for msgs in examples["messages"]
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    tokenizer.chat_template = load_chat_template(cfg["chat_template"])
    tokenizer.padding_side = "right"

    special_tokens = cfg.get("special_tokens", [])
    if special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )

    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=cfg["data_path"], split="train")
    dataset = dataset.map(
        lambda ex: apply_template(ex, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["grad_accum"],
        learning_rate=train_cfg["lr"],
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.05),
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=cfg.get("deepspeed_config"),
        report_to=train_cfg.get("report_to", "none"),
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Train only on assistant turns — mask everything else from the loss
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=cfg.get("max_seq_length", 4096),
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
