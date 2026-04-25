"""
Unsloth GRPO Training Script — Memory-efficient training for Colab T4.

Uses Unsloth + QLoRA (4-bit) to fit GRPO training into 15GB VRAM.
This is the RECOMMENDED training script for the hackathon.

Usage (Google Colab):
    !pip install unsloth trl datasets
    !python -m db_surgeon.training.train_unsloth

Usage (Local):
    python -m db_surgeon.training.train_unsloth

WARNING from rl_complete_guide.md §8:
    "Don't naively upcast a 4-bit model to 16-bit and then merge adapters,
    because that can damage model quality; use the proper merge path instead."
"""

from __future__ import annotations

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run Unsloth-optimized GRPO training."""
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer

    from db_surgeon.training.tool_env import DBSurgeonToolEnv
    from db_surgeon.training.reward_functions import reward_func
    from db_surgeon.training.dataset import create_training_dataset

    # ─── Configuration ───
    MODEL_NAME = os.environ.get("DB_SURGEON_MODEL", "Qwen/Qwen3-0.6B")
    NUM_EPISODES = int(os.environ.get("DB_SURGEON_EPISODES", "200"))
    OUTPUT_DIR = os.environ.get("DB_SURGEON_OUTPUT", "./db_surgeon_output")
    MAX_SEQ_LENGTH = 2048

    logger.info(f"Model: {MODEL_NAME} (Unsloth QLoRA 4-bit)")
    logger.info(f"Episodes: {NUM_EPISODES}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ─── Load Model with Unsloth (4-bit QLoRA) ───
    logger.info(f"Loading model with Unsloth: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # Auto-detect (float16 on T4)
    )

    # ─── Apply LoRA Adapters ───
    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    # ─── Dataset ───
    logger.info("Creating training dataset...")
    dataset = create_training_dataset(NUM_EPISODES)
    logger.info(f"Dataset: {len(dataset)} episodes")

    # ─── Training Config (T4-optimized) ───
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Generation (conservative for T4)
        max_completion_length=MAX_SEQ_LENGTH,
        num_generations=4,  # 4 not 8 to fit in VRAM

        # Chat template
        chat_template_kwargs={"enable_thinking": False},

        # Optimization
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_steps=NUM_EPISODES,
        warmup_steps=10,

        # Logging
        logging_steps=10,
        log_completions=True,
        save_steps=50,
        save_total_limit=3,

        # Memory (critical for T4)
        bf16=False,   # T4 doesn't support bf16 well
        fp16=True,    # Use fp16 instead
        gradient_checkpointing=True,
    )

    # ─── Trainer ───
    logger.info("Initializing GRPOTrainer with Unsloth model...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=DBSurgeonToolEnv,
    )

    # ─── Train ───
    logger.info("Starting Unsloth GRPO training...")
    trainer.train()

    # ─── Save (CORRECT Unsloth way) ───
    logger.info(f"Saving LoRA adapter to {OUTPUT_DIR}/lora_adapter")
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")

    # Save merged model (16-bit, correct merge path)
    logger.info(f"Saving merged model to {OUTPUT_DIR}/merged_model")
    model.save_pretrained_merged(
        f"{OUTPUT_DIR}/merged_model",
        tokenizer,
        save_method="merged_16bit",
    )

    logger.info("Unsloth training complete!")
    logger.info(f"LoRA adapter: {OUTPUT_DIR}/lora_adapter")
    logger.info(f"Merged model: {OUTPUT_DIR}/merged_model")


if __name__ == "__main__":
    main()
