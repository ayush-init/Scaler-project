"""
GRPO Training Script — Standard TRL GRPOTrainer integration.

This script trains a language model to debug databases using
Group Relative Policy Optimization (GRPO) with verifiable rewards.

Usage:
    python -m db_surgeon.training.train_grpo

For Google Colab T4 (15GB VRAM):
    Use with Qwen3-0.6B + QLoRA via train_unsloth.py instead.
"""

from __future__ import annotations

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run GRPO training."""
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from db_surgeon.training.tool_env import DBSurgeonToolEnv
    from db_surgeon.training.reward_functions import reward_func
    from db_surgeon.training.dataset import create_training_dataset

    # ─── Configuration ───
    MODEL_NAME = os.environ.get("DB_SURGEON_MODEL", "Qwen/Qwen3-0.6B")
    NUM_EPISODES = int(os.environ.get("DB_SURGEON_EPISODES", "200"))
    OUTPUT_DIR = os.environ.get("DB_SURGEON_OUTPUT", "./db_surgeon_output")

    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Episodes: {NUM_EPISODES}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # ─── Dataset ───
    logger.info("Creating training dataset...")
    dataset = create_training_dataset(NUM_EPISODES)
    logger.info(f"Dataset: {len(dataset)} episodes")

    # ─── Model ───
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    # ─── Training Config ───
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Generation
        max_completion_length=2048,
        num_generations=4,  # 4 completions per prompt for GRPO comparison

        # Chat template
        chat_template_kwargs={"enable_thinking": False},

        # Optimization
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_steps=NUM_EPISODES,

        # Logging
        logging_steps=10,
        log_completions=True,
        save_steps=50,
        save_total_limit=3,

        # Memory optimization
        bf16=True,
        gradient_checkpointing=True,
    )

    # ─── Trainer ───
    logger.info("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=DBSurgeonToolEnv,
    )

    # ─── Train ───
    logger.info("Starting GRPO training...")
    trainer.train()

    # ─── Save ───
    logger.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
