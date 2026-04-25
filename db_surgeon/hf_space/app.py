"""
DB-Surgeon HuggingFace Space — Training + Demo App

This Gradio app runs on HuggingFace Spaces with GPU.
It provides:
- Tab 1: Environment Demo (interactive manual play)
- Tab 2: Training Dashboard (run GRPO training, see live metrics)
- Tab 3: Results (plots, comparison, model download)
"""

import os
import sys
import json
import time
import random
import threading
import traceback
from datetime import datetime

import gradio as gr
import gradio_client.utils as _gu

# ═══════════════════════════════════════════════════════════════
# NUCLEAR MONKEY PATCH — Gradio schema parser bool crash fix
# ═══════════════════════════════════════════════════════════════
# The bug: huggingface_hub>=1.0 emits JSON schemas with
#   "additionalProperties": true  (a Python bool)
# Gradio's _json_schema_to_python_type() recurses into that bool
# and crashes with "argument of type 'bool' is not iterable" or
# "Cannot parse schema True". We patch BOTH the inner recursive
# function AND get_type at the MODULE level so that recursive
# calls (which resolve via module globals) also use the safe version.
# ─────────────────────────────────────────────────────────────────

_orig_inner = _gu._json_schema_to_python_type
_orig_get_type = _gu.get_type
_orig_public = _gu.json_schema_to_python_type

def _safe_inner(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    try:
        return _orig_inner(schema, defs)
    except Exception:
        return "Any"

def _safe_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    try:
        return _orig_get_type(schema)
    except Exception:
        return "Any"

def _safe_public(schema):
    if isinstance(schema, bool):
        return "Any"
    try:
        return _safe_inner(schema, schema.get("$defs") if isinstance(schema, dict) else None)
    except Exception:
        return "Any"

_gu._json_schema_to_python_type = _safe_inner
_gu.get_type = _safe_get_type
_gu.json_schema_to_python_type = _safe_public
# ═══════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ─── Add project to path ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_surgeon.models import DBSurgeonAction
from db_surgeon.client import DBSurgeonLocalEnv
from db_surgeon.training.tool_env import DBSurgeonToolEnv

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

training_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "log": [],
    "rewards": [],
    "losses": [],
    "baseline_rewards": [],
    "baseline_avg": 0,
    "completed": False,
    "error": None,
    "start_time": None,
    "model_saved": False,
}

# ═══════════════════════════════════════════════════════════════
# TAB 1: ENVIRONMENT DEMO
# ═══════════════════════════════════════════════════════════════

demo_env = None
demo_obs = None

def demo_reset():
    global demo_env, demo_obs
    demo_env = DBSurgeonLocalEnv()
    result = demo_env.reset()
    demo_obs = result.observation
    state = demo_env.state()

    status = f"**Episode Started** | Bug: `{state.initial_bug_type}` | Steps: 0/{demo_obs.max_steps}"
    schema = demo_obs.schema_snapshot
    error = demo_obs.error_log
    query = demo_obs.failing_query
    history = "Episode started. Use the tools below to diagnose and fix."
    reward = "0.0"

    return status, schema, error, query, history, reward

def demo_action(tool_name, arg1, arg2, arg3):
    global demo_env, demo_obs
    if demo_env is None:
        return "⚠️ Click 'Reset' first!", "", "", "", "No episode running.", "0.0"

    # Build action based on tool
    args = {}
    if tool_name == "inspect_schema":
        if arg1.strip(): args["table_name"] = arg1.strip()
    elif tool_name == "run_query":
        args["sql"] = arg1.strip()
    elif tool_name == "fix_column":
        args["table_name"] = arg1.strip()
        args["column_name"] = arg2.strip()
        if arg3.strip():
            # Check if it looks like a type or a name
            if arg3.strip().upper() in ("INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC"):
                args["new_type"] = arg3.strip()
            else:
                args["new_name"] = arg3.strip()
    elif tool_name == "execute_fix":
        args["sql"] = arg1.strip()
    elif tool_name == "submit":
        pass
    elif tool_name == "add_index":
        args["table_name"] = arg1.strip()
        args["column_name"] = arg2.strip()

    action = DBSurgeonAction(tool_name=tool_name, arguments=args)
    result = demo_env.step(action)
    demo_obs = result.observation
    state = demo_env.state()

    done_str = " | **DONE**" if result.done else ""
    fixed_str = " ✅ FIXED!" if state.is_fixed else ""
    status = f"Step {state.step_count}/{demo_obs.max_steps} | Reward: {result.reward:+.1f} | Total: {state.total_reward:+.1f}{done_str}{fixed_str}"

    history = "\n".join(demo_obs.action_history[-10:])  # Last 10 actions
    schema = demo_obs.schema_snapshot
    error = demo_obs.error_log
    query = demo_obs.failing_query

    return status, schema, error, query, history, f"{state.total_reward:+.1f}"


# ═══════════════════════════════════════════════════════════════
# TAB 2: TRAINING
# ═══════════════════════════════════════════════════════════════

def run_baseline():
    """Run random baseline episodes."""
    training_state["log"].append("Running random baseline (30 episodes)...")

    rewards, successes = [], 0
    for ep in range(30):
        env = DBSurgeonLocalEnv()
        result = env.reset()
        tables = env._env._db.get_table_names()
        total, steps = 0.0, 0
        while not result.done and steps < 15:
            # Random action (no submit before step 10)
            if steps < 10:
                tool = random.choice(["inspect_schema","run_query","fix_column","add_index","execute_fix"])
            else:
                tool = random.choice(["inspect_schema","run_query","fix_column","add_index","execute_fix","submit"])

            if tool == "inspect_schema":
                a = DBSurgeonAction(tool, {"table_name": random.choice(tables + [""])})
            elif tool == "run_query":
                t = random.choice(tables) if tables else "test"
                a = DBSurgeonAction(tool, {"sql": f"SELECT * FROM {t} LIMIT 5"})
            elif tool == "fix_column":
                t = random.choice(tables) if tables else "test"
                a = DBSurgeonAction(tool, {"table_name":t,"column_name":random.choice(["id","name","user_id","amount"]),"new_type":random.choice(["INTEGER","TEXT",""])})
            elif tool == "add_index":
                t = random.choice(tables) if tables else "test"
                a = DBSurgeonAction(tool, {"table_name":t,"column_name":"id"})
            elif tool == "execute_fix":
                a = DBSurgeonAction(tool, {"sql":"SELECT 1;"})
            else:
                a = DBSurgeonAction(tool, {})

            result = env.step(a)
            total += result.reward
            steps += 1

        if not result.done:
            result = env.step(DBSurgeonAction("submit", {}))
            total += result.reward
        if env.state().is_fixed: successes += 1
        rewards.append(total)
        env.close()

    training_state["baseline_rewards"] = rewards
    training_state["baseline_avg"] = sum(rewards) / len(rewards)
    msg = f"Baseline: {successes}/30 success ({100*successes/30:.0f}%), avg reward: {training_state['baseline_avg']:+.2f}"
    training_state["log"].append(msg)
    return msg


def run_training(num_episodes, model_name, learning_rate):
    """Run GRPO training in a background thread."""
    if training_state["running"]:
        return "⚠️ Training already in progress!"

    training_state["running"] = True
    training_state["progress"] = 0
    training_state["total"] = int(num_episodes)
    training_state["completed"] = False
    training_state["error"] = None
    training_state["rewards"] = []
    training_state["losses"] = []
    training_state["log"] = []
    training_state["start_time"] = time.time()
    training_state["model_saved"] = False

    def _train():
        try:
            # Step 1: Baseline
            run_baseline()

            # Step 2: Load model
            training_state["log"].append(f"Loading {model_name} with Unsloth (4-bit)...")
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=None,
            )

            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=16, lora_dropout=0,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                use_gradient_checkpointing="unsloth",
            )

            import torch
            training_state["log"].append(
                f"Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
                f"{torch.cuda.get_device_properties(0).total_mem/1024**3:.1f}GB"
            )

            # Step 3: Dataset
            from db_surgeon.training.dataset import create_training_dataset
            from db_surgeon.training.reward_functions import reward_func
            from trl import GRPOConfig, GRPOTrainer

            n = int(num_episodes)
            dataset = create_training_dataset(n)
            training_state["log"].append(f"Dataset: {len(dataset)} episodes")

            # Step 4: Configure
            output_dir = "/tmp/db_surgeon_output"
            os.makedirs(output_dir, exist_ok=True)

            config = GRPOConfig(
                output_dir=output_dir,
                max_completion_length=2048,
                num_generations=4,
                chat_template_kwargs={"enable_thinking": False},
                learning_rate=float(learning_rate),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_train_epochs=1,
                max_steps=n,
                warmup_steps=10,
                logging_steps=5,
                log_completions=True,
                save_steps=50,
                save_total_limit=3,
                bf16=False,
                fp16=True,
                gradient_checkpointing=True,
            )

            # Step 5: Train
            training_state["log"].append("Starting GRPO training...")
            trainer = GRPOTrainer(
                model=model, tokenizer=tokenizer,
                reward_funcs=reward_func,
                train_dataset=dataset,
                args=config,
                environment_factory=DBSurgeonToolEnv,
            )

            trainer.train()

            # Extract metrics from log history
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    training_state["losses"].append(entry["loss"])
                if "reward" in entry:
                    training_state["rewards"].append(entry["reward"])

            training_state["log"].append("Training complete!")

            # Step 6: Save
            training_state["log"].append("Saving model...")
            model.save_pretrained(f"{output_dir}/lora_adapter")
            tokenizer.save_pretrained(f"{output_dir}/lora_adapter")
            model.save_pretrained_merged(
                f"{output_dir}/merged_model", tokenizer, save_method="merged_16bit",
            )
            training_state["model_saved"] = True
            training_state["log"].append(f"Model saved to {output_dir}")

            # Optional: Push to HF Hub
            try:
                hf_user = os.environ.get("SPACE_AUTHOR_NAME", "")
                if hf_user:
                    repo_id = f"{hf_user}/db-surgeon-qwen3-0.6b-grpo"
                    model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
                    training_state["log"].append(f"Model pushed to Hub: {repo_id}")
            except Exception as e:
                training_state["log"].append(f"Hub push skipped: {e}")

            training_state["completed"] = True

        except Exception as e:
            training_state["error"] = str(e)
            training_state["log"].append(f"ERROR: {e}")
            training_state["log"].append(traceback.format_exc())

        finally:
            training_state["running"] = False

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return "🚀 Training started! Check the log below for progress."


def get_training_status():
    """Get current training status for UI refresh."""
    elapsed = ""
    if training_state["start_time"]:
        secs = time.time() - training_state["start_time"]
        elapsed = f" | Elapsed: {secs/60:.1f}min"

    if training_state["error"]:
        status = f"❌ Error{elapsed}"
    elif training_state["completed"]:
        status = f"✅ Training Complete!{elapsed}"
    elif training_state["running"]:
        status = f"🔄 Training in progress...{elapsed}"
    else:
        status = "⏸️ Not started"

    log_text = "\n".join(training_state["log"][-30:])  # Last 30 lines
    return status, log_text


def get_training_plot():
    """Generate training metrics plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    losses = training_state.get("losses", [])
    if losses:
        axes[0].plot(losses, color="#6366F1", linewidth=2)
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss", fontweight="bold")
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No loss data yet", ha="center", va="center", fontsize=14)
        axes[0].set_title("Training Loss", fontweight="bold")

    # Baseline vs trained comparison
    baseline = training_state.get("baseline_rewards", [])
    rewards = training_state.get("rewards", [])
    if baseline:
        axes[1].hist(baseline, bins=12, color="#EF4444", alpha=0.6, label="Random Baseline", edgecolor="white")
        avg_b = sum(baseline) / len(baseline)
        axes[1].axvline(avg_b, color="#991B1B", linewidth=2, linestyle="--", label=f"Baseline avg: {avg_b:.1f}")
    if rewards:
        axes[1].hist(rewards, bins=12, color="#6366F1", alpha=0.6, label="Trained", edgecolor="white")
        avg_t = sum(rewards) / len(rewards)
        axes[1].axvline(avg_t, color="#312E81", linewidth=2, linestyle="--", label=f"Trained avg: {avg_t:.1f}")

    axes[1].set_xlabel("Episode Reward")
    axes[1].set_title("Reward Distribution", fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "/tmp/training_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ═══════════════════════════════════════════════════════════════
# TAB 3: QUICK TEST (Scripted Agent)
# ═══════════════════════════════════════════════════════════════

def run_scripted_test(num_episodes):
    """Run scripted agent for N episodes."""
    n = int(num_episodes)
    results = []

    for ep in range(n):
        env = DBSurgeonLocalEnv()
        result = env.reset()
        obs = result.observation
        scenario = env._env._scenario

        # Smart scripted agent
        env.step(DBSurgeonAction("inspect_schema", {}))
        env.step(DBSurgeonAction("run_query", {"sql": obs.failing_query}))

        root = env.state().root_cause.lower()
        tables = env._env._db.get_table_names()

        if "renamed" in root or "usr_id" in root:
            orders = [t for t in tables if "orders" in t]
            if orders:
                env.step(DBSurgeonAction("fix_column", {"table_name": orders[0], "column_name": "usr_id", "new_name": "user_id"}))
        elif "missing" in root:
            env.step(DBSurgeonAction("execute_fix", {"sql": scenario.healthy_schema_sql}))
        elif "typo" in root:
            env.step(DBSurgeonAction("execute_fix", {"sql": scenario.healthy_schema_sql}))
        elif "text" in root or "email" in root:
            orders = [t for t in tables if "orders" in t]
            if orders:
                env.step(DBSurgeonAction("fix_column", {"table_name": orders[0], "column_name": "user_id", "new_type": "INTEGER"}))
        else:
            env.step(DBSurgeonAction("execute_fix", {"sql": scenario.healthy_schema_sql}))

        result = env.step(DBSurgeonAction("submit", {}))
        state = env.state()

        results.append({
            "episode": ep + 1,
            "bug_variant": scenario.bug_variant,
            "fixed": state.is_fixed,
            "reward": state.total_reward,
            "steps": state.step_count,
        })
        env.close()

    # Format results
    lines = [f"{'Ep':>3} | {'Variant':<17} | {'Fixed':>5} | {'Reward':>8} | {'Steps':>5}"]
    lines.append("-" * 55)
    for r in results:
        lines.append(f"{r['episode']:>3} | {r['bug_variant']:<17} | {'YES' if r['fixed'] else 'NO':>5} | {r['reward']:>+8.1f} | {r['steps']:>5}")

    successes = sum(1 for r in results if r["fixed"])
    avg_rew = sum(r["reward"] for r in results) / len(results)
    lines.append("-" * 55)
    lines.append(f"Success: {successes}/{n} ({100*successes/n:.0f}%) | Avg Reward: {avg_rew:+.1f}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(
    title="DB-Surgeon: Database Surgery RL Environment",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
) as app:

    gr.Markdown("# 🏥 DB-Surgeon — Database Surgery RL Environment")
    gr.Markdown("An RL environment where an LLM learns to diagnose and fix broken database schemas.")

    with gr.Tabs():

        # ─── TAB 1: INTERACTIVE DEMO ───
        with gr.Tab("🎮 Interactive Demo"):
            gr.Markdown("### Try fixing a broken database yourself!")

            with gr.Row():
                demo_status = gr.Markdown("Click **Reset** to start a new episode.")
                demo_reward = gr.Textbox(label="Total Reward", value="0.0", interactive=False, scale=0)

            reset_btn = gr.Button("🔄 Reset (New Episode)", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    demo_schema = gr.Textbox(label="Database Schema", lines=12, interactive=False)
                with gr.Column(scale=1):
                    demo_error = gr.Textbox(label="Error Log", lines=6, interactive=False)
                    demo_query = gr.Textbox(label="Failing Business Query", lines=6, interactive=False)

            gr.Markdown("### Execute a Tool")
            with gr.Row():
                tool_select = gr.Dropdown(
                    choices=["inspect_schema", "run_query", "fix_column", "execute_fix", "add_index", "submit"],
                    value="inspect_schema", label="Tool", scale=1
                )
                tool_arg1 = gr.Textbox(label="Arg 1 (table_name / sql)", placeholder="Leave empty for all tables", scale=2)
                tool_arg2 = gr.Textbox(label="Arg 2 (column_name)", placeholder="For fix_column", scale=1)
                tool_arg3 = gr.Textbox(label="Arg 3 (new_type / new_name)", placeholder="INTEGER, TEXT, or new name", scale=1)

            action_btn = gr.Button("▶️ Execute Tool", variant="secondary")
            demo_history = gr.Textbox(label="Action History", lines=6, interactive=False)

            reset_btn.click(
                demo_reset,
                outputs=[demo_status, demo_schema, demo_error, demo_query, demo_history, demo_reward],
            )
            action_btn.click(
                demo_action,
                inputs=[tool_select, tool_arg1, tool_arg2, tool_arg3],
                outputs=[demo_status, demo_schema, demo_error, demo_query, demo_history, demo_reward],
            )

        # ─── TAB 2: TRAINING ───
        with gr.Tab("🚀 Training"):
            gr.Markdown("### Train an LLM to fix databases using GRPO + Unsloth")
            gr.Markdown("This runs GRPO training with your HuggingFace GPU. **$30 budget = ~30 hours of A10G.**")

            with gr.Row():
                model_input = gr.Dropdown(
                    choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"],
                    value="Qwen/Qwen3-0.6B",
                    label="Model",
                )
                episodes_input = gr.Number(value=200, label="Training Episodes", minimum=10, maximum=1000)
                lr_input = gr.Number(value=5e-6, label="Learning Rate")

            with gr.Row():
                train_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                status_btn = gr.Button("🔄 Refresh Status", scale=1)
                plot_btn = gr.Button("📊 Show Plot", scale=1)

            train_status = gr.Markdown("⏸️ Not started")
            train_log = gr.Textbox(label="Training Log", lines=15, interactive=False)
            train_plot = gr.Image(label="Training Metrics", height=400)

            train_btn.click(
                run_training,
                inputs=[episodes_input, model_input, lr_input],
                outputs=[train_status],
            )
            status_btn.click(
                get_training_status,
                outputs=[train_status, train_log],
            )
            plot_btn.click(
                get_training_plot,
                outputs=[train_plot],
            )

        # ─── TAB 3: QUICK TEST ───
        with gr.Tab("🧪 Quick Test"):
            gr.Markdown("### Test the environment with a scripted agent")
            gr.Markdown("Runs a smart scripted agent that reads the root cause and applies the correct fix.")

            with gr.Row():
                test_n = gr.Number(value=10, label="Number of Episodes", minimum=1, maximum=100)
                test_btn = gr.Button("▶️ Run Test", variant="primary")

            test_output = gr.Textbox(label="Results", lines=20, interactive=False, show_copy_button=True)
            test_btn.click(run_scripted_test, inputs=[test_n], outputs=[test_output])


# ─── Launch ───
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
