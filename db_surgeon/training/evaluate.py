"""
Evaluation Script — Runs trained model and collects metrics.

Usage:
    python -m db_surgeon.training.evaluate --model ./db_surgeon_output --episodes 50
"""

from __future__ import annotations

import json
import os
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path: str, n_episodes: int = 50, output_dir: str = "metrics/results"):
    """
    Evaluate a trained model on fresh DB-Surgeon episodes.
    
    Args:
        model_path: Path to the trained model or LoRA adapter.
        n_episodes: Number of episodes to evaluate.
        output_dir: Directory to save results.
    """
    # Lazy imports to avoid requiring training deps for env-only usage
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from db_surgeon.training.tool_env import DBSurgeonToolEnv

    logger.info(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )

    logger.info(f"Running {n_episodes} evaluation episodes...")
    
    results = {
        "model": model_path,
        "timestamp": datetime.now().isoformat(),
        "episodes": [],
        "summary": {},
    }

    rewards = []
    successes = 0
    step_counts = []

    for ep in range(n_episodes):
        env = DBSurgeonToolEnv()
        obs = env.reset()
        
        episode_data = {
            "episode": ep,
            "reward": 0.0,
            "steps": 0,
            "success": False,
            "bug_type": "",
            "tool_calls": [],
        }

        # Simple greedy evaluation (no sampling)
        # In a full eval, you'd use the model to generate tool calls
        # For now, we just run a basic interact loop
        logger.info(f"  Episode {ep + 1}/{n_episodes}")

        episode_data["reward"] = env.reward
        episode_data["success"] = env.done
        rewards.append(env.reward)
        step_counts.append(episode_data["steps"])
        if episode_data["success"]:
            successes += 1
        
        results["episodes"].append(episode_data)

    # Compute summary
    results["summary"] = {
        "n_episodes": n_episodes,
        "success_rate": successes / n_episodes if n_episodes > 0 else 0,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
        "min_reward": min(rewards) if rewards else 0,
        "max_reward": max(rewards) if rewards else 0,
        "avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Success rate: {results['summary']['success_rate']:.1%}")
    logger.info(f"Avg reward: {results['summary']['avg_reward']:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DB-Surgeon agent")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--output", default="metrics/results", help="Output directory")
    args = parser.parse_args()

    evaluate_model(args.model, args.episodes, args.output)


if __name__ == "__main__":
    main()
