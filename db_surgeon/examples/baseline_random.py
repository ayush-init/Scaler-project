"""
Baseline Random Agent — Establishes the baseline performance.

This agent takes random actions with random arguments.
Expected result: ~0% success rate, negative average reward.
Used to demonstrate that the trained agent has actually learned.

Usage:
    python -m db_surgeon.examples.baseline_random
"""

from __future__ import annotations

import sys
import os
import random
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_surgeon.models import DBSurgeonAction
from db_surgeon.client import DBSurgeonLocalEnv


def random_action(table_names: list[str], step: int = 0) -> DBSurgeonAction:
    """Generate a random action. Only allows submit after step 10."""
    if step < 10:
        tool = random.choice([
            "inspect_schema", "run_query", "fix_column",
            "add_index", "execute_fix",
        ])
    else:
        tool = random.choice([
            "inspect_schema", "run_query", "fix_column",
            "add_index", "execute_fix", "submit",
        ])

    if tool == "inspect_schema":
        table = random.choice(table_names + [""])
        return DBSurgeonAction(tool_name=tool, arguments={"table_name": table})

    elif tool == "run_query":
        queries = [
            f"SELECT * FROM {random.choice(table_names)} LIMIT 5" if table_names else "SELECT 1",
            "SELECT name FROM sqlite_master WHERE type='table'",
            f"SELECT COUNT(*) FROM {random.choice(table_names)}" if table_names else "SELECT 1",
        ]
        return DBSurgeonAction(tool_name=tool, arguments={"sql": random.choice(queries)})

    elif tool == "fix_column":
        table = random.choice(table_names) if table_names else "test"
        col = random.choice(["id", "name", "user_id", "amount", "status", "email"])
        return DBSurgeonAction(tool_name=tool, arguments={
            "table_name": table,
            "column_name": col,
            "new_type": random.choice(["INTEGER", "TEXT", "REAL", ""]),
            "new_name": random.choice(["fixed_col", "", "new_name"]),
        })

    elif tool == "add_index":
        table = random.choice(table_names) if table_names else "test"
        col = random.choice(["id", "name", "user_id", "amount"])
        return DBSurgeonAction(tool_name=tool, arguments={
            "table_name": table, "column_name": col,
        })

    elif tool == "execute_fix":
        return DBSurgeonAction(tool_name=tool, arguments={
            "sql": "SELECT 1;",  # Harmless but useless
        })

    else:  # submit
        return DBSurgeonAction(tool_name=tool, arguments={})


def run_random_baseline(n_episodes: int = 20):
    """Run random agent for n episodes and report statistics."""
    print("=" * 70)
    print("🎲 DB-SURGEON — Random Baseline Agent")
    print(f"   Running {n_episodes} episodes...")
    print("=" * 70)

    rewards = []
    successes = 0
    step_counts = []

    for ep in range(n_episodes):
        env = DBSurgeonLocalEnv()
        result = env.reset()
        
        table_names = env._env._db.get_table_names()
        total_reward = 0.0
        steps = 0

        while not result.done and steps < 15:
            action = random_action(table_names, step=steps)
            result = env.step(action)
            total_reward += result.reward
            steps += 1

        # Submit if not already done
        if not result.done:
            result = env.step(DBSurgeonAction(tool_name="submit", arguments={}))
            total_reward += result.reward
            steps += 1

        state = env.state()
        rewards.append(total_reward)
        step_counts.append(steps)
        if state.is_fixed:
            successes += 1

        print(f"  Episode {ep + 1:3d}: reward={total_reward:+7.2f}, steps={steps:2d}, fixed={state.is_fixed}")
        env.close()

    # Statistics
    print("\n" + "=" * 70)
    print("📊 RANDOM BASELINE RESULTS")
    print(f"  Episodes:      {n_episodes}")
    print(f"  Success Rate:  {successes}/{n_episodes} ({100*successes/n_episodes:.1f}%)")
    print(f"  Avg Reward:    {sum(rewards)/len(rewards):+.2f}")
    print(f"  Min Reward:    {min(rewards):+.2f}")
    print(f"  Max Reward:    {max(rewards):+.2f}")
    print(f"  Avg Steps:     {sum(step_counts)/len(step_counts):.1f}")
    print("=" * 70)

    # Save results
    results = {
        "agent": "random_baseline",
        "episodes": n_episodes,
        "success_rate": successes / n_episodes,
        "avg_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "avg_steps": sum(step_counts) / len(step_counts),
        "rewards": rewards,
        "step_counts": step_counts,
    }

    os.makedirs("metrics/results", exist_ok=True)
    with open("metrics/results/baseline_random.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to metrics/results/baseline_random.json")

    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_random_baseline(n)
