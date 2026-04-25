"""
Example Episode — Step-by-step scripted agent interaction.

Demonstrates a full DB-Surgeon episode from start to finish:
1. Reset the environment
2. Inspect schema (diagnose)
3. Run the failing query (understand error)
4. Apply the fix
5. Verify the fix
6. Submit

Usage:
    python -m db_surgeon.examples.example_episode
"""

from __future__ import annotations

import sys
import os

# Add parent to path if running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_surgeon.models import DBSurgeonAction
from db_surgeon.client import DBSurgeonLocalEnv


def run_episode():
    """Run a complete episode with a scripted 'smart' agent."""
    print("=" * 70)
    print("🏥 DB-SURGEON — Example Episode")
    print("=" * 70)

    env = DBSurgeonLocalEnv()

    # ─── Step 0: Reset ───
    print("\n📋 STEP 0: Resetting environment...")
    result = env.reset()
    obs = result.observation
    print(f"  Bug type: {env.state().initial_bug_type}")
    print(f"  Root cause: {env.state().root_cause}")
    print(f"\n  Failing query:\n  {obs.failing_query[:120]}...")
    print(f"\n  Error log:\n  {obs.error_log[:200]}...")

    total_reward = 0.0

    # ─── Step 1: Inspect all tables ───
    print("\n" + "-" * 70)
    print("🔍 STEP 1: Inspecting schema...")
    result = env.step(DBSurgeonAction(
        tool_name="inspect_schema",
        arguments={},
    ))
    print(f"  Reward: {result.reward:+.1f}")
    total_reward += result.reward
    schema = result.observation.last_action_result
    print(f"  Schema:\n  {schema[:300]}...")

    # Parse table names from schema
    tables = env._env.state().root_cause  # Get hint about what's broken
    print(f"\n  Root cause hint: {tables}")

    # ─── Step 2: Run the failing business query ───
    print("\n" + "-" * 70)
    print("🔍 STEP 2: Running the failing business query...")
    result = env.step(DBSurgeonAction(
        tool_name="run_query",
        arguments={"sql": obs.failing_query},
    ))
    print(f"  Reward: {result.reward:+.1f}")
    total_reward += result.reward
    print(f"  Result:\n  {result.observation.last_action_result[:300]}")

    # ─── Step 3: Inspect tables involved in the bug ───
    print("\n" + "-" * 70)
    print("🔍 STEP 3: Inspecting specific tables...")
    state = env.state()
    # Get table names from the schema
    table_names = env._env._db.get_table_names()
    for tbl in table_names[:2]:  # Inspect first 2 tables
        result = env.step(DBSurgeonAction(
            tool_name="inspect_schema",
            arguments={"table_name": tbl},
        ))
        print(f"  [{tbl}] Reward: {result.reward:+.1f}")
        total_reward += result.reward
        print(f"  {result.observation.last_action_result[:200]}...")
        print()

    # ─── Step 4: Apply fix based on root cause ───
    print("-" * 70)
    print("🔧 STEP 4: Applying fix...")
    root_cause = env.state().root_cause.lower()
    scenario = env._env._scenario

    if "renamed" in root_cause or "usr_id" in root_cause:
        # Fix: rename usr_id back to user_id
        orders_tbl = [t for t in table_names if "orders" in t][0]
        result = env.step(DBSurgeonAction(
            tool_name="fix_column",
            arguments={
                "table_name": orders_tbl,
                "column_name": "usr_id",
                "new_name": "user_id",
            },
        ))
    elif "missing" in root_cause:
        # Fix: create missing users table
        prefix = scenario.table_prefix
        result = env.step(DBSurgeonAction(
            tool_name="execute_fix",
            arguments={
                "sql": f"""CREATE TABLE {prefix}_users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL
                );
                INSERT INTO {prefix}_users (id, name, email) VALUES (1, 'Admin', 'admin@example.com');
                INSERT INTO {prefix}_users (id, name, email) VALUES (2, 'User', 'user@example.com');
                """
            },
        ))
    elif "typo" in root_cause or "usrs" in root_cause:
        # Fix: recreate table with correct FK reference
        result = env.step(DBSurgeonAction(
            tool_name="execute_fix",
            arguments={
                "sql": scenario.healthy_schema_sql,
            },
        ))
    elif "text" in root_cause or "email" in root_cause:
        # Fix: change user_id type from TEXT to INTEGER
        orders_tbl = [t for t in table_names if "orders" in t][0]
        result = env.step(DBSurgeonAction(
            tool_name="fix_column",
            arguments={
                "table_name": orders_tbl,
                "column_name": "user_id",
                "new_type": "INTEGER",
            },
        ))
    else:
        # Generic: try execute_fix with healthy schema
        result = env.step(DBSurgeonAction(
            tool_name="execute_fix",
            arguments={"sql": scenario.healthy_schema_sql},
        ))

    print(f"  Reward: {result.reward:+.1f}")
    total_reward += result.reward
    print(f"  Result: {result.observation.last_action_result[:200]}")

    # ─── Step 5: Verify the fix ───
    print("\n" + "-" * 70)
    print("✅ STEP 5: Verifying fix...")
    result = env.step(DBSurgeonAction(
        tool_name="run_query",
        arguments={"sql": obs.failing_query},
    ))
    print(f"  Reward: {result.reward:+.1f}")
    total_reward += result.reward
    print(f"  Result:\n  {result.observation.last_action_result[:300]}")

    # ─── Step 6: Submit ───
    print("\n" + "-" * 70)
    print("📤 STEP 6: Submitting fix...")
    result = env.step(DBSurgeonAction(
        tool_name="submit",
        arguments={},
    ))
    print(f"  Reward: {result.reward:+.1f}")
    total_reward += result.reward
    print(f"\n{result.observation.last_action_result}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("📊 EPISODE SUMMARY")
    print(f"  Total Reward: {total_reward:+.1f}")
    print(f"  Steps Used: {env.state().step_count}")
    print(f"  Fixed: {env.state().is_fixed}")
    print(f"  Bug Type: {env.state().initial_bug_type}")
    print("=" * 70)

    env.close()
    return total_reward


if __name__ == "__main__":
    run_episode()
