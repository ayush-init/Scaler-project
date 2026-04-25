"""
Dataset — Generates the prompt dataset for GRPO training.

Each prompt is a system message that instructs the agent on its role
and how to interact with the DB-Surgeon environment. Since the
environment itself generates unique scenarios on each reset(),
we can use the same prompt template for all episodes.
"""

from __future__ import annotations


SYSTEM_PROMPT = """You are a skilled database engineer performing emergency database surgery.

A production database has been reported with schema failures. Business-critical queries are failing.

Your job:
1. **Diagnose** — Inspect the schema and error logs to understand what's broken
2. **Fix** — Apply targeted DDL/schema changes to repair the database 
3. **Verify** — Run the failing query to confirm it works after your fix
4. **Submit** — Submit your fix for evaluation

IMPORTANT RULES:
- Always start by inspecting the schema to understand the database structure
- Read error messages carefully — they contain clues about the root cause
- Make targeted fixes — don't drop or recreate tables unnecessarily
- You have a limited number of steps, so be efficient
- Call submit() when you're confident the fix is complete

Available tools:
- inspect_schema(table_name?) — View database schema or specific table details
- run_query(sql) — Execute a read-only SQL query
- fix_column(table_name, column_name, new_type?, new_name?) — Modify a column
- add_index(table_name, column_name) — Create an index
- add_constraint(table_name, constraint_type, column_name, reference?) — Add a constraint
- execute_fix(sql) — Execute a DDL/DML fix statement
- submit() — Submit your fix and end the episode

Begin by inspecting the database schema."""


def create_training_dataset(num_episodes: int = 200):
    """
    Create the HuggingFace Dataset for GRPO training.
    
    Each entry contains a single user message with the system prompt.
    The environment_factory provides unique scenarios per episode
    via reset(), so the same prompt can be reused.
    
    Args:
        num_episodes: Number of episodes in the dataset.
        
    Returns:
        HuggingFace Dataset with 'prompt' column.
    """
    from datasets import Dataset

    prompts = []
    for _ in range(num_episodes):
        prompts.append([
            {"role": "user", "content": SYSTEM_PROMPT},
        ])

    return Dataset.from_dict({"prompt": prompts})


def create_training_dataset_simple(num_episodes: int = 200) -> list[list[dict]]:
    """
    Create a simple list-based dataset (no HuggingFace dependency).
    
    Returns:
        List of conversation prompts.
    """
    return [
        [{"role": "user", "content": SYSTEM_PROMPT}]
        for _ in range(num_episodes)
    ]


if __name__ == "__main__":
    # Quick test
    ds = create_training_dataset(10)
    print(f"Dataset created with {len(ds)} episodes")
    print(f"First prompt:\n{ds[0]['prompt'][0]['content'][:200]}...")
