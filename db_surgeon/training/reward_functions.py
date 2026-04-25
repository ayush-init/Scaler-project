"""
Reward Functions — Used by TRL GRPOTrainer to score completions.

These functions evaluate model completions directly, providing
multi-signal rewards that guide the model toward producing
correct database diagnosis and repair responses.

Reward Strategy:
  1. reward_func: Primary reward — evaluates SQL quality, reasoning, 
     and adherence to the diagnosis-fix-verify-submit workflow
  2. format_reward_func: Format reward — tool-calling structure bonus
"""

from __future__ import annotations
import re


def reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Primary reward function: evaluates completion quality for DB surgery.
    
    Scores model outputs on:
      - SQL correctness signals (+2.0 for ALTER/CREATE/fix SQL)
      - Diagnostic reasoning (+1.0 for schema analysis)
      - Workflow adherence (+1.0 for inspect → fix → submit flow)
      - Anti-gaming penalties (-2.0 for repetition/nonsense)
    
    Args:
        prompts: List of prompt strings/messages from trainer.
        completions: List of completion strings from trainer.
        **kwargs: Additional data from GRPOTrainer.
    
    Returns:
        List of float rewards, one per completion.
    """
    # First try environment-based rewards if available
    environments = kwargs.get("environments", [])
    if environments:
        env_rewards = []
        has_nonzero = False
        for env in environments:
            r = getattr(env, "reward", 0.0)
            env_rewards.append(r)
            if r != 0.0:
                has_nonzero = True
        if has_nonzero:
            return env_rewards

    # Content-based reward evaluation
    rewards = []
    for i, completion in enumerate(completions):
        content = completion if isinstance(completion, str) else str(completion)
        content_lower = content.lower()
        reward = 0.0

        # ─── 1. SQL Fix Quality (0 to +3.0) ───
        # DDL fix statements (highest value)
        fix_patterns = [
            (r'\balter\s+table\b', 1.5),     # ALTER TABLE
            (r'\bcreate\s+table\b', 1.0),     # CREATE TABLE
            (r'\bcreate\s+index\b', 0.8),     # CREATE INDEX
            (r'\badd\s+constraint\b', 0.8),   # constraints
            (r'\brename\s+column\b', 1.0),    # column rename
            (r'\bmodify\s+column\b', 1.0),    # column modify
            (r'\badd\s+column\b', 0.8),       # adding column
        ]
        sql_score = 0.0
        for pattern, score in fix_patterns:
            if re.search(pattern, content_lower):
                sql_score += score
        reward += min(sql_score, 3.0)  # Cap at 3.0

        # ─── 2. Diagnostic Reasoning (+0 to +2.0) ───
        diag_patterns = [
            (r'\binspect_schema\b', 0.5),     # Schema inspection
            (r'\brun_query\b', 0.5),          # Query testing
            (r'\bselect\b.*\bfrom\b', 0.3),   # SELECT queries
            (r'\berror\b.*\b(type|column|table|missing)\b', 0.3),  # Error analysis
            (r'\bschema\b', 0.2),             # Schema awareness
            (r'\bforeign\s*key\b', 0.3),      # FK awareness
            (r'\bdata\s*type\b', 0.3),        # Type awareness
        ]
        diag_score = 0.0
        for pattern, score in diag_patterns:
            if re.search(pattern, content_lower):
                diag_score += score
        reward += min(diag_score, 2.0)  # Cap at 2.0

        # ─── 3. Workflow Adherence (+0 to +1.5) ───
        has_inspect = bool(re.search(r'\binspect', content_lower))
        has_fix = bool(re.search(r'\b(alter|create|fix_column|execute_fix|add_index|add_constraint)\b', content_lower))
        has_submit = bool(re.search(r'\bsubmit\b', content_lower))

        if has_inspect and has_fix:
            reward += 0.5  # Diagnose then fix
        if has_fix and has_submit:
            reward += 0.5  # Fix then submit
        if has_inspect and has_fix and has_submit:
            reward += 0.5  # Full workflow

        # ─── 4. Tool Call Format Bonus (+0 to +1.0) ───
        tool_call_patterns = [
            r'"function"',           # JSON function calling
            r'"tool_call"',          # Tool call format
            r'<tool_call>',          # XML tool call
            r'\bfunction\s*\(',      # Function call syntax
            r'"name"\s*:\s*"',       # JSON name field
            r'"arguments"\s*:\s*\{', # JSON arguments
        ]
        tool_score = 0.0
        for pattern in tool_call_patterns:
            if re.search(pattern, content_lower):
                tool_score += 0.2
        reward += min(tool_score, 1.0)

        # ─── 5. Anti-Gaming Penalties ───
        # Repetition penalty
        words = content_lower.split()
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                reward -= 2.0  # Heavy repetition

        # Too short (lazy response)
        if len(content.strip()) < 30:
            reward -= 1.0

        # Too long without substance (padding)
        if len(content) > 5000 and reward < 1.0:
            reward -= 0.5

        # Empty content
        if len(content.strip()) == 0:
            reward = -2.0

        rewards.append(round(reward, 2))

    return rewards


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Format bonus: rewards structured tool-calling responses.
    
    Args:
        prompts: List of prompts (unused but required by TRL).
        completions: List of completion strings.
        **kwargs: Additional data.
    
    Returns:
        List of format bonus rewards.
    """
    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else str(completion)
        content_lower = content.lower()
        reward = 0.0

        # Reward structured tool usage
        if any(kw in content_lower for kw in ['"function"', '"tool_call"', '<tool_call>', 'function_call']):
            reward += 1.0
        
        # Reward step-by-step reasoning markers
        if any(kw in content_lower for kw in ['step 1', 'first,', 'let me', 'i will']):
            reward += 0.3
        
        # Penalty for completely empty
        if len(content.strip()) < 10:
            reward -= 1.0

        rewards.append(round(reward, 2))

    return rewards
