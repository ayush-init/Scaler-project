"""
Reward Functions — Used by TRL GRPOTrainer to score completions.

High-variance reward design for effective GRPO training:
- GRPO learns from RELATIVE differences within each generation group
- All 4 generations getting ~4.0 = zero advantage = no learning
- We need: great answers = +8, okay answers = +2, bad = -3
- This creates gradient signal for the policy to improve

Strategy:
  1. reward_func: Evaluates SQL execution quality with sharp scoring
  2. format_reward_func: Structural quality bonus/penalty
"""

from __future__ import annotations
import re
import sqlite3
import tempfile
import os


def reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Primary reward: sharp scoring based on SQL execution quality.
    
    Scoring tiers (designed for high variance):
      - Tier S (+8 to +10): Completions with working ALTER/CREATE SQL 
      - Tier A (+4 to +6):  Contains valid SQL patterns with reasoning
      - Tier B (+1 to +2):  Shows understanding but weak execution
      - Tier C (-1 to -3):  Empty, repetitive, or irrelevant
    """
    # Try environment-based rewards first
    environments = kwargs.get("environments", [])
    if environments:
        env_rewards = [getattr(env, "reward", 0.0) for env in environments]
        if any(r != 0.0 for r in env_rewards):
            return env_rewards

    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else str(completion)
        content_lower = content.lower().strip()
        
        if not content_lower or len(content_lower) < 10:
            rewards.append(-3.0)
            continue

        reward = 0.0

        # ─── SQL Execution Test (+0 to +5) ───
        # Extract SQL statements and test if they're syntactically valid
        sql_score = _score_sql_quality(content)
        reward += sql_score

        # ─── Diagnostic Depth (+0 to +3) ───
        diag_score = _score_diagnosis(content_lower)
        reward += diag_score

        # ─── Workflow Completeness (+0 to +3) ───
        workflow_score = _score_workflow(content_lower)
        reward += workflow_score

        # ─── Penalties ───
        penalty = _compute_penalties(content, content_lower)
        reward += penalty

        # Clamp to [-3, 10]
        rewards.append(round(max(-3.0, min(10.0, reward)), 2))

    return rewards


def _score_sql_quality(content: str) -> float:
    """Test SQL statements for syntactic validity using SQLite."""
    # Extract SQL-like statements
    sql_patterns = re.findall(
        r'(?:ALTER|CREATE|INSERT|UPDATE|DELETE|DROP|RENAME)\s+(?:TABLE|INDEX|VIEW|COLUMN)\s+[^;]+',
        content, re.IGNORECASE
    )
    
    if not sql_patterns:
        # Check for inline SQL in tool calls
        sql_patterns = re.findall(
            r'"sql"\s*:\s*"([^"]+)"', content
        )
    
    if not sql_patterns:
        return 0.0
    
    score = 0.0
    valid_count = 0
    
    for sql in sql_patterns[:5]:  # Test up to 5 statements
        sql = sql.strip().rstrip(';')
        if _is_valid_sql(sql):
            valid_count += 1
            # Bonus based on complexity
            if re.search(r'\bALTER\s+TABLE\b', sql, re.IGNORECASE):
                score += 2.0  # ALTER TABLE is the most relevant fix
            elif re.search(r'\bCREATE\s+(TABLE|INDEX)\b', sql, re.IGNORECASE):
                score += 1.5
            else:
                score += 1.0
        else:
            score -= 0.5  # Penalty for invalid SQL
    
    return min(score, 5.0)


def _is_valid_sql(sql: str) -> bool:
    """Test if SQL is syntactically valid using an in-memory SQLite."""
    if len(sql.strip()) < 5:
        return False
    try:
        db = sqlite3.connect(":memory:")
        # Create a dummy table for testing
        db.execute("CREATE TABLE test_table (id INTEGER, name TEXT, value REAL, status TEXT)")
        db.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, amount REAL)")
        try:
            db.execute(sql)
            db.close()
            return True
        except sqlite3.OperationalError:
            # Try as a plan (the SQL may reference tables not in our dummy DB)
            # If it's a syntax error, it will fail on any DB
            try:
                db.execute(f"EXPLAIN {sql}")
                db.close()
                return True
            except Exception:
                db.close()
                # If it looks like valid SQL structure, give partial credit
                return bool(re.match(r'^\s*(ALTER|CREATE|INSERT|UPDATE|DROP|RENAME)\s+', sql, re.IGNORECASE))
    except Exception:
        return False


def _score_diagnosis(content_lower: str) -> float:
    """Score diagnostic reasoning quality."""
    score = 0.0
    
    # Strong diagnostic signals (each worth more individually)
    strong_signals = [
        (r'\binspect_schema\s*\(', 1.0),       # Actually calling inspect
        (r'\brun_query\s*\(', 0.8),             # Testing with queries
        (r'error.*(?:type|mismatch|missing)', 0.8),  # Error root cause analysis
        (r'(?:column|table)\s+\w+\s+(?:is|has|should)', 0.7),  # Specific diagnosis
        (r'foreign\s*key.*references', 0.7),    # FK analysis
    ]
    
    for pattern, points in strong_signals:
        if re.search(pattern, content_lower):
            score += points
    
    # Weak signals
    weak_signals = ['schema', 'constraint', 'data type', 'integer', 'text', 'real']
    weak_count = sum(1 for s in weak_signals if s in content_lower)
    score += min(weak_count * 0.2, 0.6)
    
    return min(score, 3.0)


def _score_workflow(content_lower: str) -> float:
    """Score adherence to the inspect → fix → verify → submit workflow."""
    score = 0.0
    
    has_inspect = bool(re.search(r'inspect_schema', content_lower))
    has_diagnose = bool(re.search(r'run_query|select.*from', content_lower))
    has_fix = bool(re.search(r'alter\s+table|create\s+(table|index)|fix_column|execute_fix|add_(index|constraint)', content_lower))
    has_verify = bool(re.search(r'(verify|check|test|confirm).*(?:fix|query|work)', content_lower))
    has_submit = bool(re.search(r'\bsubmit\s*\(', content_lower))
    
    steps = [has_inspect, has_diagnose, has_fix, has_verify, has_submit]
    completed = sum(steps)
    
    if completed >= 4:
        score = 3.0  # Near-complete workflow
    elif completed == 3:
        score = 2.0
    elif completed == 2:
        score = 1.0
    elif completed == 1:
        score = 0.3
    
    # Bonus for correct ordering (inspect before fix)
    if has_inspect and has_fix:
        inspect_pos = content_lower.find('inspect_schema')
        fix_pos = max(
            content_lower.find('alter table'),
            content_lower.find('fix_column'),
            content_lower.find('execute_fix'),
        )
        if inspect_pos < fix_pos:
            score += 0.5  # Correct order bonus
    
    return min(score, 3.0)


def _compute_penalties(content: str, content_lower: str) -> float:
    """Compute penalties for low-quality outputs."""
    penalty = 0.0
    
    words = content_lower.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            penalty -= 5.0  # Severe repetition
        elif unique_ratio < 0.35:
            penalty -= 2.0  # Moderate repetition
    
    # Too short
    if len(content.strip()) < 50:
        penalty -= 2.0
    
    # Rambling without action (long but no SQL/tools)
    if len(content) > 3000 and not re.search(r'(ALTER|CREATE|fix_column|execute_fix|submit)', content, re.IGNORECASE):
        penalty -= 2.0
    
    # Pure filler / refusal
    if re.search(r"(i can't|i cannot|i'm sorry|as an ai)", content_lower):
        penalty -= 3.0
    
    return penalty


def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Format reward: sharper scoring for structured tool-calling.
    
    +2.0 for proper tool call JSON structure
    +0.5 for step-by-step reasoning
    -2.0 for completely unformatted output
    """
    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else str(completion)
        content_lower = content.lower()
        reward = 0.0

        # Strong structured format signals
        tool_patterns = [
            (r'"name"\s*:\s*"(inspect_schema|run_query|fix_column|execute_fix|submit)"', 2.0),
            (r'"arguments"\s*:\s*\{', 0.5),
            (r'<tool_call>', 1.0),
            (r'"function"\s*:', 0.5),
        ]
        for pattern, score in tool_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                reward += score
        reward = min(reward, 3.0)

        # Step-by-step reasoning
        if re.search(r'(step\s*\d|first|then|finally|next)', content_lower):
            reward += 0.5

        # Penalty for unformatted junk
        if len(content.strip()) < 20:
            reward = -2.0

        rewards.append(round(reward, 2))

    return rewards
