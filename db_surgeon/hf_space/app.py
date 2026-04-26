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
import re
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


# ─── Trained Model Auto-Play ───

_trained_model = None
_trained_tokenizer = None

def _load_trained_model():
    """Load the trained model from HuggingFace Hub (cached)."""
    global _trained_model, _trained_tokenizer
    if _trained_model is not None:
        return _trained_model, _trained_tokenizer

    import torch
    try:
        from unsloth import FastLanguageModel
        _trained_model, _trained_tokenizer = FastLanguageModel.from_pretrained(
            model_name="ayush0211/db-surgeon-qwen3-0.6b-grpo",
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=torch.float16,
        )
        FastLanguageModel.for_inference(_trained_model)
    except Exception:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _trained_tokenizer = AutoTokenizer.from_pretrained("ayush0211/db-surgeon-qwen3-0.6b-grpo")
        _trained_model = AutoModelForCausalLM.from_pretrained(
            "ayush0211/db-surgeon-qwen3-0.6b-grpo",
            torch_dtype=torch.float16,
            device_map="auto",
        )

    return _trained_model, _trained_tokenizer


def _parse_tool_calls(text, turn_number=0):
    """Parse tool calls from model output text.
    
    Args:
        text: Raw model output text
        turn_number: Current turn (0-indexed). Used to prevent premature submit.
    """
    import json as _json

    # Strip <think>...</think> blocks — model reasons here, not actual tool calls
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Also strip partial thinking blocks (model didn't close the tag)
    clean_text = re.sub(r'<think>.*$', '', clean_text, flags=re.DOTALL)

    if not clean_text.strip():
        # All content was inside <think>, use full text as fallback
        clean_text = text

    tool_calls = []

    # Try JSON tool_call format: {"name": "...", "arguments": {...}}
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', clean_text):
        try:
            name = m.group(1)
            args = _json.loads(m.group(2))
            tool_calls.append((name, args))
        except Exception:
            pass

    # Try function call format: tool_name(arg1, arg2)
    if not tool_calls:
        for m in re.finditer(r'\b(inspect_schema|run_query|fix_column|execute_fix|add_index|add_constraint|submit)\s*\(([^)]*)\)', clean_text):
            name = m.group(1)
            raw_args = m.group(2).strip()
            args = {}
            if name == "inspect_schema" and raw_args:
                args["table_name"] = raw_args.strip("'\"")
            elif name == "run_query" and raw_args:
                args["sql"] = raw_args.strip("'\"")
            elif name == "fix_column":
                parts = [p.strip().strip("'\"") for p in raw_args.split(",")]
                if len(parts) >= 2:
                    args["table_name"] = parts[0]
                    args["column_name"] = parts[1]
                if len(parts) >= 3:
                    val = parts[2]
                    if val.upper() in ("INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC"):
                        args["new_type"] = val
                    else:
                        args["new_name"] = val
            elif name == "execute_fix" and raw_args:
                args["sql"] = raw_args.strip("'\"")
            elif name == "add_index":
                parts = [p.strip().strip("'\"") for p in raw_args.split(",")]
                if len(parts) >= 2:
                    args["table_name"] = parts[0]
                    args["column_name"] = parts[1]
            tool_calls.append((name, args))

    # Try to find SQL statements and wrap as execute_fix
    if not tool_calls:
        sql_matches = re.findall(
            r'(ALTER\s+TABLE\s+\w+\s+(?:ADD|RENAME|MODIFY|DROP)\s+[^;]+;?)',
            clean_text, re.IGNORECASE
        )
        for sql in sql_matches[:3]:
            tool_calls.append(("execute_fix", {"sql": sql.rstrip(";")}))

    # Prevent premature submit: on early turns, don't submit unless there are
    # other tool calls before it
    if turn_number < 3:
        non_submit = [(n, a) for n, a in tool_calls if n != "submit"]
        if non_submit:
            tool_calls = non_submit  # Drop submit, keep other tools
        elif tool_calls and all(n == "submit" for n, _ in tool_calls):
            # Only submit found on early turn — replace with inspect_schema
            tool_calls = [("inspect_schema", {})]

    # Default: inspect if nothing found
    if not tool_calls:
        if turn_number == 0:
            tool_calls = [("inspect_schema", {})]
        else:
            tool_calls = [("submit", {})]

    return tool_calls


def auto_play_model():
    """Auto-play: load trained model and solve the current episode."""
    global demo_env, demo_obs
    import torch

    log_lines = []

    # Step 1: Reset environment
    log_lines.append("=" * 60)
    log_lines.append("🤖 TRAINED MODEL AUTO-PLAY")
    log_lines.append("=" * 60)

    demo_env = DBSurgeonLocalEnv()
    result = demo_env.reset()
    demo_obs = result.observation
    state = demo_env.state()

    log_lines.append(f"\n📋 Bug Type: {state.initial_bug_type}")
    log_lines.append(f"📋 Failing Query: {demo_obs.failing_query[:100]}...")
    log_lines.append(f"📋 Error: {demo_obs.error_log[:100]}...")

    # Step 2: Load model
    log_lines.append("\n⏳ Loading trained model...")
    model = None
    tokenizer = None

    try:
        model, tokenizer = _load_trained_model()
        log_lines.append("✅ Model loaded!")
    except Exception as e:
        log_lines.append(f"❌ Model load failed: {e}")
        log_lines.append("\n🔧 Falling back to scripted agent...")
        return _run_fallback_scripted(log_lines)

    # Step 3: Generate & execute actions
    max_turns = 8
    for turn in range(max_turns):
        if result.done:
            break

        log_lines.append(f"\n{'─' * 40}")
        log_lines.append(f"🔄 Turn {turn + 1}/{max_turns}")

        # Build prompt from current observation
        prompt = f"""You are a database engineer fixing a broken database.

CURRENT SCHEMA:
{demo_obs.schema_snapshot[:500]}

FAILING QUERY:
{demo_obs.failing_query}

ERROR:
{demo_obs.error_log}

Available tools: inspect_schema(table_name), run_query(sql), fix_column(table_name, column_name, new_type/new_name), execute_fix(sql), add_index(table_name, column_name), submit()

Diagnose the issue and fix it. Call the appropriate tool."""

        messages = [{"role": "user", "content": prompt}]

        try:
            input_ids = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            )
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            log_lines.append(f"🧠 Model says:\n{response[:400]}{'...' if len(response) > 400 else ''}")

        except Exception as e:
            log_lines.append(f"❌ Generation error: {e}")
            break

        # Parse and execute tool calls
        tool_calls = _parse_tool_calls(response, turn_number=turn)
        if not tool_calls:
            log_lines.append("⚠️ No tool calls found, trying inspect_schema...")
            tool_calls = [("inspect_schema", {})]

        for tool_name, args in tool_calls:
            if result.done:
                break
            log_lines.append(f"\n  🔧 Executing: {tool_name}({args})")
            try:
                action = DBSurgeonAction(tool_name=tool_name, arguments=args)
                result = demo_env.step(action)
                demo_obs = result.observation
                state = demo_env.state()
                log_lines.append(f"  📊 Reward: {result.reward:+.1f} | Total: {state.total_reward:+.1f}")
                if result.done:
                    if state.is_fixed:
                        log_lines.append("  🎉 DATABASE FIXED!")
                    else:
                        log_lines.append("  ❌ Episode ended (not fixed)")
            except Exception as e:
                log_lines.append(f"  ❌ Action error: {e}")

    # Final status
    if not result.done:
        log_lines.append(f"\n{'─' * 40}")
        log_lines.append("⏰ Max turns reached, submitting...")
        result = demo_env.step(DBSurgeonAction("submit", {}))
        demo_obs = result.observation
        state = demo_env.state()

    log_lines.append(f"\n{'=' * 60}")
    log_lines.append("📊 FINAL RESULT")
    log_lines.append(f"  Total Reward: {state.total_reward:+.1f}")
    log_lines.append(f"  Fixed: {'✅ YES' if state.is_fixed else '❌ NO'}")
    log_lines.append(f"  Steps Used: {state.step_count}/{demo_obs.max_steps}")
    log_lines.append(f"{'=' * 60}")

    return _format_autoplay_output(log_lines, state, demo_obs)


def _run_fallback_scripted(log_lines):
    """Fallback scripted agent when model can't load (e.g., CPU mode)."""
    global demo_env, demo_obs

    scenario = demo_env._env._scenario
    tables = demo_env._env._db.get_table_names()
    root = demo_env.state().root_cause.lower()

    log_lines.append("  Inspecting schema...")
    r = demo_env.step(DBSurgeonAction("inspect_schema", {}))
    demo_obs = r.observation

    log_lines.append(f"  Root cause: {root[:80]}")

    if "renamed" in root or "usr_id" in root:
        orders = [t for t in tables if "orders" in t]
        if orders:
            log_lines.append(f"  Fixing renamed column in {orders[0]}...")
            demo_env.step(DBSurgeonAction("fix_column", {"table_name": orders[0], "column_name": "usr_id", "new_name": "user_id"}))
    elif "text" in root or "type" in root:
        orders = [t for t in tables if "orders" in t]
        if orders:
            log_lines.append(f"  Fixing column type in {orders[0]}...")
            demo_env.step(DBSurgeonAction("fix_column", {"table_name": orders[0], "column_name": "user_id", "new_type": "INTEGER"}))
    else:
        log_lines.append("  Applying schema fix...")
        demo_env.step(DBSurgeonAction("execute_fix", {"sql": scenario.healthy_schema_sql}))

    log_lines.append("  Submitting fix...")
    r = demo_env.step(DBSurgeonAction("submit", {}))
    demo_obs = r.observation
    state = demo_env.state()

    log_lines.append(f"\n{'=' * 60}")
    log_lines.append("📊 RESULT (Scripted Agent Fallback)")
    log_lines.append(f"  Total Reward: {state.total_reward:+.1f}")
    log_lines.append(f"  Fixed: {'✅ YES' if state.is_fixed else '❌ NO'}")
    log_lines.append(f"{'=' * 60}")

    return _format_autoplay_output(log_lines, state, demo_obs)


def _format_autoplay_output(log_lines, state, obs):
    """Format outputs for the autoplay UI update."""
    done_str = " | **DONE**" if state.done else ""
    fixed_str = " ✅ FIXED!" if state.is_fixed else ""
    status = f"🤖 Auto-Play | Step {state.step_count}/{obs.max_steps} | Total: {state.total_reward:+.1f}{done_str}{fixed_str}"
    schema = obs.schema_snapshot
    error = obs.error_log
    query = obs.failing_query
    history = "\n".join(log_lines[-40:])
    reward = f"{state.total_reward:+.1f}"
    return status, schema, error, query, history, reward


# ═══════════════════════════════════════════════════════════════
# NATURAL LANGUAGE TO SQL (NL2SQL)
# ═══════════════════════════════════════════════════════════════

# Dedicated database for NL2SQL queries
_nl2sql_db = None
_nl2sql_schema = ""

def nl2sql_reset_db():
    """Create a fresh sample database for NL2SQL queries."""
    global _nl2sql_db, _nl2sql_schema
    import sqlite3

    _nl2sql_db = sqlite3.connect(":memory:")

    schema_sql = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    city TEXT NOT NULL,
    age INTEGER
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL NOT NULL,
    manager TEXT
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER,
    status TEXT DEFAULT 'active',
    start_date TEXT,
    budget REAL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE sales (
    id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    product TEXT NOT NULL,
    amount REAL NOT NULL,
    sale_date TEXT NOT NULL,
    region TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);
"""

    seed_sql = """
INSERT INTO departments VALUES (1, 'Engineering', 500000, 'Rajesh Kumar');
INSERT INTO departments VALUES (2, 'Marketing', 200000, 'Priya Sharma');
INSERT INTO departments VALUES (3, 'Sales', 300000, 'Amit Patel');
INSERT INTO departments VALUES (4, 'HR', 150000, 'Sunita Verma');
INSERT INTO departments VALUES (5, 'Finance', 250000, 'Vikram Singh');

INSERT INTO employees VALUES (1, 'Rahul Sharma', 'Engineering', 85000, '2021-03-15', 'Mumbai', 28);
INSERT INTO employees VALUES (2, 'Priya Gupta', 'Marketing', 65000, '2022-06-01', 'Delhi', 26);
INSERT INTO employees VALUES (3, 'Amit Singh', 'Engineering', 92000, '2020-01-10', 'Bangalore', 32);
INSERT INTO employees VALUES (4, 'Neha Patel', 'Sales', 70000, '2021-09-20', 'Mumbai', 29);
INSERT INTO employees VALUES (5, 'Vikram Joshi', 'Engineering', 110000, '2019-05-01', 'Pune', 35);
INSERT INTO employees VALUES (6, 'Anita Desai', 'HR', 60000, '2023-01-15', 'Delhi', 24);
INSERT INTO employees VALUES (7, 'Suresh Reddy', 'Sales', 75000, '2020-11-10', 'Hyderabad', 31);
INSERT INTO employees VALUES (8, 'Kavita Nair', 'Marketing', 72000, '2021-07-22', 'Bangalore', 27);
INSERT INTO employees VALUES (9, 'Ravi Iyer', 'Finance', 88000, '2020-04-05', 'Chennai', 33);
INSERT INTO employees VALUES (10, 'Deepa Menon', 'Engineering', 95000, '2022-02-14', 'Pune', 30);
INSERT INTO employees VALUES (11, 'Arjun Kumar', 'Sales', 68000, '2023-03-01', 'Mumbai', 25);
INSERT INTO employees VALUES (12, 'Meera Krishnan', 'Finance', 82000, '2021-08-18', 'Chennai', 29);

INSERT INTO projects VALUES (1, 'Cloud Migration', 1, 'active', '2024-01-01', 200000);
INSERT INTO projects VALUES (2, 'Brand Redesign', 2, 'completed', '2023-06-01', 50000);
INSERT INTO projects VALUES (3, 'Sales Portal', 3, 'active', '2024-03-15', 150000);
INSERT INTO projects VALUES (4, 'HR Automation', 4, 'active', '2024-02-01', 80000);
INSERT INTO projects VALUES (5, 'Data Pipeline', 1, 'active', '2024-04-01', 120000);

INSERT INTO sales VALUES (1, 4, 'Widget Pro', 15000, '2024-01-15', 'North');
INSERT INTO sales VALUES (2, 7, 'Gadget Plus', 22000, '2024-01-20', 'South');
INSERT INTO sales VALUES (3, 11, 'Widget Pro', 18000, '2024-02-05', 'West');
INSERT INTO sales VALUES (4, 4, 'Mega Suite', 45000, '2024-02-10', 'North');
INSERT INTO sales VALUES (5, 7, 'Widget Pro', 12000, '2024-02-15', 'South');
INSERT INTO sales VALUES (6, 11, 'Gadget Plus', 35000, '2024-03-01', 'East');
INSERT INTO sales VALUES (7, 4, 'Gadget Plus', 28000, '2024-03-10', 'North');
INSERT INTO sales VALUES (8, 7, 'Mega Suite', 52000, '2024-03-15', 'South');
"""

    for stmt in schema_sql.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            _nl2sql_db.execute(stmt)

    for stmt in seed_sql.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            _nl2sql_db.execute(stmt)

    _nl2sql_db.commit()

    _nl2sql_schema = """Tables in database:

1. employees (id, name, department, salary, hire_date, city, age)
   - 12 employees across Engineering, Marketing, Sales, HR, Finance
   - Cities: Mumbai, Delhi, Bangalore, Pune, Hyderabad, Chennai

2. departments (id, name, budget, manager)
   - Engineering, Marketing, Sales, HR, Finance

3. projects (id, name, department_id, status, start_date, budget)
   - Status: active, completed

4. sales (id, employee_id, product, amount, sale_date, region)
   - Products: Widget Pro, Gadget Plus, Mega Suite
   - Regions: North, South, East, West"""

    return _nl2sql_schema, "✅ Fresh database loaded! You can now ask questions in any language."


def nl2sql_query(user_question):
    """Convert natural language question to SQL and execute it."""
    global _nl2sql_db, _nl2sql_schema
    import torch

    if _nl2sql_db is None:
        return "⚠️ Please click **Load Database** first!", "", ""

    if not user_question or not user_question.strip():
        return "⚠️ Please type a question!", "", ""

    # Step 1: Load model
    try:
        model, tokenizer = _load_trained_model()
    except Exception as e:
        return f"❌ Model load error: {e}", "", ""

    # Step 2: Prompt the model to generate SQL
    prompt = f"""You are a SQL expert. Convert the user's natural language question into a SQL query.

DATABASE SCHEMA:
{_nl2sql_schema}

RULES:
- Output ONLY the SQL query, nothing else
- Use standard SQLite SQL syntax
- Do NOT explain the query
- Do NOT wrap in markdown code blocks
- If the question is in Hindi or any other language, understand it and still output valid SQL

USER QUESTION: {user_question}

SQL QUERY:"""

    messages = [{"role": "user", "content": prompt}]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        # Clean up: strip thinking blocks and extract SQL
        sql = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        sql = re.sub(r'<think>.*$', '', sql, flags=re.DOTALL)
        sql = sql.strip()

        # Remove markdown code fences if present
        sql = re.sub(r'^```sql\s*', '', sql)
        sql = re.sub(r'^```\s*', '', sql)
        sql = re.sub(r'\s*```$', '', sql)
        sql = sql.strip()

        # Take only the first SQL statement
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
        
        if not sql:
            return "⚠️ Model didn't generate a SQL query. Try rephrasing.", "", ""

    except Exception as e:
        return f"❌ Generation error: {e}", "", ""

    # Step 3: Execute the SQL
    try:
        cursor = _nl2sql_db.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        # Format results as a table
        if not rows:
            result_text = "(No results returned)"
        else:
            # Header
            col_widths = [max(len(str(col)), max(len(str(row[i])) for row in rows)) for i, col in enumerate(columns)]
            header = " | ".join(str(col).ljust(w) for col, w in zip(columns, col_widths))
            separator = "-+-".join("-" * w for w in col_widths)
            
            result_lines = [header, separator]
            for row in rows[:50]:  # Limit to 50 rows
                result_lines.append(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
            
            if len(rows) > 50:
                result_lines.append(f"\n... and {len(rows) - 50} more rows")
            
            result_lines.append(f"\n({len(rows)} rows returned)")
            result_text = "\n".join(result_lines)

        return f"✅ Query executed successfully!", sql, result_text

    except Exception as e:
        return f"❌ SQL Error: {e}", sql, f"The generated SQL had an error:\n{e}\n\nTry rephrasing your question."

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
            import torch

            # IMPORTANT: Always use float16 with 4-bit quantization.
            # bitsandbytes dequantizes 4-bit weights to float16 internally.
            # Using bf16 causes dtype mismatch in LoRA matmul kernels.
            training_state["log"].append("  Using dtype: float16 (required for 4-bit quant)")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                dtype=torch.float16,
            )

            model = FastLanguageModel.get_peft_model(
                model, r=16, lora_alpha=16, lora_dropout=0,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                use_gradient_checkpointing="unsloth",
            )

            import torch
            training_state["log"].append(
                f"Model loaded! GPU: {torch.cuda.memory_allocated()/1024**3:.1f}GB / "
                f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB"
            )

            # Step 3: Dataset
            from db_surgeon.training.dataset import create_training_dataset
            from db_surgeon.training.reward_functions import reward_func, format_reward_func
            from trl import GRPOConfig, GRPOTrainer

            n = int(num_episodes)
            dataset = create_training_dataset(n)
            training_state["log"].append(f"Dataset: {len(dataset)} episodes")

            # Step 4: Configure — resilient to different TRL versions
            output_dir = "/tmp/db_surgeon_output"
            os.makedirs(output_dir, exist_ok=True)
            # Always fp16 with 4-bit quantization (bitsandbytes constraint)
            config_kwargs = dict(
                output_dir=output_dir,
                max_completion_length=2048,
                num_generations=4,
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

            # Try with optional kwargs, drop them if unsupported
            optional_kwargs = {"chat_template_kwargs": {"enable_thinking": False}}
            for key, val in optional_kwargs.items():
                try:
                    test = GRPOConfig(**{key: val, "output_dir": "/tmp/_test"})
                    config_kwargs[key] = val
                    del test
                except TypeError:
                    training_state["log"].append(f"  (Skipping unsupported config: {key})")

            config = GRPOConfig(**config_kwargs)

            # Step 5: Train with live progress callback
            from transformers import TrainerCallback

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **cb_kwargs):
                    if logs:
                        step = state.global_step
                        total = state.max_steps
                        loss = logs.get("loss", "?")
                        reward = logs.get("reward", "?")
                        lr = logs.get("learning_rate", "?")
                        msg = f"  Step {step}/{total} | loss={loss} | reward={reward} | lr={lr}"
                        training_state["log"].append(msg)
                        training_state["progress"] = step
                        if isinstance(loss, (int, float)):
                            training_state["losses"].append(loss)
                        if isinstance(reward, (int, float)):
                            training_state["rewards"].append(reward)

                def on_step_end(self, args, state, control, **cb_kwargs):
                    # Log every 25 steps even without metrics
                    if state.global_step % 25 == 0 and state.global_step > 0:
                        elapsed = time.time() - training_state["start_time"]
                        training_state["log"].append(
                            f"  [Heartbeat] Step {state.global_step}/{state.max_steps} | {elapsed/60:.1f}min elapsed"
                        )

            training_state["log"].append("Starting GRPO training...")

            trainer_kwargs = dict(
                model=model, tokenizer=tokenizer,
                reward_funcs=[reward_func, format_reward_func],
                train_dataset=dataset,
                args=config,
                callbacks=[ProgressCallback()],
            )

            # environment_factory may not exist in all TRL versions
            try:
                trainer = GRPOTrainer(**trainer_kwargs, environment_factory=DBSurgeonToolEnv)
            except TypeError:
                training_state["log"].append("  (environment_factory not supported, using default)")
                del trainer_kwargs["callbacks"]  # rebuild without it
                trainer = GRPOTrainer(
                    model=model, tokenizer=tokenizer,
                    reward_funcs=reward_func,
                    train_dataset=dataset,
                    args=config,
                )
                trainer.add_callback(ProgressCallback())

            trainer.train()

            # Extract any remaining metrics from log history
            for entry in trainer.state.log_history:
                if "loss" in entry and entry["loss"] not in training_state["losses"]:
                    training_state["losses"].append(entry["loss"])
                if "reward" in entry and entry["reward"] not in training_state["rewards"]:
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
                hf_token = os.environ.get("HF_TOKEN", "")
                hf_user = os.environ.get("SPACE_AUTHOR_NAME", "ayush0211")
                if hf_token and hf_user:
                    from huggingface_hub import login
                    login(token=hf_token)
                    repo_id = f"{hf_user}/db-surgeon-qwen3-0.6b-grpo"
                    model.push_to_hub_merged(repo_id, tokenizer, save_method="merged_16bit")
                    training_state["log"].append(f"Model pushed to Hub: {repo_id}")
                else:
                    training_state["log"].append("Hub push skipped: no HF_TOKEN secret set")
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
            demo_history = gr.Textbox(label="Action History", lines=12, interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 🤖 Or let the Trained Model solve it automatically")
            gr.Markdown("Loads `ayush0211/db-surgeon-qwen3-0.6b-grpo` and watches it diagnose & fix the database in real-time.")
            autoplay_btn = gr.Button("🤖 Auto-Play with Trained Model", variant="primary", size="lg")

            reset_btn.click(
                demo_reset,
                outputs=[demo_status, demo_schema, demo_error, demo_query, demo_history, demo_reward],
            )
            action_btn.click(
                demo_action,
                inputs=[tool_select, tool_arg1, tool_arg2, tool_arg3],
                outputs=[demo_status, demo_schema, demo_error, demo_query, demo_history, demo_reward],
            )
            autoplay_btn.click(
                auto_play_model,
                outputs=[demo_status, demo_schema, demo_error, demo_query, demo_history, demo_reward],
            )

        # ─── TAB 2: NATURAL LANGUAGE TO SQL ───
        with gr.Tab("💬 Ask in Any Language"):
            gr.Markdown("### Ask questions about a database in any language — Hindi, English, or anything!")
            gr.Markdown("The trained model converts your natural language question into SQL, runs it, and shows results.")

            with gr.Row():
                with gr.Column(scale=2):
                    nl2sql_schema_display = gr.Textbox(
                        label="📊 Database Schema",
                        lines=15, interactive=False,
                        placeholder="Click 'Load Database' to see the schema..."
                    )
                with gr.Column(scale=1):
                    gr.Markdown("**Example questions:**")
                    gr.Markdown("""
- Show all employees from Mumbai
- सबसे ज्यादा सैलरी किसकी है?
- Engineering department ka budget kitna hai?
- List products with total sales above 30000
- कौन से projects active हैं?
- Which region has the most sales?
- मुंबई में कितने employees हैं?
""")

            load_db_btn = gr.Button("🗄️ Load Database", variant="primary")
            nl2sql_db_status = gr.Markdown("")

            gr.Markdown("---")

            with gr.Row():
                nl2sql_input = gr.Textbox(
                    label="🗣️ Your Question (any language)",
                    placeholder="e.g., 'Show me all employees with salary above 80000' or 'सबसे ज्यादा सैलरी किसकी है?'",
                    lines=2, scale=4,
                )
                nl2sql_btn = gr.Button("🔍 Ask", variant="primary", scale=1)

            nl2sql_status = gr.Markdown("")
            nl2sql_generated_sql = gr.Textbox(label="🤖 Generated SQL", lines=3, interactive=False)
            nl2sql_results = gr.Textbox(label="📋 Query Results", lines=12, interactive=False, show_copy_button=True)

            load_db_btn.click(
                nl2sql_reset_db,
                outputs=[nl2sql_schema_display, nl2sql_db_status],
            )
            nl2sql_btn.click(
                nl2sql_query,
                inputs=[nl2sql_input],
                outputs=[nl2sql_status, nl2sql_generated_sql, nl2sql_results],
            )
            nl2sql_input.submit(
                nl2sql_query,
                inputs=[nl2sql_input],
                outputs=[nl2sql_status, nl2sql_generated_sql, nl2sql_results],
            )

        # ─── TAB 3: TRAINING ───
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
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, ssr_mode=False)
