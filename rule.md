# 🏗️ DB-SURGEON — MASTER RULE BOOK

> **One document to rule them all.**
> Every step, requirement, constraint, edge case, anti-pattern, and decision — synthesized from `rl_complete_guide.md`, `hackathon_rl_guide.md`, `implementation_plan.md`, OpenEnv GitHub, TRL HuggingFace docs, and Unsloth recipes.

---

## TABLE OF CONTENTS

1. [Project Identity](#1-project-identity)
2. [Golden Rules (Non-Negotiable)](#2-golden-rules-non-negotiable)
3. [Architecture Contract](#3-architecture-contract)
4. [OpenEnv API — Exact Spec](#4-openenv-api--exact-spec)
5. [TRL environment_factory — Exact Pattern](#5-trl-environment_factory--exact-pattern)
6. [File-by-File Build Order](#6-file-by-file-build-order)
7. [Broken DB Generator — Full Spec](#7-broken-db-generator--full-spec)
8. [Reward Engineering — Rules & Formulas](#8-reward-engineering--rules--formulas)
9. [Anti-Reward-Hacking — Mandatory Checklist](#9-anti-reward-hacking--mandatory-checklist)
10. [Edge Case Catalog](#10-edge-case-catalog)
11. [GRPO Training — How It Actually Works](#11-grpo-training--how-it-actually-works)
12. [Unsloth QLoRA — Exact Recipe](#12-unsloth-qlora--exact-recipe)
13. [Training Hyperparameters](#13-training-hyperparameters)
14. [Debugging Order (MANDATORY)](#14-debugging-order-mandatory)
15. [Metrics to Collect](#15-metrics-to-collect)
16. [Demo & Storytelling Requirements](#16-demo--storytelling-requirements)
17. [Judge Expectations](#17-judge-expectations)
18. [Fatal Mistakes to Avoid](#18-fatal-mistakes-to-avoid)
19. [Curriculum & Difficulty Progression](#19-curriculum--difficulty-progression)
20. [Deployment Checklist](#20-deployment-checklist)
21. [Quick Reference Card](#21-quick-reference-card)

---

## 1. PROJECT IDENTITY

| Field | Value |
|-------|-------|
| **Name** | DB-Surgeon (Database Surgery Environment) |
| **Type** | OpenEnv-compatible RL Environment |
| **Domain** | Database schema debugging & repair |
| **Agent Task** | Diagnose broken DBs, apply DDL fixes, restore business queries |
| **Model** | Qwen3-0.6B (dev) → Qwen3-1.7B (final demo, optional) |
| **Training** | TRL GRPOTrainer + Unsloth QLoRA on Google Colab T4 |
| **Episodes** | 100–300 for hackathon demo |
| **DB Engine** | SQLite `:memory:` (zero-cost isolation per episode) |
| **Framework** | OpenEnv (meta-pytorch/OpenEnv) |
| **Deployment** | Local first → HuggingFace Spaces for submission |

---

## 2. GOLDEN RULES (NON-NEGOTIABLE)

These rules come directly from the hackathon guides and RL best practices. Violating ANY of these is a project-killer.

### Rule 1: This is an RL Environment, NOT just code
> "The core idea is not just to fine-tune a text model, but to build a specialized LLM system that can act inside an environment, get feedback, and improve through reinforcement learning."
— hackathon_rl_guide.md §0

- You MUST have `reset()`, `step()`, `state()`
- You MUST show the agent improving over episodes
- You MUST show reward increasing over time

### Rule 2: Write the Verifier BEFORE the Policy
> "Write the verifier before writing the policy loop."
— rl_complete_guide.md §18

- Build and test the reward system BEFORE connecting the LLM
- Manually verify that correct fixes get positive reward
- Manually verify that hacking attempts get negative reward

### Rule 3: Reward = Task Specification
> "The reward is the task definition as far as optimization is concerned."
— rl_complete_guide.md §26

- If your reward is hackable, the model WILL hack it
- If your reward is too sparse, the model WILL NOT learn
- Multi-component reward with hidden eval queries is mandatory

### Rule 4: Don't Optimize What You Haven't Tried to Break
> "Do not optimize a reward you have not tried to break yourself first."
— rl_complete_guide.md §57

- Before training, play as the agent manually
- Try to hack your own reward
- Fix every loophole you find

### Rule 5: Success Probability MUST Be > 0
> "RL fails if success probability = 0."
— hackathon_rl_guide.md §1

> "The probability of a good answer must be greater than zero."
— rl_complete_guide.md §15

- If the task is too hard for the model to ever stumble into a positive reward, RL will waste compute
- Start with simple bugs the model can occasionally solve
- Use curriculum learning to increase difficulty

### Rule 6: Start From an Instruct Model
> "Start from a solid instruct model. Add a tiny amount of task-format SFT if needed. Build a strong verifier. Use GRPO/PPO-style RL only after the model can at least occasionally succeed."
— rl_complete_guide.md §16

- Use `Qwen/Qwen3-0.6B` (instruct variant) — NOT a raw base model
- The model must already understand tool-calling format before RL begins

### Rule 7: Monitor Behavior, Not Just Reward
> "They monitor the training reward but not actual behavior. Reward alone is not enough because the reward channel can be flawed."
— rl_complete_guide.md §52

- Log actual tool call sequences, not just reward numbers
- Sample and inspect completions during training
- Watch for degenerate patterns (repeating same action, gaming format)

---

## 3. ARCHITECTURE CONTRACT

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ Unsloth Model│  │ GRPOTrainer     │  │ Reward Funcs  │  │
│  │ (Qwen3-0.6B) │  │ (TRL)           │  │               │  │
│  │ QLoRA 4-bit  │──│ environment_    │──│ reads env.    │  │
│  │              │  │ factory pattern │  │ reward state  │  │
│  └──────────────┘  └────────┬────────┘  └───────────────┘  │
│                             │                                │
│              ┌──────────────▼──────────────┐                │
│              │   DBSurgeonToolEnv          │                │
│              │   (TRL Tool Wrapper)        │                │
│              │                             │                │
│              │   Public methods become     │                │
│              │   LLM tool calls:           │                │
│              │   • inspect_schema()        │                │
│              │   • run_query()             │                │
│              │   • fix_column()            │                │
│              │   • add_index()             │                │
│              │   • add_constraint()        │                │
│              │   • execute_fix()           │                │
│              │   • submit()               │                │
│              └──────────────┬──────────────┘                │
└─────────────────────────────┼───────────────────────────────┘
                              │ WebSocket / HTTP
┌─────────────────────────────▼───────────────────────────────┐
│                  OPENENV SERVER (Docker)                      │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastAPI Application                      │   │
│  │  ┌──────────────────────────────────────────────┐    │   │
│  │  │        DBSurgeonEnvironment                   │    │   │
│  │  │        (Environment base class)               │    │   │
│  │  │                                               │    │   │
│  │  │  reset() ←── BrokenDBGenerator               │    │   │
│  │  │  step()  ←── DBManager (SQLite :memory:)      │    │   │
│  │  │  state() ←── RewardCalculator                 │    │   │
│  │  │          ←── EvaluationOracle                  │    │   │
│  │  └──────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Two Integration Layers (CRITICAL to understand):

**Layer 1 — OpenEnv Server**: Classic `reset()` / `step(action)` / `state()` running as a FastAPI service. This is the raw environment that can run in Docker or locally.

**Layer 2 — TRL Tool Wrapper**: A Python class (`DBSurgeonToolEnv`) that wraps the OpenEnv client. Its public methods are auto-discovered by TRL's `GRPOTrainer` and exposed as function-calling tools to the LLM.

> **WHY two layers?** Because TRL's `environment_factory` pattern expects individual tool methods with docstrings and typed args — NOT a generic `step(action_json)`. The GRPOTrainer builds the tool schema from method signatures. See TRL docs: "Tools must be individual methods with descriptive names and typed arguments."

---

## 4. OPENENV API — EXACT SPEC

From the OpenEnv GitHub repo (`meta-pytorch/OpenEnv`):

### Required Models (Pydantic dataclasses):

```python
# Action — what the agent sends
@dataclass
class DBSurgeonAction:
    tool_name: str    # inspect_schema | run_query | fix_column | ...
    arguments: dict   # tool-specific args

# Observation — what the agent sees
@dataclass
class DBSurgeonObservation:
    schema_snapshot: str       # Current CREATE TABLE statements
    error_log: str             # SQL errors from last action
    failing_query: str         # The business query that needs fixing
    last_action_result: str    # Output of last tool call
    step_number: int           # Current step (0-indexed)
    max_steps: int             # Hard limit (15)
    action_history: list[str]  # Summary of past actions

# State — internal tracking
@dataclass
class DBSurgeonState:
    episode_id: str
    step_count: int
    initial_bug_type: str
    root_cause: str
    is_fixed: bool
    done: bool
```

### Required Environment Methods:

```python
class DBSurgeonEnvironment(Environment):
    def reset(self) -> DBSurgeonObservation:
        """Create fresh broken DB, return initial observation."""
    
    def step(self, action: DBSurgeonAction) -> StepResult:
        """Execute action, return (observation, reward, done)."""
    
    def state(self) -> DBSurgeonState:
        """Return current episode metadata."""
```

### Required Server Setup:

```python
# server/app.py
from openenv.core.env_server import create_app

app = create_app(
    create_db_surgeon_environment,  # factory function
    DBSurgeonAction,
    DBSurgeonObservation,
    max_concurrent_envs=64,  # for parallel training
)
```

### Required Client:

```python
# client.py
from openenv import EnvClient

class DBSurgeonEnv(EnvClient):
    """Client that connects to the FastAPI server."""
    # Handles WebSocket connection, serialization, type parsing
```

### Required Directory Structure:

```
db_surgeon/
├── __init__.py          # Exports: DBSurgeonAction, DBSurgeonObservation, DBSurgeonEnv
├── models.py            # Action, Observation, State dataclasses
├── client.py            # DBSurgeonEnv(EnvClient)
├── openenv.yaml         # Environment manifest
├── pyproject.toml       # Dependencies
├── README.md
└── server/
    ├── __init__.py
    ├── app.py           # FastAPI app
    ├── db_surgeon_environment.py  # Core logic
    ├── db_manager.py    # SQLite lifecycle
    ├── broken_db_generator.py     # Bug injection
    ├── reward.py        # Reward calculator
    ├── evaluation_oracle.py       # Hidden eval queries
    ├── requirements.txt
    └── Dockerfile
```

---

## 5. TRL ENVIRONMENT_FACTORY — EXACT PATTERN

From TRL docs (`huggingface.co/docs/trl/openenv`):

### The Pattern (copy this exactly):

```python
class DBSurgeonToolEnv:
    """Wraps the OpenEnv environment for TRL GRPOTrainer."""
    
    def __init__(self):
        # __init__ MUST take no arguments
        # Use module-level variables for config
        self.client = DBSurgeonEnv(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False
    
    def reset(self, **kwargs) -> str | None:
        # MUST accept **kwargs (dataset columns passed here)
        # MUST return str or None
        result = self.client.reset()
        self.reward = 0.0
        self.done = False
        return self._format_observation(result.observation)
    
    # Every public method (not starting with _) becomes a tool
    # MUST have docstring with Args: section
    # MUST have typed arguments
    
    def inspect_schema(self, table_name: str = "") -> str:
        """
        Inspect the database schema.
        
        Args:
            table_name: Optional table name for detailed info
            
        Returns:
            Schema information as text.
        """
        if self.done:
            raise ValueError("Episode is over.")
        result = self.client.step(DBSurgeonAction(
            tool_name="inspect_schema",
            arguments={"table_name": table_name}
        ))
        self._update_state(result)
        return result.observation.last_action_result
    
    # ... other tool methods follow same pattern ...
    
    def _update_state(self, result):
        """Private method — NOT exposed as tool (starts with _)."""
        self.reward = result.reward
        self.done = result.done
    
    def _format_observation(self, obs):
        """Private — format observation for LLM consumption."""
        return f"""DATABASE STATUS:
Schema: {obs.schema_snapshot}
Error Log: {obs.error_log}
Failing Query: {obs.failing_query}
Steps Remaining: {obs.max_steps - obs.step_number}"""
```

### Reward Function Pattern:

```python
def reward_func(environments, **kwargs) -> list[float]:
    """Read accumulated reward from each environment instance."""
    return [env.reward for env in environments]
```

### Trainer Setup:

```python
trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=GRPOConfig(...),
    environment_factory=DBSurgeonToolEnv,  # Pass the CLASS, not instance
)
```

### Key Rules from TRL Docs:

| Rule | Source |
|------|--------|
| `__init__` must take no arguments | TRL OpenEnv docs: Environment class requirements |
| Pass the **class**, not an instance, to `environment_factory` | TRL OpenEnv docs: How environment_factory works |
| `reset()` receives dataset columns as `**kwargs` | TRL OpenEnv docs: Environment class requirements |
| Tool methods MUST have docstrings with `Args:` | TRL OpenEnv docs: Environment class requirements |
| Raise `ValueError` to signal game over | TRL OpenEnv docs: Tips for environment classes |
| Private methods start with `_` and are not exposed | TRL OpenEnv docs: Environment class requirements |
| Store reward on `self.reward` for the reward function | TRL OpenEnv docs: Reward functions |

---

## 6. FILE-BY-FILE BUILD ORDER

### Phase 1: Foundation (Build FIRST, test standalone)

| Order | File | Purpose | Test Criteria |
|-------|------|---------|---------------|
| 1.1 | `pyproject.toml` | Dependencies | `pip install -e .` succeeds |
| 1.2 | `models.py` | Data structures | Import without errors |
| 1.3 | `__init__.py` | Package exports | `from db_surgeon import *` works |
| 1.4 | `openenv.yaml` | Manifest | Valid YAML |
| 1.5 | `server/db_manager.py` | SQLite lifecycle | Create DB, run queries, get schema |
| 1.6 | `server/broken_db_generator.py` | Bug injection | Generate broken DB, verify query fails |
| 1.7 | `server/evaluation_oracle.py` | Hidden eval | Score healthy DB = 1.0, broken = < 1.0 |
| 1.8 | `server/reward.py` | Reward calculator | Known inputs → expected rewards |

### Phase 2: Environment Core (Test with scripted agent)

| Order | File | Purpose | Test Criteria |
|-------|------|---------|---------------|
| 2.1 | `server/db_surgeon_environment.py` | Core env logic | `reset()` returns valid obs, `step()` executes |
| 2.2 | `server/app.py` | FastAPI server | Server starts, health check passes |
| 2.3 | `client.py` | OpenEnv client | Connect, reset, step cycle works |
| 2.4 | `server/Dockerfile` | Container | `docker build` succeeds |

### Phase 3: Training Integration (Test with tiny training run)

| Order | File | Purpose | Test Criteria |
|-------|------|---------|---------------|
| 3.1 | `training/tool_env.py` | TRL wrapper | All tools callable, reward accumulates |
| 3.2 | `training/dataset.py` | Prompt dataset | Valid HF Dataset object |
| 3.3 | `training/reward_functions.py` | Reward for GRPO | Returns list[float] of correct length |
| 3.4 | `training/train_grpo.py` | GRPO script | 5 episodes complete without crash |
| 3.5 | `training/train_unsloth.py` | Unsloth variant | Model loads in 4-bit, training starts |

### Phase 4: Evaluation & Demo

| Order | File | Purpose | Test Criteria |
|-------|------|---------|---------------|
| 4.1 | `examples/example_episode.py` | Step-by-step demo | Full episode plays out correctly |
| 4.2 | `examples/baseline_random.py` | Random baseline | Negative avg reward confirmed |
| 4.3 | `training/evaluate.py` | Eval script | Metrics saved to JSON |
| 4.4 | `metrics/plot_rewards.py` | Visualizations | Plots saved as PNG |
| 4.5 | `demo/gradio_app.py` | HF Space UI | Gradio launches locally |
| 4.6 | `README.md` | Documentation | Complete explanation + results |

---

## 7. BROKEN DB GENERATOR — FULL SPEC

### Phase 1 Bug Type (MVP): Foreign Key Violation

**What's broken:** A table's FOREIGN KEY references a column with mismatched name or non-existent table.

**Generation algorithm:**
```
1. Create "healthy" schema template:
   - tbl_{hex}_users(id INTEGER PK, name TEXT, email TEXT)
   - tbl_{hex}_orders(id INTEGER PK, user_id INTEGER, amount REAL, ...)
   - tbl_{hex}_products(id INTEGER PK, ...)
   - FK: orders.user_id → users.id

2. Inject bug (pick one variant randomly):
   a) Rename the FK source column: user_id → usr_id (but query still uses user_id)
   b) Change FK target: REFERENCES users(id) → REFERENCES users(email)
   c) Drop the referenced table entirely
   d) Mistype the referenced table name: users → usrs

3. Create business query that fails:
   SELECT u.name, o.amount 
   FROM tbl_{hex}_orders o 
   JOIN tbl_{hex}_users u ON o.user_id = u.id
   WHERE o.amount > 100;

4. Create hidden eval queries (3-5):
   - The business query above
   - An INSERT that would violate the broken FK
   - A COUNT aggregation across the JOIN
   - A subquery referencing the FK relationship

5. Record ground truth:
   - bug_type: "fk_violation"
   - root_cause: "orders.user_id column renamed to usr_id"
   - expected_fix: "ALTER TABLE ... RENAME COLUMN usr_id TO user_id"
```

### Randomization (Anti-Hacking):

```python
import secrets

def generate_prefix():
    return f"tbl_{secrets.token_hex(2)}"  # e.g., "tbl_a3f2"

# Every episode gets different:
# - Table name prefixes
# - Column order within tables
# - Number of dummy columns (2-5 extra per table)
# - Dummy data rows (5-20 per table)
# - Which FK variant is broken
```

### Phase 2+ Bug Types (expand if time permits):

| Bug | Variant Count | Difficulty |
|-----|--------------|------------|
| FK Violation | 4 variants | Easy |
| Datatype Mismatch | 3 variants | Medium |
| Missing Index (simulated) | 2 variants | Medium |
| Constraint Conflict | 3 variants | Hard |
| Schema Drift | 3 variants | Hard |

---

## 8. REWARD ENGINEERING — RULES & FORMULAS

### Core Principle:
> "Start simple, often with sparse success/failure reward, before layering in shaping terms."
— rl_complete_guide.md §39 (from OpenEnv docs)

### Reward Formula (per step):

```python
def calculate_step_reward(action, result, env_state) -> float:
    reward = 0.0
    
    # ─── OUTCOME REWARDS (most important) ───
    
    # Business query now passes → BIG reward
    if business_query_now_passes and not business_query_passed_before:
        reward += 5.0
    
    # All hidden eval queries pass (on submit) → BIG reward  
    if action == "submit":
        eval_score = oracle.score(db)  # 0.0 to 1.0
        reward += eval_score * 5.0     # Up to +5.0
    
    # ─── PROCESS REWARDS (secondary) ───
    
    # Fix addresses root cause → causal bonus
    if fix_matches_root_cause(action, ground_truth):
        reward += 3.0
    
    # Partial improvement (fewer errors than before)
    if error_count_decreased:
        reward += 2.0
    
    # Good diagnostic (inspected a relevant table)
    if action == "inspect_schema" and table_is_involved_in_bug:
        reward += 1.0
    
    # Efficiency bonus (solved in < step_limit/2 steps)
    if is_solved and step_count < max_steps / 2:
        reward += 1.0
    
    # ─── PENALTIES (anti-gaming) ───
    
    # Invalid SQL syntax
    if sql_syntax_error:
        reward -= 1.0
    
    # Broke something that was working
    if previously_passing_query_now_fails:
        reward -= 3.0
    
    # Repeated exact same action
    if action_is_duplicate:
        reward -= 1.0
    
    # Step tax (small, continuous)
    reward -= 0.1
    
    return reward
```

### Reward Accumulation Strategy:

The **TRL reward function** reads `env.reward` after the episode ends. 
Two options:

**Option A — Final reward only (RECOMMENDED for hackathon):**
```python
# In tool_env.py, only set self.reward on submit()
def submit(self):
    result = self.client.step(...)
    self.reward = result.reward  # Final cumulative score
```

**Option B — Accumulated reward:**
```python
# Sum up rewards across all steps
def _update_state(self, result):
    self.reward += result.reward  # Accumulate
```

> **Hackathon recommendation:** Use Option A (final reward only) because TRL's GRPO docs say "binary rewards gave cleaner training signals than shaped rewards with partial credit."

### Reward Validation Checklist:

Before training, manually verify these scenarios:

| Scenario | Expected Reward | Verified? |
|---------|----------------|-----------|
| Agent inspects schema, finds bug, fixes it, submits | +10 to +14 | |
| Agent applies correct fix immediately (no diagnosis) | +8 to +10 | |
| Agent inspects but fixes wrong thing, submits | -2 to +2 | |
| Agent does nothing and submits | -1 to 0 | |
| Agent breaks everything, submits | -5 to -3 | |
| Agent tries `DROP TABLE *`, submits | -4 to -2 | |
| Agent repeats same action 5 times | -5 | |

---

## 9. ANTI-REWARD-HACKING — MANDATORY CHECKLIST

> "Do not optimize a reward you have not tried to break yourself first."
— rl_complete_guide.md §57

### Must-Have Protections:

- [ ] **Randomized schema names** — Table/column names change every episode
- [ ] **Hidden evaluation queries** — Agent only sees the business query, not the full eval set
- [ ] **Step limit (15)** — Episode forcibly ends after 15 actions
- [ ] **DB reset per episode** — Fresh SQLite `:memory:` every reset
- [ ] **DDL whitelist** — Only allow safe DDL: ALTER, CREATE INDEX, CREATE TABLE (no DROP DATABASE, no TRUNCATE on critical tables)
- [ ] **Regression detection** — Track which queries pass before and after each action; penalize breaking working things
- [ ] **Execution sandbox** — SQLite `:memory:` is inherently sandboxed (no file access, no network)
- [ ] **Duplicate action detection** — Same tool call with same args = penalty
- [ ] **Output sampling** — Log and inspect agent completions during training

### Attack Vectors We've Mitigated:

| Attack | Mitigation |
|--------|------------|
| `DROP TABLE *; CREATE TABLE *` (nuke and rebuild) | Regression penalty (-3.0), hidden eval queries test structural integrity |
| Memorize table names across episodes | Randomized prefixes per episode |
| Spam `submit()` until lucky | Only one submit per episode (done=True after submit) |
| Ignore diagnosis, try all fix types | Efficiency penalty + step limit makes brute-force expensive |
| Inject SQL to read ground truth | Ground truth stored in Python memory, not in DB |
| Format gaming (output looks good but does nothing) | Only execution results matter, not text quality |

---

## 10. EDGE CASE CATALOG

### Environment Edge Cases:

| # | Edge Case | What Happens | How We Handle It |
|---|-----------|-------------|------------------|
| E1 | Agent sends empty SQL string | SQLite error | Catch, return "Error: empty SQL", reward -1.0 |
| E2 | Agent sends SQL with syntax error | SQLite parse error | Catch, return error message, reward -1.0 |
| E3 | Agent tries `DROP DATABASE` | Not supported in SQLite | Blocked by DDL whitelist |
| E4 | Agent tries to read PRAGMA or sqlite_master | Should work (diagnostic) | Allow — it's inspection |
| E5 | Agent calls `submit()` without fixing anything | Low eval score | Return 0/5 queries pass, low reward |
| E6 | Agent calls tool after `submit()` | Episode is over | Raise `ValueError("Episode is over.")` |
| E7 | Agent repeats exact same action 3+ times | Stuck in loop | Force `done=True` after 3 consecutive repeats |
| E8 | Agent finds a valid fix different from ground truth | Correct behavior | Accept — eval queries are the judge, not ground truth text |
| E9 | Agent fixes the bug AND breaks something else | Partial success | Net reward from eval queries (some pass, some fail) |
| E10 | SQLite doesn't support `ALTER COLUMN` | Real SQLite limitation | Our `fix_column()` tool handles this via table recreation internally |
| E11 | Agent sends extremely long SQL (>10KB) | Performance issue | Truncate at 5KB, return error |
| E12 | Agent calls `inspect_schema()` with non-existent table | No results | Return "Table not found: {name}" |
| E13 | Multiple bugs exist simultaneously (Phase 2+) | Complex scenario | Partial reward for each bug fixed |

### Training Edge Cases:

| # | Edge Case | What Happens | How We Handle It |
|---|-----------|-------------|------------------|
| T1 | All 8 GRPO completions get same reward | Zero variance → no gradient | This is normal early on; curriculum helps |
| T2 | Model never calls any tools | No environment interaction | System prompt must instruct tool use explicitly |
| T3 | Model calls wrong tool for the task | Wasted step | Return informative error, let model learn |
| T4 | Model generates malformed tool call JSON | Parse error | TRL handles this — feeds error back to model |
| T5 | WebSocket connection drops mid-episode | Training crash | Use `try/except` with reconnect in tool wrapper |
| T6 | VRAM overflow on Colab T4 | OOM | Use Unsloth 4-bit QLoRA, `num_generations=4` |
| T7 | Training reward goes up but quality goes down | Reward hacking | Sample outputs, check eval queries, increase penalty |
| T8 | Model outputs thinking tokens | Wasted context | Set `enable_thinking=False` in GRPOConfig |

---

## 11. GRPO TRAINING — HOW IT ACTUALLY WORKS

### The Algorithm (from rl_complete_guide.md §9 + TRL docs):

```
For each training batch:
  1. Sample a prompt from the dataset
  2. Generate G=4-8 completions (full episodes, multi-turn)
  3. Each completion = a sequence of tool calls + responses
  4. Score each completion using reward_func()
  5. Calculate advantage for each completion:
     advantage_i = (reward_i - mean(rewards)) / std(rewards)
  6. Update model weights:
     - Increase probability of high-advantage completions
     - Decrease probability of low-advantage completions
  7. Clip updates to prevent instability (like PPO)
```

### Why GRPO Works for DB-Surgeon:

- **No critic model needed** → saves VRAM (important for Colab T4)
- **Group comparison** → the model learns that "inspect first, then fix" is better than "fix randomly" because inspect-first episodes score higher within the same group
- **Verifiable rewards** → SQL pass/fail is deterministic, no learned reward model needed

### Multi-Turn Episode Flow:

```
Prompt: "You are a database engineer. Diagnose and fix the broken database."

Completion 1 (reward = -2.0):
  → inspect_schema() → "Tables: tbl_f1_users, tbl_f1_orders"
  → fix_column("tbl_f1_users", "id", new_type="TEXT") → "Error: can't change PK type"
  → submit() → 1/5 queries pass

Completion 2 (reward = +8.0):
  → inspect_schema() → "Tables: tbl_f1_users, tbl_f1_orders"
  → run_query("SELECT * FROM tbl_f1_orders LIMIT 1") → "Error: no column user_id"
  → inspect_schema("tbl_f1_orders") → "Columns: id, usr_id, amount"
  → fix_column("tbl_f1_orders", "usr_id", new_name="user_id") → "Success"
  → submit() → 5/5 queries pass

GRPO: Completion 2 gets positive advantage → reinforce those actions
GRPO: Completion 1 gets negative advantage → suppress those actions
```

---

## 12. UNSLOTH QLoRA — EXACT RECIPE

From Unsloth docs + rl_complete_guide.md §59:

```python
from unsloth import FastLanguageModel

# Load model in 4-bit for memory efficiency
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-0.6B",
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                  # LoRA rank
    lora_alpha=16,         # LoRA alpha
    lora_dropout=0,        # No dropout for RL
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # Memory optimization
)
```

### Critical Warning from rl_complete_guide.md §8:
> "Don't naively upcast a 4-bit model to 16-bit and then merge adapters, because that can damage model quality; use the proper merge path instead."

### Saving the Model:

```python
# CORRECT way to save
model.save_pretrained_merged("output_dir", tokenizer, save_method="merged_16bit")

# Or save LoRA adapter only
model.save_pretrained("output_adapter_dir")
```

---

## 13. TRAINING HYPERPARAMETERS

### For Google Colab T4 (15GB VRAM):

```python
GRPOConfig(
    # Model & Generation
    use_vllm=False,            # No vLLM on Colab (use native generation)
    max_completion_length=2048, # DB schemas can be long
    num_generations=4,          # 4 completions per prompt (not 8, saves VRAM)
    
    # Chat template
    chat_template_kwargs={"enable_thinking": False},
    
    # Optimization
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    max_steps=300,             # 100-300 episodes for hackathon
    
    # Logging
    log_completions=True,
    logging_steps=10,
    save_steps=50,
    
    # Memory
    bf16=True,                 # If supported
    gradient_checkpointing=True,
)
```

### Why These Values:

| Param | Value | Reason |
|-------|-------|--------|
| `num_generations=4` | Lower than default 8 | Fits in T4 VRAM |
| `max_completion_length=2048` | Higher than default | DB schemas + multi-turn conversation is verbose |
| `enable_thinking=False` | Off | Small models waste tokens on thinking; saves context window |
| `learning_rate=5e-6` | Conservative | RL is sensitive to LR; too high → instability |
| `gradient_accumulation_steps=8` | Moderate | Effective batch = 1 * 8 = 8 episodes per update |
| `max_steps=300` | Bounded | Enough to show improvement, fits hackathon timeline |

---

## 14. DEBUGGING ORDER (MANDATORY)

> "First debug the environment manually. Then debug the verifier. Then run scripted baseline policies. Then run a frozen model. Then run a tiny RL experiment. Only then scale."
— rl_complete_guide.md §56

### Step-by-step:

```
1. TEST DB MANAGER STANDALONE
   → Create DB, insert data, run queries, verify results
   → Test ALTER TABLE, CREATE INDEX, etc.
   → Time: 15 minutes

2. TEST BROKEN DB GENERATOR STANDALONE
   → Generate 10 broken DBs
   → For each: verify the business query fails
   → For each: verify the eval queries would pass on healthy DB
   → Time: 15 minutes

3. TEST REWARD CALCULATOR STANDALONE
   → Feed known (action, result, state) tuples
   → Verify correct reward output
   → Try to hack it manually
   → Time: 15 minutes

4. TEST ENVIRONMENT (reset/step) WITH SCRIPTED AGENT
   → Write a script that calls reset(), then step() with correct fix
   → Verify: positive reward, done=True
   → Write a script that calls reset(), then step() with wrong fix
   → Verify: negative or low reward
   → Time: 20 minutes

5. TEST TRL TOOL WRAPPER STANDALONE
   → Create DBSurgeonToolEnv()
   → Call reset()
   → Call inspect_schema()
   → Call fix_column() with correct fix
   → Call submit()
   → Verify: self.reward > 0
   → Time: 20 minutes

6. RUN TINY TRAINING (5 steps)
   → Set max_steps=5
   → Verify: training loop completes without crash
   → Verify: reward values are logged
   → Time: 10 minutes

7. RUN SMALL TRAINING (50 steps)
   → Verify: reward trend is not degenerate
   → Sample 5 completions, read them manually
   → Time: 30 minutes

8. FULL TRAINING (100-300 steps)
   → Monitor reward curve
   → Sample completions periodically
   → Save checkpoint
   → Time: 1-3 hours
```

---

## 15. METRICS TO COLLECT

### During Training:

| Metric | How to Compute | Why It Matters |
|--------|---------------|----------------|
| `avg_reward` per step | Mean reward across all episodes | Shows learning trend |
| `success_rate` | % of episodes where all eval queries pass | The real measure |
| `avg_steps_to_solve` | Mean step count for successful episodes | Shows efficiency |
| `tool_call_distribution` | Count of each tool type used | Shows behavior change |
| `invalid_sql_rate` | % of steps with SQL errors | Should decrease |
| `submit_early_rate` | % of episodes that submit before step limit | Should increase |

### For Demo/Judges:

| Metric | Presentation |
|--------|-------------|
| Reward vs. training steps (smoothed) | Line plot |
| Success rate vs. training steps | Line plot |
| Before/after episode comparison | Side-by-side text |
| Tool call pattern evolution | Stacked bar chart |

### Data Collection Code:

```python
import json
from datetime import datetime

class MetricsLogger:
    def __init__(self):
        self.episodes = []
    
    def log_episode(self, episode_data: dict):
        self.episodes.append({
            "timestamp": datetime.now().isoformat(),
            "episode_id": episode_data["episode_id"],
            "reward": episode_data["reward"],
            "steps": episode_data["steps"],
            "success": episode_data["all_eval_pass"],
            "bug_type": episode_data["bug_type"],
            "tool_calls": episode_data["tool_call_sequence"],
        })
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.episodes, f, indent=2)
```

---

## 16. DEMO & STORYTELLING REQUIREMENTS

### From hackathon_rl_guide.md §19 — Judges Expect:

1. **Clear system** — Architecture diagram, data flow
2. **Good rewards** — Multi-component, hard to hack
3. **Improvement proof** — Before/after training comparison
4. **Demo** — Interactive, visual

### Storytelling Structure (5 acts):

```
ACT 1 — THE PROBLEM
"Database debugging is a $X billion cost center. Engineers spend hours
tracing cascading failures through schema misconfigurations."
→ Show a broken query error message

ACT 2 — THE ENVIRONMENT
"We built DB-Surgeon: an RL environment where an LLM agent diagnoses
and fixes broken databases through multi-step tool use."
→ Show architecture diagram
→ Show initial observation the agent sees

ACT 3 — THE LEARNING
"Using GRPO with verifiable rewards, the agent learns to inspect
before fixing, identify root causes, and apply targeted DDL fixes."
→ Show reward curve (going UP)
→ Show before/after behavior comparison

ACT 4 — THE RESULTS
"After 200 episodes: 45% success rate (from 0%), average 5 steps
(from 15), and structured diagnostic behavior."
→ Show metrics table
→ Show qualitative example

ACT 5 — THE IMPACT
"This approach generalizes to any system with verifiable state:
CI/CD pipelines, cloud infrastructure, network debugging."
→ Show extension possibilities
```

### Gradio Demo Plan:

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# DB-Surgeon: RL for Database Debugging")
    
    with gr.Row():
        # Left: Episode controls
        with gr.Column():
            new_episode_btn = gr.Button("New Episode")
            run_agent_btn = gr.Button("Run Agent")
            run_baseline_btn = gr.Button("Run Baseline")
            episode_log = gr.Textbox(label="Episode Log", lines=20)
        
        # Right: DB state
        with gr.Column():
            schema_display = gr.Code(label="Current Schema", language="sql")
            error_display = gr.Textbox(label="Error Log")
            reward_display = gr.Number(label="Reward")
            step_display = gr.Number(label="Step")
    
    # Bottom: Metrics
    reward_plot = gr.Plot(label="Reward Over Training")
```

---

## 17. JUDGE EXPECTATIONS

### What Judges Score (from hackathon guides + OpenEnv docs):

| Criterion | Weight | How We Satisfy It |
|-----------|--------|-------------------|
| **Task Definition** | High | Clear: diagnose broken DB, apply fix, restore queries |
| **Grading Mechanisms** | High | Multi-component reward with eval oracle |
| **Reward Logic** | High | Verifiable (SQL pass/fail), multi-signal, anti-hacking |
| **Functionality** | High | Full OpenEnv API: reset/step/state, Dockerized |
| **Learning Demonstration** | Critical | Reward curve, success rate, before/after behavior |
| **Compatibility** | Medium | Standard OpenEnv structure, TRL integration |
| **Reusability** | Medium | Can add new bug types, configurable difficulty |
| **Demo Quality** | Medium | Gradio UI, interactive exploration |

### What Will Impress:

1. **Reward going UP** — The single most important visual
2. **Behavior change** — Show the agent doing something DIFFERENT after training
3. **Anti-hacking** — Explain what shortcuts you prevented
4. **Real-world relevance** — "This isn't a toy problem"

### What Will NOT Impress:

1. "We built a cool environment but didn't train anything"
2. "The reward goes up but we can't explain why"
3. "We have one reward signal and no anti-hacking"
4. "Our environment works but has no demo"

---

## 18. FATAL MISTAKES TO AVOID

> From rl_complete_guide.md and hackathon_rl_guide.md, synthesized:

### Architecture Mistakes:

| Mistake | Why It's Fatal | Fix |
|---------|---------------|-----|
| Using raw `step(action_json)` with TRL | TRL needs named tool methods | Use `environment_factory` pattern |
| Not using an instruct model | Base model can't follow tool-calling format | Use `Qwen3-0.6B` (instruct) |
| Making the task too hard initially | Model never gets positive reward → RL stalls | Start with simple FK bugs |
| Skipping the verifier | No way to know if reward is correct | Build + test verifier before training |

### Reward Mistakes:

| Mistake | Why It's Fatal | Fix |
|---------|---------------|-----|
| Single reward signal | Easy to game | Multi-component with hidden queries |
| Rewarding format over substance | Model optimizes text appearance | Only reward execution results |
| No negative reward for breaking things | Model learns destructive shortcuts | -3.0 for breaking working queries |
| Sparse reward only (no partial credit) | Model never finds positive signal | Add diagnostic rewards (+1.0) |
| Dense reward that conflicts | Model oscillates | Start sparse, add shaping carefully |

### Training Mistakes:

| Mistake | Why It's Fatal | Fix |
|---------|---------------|-----|
| Training too long without monitoring | Reward hacking goes undetected | Sample outputs every 50 steps |
| Not saving checkpoints | Lose best model when it degrades | Save every 50 steps |
| OOM on Colab | Training crashes | Use Unsloth 4-bit, num_generations=4 |
| Using `enable_thinking=True` with small models | Wastes tokens on poor reasoning | Set `enable_thinking=False` |

### Demo Mistakes:

| Mistake | Why It's Fatal | Fix |
|---------|---------------|-----|
| No reward curve visualization | Judges can't see improvement | Plot reward and success rate |
| No before/after comparison | Judges can't see behavior change | Show episode transcripts |
| No architecture diagram | Judges can't understand the system | Include in README and demo |

---

## 19. CURRICULUM & DIFFICULTY PROGRESSION

> "Start with short horizons, fewer tools, simpler state spaces, stronger hints, easier test cases, then gradually remove scaffolding."
— rl_complete_guide.md §14

### For DB-Surgeon:

```
LEVEL 1 (Start here):
├── Bug Types: FK violation only (1 type)
├── Schema Size: 2-3 tables
├── Randomization: Table prefix only
├── Hints: Error message explicitly mentions the broken column
├── Step Limit: 15 steps
└── Goal: Agent occasionally solves it → reward signal exists

LEVEL 2 (After Level 1 shows improvement):
├── Bug Types: FK + Datatype mismatch (2 types)
├── Schema Size: 3-4 tables
├── Randomization: Table prefix + column order shuffled
├── Hints: Error message is realistic (not dumbed down)
├── Step Limit: 12 steps
└── Goal: Agent reliably diagnoses before fixing

LEVEL 3 (If time allows):
├── Bug Types: All 5 types
├── Schema Size: 4-6 tables
├── Randomization: Full (names, order, dummy cols, red herrings)
├── Hints: Standard SQLite error messages only
├── Step Limit: 10 steps
└── Goal: Agent generalizes across bug types
```

### Key Insight:
> "RL fails if success probability = 0."

If the model NEVER solves a Level 1 bug, RL has no signal. In that case:
- Make bugs even simpler
- Add stronger hints in the observation
- Consider a tiny SFT warmup on 10-20 solved examples

---

## 20. DEPLOYMENT CHECKLIST

### For HuggingFace Spaces Submission:

- [ ] Environment runs in Docker container
- [ ] `openenv.yaml` manifest is valid
- [ ] `pyproject.toml` has all dependencies
- [ ] `Dockerfile` builds on Linux
- [ ] Server starts on port 7860 (HF Spaces default)
- [ ] Concurrent sessions supported (set `max_concurrent_envs`)
- [ ] Client installable via `pip install git+https://huggingface.co/spaces/...`
- [ ] Gradio demo works standalone
- [ ] README.md has clear instructions
- [ ] Model adapter uploaded to HF Hub
- [ ] Metrics/plots saved in results/

### Deployment Command:

```bash
cd db_surgeon
openenv push --repo-id yourusername/db-surgeon --private
```

---

## 21. QUICK REFERENCE CARD

### Environment Lifecycle:

```
reset() → Observation
  ↓
step(Action) → (Observation, Reward, Done)
  ↓ (repeat until done)
step(submit) → Final reward, Done=True
```

### TRL Integration:

```python
# Class with tool methods → passed to GRPOTrainer
environment_factory=DBSurgeonToolEnv

# Reward function reads env state
reward_funcs=lambda envs, **kw: [e.reward for e in envs]
```

### Key File Paths:

```
server/db_surgeon_environment.py  → Core environment logic
server/broken_db_generator.py     → Bug injection
server/reward.py                  → Reward calculation
training/tool_env.py              → TRL wrapper
training/train_grpo.py            → Training script
```

### Commands:

```bash
# Install
pip install -e .

# Run server locally
python -m db_surgeon.server.app

# Run example episode
python examples/example_episode.py

# Train (Colab)
python training/train_grpo.py

# Deploy
openenv push
```

### Emergency Fixes:

| Problem | Solution |
|---------|----------|
| OOM on Colab | Reduce `num_generations` to 2, reduce `max_completion_length` to 1024 |
| Reward stuck at negative | Simplify the bug type, add more hints |
| Model doesn't call tools | Improve system prompt, consider SFT warmup |
| Server crashes on step | Add try/except around SQLite execution |
| Training reward goes down | Save last good checkpoint, reduce learning rate |

---

> **Remember the one rule:** *If you can build a task where success is verifiable, difficulty is controllable, and loopholes are monitored, RL can turn an LLM from "good at answering" into "better at acting."* — rl_complete_guide.md §20
