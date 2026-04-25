# DB-SURGEON — Complete Project Guide

> Everything you need to understand, test, run, and train this project.

---

## TABLE OF CONTENTS

1. [What This Project Is](#1-what-this-project-is)
2. [How The Whole System Works (Big Picture)](#2-how-the-whole-system-works-big-picture)
3. [Complete Data Flow — Step by Step](#3-complete-data-flow--step-by-step)
4. [Every File Explained](#4-every-file-explained)
5. [How to Set Up The Project](#5-how-to-set-up-the-project)
6. [How to Run & Test Everything](#6-how-to-run--test-everything)
7. [How to Train the Agent](#7-how-to-train-the-agent)
8. [How to Deploy](#8-how-to-deploy)
9. [Troubleshooting Common Issues](#9-troubleshooting-common-issues)

---

## 1. WHAT THIS PROJECT IS

DB-Surgeon is a **Reinforcement Learning environment** where an LLM agent learns to fix broken databases.

Think of it like a video game:
- The **game world** = a broken SQLite database
- The **player** = an LLM (like Qwen3-0.6B)
- The **moves** = SQL tools (inspect, query, fix, submit)
- The **score** = reward based on whether the database is repaired
- The **training** = GRPO algorithm makes the LLM better at the game over time

**What makes this an RL project (not just code)?**

```
Without RL:  Human writes rules → LLM follows rules → fixed output
With RL:     LLM tries stuff → gets reward/penalty → learns what works → gets better
```

The LLM starts terrible (random actions, negative reward). After 200+ episodes of trial-and-error with GRPO, it learns patterns like "always inspect first, then fix the right column, then submit."

---

## 2. HOW THE WHOLE SYSTEM WORKS (BIG PICTURE)

There are **3 layers** in this project. Each layer talks to the one below it:

```
╔══════════════════════════════════════════════════════════════╗
║  LAYER 3: TRAINING PIPELINE                                  ║
║                                                              ║
║  GRPOTrainer + Unsloth (QLoRA)                              ║
║  - Generates 4 completions per prompt                        ║
║  - Each completion = a full multi-turn episode               ║
║  - Compares rewards across completions                       ║
║  - Updates model weights toward better strategies            ║
║                                                              ║
║  Files: train_grpo.py, train_unsloth.py, dataset.py,        ║
║         reward_functions.py                                   ║
╠══════════════════════════════════════════════════════════════╣
║  LAYER 2: TRL TOOL WRAPPER                                   ║
║                                                              ║
║  DBSurgeonToolEnv (tool_env.py)                              ║
║  - Public methods = tools the LLM can call                   ║
║  - inspect_schema(), run_query(), fix_column(), submit()...  ║
║  - Translates LLM function calls → environment actions       ║
║  - Accumulates reward for the training pipeline              ║
║                                                              ║
║  WHY: TRL GRPOTrainer needs named methods, not raw step()    ║
╠══════════════════════════════════════════════════════════════╣
║  LAYER 1: OPENENV ENVIRONMENT                                ║
║                                                              ║
║  DBSurgeonEnvironment (db_surgeon_environment.py)            ║
║  - reset() → generates broken DB from scratch                ║
║  - step(action) → executes SQL, calculates reward            ║
║  - state() → returns episode metadata                        ║
║                                                              ║
║  Sub-components:                                              ║
║  - DBManager → manages the SQLite :memory: database          ║
║  - BrokenDBGenerator → injects bugs into schemas             ║
║  - RewardCalculator → multi-component reward scoring         ║
║  - EvaluationOracle → hidden eval queries for anti-hacking   ║
║                                                              ║
║  Served as: FastAPI app (app.py) or direct Python (client.py)║
╚══════════════════════════════════════════════════════════════╝
```

---

## 3. COMPLETE DATA FLOW — STEP BY STEP

### Episode Lifecycle (what happens from start to finish):

```
STEP 1: RESET
─────────────
GRPOTrainer calls → DBSurgeonToolEnv.reset()
                   → DBSurgeonLocalEnv.reset()
                   → DBSurgeonEnvironment.reset()
                   → BrokenDBGenerator.generate()
                       → Picks random FK violation variant
                       → Creates randomized table names (tbl_a3f2_users, etc.)
                       → Builds healthy schema SQL
                       → Injects bug (rename column, remove table, etc.)
                       → Creates business query that will fail
                       → Creates 5 hidden eval queries
                   → DBManager.create_database(broken_schema, seed_data)
                       → sqlite3.connect(":memory:")
                       → Executes CREATE TABLE statements
                       → Inserts seed data
                   → EvaluationOracle.set_baseline(db)
                       → Runs all eval queries to record initial pass/fail
                   → RewardCalculator.__init__(db, oracle, business_query, ...)
                       → Records initial state for reward tracking
                   → Returns Observation:
                       {schema_snapshot, error_log, failing_query, ...}

GRPOTrainer shows the observation to the LLM as a message.
The LLM now sees: "Here's a broken database. The query fails. Fix it."


STEP 2-N: AGENT ACTIONS (the LLM makes tool calls)
────────────────────────────────────────────────────
LLM generates → function call: inspect_schema(table_name="tbl_a3f2_orders")

GRPOTrainer calls → DBSurgeonToolEnv.inspect_schema("tbl_a3f2_orders")
                   → DBSurgeonLocalEnv.step(Action("inspect_schema", {table_name: "..."}))
                   → DBSurgeonEnvironment.step(action)
                       → _execute_action(action)
                           → _action_inspect_schema(args)
                           → DBManager.get_table_info("tbl_a3f2_orders")
                               → PRAGMA table_info(tbl_a3f2_orders)
                               → Returns column names, types, constraints
                       → RewardCalculator.calculate(action, result, step)
                           → Checks: is this a relevant table? (+1.0 if yes)
                           → Checks: invalid SQL? (-1.0 if yes)
                           → Checks: repeated action? (-1.0 if yes)
                           → Applies step tax (-0.1)
                           → Returns: reward = +0.9 (good diagnostic)
                       → Updates state (step_count++, total_reward += reward)
                       → Returns StepResult(observation, reward, done=False)

GRPOTrainer appends the tool result to the conversation.
LLM sees: "Table tbl_a3f2_orders: id INTEGER PK, usr_id INTEGER, ..."
LLM thinks: "Hmm, usr_id should be user_id. Let me fix that."


STEP N+1: FIX
─────────────
LLM generates → function call: fix_column("tbl_a3f2_orders", "usr_id", new_name="user_id")

GRPOTrainer calls → DBSurgeonToolEnv.fix_column(...)
                   → Environment.step(Action("fix_column", {...}))
                       → DBManager.fix_column("tbl_a3f2_orders", "usr_id", new_name="user_id")
                           → ALTER TABLE tbl_a3f2_orders RENAME COLUMN usr_id TO user_id
                       → RewardCalculator.calculate(...)
                           → Business query now passes! (+5.0)
                           → Fix addresses root cause! (+3.0)
                           → Partial improvement! (+2.0)
                           → Efficient! (+1.0)
                           → Step tax (-0.1)
                           → Returns: reward = +10.9
                       → Returns StepResult(observation, reward=+10.9, done=False)


STEP FINAL: SUBMIT
───────────────────
LLM generates → function call: submit()

GRPOTrainer calls → DBSurgeonToolEnv.submit()
                   → Environment.step(Action("submit", {}))
                       → EvaluationOracle.detailed_score(db)
                           → Runs all 5 hidden eval queries
                           → 5/5 pass = 100% score
                       → RewardCalculator.calculate(..., is_submit=True)
                           → eval_score * 5.0 = +5.0
                           → Step tax: -0.1
                           → Returns: reward = +4.9
                       → Sets done=True, is_fixed=True
                   → DBSurgeonToolEnv.reward = total accumulated reward (+16.3)
                   → Returns "5/5 queries passed! Database fixed!"

GRPOTrainer reads env.reward = +16.3 for this completion.


STEP GRPO UPDATE:
─────────────────
GRPOTrainer has 4 completions for the same prompt:
  Completion 1: reward = +16.3 (inspected → diagnosed → fixed → submitted)
  Completion 2: reward = -3.5  (random fixes → broke things → submitted)
  Completion 3: reward = +2.1  (inspected → wrong fix → submitted)
  Completion 4: reward = -1.2  (inspected → gave up → submitted)

GRPO calculates advantage:
  mean = 3.43, std = 8.1
  Advantage 1: (16.3 - 3.43) / 8.1 = +1.59 (REINFORCE these actions)
  Advantage 2: (-3.5 - 3.43) / 8.1 = -0.86 (SUPPRESS these actions)
  Advantage 3: (2.1 - 3.43) / 8.1 =  -0.16 (slightly suppress)
  Advantage 4: (-1.2 - 3.43) / 8.1 = -0.57 (suppress)

Model weights updated: "inspect then fix" strategy becomes more likely.
Over 200+ episodes, the model consistently learns:
  1. Always inspect schema first
  2. Read error messages for clues
  3. Apply targeted fix
  4. Verify before submitting
```

---

## 4. EVERY FILE EXPLAINED

### Root Files

| File | What It Does | Why It Exists |
|------|-------------|---------------|
| `pyproject.toml` | Lists all Python dependencies, defines the package name (`db-surgeon-env`), and sets up install groups (core, training, unsloth, demo, dev) | So you can `pip install -e .` and all imports work |
| `openenv.yaml` | Declares this project as an OpenEnv environment — specifies entry points for server, client, models | Required by OpenEnv framework for deployment and discovery |
| `models.py` | Defines 4 dataclasses: `DBSurgeonAction` (what the agent sends), `DBSurgeonObservation` (what the agent sees), `DBSurgeonState` (internal tracking), `StepResult` (step output) | All other files import these — they're the shared language |
| `__init__.py` | Exports public classes so you can do `from db_surgeon import DBSurgeonAction` | Standard Python packaging |
| `client.py` | Two client classes: `DBSurgeonEnv` (HTTP client for remote server) and `DBSurgeonLocalEnv` (direct Python, no HTTP overhead) | LocalEnv used during training for speed; HTTP client for deployment |

### Server Files (`server/`)

| File | What It Does | Key Classes/Functions |
|------|-------------|----------------------|
| `db_manager.py` | Manages one SQLite `:memory:` database. Creates it, runs queries, modifies schema, inspects tables, validates fixes. Has DDL whitelist to block destructive operations like `DROP DATABASE`. | `DBManager` with methods: `create_database()`, `execute_query()`, `execute_ddl()`, `fix_column()`, `add_index()`, `get_schema()`, `get_table_info()`, `validate_fix()`, `reset()` |
| `broken_db_generator.py` | Generates a random broken database scenario per episode. Currently implements 4 FK violation variants: column renamed, wrong FK reference, missing table, table name typo. Each scenario includes randomized table prefixes (anti-memorization), seed data, a business query that fails, and 5 hidden eval queries. | `BrokenDBGenerator.generate()` → returns `BrokenDBScenario` dataclass with everything needed for one episode |
| `evaluation_oracle.py` | Holds the hidden eval queries (the agent only sees 1 business query, oracle tests 5). Tracks which queries pass before/after each action to detect regressions. Used for final scoring on submit. | `EvaluationOracle` with methods: `set_baseline()`, `score()`, `detailed_score()`, `count_regressions()`, `update_baseline()` |
| `reward.py` | Calculates the multi-component reward for each step. 10 signals: business query pass (+5), eval score (+5), causal fix (+3), improvement (+2), good diagnostic (+1), efficiency (+1), invalid SQL (-1), regression (-3), repeats (-1), step tax (-0.1). | `RewardCalculator.calculate(action, result, step_number, is_submit)` → returns float |
| `db_surgeon_environment.py` | The **core environment**. Wires everything together. `reset()` generates broken DB + sets up oracle + reward calculator. `step()` routes actions to the right handler, calculates reward, checks if done. `state()` returns metadata. | `DBSurgeonEnvironment` with `reset()`, `step()`, `state()`, plus 7 action handlers: `_action_inspect_schema`, `_action_run_query`, `_action_fix_column`, `_action_add_index`, `_action_add_constraint`, `_action_execute_fix`, `_action_submit` |
| `app.py` | FastAPI web server with both WebSocket (for OpenEnv protocol) and HTTP (for testing). WebSocket supports concurrent sessions via session_id. HTTP endpoints: `/health`, `/info`, `/reset`, `/step`, `/state`. | `create_app()` returns a FastAPI app. Run with `python -m db_surgeon.server.app` |
| `Dockerfile` | Container definition for deploying the server in Docker/HF Spaces. | Standard Python 3.11 slim image |
| `requirements.txt` | Server-only pip dependencies (fastapi, uvicorn, pydantic, websockets, requests) | Used by Dockerfile |

### Training Files (`training/`)

| File | What It Does | Key Classes/Functions |
|------|-------------|----------------------|
| `tool_env.py` | **The most important training file.** Wraps the environment for TRL's GRPOTrainer. Every public method becomes a tool the LLM can call. Private methods (starting with `_`) are hidden. Stores accumulated reward on `self.reward` for the reward function to read. | `DBSurgeonToolEnv` with 7 public tool methods: `inspect_schema()`, `run_query()`, `fix_column()`, `add_index()`, `add_constraint()`, `execute_fix()`, `submit()` |
| `dataset.py` | Creates the HuggingFace Dataset for GRPO training. Each entry is a single user message with a system prompt that teaches the agent how to use the tools. Same prompt repeated N times (environment randomizes scenarios on each reset). | `create_training_dataset(num_episodes=200)` → HF Dataset; `SYSTEM_PROMPT` constant |
| `reward_functions.py` | Reward functions passed to GRPOTrainer. Primary function reads `env.reward` from each environment instance. Optional format reward gives small bonus for proper tool-calling. | `reward_func(environments, **kwargs)` → `list[float]` |
| `train_grpo.py` | Standard GRPO training script. Loads model, creates dataset, configures GRPOTrainer with environment_factory, runs training, saves model. Uses env vars for configuration. | `main()` function; configurable via `DB_SURGEON_MODEL`, `DB_SURGEON_EPISODES`, `DB_SURGEON_OUTPUT` |
| `train_unsloth.py` | **Recommended training script for Colab T4.** Same as train_grpo but uses Unsloth for 4-bit QLoRA loading, fp16 instead of bf16 (T4 compatibility), and correct LoRA merge path for saving. | `main()` function; uses `FastLanguageModel.from_pretrained(load_in_4bit=True)` |
| `evaluate.py` | Runs a trained model on fresh episodes and saves metrics to JSON. | `evaluate_model(model_path, n_episodes, output_dir)` |

### Examples (`examples/`)

| File | What It Does |
|------|-------------|
| `example_episode.py` | Runs one complete episode with a **scripted "smart" agent** that reads the root cause and applies the correct fix. Demonstrates the full lifecycle: reset → inspect → diagnose → fix → verify → submit. Shows expected positive reward. |
| `baseline_random.py` | Runs N episodes with a **random agent** that picks random tools with random arguments. Establishes the baseline: ~0% success rate, negative average reward. Saves results to `metrics/results/baseline_random.json`. |

### Metrics (`metrics/`)

| File | What It Does |
|------|-------------|
| `plot_rewards.py` | Generates publication-quality matplotlib plots: reward curve (with smoothing), success rate over time, baseline vs. trained comparison. Has `--demo` mode that generates plots with simulated data (for presentation setup before actual training). |

---

## 5. HOW TO SET UP THE PROJECT

### Prerequisites

- Python 3.10+ installed
- pip installed
- (For training) Google Colab with T4 GPU, or local GPU with 16GB+ VRAM

### Step 1: Navigate to the project

```powershell
cd E:\Scaler
```

### Step 2: Install the package in development mode

```powershell
# Core environment only (no GPU needed)
pip install -e ./db_surgeon

# If you want to train locally (needs GPU + CUDA)
pip install -e "./db_surgeon[training]"

# If you want to use Unsloth (recommended for Colab)
pip install -e "./db_surgeon[training,unsloth]"

# For visualization
pip install -e "./db_surgeon[dev]"
```

### Step 3: Verify installation

```powershell
python -c "from db_surgeon import DBSurgeonAction, DBSurgeonObservation; print('Import OK')"
```

**Expected output:** `Import OK`

---

## 6. HOW TO RUN & TEST EVERYTHING

### Test 1: Verify individual components work

Run these commands one at a time from `E:\Scaler`:

```powershell
# Set UTF-8 encoding (Windows needs this for emoji output)
$env:PYTHONIOENCODING="utf-8"
```

#### Test 1a: DB Manager (SQLite works correctly)

```powershell
python -c "
from db_surgeon.server.db_manager import DBManager

db = DBManager()
db.create_database('''
CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
''', '''
INSERT INTO users VALUES (1, 'Alice', 'alice@test.com');
INSERT INTO users VALUES (2, 'Bob', 'bob@test.com');
INSERT INTO orders VALUES (1, 1, 99.99);
INSERT INTO orders VALUES (2, 2, 149.50);
''')

# Test query
success, result = db.execute_query('SELECT u.name, o.amount FROM orders o JOIN users u ON o.user_id = u.id')
print('Query test:', 'PASS' if success else 'FAIL')
print(result)

# Test schema
print('\nSchema:')
print(db.get_schema())

# Test DDL whitelist
success, result = db.execute_ddl('DROP DATABASE test')
print('\nDDL block test:', 'PASS' if not success else 'FAIL (should be blocked!)')
print(result)

# Test fix_column
success, result = db.fix_column('orders', 'user_id', new_name='uid')
print('\nFix column test:', 'PASS' if success else 'FAIL')
print(result)

db.reset()
print('\nAll DB Manager tests passed!')
"
```

**Expected:** All tests show PASS, DDL block shows blocked, fix_column succeeds.

#### Test 1b: Broken DB Generator (creates valid broken scenarios)

```powershell
python -c "
from db_surgeon.server.broken_db_generator import BrokenDBGenerator
from db_surgeon.server.db_manager import DBManager

gen = BrokenDBGenerator(seed=42)

for i in range(5):
    scenario = gen.generate()
    
    # Verify: business query should FAIL on broken schema
    db = DBManager()
    db.create_database(scenario.schema_sql, scenario.seed_data_sql)
    success, error = db.execute_query(scenario.business_query)
    
    # Verify: business query should PASS on healthy schema
    db2 = DBManager()
    db2.create_database(scenario.healthy_schema_sql, scenario.healthy_seed_data_sql)
    success2, _ = db2.execute_query(scenario.business_query)
    
    status = 'OK' if (not success and success2) else 'PROBLEM'
    print(f'  Scenario {i+1}: {status} | variant={scenario.bug_variant} | prefix={scenario.table_prefix}')
    if status == 'PROBLEM':
        print(f'    broken_fails={not success}, healthy_passes={success2}')
        print(f'    error: {error[:100]}')
    
    db.reset()
    db2.reset()

print('\nAll generator tests passed!')
"
```

**Expected:** 5 scenarios, all show `OK`, each with different variant and prefix.

#### Test 1c: Reward Calculator (correct rewards for known inputs)

```powershell
python -c "
from db_surgeon.server.broken_db_generator import BrokenDBGenerator
from db_surgeon.server.db_manager import DBManager
from db_surgeon.server.evaluation_oracle import EvaluationOracle
from db_surgeon.server.reward import RewardCalculator
from db_surgeon.models import DBSurgeonAction

gen = BrokenDBGenerator(seed=99)
scenario = gen.generate()

db = DBManager()
db.create_database(scenario.schema_sql, scenario.seed_data_sql)

oracle = EvaluationOracle(scenario.eval_queries)
oracle.set_baseline(db)

calc = RewardCalculator(
    db=db, oracle=oracle,
    business_query=scenario.business_query,
    root_cause=scenario.root_cause,
    involved_tables=scenario.involved_tables,
)

# Test 1: Inspect schema (should be positive - good diagnostic)
action = DBSurgeonAction(tool_name='inspect_schema', arguments={'table_name': ''})
r1 = calc.calculate(action, (True, 'schema output'), step_number=0)
print(f'Inspect schema reward: {r1:+.1f} (expected ~+0.9)')

# Test 2: Invalid SQL (should be negative)
action2 = DBSurgeonAction(tool_name='run_query', arguments={'sql': 'SELEC WRONG'})
db.execute_query('SELEC WRONG')  # This will fail
r2 = calc.calculate(action2, (False, 'syntax error'), step_number=1)
print(f'Invalid SQL reward:    {r2:+.1f} (expected ~-1.1)')

# Test 3: Repeated action (should be penalty)
action3 = DBSurgeonAction(tool_name='inspect_schema', arguments={'table_name': ''})
r3 = calc.calculate(action3, (True, 'schema'), step_number=2)
print(f'Repeated action reward: {r3:+.1f} (expected ~-1.1)')

print()
print(f'Bug type: {scenario.bug_type}, variant: {scenario.bug_variant}')
print(f'Root cause: {scenario.root_cause}')

db.reset()
print('\nReward calculator tests passed!')
"
```

**Expected:** Inspect gets ~+0.9, Invalid SQL gets ~-1.1, Repeated gets ~-1.1.

#### Test 1d: Evaluation Oracle (scoring works)

```powershell
python -c "
from db_surgeon.server.broken_db_generator import BrokenDBGenerator
from db_surgeon.server.db_manager import DBManager
from db_surgeon.server.evaluation_oracle import EvaluationOracle

gen = BrokenDBGenerator(seed=7)
scenario = gen.generate()

# Score broken DB
db_broken = DBManager()
db_broken.create_database(scenario.schema_sql, scenario.seed_data_sql)
oracle = EvaluationOracle(scenario.eval_queries)
broken_score = oracle.score(db_broken)
print(f'Broken DB score:  {broken_score:.1%} (should be < 100%)')

# Score healthy DB
db_healthy = DBManager()
db_healthy.create_database(scenario.healthy_schema_sql, scenario.healthy_seed_data_sql)
healthy_score = oracle.score(db_healthy)
print(f'Healthy DB score: {healthy_score:.1%} (should be ~100%)')

# Detailed score
details = oracle.detailed_score(db_broken)
print(f'\nDetailed (broken):')
for d in details['details']:
    status = 'PASS' if d['passed'] else 'FAIL'
    print(f'  [{status}] {d[\"query\"]}')

db_broken.reset()
db_healthy.reset()
print('\nOracle tests passed!')
"
```

**Expected:** Broken DB < 100%, Healthy DB ~100%.

---

### Test 2: Full Environment (end-to-end episode)

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m db_surgeon.examples.example_episode
```

**Expected output:**
```
======================================================================
DB-SURGEON - Example Episode
======================================================================

STEP 0: Resetting environment...
  Bug type: fk_violation
  Root cause: [description of the bug]

STEP 1: Inspecting schema... [+0.9 reward]
STEP 2: Running failing query... [-1.1 reward]
STEP 3: Inspecting tables...
STEP 4: Applying fix... [+10.9 reward]
STEP 5: Verifying fix...
STEP 6: Submitting fix... [+4.9 reward]

EPISODE SUMMARY
  Total Reward: ~+16
  Steps Used: 7
  Fixed: True
```

---

### Test 3: Random Baseline (establishes performance floor)

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m db_surgeon.examples.baseline_random 20
```

**Expected output:**
```
RANDOM BASELINE RESULTS
  Episodes:      20
  Success Rate:  0/20 (0.0%)
  Avg Reward:    ~-1 to -3
  Avg Steps:     ~13
```

This proves that random behavior doesn't solve the task. A trained agent must do better.

---

### Test 4: TRL Tool Wrapper (training integration works)

```powershell
python -c "
from db_surgeon.training.tool_env import DBSurgeonToolEnv

env = DBSurgeonToolEnv()

# 1. Reset
obs = env.reset()
print(f'1. RESET: Got {len(obs)} char observation')
print(f'   Done={env.done}, Reward={env.reward}')

# 2. Inspect
result = env.inspect_schema()
print(f'2. INSPECT: Got {len(result)} char result')

# 3. Run query
result = env.run_query('SELECT 1')
print(f'3. QUERY: {result[:50]}')

# 4. Submit
result = env.submit()
print(f'4. SUBMIT: Done={env.done}, Reward={env.reward:.2f}')

# 5. Call after done (should raise ValueError)
try:
    env.inspect_schema()
    print('5. ERROR: Should have raised ValueError!')
except ValueError as e:
    print(f'5. VALUEERROR: Correctly raised')

print('\nAll TRL wrapper tests passed!')
"
```

**Expected:** All 5 tests pass, ValueError raised on step 5.

---

### Test 5: FastAPI Server (API works)

```powershell
# Terminal 1: Start the server
$env:PYTHONIOENCODING="utf-8"
python -m db_surgeon.server.app
```

In a separate terminal:

```powershell
# Terminal 2: Test the API
curl http://localhost:7860/health
# Expected: {"status":"ok","environment":"db-surgeon","version":"0.1.0"}

curl -X POST http://localhost:7860/reset
# Expected: JSON with observation data

curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d "{\"tool_name\":\"inspect_schema\",\"arguments\":{}}"
# Expected: JSON with observation, reward, done
```

Or test without curl:

```powershell
python -c "
from db_surgeon.client import DBSurgeonEnv
from db_surgeon.models import DBSurgeonAction

# This connects to the running server
client = DBSurgeonEnv('http://localhost:7860')
print('Health:', client.health())

result = client.reset()
print('Reset OK, observation length:', len(result.observation.schema_snapshot))

result = client.step(DBSurgeonAction('inspect_schema', {}))
print('Step OK, reward:', result.reward)

client.close()
"
```

---

### Test 6: Generate Demo Plots

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m db_surgeon.metrics.plot_rewards --demo
```

**Expected:** 3 PNG files created in `metrics/results/`:
- `demo_reward_curve.png` — Reward going UP over training
- `demo_success_rate.png` — Success rate increasing
- `demo_comparison.png` — Random baseline vs. trained agent histograms

---

### Test 7: Dataset Generation

```powershell
python -c "
from db_surgeon.training.dataset import create_training_dataset, SYSTEM_PROMPT

ds = create_training_dataset(10)
print(f'Dataset rows: {len(ds)}')
print(f'Columns: {ds.column_names}')
print(f'First prompt (first 100 chars):')
print(f'  {ds[0][\"prompt\"][0][\"content\"][:100]}...')
print(f'\nSystem prompt length: {len(SYSTEM_PROMPT)} chars')
"
```

**Expected:** Dataset with 10 rows, "prompt" column, each containing the system message.

---

## 7. HOW TO TRAIN THE AGENT

### Option A: Google Colab (Recommended — Free T4 GPU)

1. Upload the `db_surgeon/` folder to Google Drive or Colab

2. In a Colab notebook:

```python
# Cell 1: Install dependencies
!pip install unsloth trl datasets transformers accelerate peft

# Cell 2: Upload/clone your project
# (upload db_surgeon folder or git clone)

# Cell 3: Install the package
!pip install -e ./db_surgeon

# Cell 4: Run training
!python -m db_surgeon.training.train_unsloth
```

3. Training will:
   - Load Qwen3-0.6B in 4-bit QLoRA
   - Create 200 episodes
   - Run GRPO with 4 completions per prompt
   - Save LoRA adapter + merged model

4. Expected runtime: 2-4 hours on T4

### Option B: Local GPU (16GB+ VRAM)

```powershell
# Install training deps
pip install -e "./db_surgeon[training]"

# Optional: set custom config
$env:DB_SURGEON_MODEL="Qwen/Qwen3-0.6B"
$env:DB_SURGEON_EPISODES="200"
$env:DB_SURGEON_OUTPUT="./db_surgeon_output"

# Run standard GRPO training
python -m db_surgeon.training.train_grpo
```

### Option C: Quick Smoke Test (verify training pipeline connects)

```powershell
# Set to just 3 episodes to verify it doesn't crash
$env:DB_SURGEON_EPISODES="3"
python -m db_surgeon.training.train_grpo
```

### After Training: Generate Real Plots

```powershell
python -m db_surgeon.metrics.plot_rewards --results metrics/results/training_log.json
```

---

## 8. HOW TO DEPLOY

### Deploy to HuggingFace Spaces

```bash
cd db_surgeon

# Login to HuggingFace
huggingface-cli login

# Push as an OpenEnv Space
openenv push --repo-id yourusername/db-surgeon --private
```

### Deploy with Docker

```bash
cd db_surgeon
docker build -t db-surgeon -f server/Dockerfile .
docker run -p 7860:7860 db-surgeon
```

---

## 9. TROUBLESHOOTING COMMON ISSUES

### Issue: `UnicodeEncodeError` on Windows

```powershell
# Fix: Set UTF-8 encoding before running
$env:PYTHONIOENCODING="utf-8"
```

### Issue: `ModuleNotFoundError: No module named 'db_surgeon'`

```powershell
# Fix: Install in development mode
cd E:\Scaler
pip install -e ./db_surgeon
```

### Issue: `ImportError: cannot import name 'GRPOTrainer' from 'trl'`

```powershell
# Fix: Install training dependencies
pip install -e "./db_surgeon[training]"
# Or: pip install trl>=0.15.0
```

### Issue: OOM (Out of Memory) on Colab T4

```python
# Fix: Reduce generation count in train_unsloth.py
# Change: num_generations=4 → num_generations=2
# Change: max_completion_length=2048 → max_completion_length=1024
```

### Issue: Training reward stays flat/negative

Possible causes:
1. **Task too hard** → Simplify the bug type (start with just `rename_column` variant)
2. **Model too small** → Try `Qwen3-1.7B` if compute allows
3. **Learning rate too high** → Reduce from `5e-6` to `1e-6`
4. **Not enough episodes** → Increase from 200 to 500

### Issue: Server won't start (port 7860 in use)

```powershell
# Fix: Change port in app.py or kill existing process
netstat -ano | findstr "7860"
taskkill /PID <PID> /F
```

### Issue: Random baseline shows high success rate

This means the bugs are too easy. Fix by:
1. Ensuring the random agent doesn't submit early (already fixed)
2. Adding more complex bug variants
3. Making eval queries stricter

---

## QUICK COMMAND REFERENCE

```powershell
# Always set this on Windows first
$env:PYTHONIOENCODING="utf-8"

# Run example episode (shows environment works)
python -m db_surgeon.examples.example_episode

# Run random baseline (shows performance floor)
python -m db_surgeon.examples.baseline_random 20

# Start the server
python -m db_surgeon.server.app

# Generate demo plots
python -m db_surgeon.metrics.plot_rewards --demo

# Train with Unsloth on Colab
python -m db_surgeon.training.train_unsloth

# Train with standard GRPO
python -m db_surgeon.training.train_grpo
```
