# DB-Surgeon: Database Surgery RL Environment 🏥

> **An OpenEnv-compatible reinforcement learning environment where an LLM agent diagnoses and fixes broken database schemas.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![TRL](https://img.shields.io/badge/TRL-GRPOTrainer-green)](https://huggingface.co/docs/trl)
[![Unsloth](https://img.shields.io/badge/Unsloth-QLoRA-orange)](https://github.com/unslothai/unsloth)

---

## 🧠 What is DB-Surgeon?

DB-Surgeon is a **real-world professional task environment** for training LLM agents to debug database schema issues. The environment simulates:

- 🔴 **Broken schemas** — FK violations, type mismatches, missing tables
- 📝 **Error logs** — Realistic SQLite error messages  
- ❌ **Failing business queries** — SQL that breaks due to schema bugs

The agent must:
1. **Diagnose** — Inspect the schema and understand what's broken
2. **Fix** — Apply targeted DDL changes to repair the database
3. **Verify** — Confirm the business query now works
4. **Submit** — Get scored on hidden evaluation queries

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│           Training Pipeline             │
│  Unsloth (QLoRA 4-bit) + GRPOTrainer   │
│              ↕                          │
│    DBSurgeonToolEnv (TRL Wrapper)       │
│    - inspect_schema()                   │
│    - run_query()                        │
│    - fix_column()                       │
│    - execute_fix()                      │
│    - submit()                           │
└──────────────┬──────────────────────────┘
               ↕
┌──────────────┴──────────────────────────┐
│       OpenEnv Server (FastAPI)          │
│  DBSurgeonEnvironment                   │
│    reset() → Broken DB scenario         │
│    step()  → Execute + reward           │
│    state() → Episode metadata           │
│                                         │
│  Components:                            │
│    SQLite :memory: │ BrokenDBGenerator  │
│    RewardCalculator │ EvalOracle        │
└─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Core environment only
pip install -e .

# With training dependencies
pip install -e ".[training]"

# With Unsloth (for Colab T4)
pip install -e ".[training,unsloth]"
```

### Run an Example Episode

```bash
python -m db_surgeon.examples.example_episode
```

### Run Random Baseline

```bash
python -m db_surgeon.examples.baseline_random
```

### Start the Server

```bash
python -m db_surgeon.server.app
```

### Train with GRPO (Standard GPU)

```bash
python -m db_surgeon.training.train_grpo
```

### Train with Unsloth (Colab T4)

```bash
python -m db_surgeon.training.train_unsloth
```

## 🎯 Reward System

Multi-component reward designed to prevent reward hacking:

| Component | Reward | Description |
|-----------|--------|-------------|
| Business query passes | +5.0 | The failing query now executes |
| Eval queries pass (submit) | +5.0 | Hidden test queries score |
| Causal fix | +3.0 | Fix addresses root cause |
| Partial improvement | +2.0 | Error count decreased |
| Good diagnostic | +1.0 | Inspected relevant table |
| Efficiency bonus | +1.0 | Solved in < half the steps |
| Invalid SQL | -1.0 | Syntax error or bad command |
| Regression | -3.0 | Broke something that worked |
| Repeated action | -1.0 | Same action repeated |
| Step tax | -0.1 | Small per-step penalty |

### Anti-Reward-Hacking Measures

- ✅ Randomized table/column names per episode
- ✅ Hidden evaluation queries (agent doesn't see all tests)
- ✅ Regression detection and heavy penalty
- ✅ DDL whitelist (destructive operations blocked)
- ✅ Step limit (15 steps max)
- ✅ Duplicate action detection

## 📊 Results

### Random Baseline vs. Trained Agent

| Metric | Random Baseline | Trained Agent |
|--------|:--------------:|:------------:|
| Success Rate | ~0% | ~45% |
| Avg Reward | -2.5 | +6.0 |
| Avg Steps | 15 (max) | 5 |

## 🗂️ Project Structure

```
db_surgeon/
├── models.py                      # Action, Observation, State
├── client.py                      # HTTP + Local clients
├── server/
│   ├── app.py                     # FastAPI server
│   ├── db_surgeon_environment.py  # Core environment
│   ├── db_manager.py              # SQLite lifecycle
│   ├── broken_db_generator.py     # Bug injection
│   ├── reward.py                  # Reward calculator
│   └── evaluation_oracle.py       # Hidden eval queries
├── training/
│   ├── tool_env.py                # TRL wrapper
│   ├── train_grpo.py              # GRPO training
│   ├── train_unsloth.py           # Unsloth variant
│   ├── dataset.py                 # Prompt dataset
│   └── reward_functions.py        # TRL reward funcs
├── examples/
│   ├── example_episode.py         # Scripted demo
│   └── baseline_random.py         # Random baseline
└── metrics/
    └── plot_rewards.py            # Visualization
```

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_SURGEON_MODEL` | `Qwen/Qwen3-0.6B` | Model to train |
| `DB_SURGEON_EPISODES` | `200` | Number of training episodes |
| `DB_SURGEON_OUTPUT` | `./db_surgeon_output` | Output directory |

## 📝 License

BSD-3-Clause
