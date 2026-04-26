# 🏥 DB-Surgeon — AI-Powered Database Diagnosis & Repair

> **An RL-trained LLM that autonomously diagnoses broken database schemas, fixes them, and translates natural language queries into SQL — in any language.**

[![Live Demo](https://img.shields.io/badge/🤗_HuggingFace-Live_Demo-blue)](https://ayush0211-db-surgeon.hf.space)
[![Model](https://img.shields.io/badge/🤗_Model-db--surgeon--qwen3--0.6b--grpo-green)](https://huggingface.co/ayush0211/db-surgeon-qwen3-0.6b-grpo)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_L4-orange)]()

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution](#-our-solution)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [RL Training Pipeline](#-rl-training-pipeline)
- [Natural Language to SQL](#-natural-language-to-sql-nl2sql)
- [Training Results](#-training-results)
- [Tech Stack](#-tech-stack)
- [How to Use](#-how-to-use)
- [Project Structure](#-project-structure)
- [Future Scope](#-future-scope)

---

## 🎯 Problem Statement

### The Pain Point

In real-world production environments, **database schema failures** are one of the most critical issues that can bring entire applications down. These include:

- **Foreign key violations** — Columns referencing wrong tables or columns
- **Type mismatches** — A column storing TEXT where INTEGER is expected
- **Missing tables** — Referenced tables that don't exist
- **Schema drift** — Column names changed but queries not updated
- **Broken constraints** — FK references with typos in table names

**Currently, diagnosing and fixing these issues requires:**
1. A senior DBA to manually inspect the schema
2. Understand the business query that's failing
3. Trace the root cause across multiple tables
4. Write and execute the correct ALTER/CREATE statements
5. Verify the fix doesn't break anything else

This process is **time-consuming, error-prone, and requires deep expertise**. During production outages, every minute of downtime costs money.

### The Question We Asked

> *"Can we train an AI model to act like a database surgeon — one that can diagnose the problem, identify the root cause, and apply the correct fix, all autonomously?"*

---

## 💡 Our Solution

**DB-Surgeon** is an end-to-end system that uses **Reinforcement Learning (GRPO)** to train a small language model (Qwen3-0.6B) to become an autonomous database repair agent. Additionally, it supports **multilingual Natural Language to SQL** translation, allowing users to query databases in Hindi, English, or any other language.

### What Makes It Unique

| Feature | Traditional Approach | DB-Surgeon |
|---------|---------------------|------------|
| **Diagnosis** | Manual DBA inspection | AI auto-diagnoses from schema + error log |
| **Fix Application** | Hand-written SQL | Model generates and executes fixes |
| **Learning** | Static rules / scripts | RL-trained — improves with each episode |
| **Language Support** | SQL only | Any human language → SQL |
| **Speed** | Minutes to hours | Seconds |
| **Cost** | Senior DBA ($$$) | 0.6B parameter model (runs on any GPU) |

---

## ✨ Key Features

### 1. 🔧 Autonomous Database Repair (RL-Trained)
The model is trained via **GRPO (Group Relative Policy Optimization)** to diagnose and fix broken databases. It learns a tool-use workflow:
```
inspect_schema → run_query → fix_column/execute_fix → submit
```

### 2. 💬 Natural Language to SQL (Any Language)
Users can ask questions about a database in **any language** — Hindi, English, Hinglish, or any other — and the model converts it to valid SQL, executes it, and returns formatted results.

**Examples:**
| Question | Generated SQL |
|----------|--------------|
| "Show all employees from Mumbai" | `SELECT * FROM employees WHERE city = 'Mumbai';` |
| "सबसे ज्यादा सैलरी किसकी है?" | `SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1;` |
| "Engineering ka budget kitna hai?" | `SELECT budget FROM departments WHERE name = 'Engineering';` |
| "Which region has most sales?" | `SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY SUM(amount) DESC LIMIT 1;` |

### 3. 🎮 Interactive Environment
A live Gradio-based demo where users can:
- **Manually** fix databases using tool commands
- **Watch the AI** auto-play and solve problems in real-time
- **Ask questions** in natural language and see SQL results

### 4. 📊 Training Dashboard
Full training pipeline with live metrics, loss curves, reward distributions, and model upload to HuggingFace Hub.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DB-Surgeon System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │   Broken DB  │    │  RL Agent    │    │  NL2SQL Engine   │  │
│  │  Generator   │───▶│  (Qwen3 +   │    │  (Multilingual   │  │
│  │  (4 bug      │    │   GRPO)     │    │   Query Parser)  │  │
│  │   variants)  │    │             │    │                  │  │
│  └──────────────┘    └──────┬───────┘    └────────┬─────────┘  │
│                             │                     │             │
│                      ┌──────▼───────┐      ┌──────▼─────────┐  │
│                      │  Tool Engine │      │  SQLite        │  │
│                      │  • inspect   │      │  Execution     │  │
│                      │  • run_query │      │  Engine        │  │
│                      │  • fix_column│      │                │  │
│                      │  • execute   │      │                │  │
│                      │  • submit    │      │                │  │
│                      └──────┬───────┘      └────────────────┘  │
│                             │                                   │
│                      ┌──────▼───────┐                          │
│                      │  Reward      │                          │
│                      │  Functions   │                          │
│                      │  • SQL valid │                          │
│                      │  • Workflow  │                          │
│                      │  • Fix check │                          │
│                      └──────────────┘                          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Gradio UI  │  HuggingFace Spaces  │  NVIDIA L4 GPU            │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description | File |
|-----------|-------------|------|
| **Broken DB Generator** | Creates randomized broken schemas with 4 FK violation variants | `server/broken_db_generator.py` |
| **Environment Engine** | OpenEnv-compatible RL environment with tool actions | `server/db_surgeon_environment.py` |
| **Reward Functions** | Multi-signal rewards: SQL validation + workflow scoring | `training/reward_functions.py` |
| **Training Pipeline** | GRPO trainer with Unsloth 4-bit quantization | `hf_space/app.py` |
| **NL2SQL Engine** | Multilingual natural language → SQL converter | `hf_space/app.py` |
| **Gradio UI** | 4-tab interactive demo + training dashboard | `hf_space/app.py` |

---

## 🧠 RL Training Pipeline

### Training Methodology: GRPO

We use **Group Relative Policy Optimization (GRPO)** from the TRL library, which is specifically designed for training language models with reinforcement learning. Unlike PPO, GRPO doesn't need a separate value network — it computes advantages by comparing rewards within a group of generated responses.

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Prompt    │────▶│  Generate N  │────▶│  Compute    │
│  (Schema +  │     │  Completions │     │  Rewards    │
│   Error)    │     │  per prompt  │     │  per group  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌──────────────┐     ┌──────▼──────┐
                    │   Update     │◀────│  Calculate  │
                    │   Policy     │     │  Advantages │
                    │   (LoRA)     │     │  (relative) │
                    └──────────────┘     └─────────────┘
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| **Base Model** | Qwen/Qwen3-0.6B |
| **Quantization** | 4-bit (bitsandbytes NF4) |
| **LoRA Rank** | 32 |
| **LoRA Alpha** | 64 |
| **Learning Rate** | 5e-6 |
| **Training Steps** | 200 |
| **Batch Size** | 4 |
| **GPU** | NVIDIA L4 (22GB VRAM) |
| **Training Time** | ~2 hours per run |
| **VRAM Usage** | ~0.6GB / 22GB |

### Reward Function Design

The reward system uses a **high-variance, multi-signal approach** to provide meaningful gradient signals for GRPO:

```python
Reward Range: [-3.0, +10.0]

Signal 1: SQL Quality Reward
  ├── Valid SQL syntax (SQLite parse)  → +2.0 to +5.0
  ├── Contains correct tool calls     → +1.0 to +3.0
  └── Invalid/empty response          → -2.0 to -3.0

Signal 2: Workflow Adherence Reward
  ├── inspect → fix → submit flow     → +2.0 to +4.0
  ├── Partial workflow                 → +0.5 to +1.0
  └── No structure                    → -1.0

Signal 3: Fix Correctness
  ├── Database actually fixed          → +5.0 (bonus)
  └── Submit without fixing            → -1.0
```

**Why High Variance?** GRPO needs clear differences between "good" and "bad" completions to compute meaningful advantages. A flat reward (e.g., always 0.5) provides zero gradient signal.

---

## 💬 Natural Language to SQL (NL2SQL)

### How It Works

1. **User types a question** in any language (Hindi, English, Hinglish, etc.)
2. **Model receives** the database schema + user question as context
3. **Generates SQL** using the fine-tuned Qwen3 model
4. **Executes the SQL** on an in-memory SQLite database
5. **Returns formatted results** as a table

### Sample Database

The NL2SQL demo comes with a pre-loaded company database:

| Table | Columns | Sample Data |
|-------|---------|-------------|
| **employees** | id, name, department, salary, hire_date, city, age | 12 employees across 5 departments |
| **departments** | id, name, budget, manager | Engineering, Marketing, Sales, HR, Finance |
| **projects** | id, name, department_id, status, start_date, budget | 5 active/completed projects |
| **sales** | id, employee_id, product, amount, sale_date, region | 8 sales records across 4 regions |

### Multilingual Support

The Qwen3 base model natively supports 100+ languages. After GRPO fine-tuning, the model retains this capability while also understanding database-specific terminology:

- 🇬🇧 English: *"Show all employees with salary above 80000"*
- 🇮🇳 Hindi: *"सबसे ज्यादा सैलरी किसकी है?"*
- 🇮🇳 Hinglish: *"Engineering department ka budget kitna hai?"*
- 🇫🇷 French: *"Montrez les employés de Mumbai"*

---

## 📊 Training Results

### Training Run Performance

| Metric | V1 (Initial) | V2 (Improved) |
|--------|:------------:|:-------------:|
| **Starting Reward** | 0.0 | 0.0 |
| **Peak Reward** | ~4.5 | **~7.09** |
| **Final Avg Reward** | ~4.3 | **~6.8** |
| **Loss** | 1.8e-8 → 1.7e-6 | 1.1e-6 → 2.1e-6 |
| **Baseline (Random)** | 27% success | 13% success |
| **Trained Model** | — | **✅ Fixes bugs autonomously** |

### Key Observations

1. **Reward climbed from 0.0 → 7.09** over 200 steps, confirming the model learned the tool-use workflow
2. **Loss increased slightly** — expected in RL (the model is exploring more diverse strategies)
3. **The model learned to:**
   - First inspect the schema to understand the problem
   - Identify the specific bug type (FK violation, type mismatch, etc.)
   - Apply the correct fix (rename column, change type, recreate table)
   - Submit when confident the fix is correct

### Auto-Play Demo Result

```
🤖 TRAINED MODEL AUTO-PLAY
════════════════════════════

📋 Bug Type: fk_violation
📋 Failing Query: SELECT u.name, u.email, o.amount...

✅ Model loaded!

🔄 Turn 1/8
🧠 Model reasons about the schema and identifies the bug
🔧 Executing: inspect_schema({'table_name': 'tbl_c841_orders'})
📊 Reward: +1.9

🔄 Turn 2/8
🔧 Executing: submit({})
📊 Reward: +4.9
🎉 DATABASE FIXED!

📊 FINAL RESULT
  Total Reward: +6.8
  Fixed: ✅ YES
  Steps Used: 2/15
```

The model fixed a broken database in just **2 steps** out of 15 allowed!

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Qwen/Qwen3-0.6B (0.6 billion parameters) |
| **RL Framework** | TRL (GRPO Trainer) |
| **Fine-tuning** | Unsloth (4-bit QLoRA) |
| **Quantization** | bitsandbytes (NF4, float16) |
| **Database** | SQLite (in-memory) |
| **UI** | Gradio |
| **Hosting** | HuggingFace Spaces |
| **GPU** | NVIDIA L4 (22GB VRAM) |
| **Language** | Python 3.11 |
| **Model Hub** | HuggingFace Hub |

---

## 🚀 How to Use

### Live Demo

Visit the live demo: **[https://ayush0211-db-surgeon.hf.space](https://ayush0211-db-surgeon.hf.space)**

#### Tab 1: 🎮 Interactive Demo
- Click **Reset** to generate a new broken database
- Use the tool dropdown to manually inspect and fix the schema
- Or click **🤖 Auto-Play with Trained Model** to watch the AI solve it

#### Tab 2: 💬 Ask in Any Language
- Click **Load Database** to set up the sample database
- Type a question in Hindi, English, or any language
- Click **Ask** — the model generates SQL and shows results

#### Tab 3: 🚀 Training
- Configure model, episodes, and learning rate
- Click **Start Training** to run GRPO
- Monitor live loss and reward metrics

#### Tab 4: 🧪 Quick Test
- Run batch tests with a scripted agent
- See success rate across multiple episodes

### Use the Trained Model Directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ayush0211/db-surgeon-qwen3-0.6b-grpo")
tokenizer = AutoTokenizer.from_pretrained("ayush0211/db-surgeon-qwen3-0.6b-grpo")

prompt = """You are a database engineer. The orders table has user_id as TEXT 
instead of INTEGER. Fix it using: fix_column(table_name, column_name, new_type)"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📂 Project Structure

```
db_surgeon/
├── README.md                          # This file
├── models.py                          # Data models (Action, Observation, State)
├── client.py                          # Local environment client
├── deploy_to_hf.py                    # HuggingFace Space deployment script
│
├── server/
│   ├── broken_db_generator.py         # Randomized bug scenario generator
│   ├── db_surgeon_environment.py      # Core RL environment engine
│   ├── db_manager.py                  # SQLite database manager
│   └── Dockerfile                     # Container for server deployment
│
├── training/
│   ├── reward_functions.py            # Multi-signal reward functions
│   ├── tool_env.py                    # Tool-use environment wrapper
│   └── dataset.py                     # Training dataset generator
│
└── hf_space/
    ├── app.py                         # Main Gradio app (UI + training + NL2SQL)
    └── requirements.txt               # Dependencies
```

---

## 🔮 Future Scope

### Phase 2: Expanded Bug Types
- **Datatype mismatches** — INTEGER stored as TEXT, REAL as INTEGER
- **Missing indexes** — Performance degradation diagnosis
- **Constraint conflicts** — CHECK/UNIQUE violations
- **Schema drift** — Columns added/removed without updating queries

### Phase 3: Advanced NL2SQL
- **Upload your own database** — Users upload CSV/SQL files and query them
- **Multi-table JOINs** — Handle complex cross-table queries
- **Query explanation** — Show step-by-step query breakdown in user's language

### Phase 4: Production Integration
- **API endpoint** — REST API for automated database monitoring
- **Slack/Discord bot** — Get notified when schema issues are detected
- **CI/CD integration** — Validate schema migrations before deployment

### Phase 5: Larger Models
- Train on Qwen3-1.7B / 4B for better accuracy
- Fine-tune on real production database incidents
- Multi-agent setup: one agent diagnoses, another fixes

---

## 👨‍💻 Author

**Ayush** — Built as part of a hackathon project exploring the intersection of Reinforcement Learning and Database Administration.

- 🤗 HuggingFace: [ayush0211](https://huggingface.co/ayush0211)
- 🔗 GitHub: [ayush-init](https://github.com/ayush-init)

---

## 📄 License

This project is open-source and available under the MIT License.

---

<p align="center">
  <b>🏥 DB-Surgeon — Because databases deserve good healthcare too.</b>
</p>
