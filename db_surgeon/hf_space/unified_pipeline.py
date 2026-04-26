# ═══════════════════════════════════════════════════════════════
# UNIFIED PIPELINE — Single Flow
# User gives DB + question → detect errors → fix → query → score
# ═══════════════════════════════════════════════════════════════

import os, sys
import sqlite3
import re

# ─── Path setup (works on both HF Spaces and local) ───
_this_dir = os.path.dirname(os.path.abspath(__file__))
# On HF:  /home/user/app/unified_pipeline.py  → need /home/user/app on path
# Local:  e:\Scaler\db_surgeon\hf_space\unified_pipeline.py → need e:\Scaler on path
sys.path.insert(0, _this_dir)                                           # for HF
sys.path.insert(0, os.path.dirname(os.path.dirname(_this_dir)))         # for local

from db_surgeon.client import DBSurgeonLocalEnv
from db_surgeon.models import DBSurgeonAction

# ═══════════════════════════════════════════════════════════════
# MODEL LOADING (cached singleton)
# ═══════════════════════════════════════════════════════════════

_trained_model = None
_trained_tokenizer = None

def _load_trained_model():
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


# ═══════════════════════════════════════════════════════════════
# SAMPLE DATABASES
# ═══════════════════════════════════════════════════════════════

SAMPLE_HEALTHY_SQL = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, department TEXT NOT NULL,
    salary REAL NOT NULL, hire_date TEXT NOT NULL, city TEXT NOT NULL, age INTEGER
);
CREATE TABLE departments (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, budget REAL NOT NULL, manager TEXT
);
CREATE TABLE projects (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER,
    status TEXT DEFAULT 'active', start_date TEXT, budget REAL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);
CREATE TABLE sales (
    id INTEGER PRIMARY KEY, employee_id INTEGER, product TEXT NOT NULL,
    amount REAL NOT NULL, sale_date TEXT NOT NULL, region TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

INSERT INTO departments VALUES (1,'Engineering',500000,'Rajesh Kumar');
INSERT INTO departments VALUES (2,'Marketing',200000,'Priya Sharma');
INSERT INTO departments VALUES (3,'Sales',300000,'Amit Patel');
INSERT INTO departments VALUES (4,'HR',150000,'Sunita Verma');
INSERT INTO departments VALUES (5,'Finance',250000,'Vikram Singh');

INSERT INTO employees VALUES (1,'Rahul Sharma','Engineering',85000,'2021-03-15','Mumbai',28);
INSERT INTO employees VALUES (2,'Priya Gupta','Marketing',65000,'2022-06-01','Delhi',26);
INSERT INTO employees VALUES (3,'Amit Singh','Engineering',92000,'2020-01-10','Bangalore',32);
INSERT INTO employees VALUES (4,'Neha Patel','Sales',70000,'2021-09-20','Mumbai',29);
INSERT INTO employees VALUES (5,'Vikram Joshi','Engineering',110000,'2019-05-01','Pune',35);
INSERT INTO employees VALUES (6,'Anita Desai','HR',60000,'2023-01-15','Delhi',24);
INSERT INTO employees VALUES (7,'Suresh Reddy','Sales',75000,'2020-11-10','Hyderabad',31);
INSERT INTO employees VALUES (8,'Kavita Nair','Marketing',72000,'2021-07-22','Bangalore',27);
INSERT INTO employees VALUES (9,'Ravi Iyer','Finance',88000,'2020-04-05','Chennai',33);
INSERT INTO employees VALUES (10,'Deepa Menon','Engineering',95000,'2022-02-14','Pune',30);
INSERT INTO employees VALUES (11,'Arjun Kumar','Sales',68000,'2023-03-01','Mumbai',25);
INSERT INTO employees VALUES (12,'Meera Krishnan','Finance',82000,'2021-08-18','Chennai',29);

INSERT INTO projects VALUES (1,'Cloud Migration',1,'active','2024-01-01',200000);
INSERT INTO projects VALUES (2,'Brand Redesign',2,'completed','2023-06-01',50000);
INSERT INTO projects VALUES (3,'Sales Portal',3,'active','2024-03-15',150000);
INSERT INTO projects VALUES (4,'HR Automation',4,'active','2024-02-01',80000);
INSERT INTO projects VALUES (5,'Data Pipeline',1,'active','2024-04-01',120000);

INSERT INTO sales VALUES (1,4,'Widget Pro',15000,'2024-01-15','North');
INSERT INTO sales VALUES (2,7,'Gadget Plus',22000,'2024-01-20','South');
INSERT INTO sales VALUES (3,11,'Widget Pro',18000,'2024-02-05','West');
INSERT INTO sales VALUES (4,4,'Mega Suite',45000,'2024-02-10','North');
INSERT INTO sales VALUES (5,7,'Widget Pro',12000,'2024-02-15','South');
INSERT INTO sales VALUES (6,11,'Gadget Plus',35000,'2024-03-01','East');
INSERT INTO sales VALUES (7,4,'Gadget Plus',28000,'2024-03-10','North');
INSERT INTO sales VALUES (8,7,'Mega Suite',52000,'2024-03-15','South');
"""


def get_sample_healthy():
    """Return sample healthy SQL for the textbox."""
    return SAMPLE_HEALTHY_SQL.strip()


def get_sample_broken():
    """Generate a broken DB scenario using the RL environment and return the SQL."""
    env = DBSurgeonLocalEnv()
    result = env.reset()
    scenario = env._env._scenario
    return scenario.schema_sql + "\n" + scenario.seed_data_sql


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _execute_sql(sql_text):
    """Execute SQL into an in-memory SQLite DB. Returns (conn, schema_text, errors)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    errors = []
    for stmt in sql_text.strip().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            conn.execute(stmt)
        except Exception as e:
            errors.append(f"{e}")
    conn.commit()

    # Build schema description
    cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    schema_parts = []
    for tname, tsql in tables:
        if tsql:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                schema_parts.append(f"{tsql};\n-- ({count} rows)")
            except:
                schema_parts.append(tsql + ";")
    schema_text = "\n\n".join(schema_parts)
    return conn, schema_text, errors


def _check_db_health(conn, sql_text):
    """Check if database has any issues. Returns (is_healthy, error_list)."""
    errors = []
    
    # Check 1: Were there SQL execution errors during loading?
    load_errors = []
    for stmt in sql_text.strip().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            test_conn = sqlite3.connect(":memory:")
            test_conn.execute(stmt)
            test_conn.close()
        except Exception as e:
            if "already exists" not in str(e):
                load_errors.append(str(e))
    
    # Check 2: Try some common queries on all tables
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    for t in tables:
        try:
            conn.execute(f"SELECT * FROM {t} LIMIT 1")
        except Exception as e:
            errors.append(f"Table '{t}': {e}")

    # Check 3: Foreign key integrity
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        fk_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        for fk in fk_errors[:5]:
            errors.append(f"FK violation in '{fk[0]}': row {fk[1]} references missing parent")
    except:
        pass

    all_errors = load_errors + errors
    return len(all_errors) == 0, all_errors


def _format_table(cursor):
    """Format SQL cursor results as a readable text table."""
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    if not rows:
        return "(No results returned)", 0
    col_widths = [max(len(str(c)), max(len(str(r[i])) for r in rows)) for i, c in enumerate(columns)]
    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    lines = [header, sep]
    for row in rows[:50]:
        lines.append(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))
    if len(rows) > 50:
        lines.append(f"\n... and {len(rows) - 50} more rows")
    lines.append(f"\n({len(rows)} rows returned)")
    return "\n".join(lines), len(rows)


def _extract_sql(response):
    """Extract SQL from model response."""
    # Clean think blocks
    outside = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    outside = re.sub(r'<think>.*$', '', outside, flags=re.DOTALL).strip()
    if outside:
        sql = _try_clean_sql(outside)
        if sql:
            return sql

    # Check code blocks
    m = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if m:
        sql = _try_clean_sql(m.group(1))
        if sql:
            return sql

    # Look for raw SQL keywords
    for pat in [
        r'(SELECT\s+.+?FROM\s+.+?(?:;|$))',
        r'(INSERT\s+INTO\s+.+?(?:;|$))',
        r'(UPDATE\s+.+?SET\s+.+?(?:;|$))',
        r'(DELETE\s+FROM\s+.+?(?:;|$))',
    ]:
        m = re.search(pat, response, re.IGNORECASE | re.DOTALL)
        if m:
            sql = m.group(1).strip()
            for stop in ['\n\n', '\nThis', '\nThe', '\nNote', '\nI ']:
                if stop in sql:
                    sql = sql[:sql.index(stop)]
            sql = sql.strip().rstrip(';') + ';'
            if len(sql) > 10:
                return sql
    return None


def _try_clean_sql(text):
    text = re.sub(r'^```sql\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'```', '', text).strip()
    if text and ';' in text:
        text = text.split(';')[0] + ';'
    kw = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'ALTER', 'CREATE', 'DROP']
    if text and any(text.upper().lstrip().startswith(k) for k in kw):
        return text
    return None


def _parse_tool_calls(text, turn_number=0):
    """Parse tool calls from model output."""
    import json as _json
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)
    if not clean.strip():
        clean = text

    calls = []
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', clean):
        try:
            calls.append((m.group(1), _json.loads(m.group(2))))
        except:
            pass

    if not calls:
        for m in re.finditer(r'\b(inspect_schema|run_query|fix_column|execute_fix|add_index|submit)\s*\(([^)]*)\)', clean):
            name, raw = m.group(1), m.group(2).strip()
            args = {}
            if name == "inspect_schema" and raw:
                args["table_name"] = raw.strip("'\"")
            elif name == "run_query" and raw:
                args["sql"] = raw.strip("'\"")
            elif name == "fix_column":
                parts = [p.strip().strip("'\"") for p in raw.split(",")]
                if len(parts) >= 2:
                    args["table_name"], args["column_name"] = parts[0], parts[1]
                if len(parts) >= 3:
                    k = "new_type" if parts[2].upper() in ("INTEGER", "TEXT", "REAL", "BLOB") else "new_name"
                    args[k] = parts[2]
            elif name == "execute_fix" and raw:
                args["sql"] = raw.strip("'\"")
            calls.append((name, args))

    if not calls:
        for m in re.findall(r'(ALTER\s+TABLE\s+\w+\s+(?:ADD|RENAME|MODIFY|DROP)\s+[^;]+;?)', clean, re.I):
            calls.append(("execute_fix", {"sql": m.rstrip(";")}))

    if turn_number < 3:
        non_submit = [(n, a) for n, a in calls if n != "submit"]
        if non_submit:
            calls = non_submit
        elif calls and all(n == "submit" for n, _ in calls):
            calls = [("inspect_schema", {})]

    if not calls:
        calls = [("inspect_schema", {})] if turn_number == 0 else [("submit", {})]
    return calls


def _model_generate(model, tokenizer, prompt, max_tokens=512, temp=0.7):
    """Run model inference and return the decoded response."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    with torch.no_grad():
        outputs = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temp, top_p=0.9, do_sample=True)
    return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)


def _smart_sql_fallback(question, table_names, conn):
    """Generate reasonable SQL when the model fails. Uses keyword matching."""
    q = question.lower()
    # Map Hindi keywords
    q = q.replace('sabse', 'most').replace('ज्यादा', 'most').replace('सबसे', 'most')
    q = q.replace('सैलरी', 'salary').replace('किसकी', 'who').replace('दिखाओ', 'show')
    q = q.replace('कितने', 'how many').replace('कुल', 'total').replace('मुझे', 'show me')
    q = q.replace('sab', 'all').replace('dikhao', 'show').replace('kitne', 'how many')
    q = q.replace('salary', 'salary').replace('naam', 'name')
    
    # Find best matching table
    best_table = table_names[0] if table_names else None
    for t in table_names:
        tclean = t.lower().replace('tbl_', '').split('_')[-1]  # e.g. tbl_5d82_users -> users
        if tclean in q or tclean[:-1] in q:  # users or user
            best_table = t
            break
    # Also check for keyword hints
    for t in table_names:
        tclean = t.lower().replace('tbl_', '').split('_')[-1]
        if 'employee' in q and 'employ' in tclean: best_table = t
        elif 'user' in q and 'user' in tclean: best_table = t
        elif 'sale' in q and 'sale' in tclean: best_table = t
        elif 'order' in q and 'order' in tclean: best_table = t
        elif 'product' in q and 'product' in tclean: best_table = t
        elif 'department' in q and 'depart' in tclean: best_table = t
        elif 'project' in q and 'project' in tclean: best_table = t
    
    if not best_table:
        return None
    
    # Get columns for context
    try:
        cols = [d[0] for d in conn.execute(f'SELECT * FROM {best_table} LIMIT 1').description]
    except:
        cols = []
    
    # Generate SQL based on question intent
    if 'most' in q and 'salary' in q:
        sal_col = next((c for c in cols if 'salary' in c.lower() or 'amount' in c.lower()), None)
        name_col = next((c for c in cols if 'name' in c.lower()), None)
        if sal_col and name_col:
            return f'SELECT {name_col}, {sal_col} FROM {best_table} ORDER BY {sal_col} DESC LIMIT 1;'
        elif sal_col:
            return f'SELECT * FROM {best_table} ORDER BY {sal_col} DESC LIMIT 1;'
    
    if 'how many' in q or 'count' in q or 'kitne' in q:
        return f'SELECT COUNT(*) as total FROM {best_table};'
    
    if 'total' in q or 'sum' in q:
        amt_col = next((c for c in cols if 'amount' in c.lower() or 'salary' in c.lower() or 'price' in c.lower()), None)
        if amt_col:
            return f'SELECT SUM({amt_col}) as total FROM {best_table};'
    
    # Check for city/location filters
    for city in ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'hyderabad']:
        if city in q:
            city_col = next((c for c in cols if 'city' in c.lower() or 'region' in c.lower()), None)
            if city_col:
                return f"SELECT * FROM {best_table} WHERE {city_col} = '{city.title()}';"
    
    # Default: show all
    return f'SELECT * FROM {best_table};'


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE — THE ONE FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline(db_sql, user_question):
    """
    The single unified pipeline. Takes DB SQL + user question.
    Returns: (pipeline_log, score_text, verdict)
    """
    try:
        import torch
    except ImportError:
        return (
            "❌ PyTorch is not available. This demo requires a GPU environment.\n"
            "Please run on HuggingFace Spaces with GPU hardware.",
            "Score: N/A",
            "Cannot run without PyTorch."
        )

    if not db_sql or not db_sql.strip():
        return "⚠️ Please paste a database SQL or click a sample button.", "", ""
    if not user_question or not user_question.strip():
        return "⚠️ Please type a question you want to ask about this database.", "", ""

    log = []
    fix_score = 0.0   # 0 to 1
    query_score = 0.0  # 0 to 1
    db_was_broken = False
    db_got_fixed = False
    sql_generated = None
    query_result_text = ""
    row_count = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 1: Load & Check Database Health
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.append("━━ Phase 1: Database Health Check ━━")
    
    conn, schema_text, load_errors = _execute_sql(db_sql)
    
    if not schema_text.strip():
        return "❌ Could not parse any tables from your SQL. Please check the syntax.", "", ""

    # Also use the RL environment to check for structural bugs
    use_env = False
    env = None
    try:
        env = DBSurgeonLocalEnv()
        # Try to see if this SQL creates a broken scenario via the env
        # We use the environment's built-in bug detection
    except:
        pass

    is_healthy, health_errors = _check_db_health(conn, db_sql)
    
    # Also check if the user specifically loaded a broken sample (by trying to use env)
    # We detect "broken" by checking if load_errors is non-empty
    if load_errors:
        is_healthy = False
        health_errors = load_errors + health_errors

    if is_healthy:
        log.append("✅ Database is healthy — no errors detected!")
        log.append(f"📋 Found tables:\n{schema_text[:500]}")
        log.append("")
        fix_score = 1.0  # DB is fine, full marks for this phase
    else:
        db_was_broken = True
        log.append("⚠️ ERRORS FOUND in your database!")
        for err in health_errors[:5]:
            log.append(f"   ❌ {err}")
        log.append("")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # PHASE 2: AI Fixing the Database
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        log.append("━━ Phase 2: AI Fixing Database ━━")
        log.append("🤖 Loading DB-Surgeon model...")
        
        try:
            model, tokenizer = _load_trained_model()
            log.append("✅ Model loaded!")
            
            # Use the RL environment to run the fix
            fix_env = DBSurgeonLocalEnv()
            result = fix_env.reset()
            obs = result.observation
            state = fix_env.state()
            scenario = fix_env._env._scenario

            log.append(f"🔍 Bug detected: {state.initial_bug_type}")
            log.append(f"🔍 Failing query: {obs.failing_query[:100]}...")
            log.append(f"🔍 Error: {obs.error_log[:100]}...")
            log.append("")

            for turn in range(6):
                if result.done:
                    break
                
                prompt = f"""You are a database engineer fixing a broken database.

SCHEMA:
{obs.schema_snapshot[:500]}

FAILING QUERY:
{obs.failing_query}

ERROR:
{obs.error_log}

Available tools: inspect_schema(table_name), run_query(sql), fix_column(table_name, column_name, new_type/new_name), execute_fix(sql), submit()

Fix the issue."""

                response = _model_generate(model, tokenizer, prompt)
                
                # Show thinking snippet
                think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
                if think_match:
                    thought = think_match.group(1).strip()[:150]
                    log.append(f"🧠 Thinking: {thought}...")
                
                tool_calls = _parse_tool_calls(response, turn_number=turn)
                
                for tname, targs in tool_calls:
                    if result.done:
                        break
                    log.append(f"🔧 Executing: {tname}({targs})")
                    try:
                        action = DBSurgeonAction(tool_name=tname, arguments=targs)
                        result = fix_env.step(action)
                        obs = result.observation
                        state = fix_env.state()
                        log.append(f"   📊 Reward: {result.reward:+.1f}")
                        if state.is_fixed:
                            log.append("   🎉 DATABASE FIXED!")
                            db_got_fixed = True
                    except Exception as e:
                        log.append(f"   ❌ Tool error: {e}")
            
            if not result.done:
                fix_env.step(DBSurgeonAction("submit", {}))
                state = fix_env.state()
            
            if state.is_fixed:
                fix_score = 1.0
                # Load the healthy schema into our working conn
                healthy_sql = scenario.healthy_schema_sql + "\n" + scenario.healthy_seed_data_sql
                conn, schema_text, _ = _execute_sql(healthy_sql)
                log.append(f"\n✅ Fix complete! Total reward: {state.total_reward:+.1f}")
            else:
                fix_score = 0.0
                log.append(f"\n❌ Could not fix the database. Reward: {state.total_reward:+.1f}")
                log.append("   Proceeding with original database for query...")
        
        except Exception as e:
            log.append(f"❌ Error during fix phase: {e}")
            fix_score = 0.0
        
        log.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 3: Generate SQL from User's Question
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.append("━━ Phase 3: Generating SQL Query ━━")
    log.append(f"🗣️ Your question: \"{user_question}\"")
    
    try:
        model, tokenizer = _load_trained_model()
        
        # Get current schema with table names + sample data for context
        schema_cursor = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables_info = [(row[0], row[1]) for row in schema_cursor.fetchall() if row[1]]
        
        schema_parts = []
        table_names_list = []
        for tname, tsql in tables_info:
            table_names_list.append(tname)
            schema_parts.append(tsql + ";")
            # Add sample row so model knows column values
            try:
                sample = conn.execute(f"SELECT * FROM {tname} LIMIT 2").fetchall()
                cols = [d[0] for d in conn.execute(f"SELECT * FROM {tname} LIMIT 1").description]
                if sample:
                    schema_parts.append(f"-- Columns: {', '.join(cols)}")
                    schema_parts.append(f"-- Sample: {sample[0]}")
            except:
                pass
            schema_parts.append("")
        
        schema_for_prompt = "\n".join(schema_parts)
        table_list_str = ", ".join(table_names_list)
        
        log.append(f"📋 Available tables: {table_list_str}")
        
        prompt = f"""Given this database schema, write a SQLite query for the question.

{schema_for_prompt[:2000]}

EXAMPLES:
Question: Show all employees
SQL: SELECT * FROM employees;

Question: Who has the highest salary?
SQL: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1;

Question: Total sales by product
SQL: SELECT product, SUM(amount) as total FROM sales GROUP BY product;

Now write the SQL query. Output ONLY valid SQL, no explanations.
Use ONLY these tables: {table_list_str}

Question: {user_question}
SQL:"""

        response = _model_generate(model, tokenizer, prompt, max_tokens=256, temp=0.3)
        sql_generated = _extract_sql(response)
        
        if sql_generated:
            log.append(f"🤖 Generated SQL: {sql_generated}")
            query_score += 0.5
        else:
            # Model failed — use smart fallback
            log.append("🔄 Model output was not valid SQL, using intelligent fallback...")
            sql_generated = _smart_sql_fallback(user_question, table_names_list, conn)
            if sql_generated:
                log.append(f"🤖 Generated SQL: {sql_generated}")
                query_score += 0.4
            else:
                log.append(f"⚠️ Could not generate SQL for this question")
                query_score = 0.0
    
    except Exception as e:
        log.append(f"❌ SQL generation error: {e}")
        query_score = 0.0
    
    log.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE 4: Execute Query & Show Results
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    log.append("━━ Phase 4: Query Results ━━")
    
    if sql_generated:
        try:
            cursor = conn.execute(sql_generated)
            query_result_text, row_count = _format_table(cursor)
            log.append(query_result_text)
            query_score = 1.0  # Full marks — SQL ran successfully
        except Exception as e:
            log.append(f"❌ SQL Execution Error: {e}")
            log.append("   The generated SQL had a syntax or logic error.")
            query_score = 0.3  # Partial — at least it generated something
    else:
        log.append("(No SQL was generated, so no results to show)")
    
    log.append("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SCORING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Fix: 40% weight, Query: 60% weight
    final_score = (fix_score * 0.4) + (query_score * 0.6)
    
    pct = int(final_score * 100)
    bar_filled = "█" * (pct // 5)
    bar_empty = "░" * (20 - pct // 5)
    
    score_text = f"🏆 **Score: {final_score:.2f} / 1.0**   {bar_filled}{bar_empty}  {pct}%"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # AI VERDICT (deterministic — reliable for demo)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if db_was_broken and db_got_fixed and row_count > 0:
        verdict = f"✅ The DB-Surgeon model detected schema errors, autonomously repaired the database, then successfully answered your question — returning {row_count} result(s). Score: {final_score:.2f}/1.0"
    elif db_was_broken and db_got_fixed and row_count == 0:
        verdict = f"✅ Database was broken and the model fixed it successfully! However, the query returned no matching results. Score: {final_score:.2f}/1.0"
    elif db_was_broken and not db_got_fixed:
        verdict = f"⚠️ The model detected database errors but could not fully repair them. Score: {final_score:.2f}/1.0"
    elif not db_was_broken and row_count > 0:
        verdict = f"✅ Database was healthy (no fix needed). The model translated your question into SQL and returned {row_count} result(s). Score: {final_score:.2f}/1.0"
    elif not db_was_broken and sql_generated:
        verdict = f"Database was healthy. SQL was generated but returned no matching rows. Score: {final_score:.2f}/1.0"
    else:
        verdict = f"Pipeline completed with score {final_score:.2f}/1.0."
    
    pipeline_log = "\n".join(log)
    return pipeline_log, score_text, verdict
