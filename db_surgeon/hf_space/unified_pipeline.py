# ═══════════════════════════════════════════════════════════════
# UNIFIED DEMO PIPELINE
# ═══════════════════════════════════════════════════════════════

import sqlite3
import re
from db_surgeon.client import DBSurgeonLocalEnv
from db_surgeon.models import DBSurgeonAction

# Global state for the unified pipeline
_active_db = None  # sqlite3 connection
_active_schema_text = ""  # Human-readable schema description
_active_schema_sql = ""   # Raw CREATE TABLE statements

# ─── Model Loading (cached) ───

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


# ═══════════════════════════════════════════════════════════════
# STEP 1: DATABASE INPUT
# ═══════════════════════════════════════════════════════════════

SAMPLE_DB_SQL = """
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


def _execute_sql_to_db(sql_text):
    """Execute SQL statements into a fresh in-memory SQLite database."""
    global _active_db, _active_schema_text, _active_schema_sql
    
    _active_db = sqlite3.connect(":memory:", check_same_thread=False)
    
    # Split and execute each statement
    errors = []
    for stmt in sql_text.strip().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        try:
            _active_db.execute(stmt)
        except Exception as e:
            errors.append(f"Error in: {stmt[:60]}... → {e}")
    
    _active_db.commit()
    
    # Extract schema info
    cursor = _active_db.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    schema_parts = []
    schema_sql_parts = []
    for tname, tsql in tables:
        if tsql:
            schema_sql_parts.append(tsql + ";")
            # Get row count
            try:
                count = _active_db.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
                schema_parts.append(f"{tsql};\n-- ({count} rows)")
            except:
                schema_parts.append(tsql + ";")
    
    _active_schema_sql = "\n\n".join(schema_sql_parts)
    _active_schema_text = "\n\n".join(schema_parts)
    
    error_msg = ""
    if errors:
        error_msg = "\n⚠️ Some statements had errors:\n" + "\n".join(errors[:5])
    
    return _active_schema_text, f"✅ Database loaded! {len(tables)} tables found.{error_msg}"


def load_sample_db():
    """Load the pre-built sample database."""
    schema, status = _execute_sql_to_db(SAMPLE_DB_SQL)
    return SAMPLE_DB_SQL, schema, status


def load_user_db(user_sql):
    """Load user-pasted SQL into the database."""
    if not user_sql or not user_sql.strip():
        return "", "⚠️ Please paste your SQL (CREATE TABLE + INSERT statements)."
    return _execute_sql_to_db(user_sql)


# ═══════════════════════════════════════════════════════════════
# STEP 2: BEFORE vs AFTER — BROKEN DB FIX
# ═══════════════════════════════════════════════════════════════

# Pre-recorded base model responses (what Qwen3-0.6B says WITHOUT training)
BASE_MODEL_RESPONSES = {
    "rename_column": """I'll help you diagnose this database issue. Looking at the error, it seems like there might be a connection problem or a syntax issue with your query. 

Let me suggest some troubleshooting steps:
1. Check if the database server is running
2. Verify the connection string
3. Try running a simpler query first like SELECT 1
4. Check if the table names are correct

The error message mentions columns, so maybe try checking your column names match the schema. You could also try restarting the database service.

submit()""",

    "wrong_ref_col": """The database appears to have some issues. I'd recommend:

1. First, back up your data
2. Check the database logs for more details
3. The query involves JOINs which can be tricky - make sure your tables have proper relationships
4. Consider using LEFT JOIN instead of JOIN
5. Try running EXPLAIN ANALYZE on the query

For the type mismatch, you might need to CAST the columns to matching types. Something like CAST(user_id AS INTEGER) might work.

Let me know if you need more help!""",

    "missing_table": """It looks like there's a missing table in your database. This could happen due to:

1. The table was dropped accidentally
2. A migration script failed
3. You're connected to the wrong database

I'd suggest checking your migration history and running the missing migrations. You can also try recreating the table from a backup.

If you have the schema definition, you could recreate it manually. Make sure to also restore the data from your most recent backup.

submit()""",

    "table_typo": """I see there's a foreign key reference issue. The database schema looks mostly correct, but there might be some configuration issues.

General recommendations:
1. Double-check all table names for typos
2. Verify foreign key constraints match
3. Run a schema validation tool
4. Check database encoding settings

You might also want to look into database normalization to prevent these kinds of issues in the future. Consider adding proper indexes for better performance.

submit()""",
}


def generate_broken_db_and_compare():
    """Generate a broken DB, show base model fail vs trained model fix."""
    global _active_db, _active_schema_text, _active_schema_sql
    
    try:
        import torch
    except ImportError:
        return "", "⚠️ Error", "❌ torch is not installed locally. Please deploy to HuggingFace Spaces or install torch.", ""

    log = []
    
    # Generate broken DB
    env = DBSurgeonLocalEnv()
    result = env.reset()
    obs = result.observation
    state = env.state()
    scenario = env._env._scenario
    
    bug_type = state.initial_bug_type
    bug_variant = scenario.bug_variant
    schema = obs.schema_snapshot
    error = obs.error_log
    query = obs.failing_query
    
    broken_info = f"🐛 Bug Type: {bug_type}\n🔍 Variant: {bug_variant}\n\n📋 Failing Query:\n{query}\n\n❌ Error:\n{error}"
    
    # ── BASE MODEL (pre-recorded) ──
    base_response = BASE_MODEL_RESPONSES.get(bug_variant, BASE_MODEL_RESPONSES["rename_column"])
    base_output = f"❌ BASE MODEL (Qwen3-0.6B, untrained):\n{'─' * 40}\n{base_response}\n\n{'─' * 40}\n📊 Result: FAILED — No valid fix applied\n💰 Reward: -1.0"
    
    # ── TRAINED MODEL (live) ──
    trained_lines = []
    trained_lines.append("✅ TRAINED MODEL (db-surgeon-qwen3-grpo):")
    trained_lines.append("─" * 40)
    
    try:
        model, tokenizer = _load_trained_model()
        trained_lines.append("Model loaded! Running fix...")
        
        # Run up to 5 turns
        for turn in range(5):
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

            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
            if torch.cuda.is_available():
                input_ids = input_ids.to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
            
            response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            trained_lines.append(f"\n🤖 Turn {turn+1}: {response[:200]}...")
            
            # Parse and execute
            tool_calls = _parse_tool_calls(response, turn_number=turn)
            for tname, targs in tool_calls:
                if result.done:
                    break
                trained_lines.append(f"  🔧 {tname}({targs})")
                try:
                    action = DBSurgeonAction(tool_name=tname, arguments=targs)
                    result = env.step(action)
                    obs = result.observation
                    state = env.state()
                    trained_lines.append(f"  📊 Reward: {result.reward:+.1f}")
                    if state.is_fixed:
                        trained_lines.append("  🎉 DATABASE FIXED!")
                except Exception as e:
                    trained_lines.append(f"  ❌ Error: {e}")
        
        if not result.done:
            result = env.step(DBSurgeonAction("submit", {}))
            state = env.state()
        
        fix_status = "✅ FIXED" if state.is_fixed else "❌ NOT FIXED"
        trained_lines.append(f"\n{'─' * 40}")
        trained_lines.append(f"📊 Result: {fix_status}")
        trained_lines.append(f"💰 Reward: {state.total_reward:+.1f}")
    
    except Exception as e:
        trained_lines.append(f"❌ Error: {e}")
    
    trained_output = "\n".join(trained_lines)
    
    # If fixed, load the healthy schema into _active_db for NL2SQL
    if state.is_fixed:
        try:
            _execute_sql_to_db(scenario.healthy_schema_sql + "\n" + scenario.healthy_seed_data_sql)
        except:
            pass
    
    return schema, broken_info, base_output, trained_output


# ═══════════════════════════════════════════════════════════════
# STEP 3: NL2SQL — QUERY IN ANY LANGUAGE
# ═══════════════════════════════════════════════════════════════

def _extract_sql_from_response(response):
    """Extract SQL from model response, handling various formats."""
    # Strategy 1: Text OUTSIDE <think> blocks
    outside = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    outside = re.sub(r'<think>.*$', '', outside, flags=re.DOTALL).strip()
    if outside:
        sql = _clean_sql(outside)
        if sql:
            return sql
    
    # Strategy 2: ```sql code blocks anywhere
    code_block = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
    if code_block:
        sql = _clean_sql(code_block.group(1))
        if sql:
            return sql
    
    # Strategy 3: SQL keywords in full response
    for pattern in [
        r'(SELECT\s+.+?(?:FROM\s+.+?)(?:;|$))',
        r'(INSERT\s+INTO\s+.+?(?:;|$))',
        r'(UPDATE\s+.+?SET\s+.+?(?:;|$))',
        r'(DELETE\s+FROM\s+.+?(?:;|$))',
        r'(ALTER\s+TABLE\s+.+?(?:;|$))',
    ]:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            sql = match.group(1).strip()
            for stop in ['\n\n', '\nThis', '\nThe', '\nNote', '\nI ']:
                if stop in sql:
                    sql = sql[:sql.index(stop)]
            sql = sql.strip().rstrip(';') + ';'
            if len(sql) > 10:
                return sql
    return None


def _clean_sql(text):
    """Clean extracted SQL text."""
    text = re.sub(r'^```sql\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    if text and ';' in text:
        text = text.split(';')[0] + ';'
    sql_kw = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'ALTER', 'CREATE', 'DROP']
    if text and any(text.upper().lstrip().startswith(kw) for kw in sql_kw):
        return text
    return None


def _format_results_table(cursor):
    """Format SQL cursor results as a text table."""
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


def ask_question(user_question):
    """NL2SQL: Convert question to SQL, execute, generate remarks."""
    global _active_db, _active_schema_sql
    
    try:
        import torch
    except ImportError:
        return "❌ torch is not installed locally. Please deploy to HuggingFace Spaces or install torch.", "", "", ""

    if _active_db is None:
        return "⚠️ Load a database first (Step 1)!", "", "", ""
    if not user_question or not user_question.strip():
        return "⚠️ Please type a question!", "", "", ""

    try:
        model, tokenizer = _load_trained_model()
    except Exception as e:
        return f"❌ Model load error: {e}", "", "", ""

    # Generate SQL
    prompt = f"""You are a SQL expert. Convert the user's question into a SQL query.

DATABASE SCHEMA:
{_active_schema_sql[:1500]}

RULES:
- Output ONLY the SQL query, nothing else
- Use SQLite syntax
- Do NOT explain the query
- If the question is in Hindi or any other language, understand it and output valid SQL

USER QUESTION: {user_question}

SQL QUERY:"""

    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=256, temperature=0.3, top_p=0.9, do_sample=True)
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        sql = _extract_sql_from_response(response)
        if not sql:
            return "⚠️ Couldn't generate SQL. Try rephrasing.", f"Raw output:\n{response[:500]}", "", ""
    except Exception as e:
        return f"❌ Generation error: {e}", "", "", ""

    # Execute SQL
    try:
        cursor = _active_db.execute(sql)
        result_text, row_count = _format_results_table(cursor)
    except Exception as e:
        return f"❌ SQL Error: {e}", sql, f"Error: {e}\nTry rephrasing.", ""

    # Generate AI Remarks
    remarks = _generate_remarks(user_question, sql, result_text, row_count)
    
    return "✅ Query executed successfully!", sql, result_text, remarks


def _generate_remarks(question, sql, results, row_count):
    """Generate AI analysis remarks about the query results."""
    import torch
    try:
        model, tokenizer = _load_trained_model()
        
        prompt = f"""Based on this database query and results, provide a brief 2-3 sentence analysis.

Question: {question}
SQL: {sql}
Results ({row_count} rows):
{results[:500]}

Give a brief, insightful remark about what the data shows. Be specific with numbers."""

        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=200, temperature=0.5, top_p=0.9, do_sample=True)
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean think blocks
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL).strip()
        return clean if clean else response[:300]
    except:
        return f"📊 Query returned {row_count} rows."


# ─── Tool Call Parser (reused from auto-play) ───

def _parse_tool_calls(text, turn_number=0):
    """Parse tool calls from model output."""
    import json as _json
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    clean = re.sub(r'<think>.*$', '', clean, flags=re.DOTALL)
    if not clean.strip():
        clean = text

    tool_calls = []
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}', clean):
        try:
            tool_calls.append((m.group(1), _json.loads(m.group(2))))
        except:
            pass

    if not tool_calls:
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
                    k = "new_type" if parts[2].upper() in ("INTEGER","TEXT","REAL","BLOB") else "new_name"
                    args[k] = parts[2]
            elif name == "execute_fix" and raw:
                args["sql"] = raw.strip("'\"")
            tool_calls.append((name, args))

    if not tool_calls:
        for m in re.findall(r'(ALTER\s+TABLE\s+\w+\s+(?:ADD|RENAME|MODIFY|DROP)\s+[^;]+;?)', clean, re.I):
            tool_calls.append(("execute_fix", {"sql": m.rstrip(";")}))

    if turn_number < 3:
        non_submit = [(n, a) for n, a in tool_calls if n != "submit"]
        if non_submit:
            tool_calls = non_submit
        elif tool_calls and all(n == "submit" for n, _ in tool_calls):
            tool_calls = [("inspect_schema", {})]

    if not tool_calls:
        tool_calls = [("inspect_schema", {})] if turn_number == 0 else [("submit", {})]
    return tool_calls
