"""
Broken DB Generator — Creates broken database scenarios for each episode.

Phase 1 (MVP): Foreign Key Violation bugs only.
Phase 2+: Datatype mismatch, missing index, constraint conflict, schema drift.

Each episode generates:
- A randomized schema with injected bugs
- A business query that fails due to the bug
- Hidden evaluation queries for scoring
- Ground truth for reward calculation
"""

from __future__ import annotations

import random
import secrets
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BrokenDBScenario:
    """Complete scenario for one episode."""
    schema_sql: str
    """CREATE TABLE statements (may contain intentional bugs)."""
    
    seed_data_sql: str
    """INSERT statements for seed data."""
    
    healthy_schema_sql: str
    """The correct schema (for reference / oracle scoring)."""
    
    healthy_seed_data_sql: str
    """Seed data that works with the healthy schema."""
    
    business_query: str
    """The SQL query that the agent must make work."""
    
    eval_queries: list[str]
    """Hidden queries for evaluation scoring."""
    
    bug_type: str
    """Category of the injected bug."""
    
    bug_variant: str
    """Specific variant within the bug type."""
    
    root_cause: str
    """Human-readable description of what's broken."""
    
    expected_fix: str
    """Description of the fix needed."""
    
    table_prefix: str
    """Randomized prefix used for table names."""
    
    involved_tables: list[str] = field(default_factory=list)
    """Tables involved in the bug (for diagnostic reward)."""


class BrokenDBGenerator:
    """
    Generates randomized broken database scenarios.
    
    Each call to generate() produces a fresh scenario with:
    - Randomized table name prefixes (anti-memorization)
    - Randomized column order and dummy columns
    - Randomized seed data
    - One injected bug from the configured bug types
    """

    # Available bug generators (Phase 1: FK only)
    BUG_TYPES = ["fk_violation"]

    # Column name pools for dummy columns
    DUMMY_TEXT_COLUMNS = [
        "description", "notes", "category", "label", "tag",
        "comment", "memo", "remark", "summary", "code",
    ]
    DUMMY_INT_COLUMNS = [
        "priority", "count", "rank", "score", "level",
        "version", "quantity", "rating", "position", "sequence",
    ]

    # Name pools for seed data
    FIRST_NAMES = [
        "Alice", "Bob", "Charlie", "Diana", "Eve",
        "Frank", "Grace", "Henry", "Ivy", "Jack",
        "Karen", "Leo", "Mona", "Nick", "Olivia",
    ]
    LAST_NAMES = [
        "Smith", "Jones", "Brown", "Davis", "Wilson",
        "Moore", "Taylor", "Thomas", "White", "Harris",
    ]
    PRODUCTS = [
        "Widget", "Gadget", "Gizmo", "Doohickey", "Thingamajig",
        "Contraption", "Device", "Apparatus", "Module", "Component",
    ]

    def __init__(self, bug_types: Optional[list[str]] = None, seed: Optional[int] = None):
        """
        Args:
            bug_types: List of bug types to generate. Defaults to all available.
            seed: Random seed for reproducibility (None = random).
        """
        self._bug_types = bug_types or self.BUG_TYPES
        self._rng = random.Random(seed)

    def generate(self) -> BrokenDBScenario:
        """Generate a fresh broken database scenario."""
        bug_type = self._rng.choice(self._bug_types)
        
        if bug_type == "fk_violation":
            return self._generate_fk_violation()
        else:
            raise ValueError(f"Unknown bug type: {bug_type}")

    # ─── FK Violation Generator ─────────────────────────────────

    def _generate_fk_violation(self) -> BrokenDBScenario:
        """
        Generate a schema with a foreign key violation.
        
        Variants:
        A) Column renamed: orders.user_id → orders.usr_id
        B) Wrong FK reference: REFERENCES users(email) instead of users(id)
        C) Referenced table missing: users table doesn't exist
        D) Table name typo: REFERENCES usrs(id) instead of users(id)
        """
        prefix = f"tbl_{secrets.token_hex(2)}"
        variant = self._rng.choice(["rename_column", "wrong_ref_col", "missing_table", "table_typo"])

        # Table names
        users_tbl = f"{prefix}_users"
        orders_tbl = f"{prefix}_orders"
        products_tbl = f"{prefix}_products"

        # Generate dummy columns
        n_dummy = self._rng.randint(1, 3)
        user_dummy = self._pick_dummy_columns(n_dummy)
        order_dummy = self._pick_dummy_columns(n_dummy)

        # ─── HEALTHY schema (ground truth) ───
        healthy_schema = f"""
CREATE TABLE {users_tbl} (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    {', '.join(f'{c[0]} {c[1]}' for c in user_dummy)}
);

CREATE TABLE {products_tbl} (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE {orders_tbl} (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    status TEXT DEFAULT 'pending',
    {', '.join(f'{c[0]} {c[1]}' for c in order_dummy)},
    FOREIGN KEY (user_id) REFERENCES {users_tbl}(id),
    FOREIGN KEY (product_id) REFERENCES {products_tbl}(id)
);
"""

        # ─── Seed data ───
        seed_data = self._generate_seed_data(users_tbl, orders_tbl, products_tbl)

        # ─── Business query (the one the agent must fix) ───
        business_query = f"""SELECT u.name, u.email, o.amount, o.status, p.name as product_name
FROM {orders_tbl} o
JOIN {users_tbl} u ON o.user_id = u.id
JOIN {products_tbl} p ON o.product_id = p.id
WHERE o.amount > 50.0
ORDER BY o.amount DESC;"""

        # ─── Eval queries (hidden from agent) ───
        eval_queries = [
            business_query,
            f"SELECT COUNT(*) FROM {orders_tbl} o JOIN {users_tbl} u ON o.user_id = u.id;",
            f"SELECT u.name, SUM(o.amount) as total FROM {orders_tbl} o JOIN {users_tbl} u ON o.user_id = u.id GROUP BY u.name;",
            f"INSERT INTO {orders_tbl} (id, user_id, product_id, amount) VALUES (999, 1, 1, 75.0);",
            f"SELECT * FROM {users_tbl} WHERE id IN (SELECT user_id FROM {orders_tbl});",
        ]

        # ─── Apply the bug ───
        if variant == "rename_column":
            # Rename user_id → usr_id in orders table
            broken_schema = healthy_schema.replace(
                f"user_id INTEGER NOT NULL",
                f"usr_id INTEGER NOT NULL",
            ).replace(
                f"FOREIGN KEY (user_id) REFERENCES {users_tbl}(id)",
                f"FOREIGN KEY (usr_id) REFERENCES {users_tbl}(id)",
            )
            # Seed data also needs the column name change
            broken_seed = seed_data.replace("user_id,", "usr_id,")
            
            root_cause = f"Column 'user_id' in table '{orders_tbl}' was renamed to 'usr_id', but the business query still references 'user_id'."
            expected_fix = f"Rename column 'usr_id' back to 'user_id' in table '{orders_tbl}'."

        elif variant == "wrong_ref_col":
            # FK references users(email) instead of users(id)
            broken_schema = healthy_schema.replace(
                f"FOREIGN KEY (user_id) REFERENCES {users_tbl}(id)",
                f"FOREIGN KEY (user_id) REFERENCES {users_tbl}(email)",
            )
            # Also change user_id type to TEXT to match email
            broken_schema = broken_schema.replace(
                f"user_id INTEGER NOT NULL",
                f"user_id TEXT NOT NULL",
            )
            broken_seed = seed_data.replace("user_id,", "user_id,")
            # Replace numeric user_id values with email strings in seed
            for i in range(1, 16):
                name = self.FIRST_NAMES[i - 1] if i <= len(self.FIRST_NAMES) else f"User{i}"
                email = f"{name.lower()}@example.com"
                broken_seed = broken_seed.replace(
                    f"VALUES ({i * 100 + 1}, {i},",
                    f"VALUES ({i * 100 + 1}, '{email}',",
                )
                broken_seed = broken_seed.replace(
                    f"VALUES ({i * 100 + 2}, {i},",
                    f"VALUES ({i * 100 + 2}, '{email}',",
                )

            root_cause = f"Column 'user_id' in table '{orders_tbl}' is type TEXT referencing users(email) instead of type INTEGER referencing users(id). The JOIN fails due to type mismatch."
            expected_fix = f"Change 'user_id' column in '{orders_tbl}' from TEXT to INTEGER and fix the FOREIGN KEY to reference {users_tbl}(id)."

        elif variant == "missing_table":
            # Remove the users table entirely
            lines = healthy_schema.split("\n")
            broken_lines = []
            skip = False
            for line in lines:
                if f"CREATE TABLE {users_tbl}" in line:
                    skip = True
                    continue
                if skip and line.strip() == ");":
                    skip = False
                    continue
                if not skip:
                    broken_lines.append(line)
            broken_schema = "\n".join(broken_lines)
            broken_seed = "\n".join(
                line for line in seed_data.split("\n")
                if users_tbl not in line
            )

            root_cause = f"Table '{users_tbl}' is missing from the database. The business query JOINs on this table."
            expected_fix = f"Create the missing '{users_tbl}' table with columns: id INTEGER PK, name TEXT, email TEXT."

        elif variant == "table_typo":
            # Typo in table name: users → usrs in FK reference
            typo_name = f"{prefix}_usrs"
            broken_schema = healthy_schema.replace(
                f"FOREIGN KEY (user_id) REFERENCES {users_tbl}(id)",
                f"FOREIGN KEY (user_id) REFERENCES {typo_name}(id)",
            )
            broken_seed = seed_data  # Seed data unchanged, but FK is broken

            root_cause = f"FOREIGN KEY in table '{orders_tbl}' references '{typo_name}' (typo) instead of '{users_tbl}'."
            expected_fix = f"Fix the FOREIGN KEY reference from '{typo_name}' to '{users_tbl}'."
        
        else:
            raise ValueError(f"Unknown FK variant: {variant}")

        return BrokenDBScenario(
            schema_sql=broken_schema,
            seed_data_sql=broken_seed if variant != "missing_table" else broken_seed,
            healthy_schema_sql=healthy_schema,
            healthy_seed_data_sql=seed_data,
            business_query=business_query,
            eval_queries=eval_queries,
            bug_type="fk_violation",
            bug_variant=variant,
            root_cause=root_cause,
            expected_fix=expected_fix,
            table_prefix=prefix,
            involved_tables=[users_tbl, orders_tbl],
        )

    # ─── Helpers ────────────────────────────────────────────────

    def _pick_dummy_columns(self, n: int) -> list[tuple[str, str]]:
        """Pick n random dummy columns (name, type)."""
        text_cols = self._rng.sample(self.DUMMY_TEXT_COLUMNS, min(n, len(self.DUMMY_TEXT_COLUMNS)))
        int_cols = self._rng.sample(self.DUMMY_INT_COLUMNS, min(n, len(self.DUMMY_INT_COLUMNS)))
        
        result = []
        for i in range(n):
            if i % 2 == 0 and text_cols:
                result.append((text_cols.pop(), "TEXT"))
            elif int_cols:
                result.append((int_cols.pop(), "INTEGER DEFAULT 0"))
            elif text_cols:
                result.append((text_cols.pop(), "TEXT"))
        return result

    def _generate_seed_data(
        self, users_tbl: str, orders_tbl: str, products_tbl: str
    ) -> str:
        """Generate realistic-looking seed data."""
        lines = []
        
        # Users (5-10)
        n_users = self._rng.randint(5, 10)
        for i in range(1, n_users + 1):
            first = self._rng.choice(self.FIRST_NAMES)
            last = self._rng.choice(self.LAST_NAMES)
            name = f"{first} {last}"
            email = f"{first.lower()}.{last.lower()}@example.com"
            lines.append(
                f"INSERT INTO {users_tbl} (id, name, email) VALUES ({i}, '{name}', '{email}');"
            )

        # Products (3-6)
        n_products = self._rng.randint(3, 6)
        for i in range(1, n_products + 1):
            prod_name = self._rng.choice(self.PRODUCTS)
            price = round(self._rng.uniform(10.0, 500.0), 2)
            lines.append(
                f"INSERT INTO {products_tbl} (id, name, price) VALUES ({i}, '{prod_name}', {price});"
            )

        # Orders (8-15)
        n_orders = self._rng.randint(8, 15)
        for i in range(1, n_orders + 1):
            user_id = self._rng.randint(1, n_users)
            product_id = self._rng.randint(1, n_products)
            amount = round(self._rng.uniform(10.0, 500.0), 2)
            status = self._rng.choice(["pending", "shipped", "delivered", "cancelled"])
            lines.append(
                f"INSERT INTO {orders_tbl} (id, user_id, product_id, amount, status) "
                f"VALUES ({i * 100 + 1}, {user_id}, {product_id}, {amount}, '{status}');"
            )

        return "\n".join(lines)
