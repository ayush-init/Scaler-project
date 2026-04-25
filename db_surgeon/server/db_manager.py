"""
DB Manager — SQLite in-memory database lifecycle management.

Handles creation, querying, schema inspection, DDL execution,
and validation of the database within a single episode.
Each episode gets a completely fresh :memory: database.
"""

from __future__ import annotations

import sqlite3
import re
from typing import Optional


class DBManager:
    """
    Manages an SQLite in-memory database for one episode.
    
    Provides safe methods for:
    - Creating and populating databases
    - Executing read-only queries
    - Applying DDL/DML fixes
    - Inspecting schema
    - Validating fixes against eval queries
    """

    # DDL commands that are ALLOWED
    ALLOWED_DDL_PREFIXES = (
        "ALTER", "CREATE INDEX", "CREATE TABLE",
        "DROP INDEX", "INSERT", "UPDATE", "DELETE",
    )

    # Commands that are BLOCKED (destructive)
    BLOCKED_PATTERNS = re.compile(
        r"\b(DROP\s+DATABASE|DROP\s+TABLE|TRUNCATE|ATTACH|DETACH|VACUUM)\b",
        re.IGNORECASE,
    )

    MAX_SQL_LENGTH = 5000  # Truncate overly long SQL

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None

    def create_database(self, schema_sql: str, seed_data_sql: str = "") -> None:
        """
        Create a fresh in-memory database with the given schema and seed data.
        
        Args:
            schema_sql: CREATE TABLE statements (semicolon-separated).
            seed_data_sql: INSERT statements for seed data (semicolon-separated).
        """
        self.reset()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._cursor = self._conn.cursor()

        # Execute schema statements
        try:
            self._conn.executescript(schema_sql)
        except sqlite3.Error as e:
            # Schema itself may be intentionally broken — that's fine
            pass

        # Execute seed data
        if seed_data_sql.strip():
            try:
                self._conn.executescript(seed_data_sql)
            except sqlite3.Error:
                # Some inserts may fail due to broken schema — expected
                pass

    def execute_query(self, sql: str) -> tuple[bool, str]:
        """
        Execute a read-only SQL query (SELECT, PRAGMA, etc.).
        
        Args:
            sql: The SQL query to execute.
            
        Returns:
            (success, result_or_error) tuple.
        """
        if not self._conn:
            return False, "Error: No database connection."

        sql = sql.strip()
        if not sql:
            return False, "Error: Empty SQL query."

        if len(sql) > self.MAX_SQL_LENGTH:
            return False, f"Error: SQL too long ({len(sql)} chars, max {self.MAX_SQL_LENGTH})."

        try:
            self._cursor.execute(sql)
            if self._cursor.description:
                columns = [desc[0] for desc in self._cursor.description]
                rows = self._cursor.fetchmany(50)  # Limit output rows
                if not rows:
                    return True, f"Columns: {', '.join(columns)}\n(0 rows)"
                
                # Format as simple table
                lines = [" | ".join(columns)]
                lines.append("-" * len(lines[0]))
                for row in rows:
                    lines.append(" | ".join(str(v) for v in row))
                
                total = self._cursor.fetchall()
                if total:
                    lines.append(f"... and {len(total)} more rows")
                
                return True, "\n".join(lines)
            else:
                return True, f"OK. Rows affected: {self._cursor.rowcount}"
        except sqlite3.Error as e:
            return False, f"SQL Error: {str(e)}"

    def execute_ddl(self, sql: str) -> tuple[bool, str]:
        """
        Execute a DDL/DML statement to modify the database.
        
        Validates against blocked patterns before execution.
        
        Args:
            sql: The DDL/DML statement to execute.
            
        Returns:
            (success, result_or_error) tuple.
        """
        if not self._conn:
            return False, "Error: No database connection."

        sql = sql.strip()
        if not sql:
            return False, "Error: Empty SQL statement."

        if len(sql) > self.MAX_SQL_LENGTH:
            return False, f"Error: SQL too long ({len(sql)} chars, max {self.MAX_SQL_LENGTH})."

        # Check for blocked commands
        if self.BLOCKED_PATTERNS.search(sql):
            return False, "Error: Destructive operation not permitted. Use specific fix tools instead."

        try:
            self._conn.executescript(sql)
            self._conn.commit()
            return True, "DDL executed successfully."
        except sqlite3.Error as e:
            return False, f"DDL Error: {str(e)}"

    def fix_column(
        self,
        table_name: str,
        column_name: str,
        new_type: str = "",
        new_name: str = "",
    ) -> tuple[bool, str]:
        """
        Modify a column's type or rename it.
        
        SQLite has limited ALTER TABLE support, so we handle column
        modifications by recreating the table when necessary.
        
        Args:
            table_name: The table containing the column.
            column_name: The column to modify.
            new_type: New data type (e.g., 'INTEGER', 'TEXT').
            new_name: New column name if renaming.
            
        Returns:
            (success, result_or_error) tuple.
        """
        if not self._conn:
            return False, "Error: No database connection."

        if not new_type and not new_name:
            return False, "Error: Must specify new_type or new_name."

        # Check table exists
        table_info = self._get_table_columns(table_name)
        if table_info is None:
            return False, f"Error: Table '{table_name}' not found."

        # Check column exists
        col_found = False
        for col in table_info:
            if col["name"] == column_name:
                col_found = True
                break
        
        if not col_found:
            return False, f"Error: Column '{column_name}' not found in table '{table_name}'."

        try:
            if new_name and not new_type:
                # Simple rename — SQLite supports this directly
                self._conn.execute(
                    f"ALTER TABLE {table_name} RENAME COLUMN {column_name} TO {new_name}"
                )
                self._conn.commit()
                return True, f"Column '{column_name}' renamed to '{new_name}' in table '{table_name}'."

            elif new_type:
                # Type change requires table recreation
                actual_new_name = new_name or column_name
                
                # Get current CREATE TABLE statement
                create_sql = self._get_create_table_sql(table_name)
                if not create_sql:
                    return False, f"Error: Could not read schema for table '{table_name}'."

                # Build new column definitions
                new_columns = []
                old_col_names = []
                for col in table_info:
                    old_col_names.append(col["name"])
                    if col["name"] == column_name:
                        col_type = new_type
                        col_nm = actual_new_name
                    else:
                        col_type = col["type"] or "TEXT"
                        col_nm = col["name"]
                    
                    pk_str = " PRIMARY KEY" if col["pk"] else ""
                    notnull_str = " NOT NULL" if col["notnull"] and not col["pk"] else ""
                    default_str = f" DEFAULT {col['dflt_value']}" if col["dflt_value"] else ""
                    new_columns.append(f"{col_nm} {col_type}{pk_str}{notnull_str}{default_str}")

                # Recreate table
                temp_name = f"_temp_{table_name}"
                new_create = f"CREATE TABLE {temp_name} ({', '.join(new_columns)})"
                
                col_mapping = ", ".join(old_col_names)
                
                self._conn.execute(new_create)
                self._conn.execute(f"INSERT INTO {temp_name} SELECT {col_mapping} FROM {table_name}")
                self._conn.execute(f"DROP TABLE {table_name}")
                self._conn.execute(f"ALTER TABLE {temp_name} RENAME TO {table_name}")
                self._conn.commit()

                result_parts = []
                if new_type:
                    result_parts.append(f"type changed to {new_type}")
                if new_name:
                    result_parts.append(f"renamed to '{new_name}'")
                
                return True, f"Column '{column_name}' in table '{table_name}': {', '.join(result_parts)}."

        except sqlite3.Error as e:
            return False, f"Fix Error: {str(e)}"

    def add_index(self, table_name: str, column_name: str) -> tuple[bool, str]:
        """
        Create an index on a column.
        
        Args:
            table_name: The table to index.
            column_name: The column to create an index on.
            
        Returns:
            (success, result_or_error) tuple.
        """
        if not self._conn:
            return False, "Error: No database connection."

        index_name = f"idx_{table_name}_{column_name}"
        try:
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name})"
            )
            self._conn.commit()
            return True, f"Index '{index_name}' created on {table_name}({column_name})."
        except sqlite3.Error as e:
            return False, f"Index Error: {str(e)}"

    def get_schema(self) -> str:
        """
        Get all CREATE TABLE statements for the current database.
        
        Returns:
            Formatted schema string.
        """
        if not self._conn:
            return "No database connection."

        try:
            self._cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
            )
            tables = self._cursor.fetchall()
            if not tables:
                return "No tables found."
            return "\n\n".join(row[0] + ";" for row in tables)
        except sqlite3.Error as e:
            return f"Schema Error: {str(e)}"

    def get_table_info(self, table_name: str) -> str:
        """
        Get detailed column information for a specific table.
        
        Args:
            table_name: The table to inspect.
            
        Returns:
            Formatted table info string.
        """
        if not self._conn:
            return "No database connection."

        cols = self._get_table_columns(table_name)
        if cols is None:
            return f"Table '{table_name}' not found."

        lines = [f"Table: {table_name}", "=" * 50]
        lines.append(f"{'Column':<20} {'Type':<12} {'PK':<4} {'NotNull':<8} {'Default':<10}")
        lines.append("-" * 54)
        for col in cols:
            pk = "YES" if col["pk"] else ""
            nn = "YES" if col["notnull"] else ""
            df = str(col["dflt_value"]) if col["dflt_value"] else ""
            lines.append(f"{col['name']:<20} {col['type'] or 'N/A':<12} {pk:<4} {nn:<8} {df:<10}")

        # Also show row count
        try:
            self._cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self._cursor.fetchone()[0]
            lines.append(f"\nRow count: {count}")
        except sqlite3.Error:
            lines.append("\nRow count: (error reading)")

        return "\n".join(lines)

    def get_table_names(self) -> list[str]:
        """Get list of all table names in the database."""
        if not self._conn:
            return []
        try:
            self._cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            return [row[0] for row in self._cursor.fetchall()]
        except sqlite3.Error:
            return []

    def validate_fix(self, eval_queries: list[str]) -> tuple[float, list[dict]]:
        """
        Run evaluation queries and return a score.
        
        Args:
            eval_queries: List of SQL queries to test.
            
        Returns:
            (score, details) where score is 0.0-1.0 and details
            is a list of {query, passed, error} dicts.
        """
        if not eval_queries:
            return 0.0, []

        details = []
        passed = 0
        for q in eval_queries:
            success, result = self.execute_query(q)
            details.append({
                "query": q[:100],  # Truncate for readability
                "passed": success,
                "error": "" if success else result,
            })
            if success:
                passed += 1

        score = passed / len(eval_queries)
        return score, details

    def reset(self) -> None:
        """Destroy the current database connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None
        self._cursor = None

    # ─── Private helpers ────────────────────────────────────────

    def _get_table_columns(self, table_name: str) -> Optional[list[dict]]:
        """Get column info via PRAGMA table_info."""
        if not self._conn:
            return None
        try:
            self._cursor.execute(f"PRAGMA table_info({table_name})")
            rows = self._cursor.fetchall()
            if not rows:
                return None
            return [
                {
                    "cid": row[0],
                    "name": row[1],
                    "type": row[2],
                    "notnull": bool(row[3]),
                    "dflt_value": row[4],
                    "pk": bool(row[5]),
                }
                for row in rows
            ]
        except sqlite3.Error:
            return None

    def _get_create_table_sql(self, table_name: str) -> Optional[str]:
        """Get the CREATE TABLE statement for a table."""
        if not self._conn:
            return None
        try:
            self._cursor.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            row = self._cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None
