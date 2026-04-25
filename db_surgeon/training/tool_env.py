"""
TRL Tool Environment Wrapper — The critical integration layer.

This class wraps the DB-Surgeon environment for TRL's GRPOTrainer.
Each public method becomes a tool that the LLM can call via
function-calling (auto-discovered by GRPOTrainer).

Rules from TRL docs:
- __init__ takes no arguments
- reset() accepts **kwargs (dataset columns passed here)
- Public methods = tools (must have docstrings with Args:)
- Private methods (start with _) are NOT exposed
- Raise ValueError to signal game over
- Store reward on self.reward for the reward function
"""

from __future__ import annotations

from db_surgeon.models import DBSurgeonAction
from db_surgeon.client import DBSurgeonLocalEnv


class DBSurgeonToolEnv:
    """
    TRL-compatible environment wrapper for DB-Surgeon.
    
    The GRPOTrainer discovers all public methods (except reset)
    and exposes them as function-calling tools for the LLM agent.
    
    Usage with TRL:
        trainer = GRPOTrainer(
            ...,
            environment_factory=DBSurgeonToolEnv,
        )
    """

    def __init__(self):
        """Initialize the environment. Must take no arguments."""
        self._env = DBSurgeonLocalEnv()
        self.reward = 0.0
        self.done = False
        self._total_reward = 0.0

    def reset(self, **kwargs) -> str | None:
        """
        Reset the environment and return the initial observation.
        
        Called automatically by GRPOTrainer at the start of each episode.
        Dataset columns are passed as kwargs.
        
        Returns:
            Formatted observation string, or None if reset fails.
        """
        try:
            result = self._env.reset()
            self.reward = 0.0
            self._total_reward = 0.0
            self.done = False
            return self._format_observation(result.observation)
        except Exception as e:
            return f"Reset failed: {str(e)}"

    # ═══════════════════════════════════════════════════════════
    # PUBLIC TOOL METHODS — Auto-discovered by GRPOTrainer
    # Each becomes a callable tool for the LLM agent
    # ═══════════════════════════════════════════════════════════

    def inspect_schema(self, table_name: str = "") -> str:
        """
        Inspect the database schema. Shows all tables if no table_name given,
        or detailed column info for a specific table.

        Args:
            table_name: Optional name of a specific table to inspect.
                       Leave empty to see all tables and their CREATE statements.

        Returns:
            Schema information as formatted text showing table structure,
            column types, constraints, and row counts.
        """
        self._check_done()
        result = self._env.step(DBSurgeonAction(
            tool_name="inspect_schema",
            arguments={"table_name": table_name},
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def run_query(self, sql: str) -> str:
        """
        Execute a read-only SQL query against the database.
        Use this to test queries, check data, or verify your fixes.

        Args:
            sql: A SQL SELECT query to execute. Only read operations
                 are allowed through this tool.

        Returns:
            Query results as a formatted table, or an error message
            if the query fails. Error messages contain useful diagnostic
            information about what went wrong.
        """
        self._check_done()
        result = self._env.step(DBSurgeonAction(
            tool_name="run_query",
            arguments={"sql": sql},
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def fix_column(
        self,
        table_name: str,
        column_name: str,
        new_type: str = "",
        new_name: str = "",
    ) -> str:
        """
        Modify a column's data type or rename it. Use this to fix
        type mismatches or column naming issues.

        Args:
            table_name: The table containing the column to fix.
            column_name: The current name of the column to modify.
            new_type: New data type for the column (e.g., 'INTEGER', 'TEXT', 'REAL').
                     Leave empty if only renaming.
            new_name: New name for the column. Leave empty if only changing type.

        Returns:
            Success message confirming the change, or error message
            explaining why the fix failed.
        """
        self._check_done()
        args = {"table_name": table_name, "column_name": column_name}
        if new_type:
            args["new_type"] = new_type
        if new_name:
            args["new_name"] = new_name

        result = self._env.step(DBSurgeonAction(
            tool_name="fix_column",
            arguments=args,
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def add_index(self, table_name: str, column_name: str) -> str:
        """
        Create an index on a specific column to improve query performance.

        Args:
            table_name: The table to add the index to.
            column_name: The column to create an index on.

        Returns:
            Success message confirming the index was created, or error
            message if the operation failed.
        """
        self._check_done()
        result = self._env.step(DBSurgeonAction(
            tool_name="add_index",
            arguments={"table_name": table_name, "column_name": column_name},
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def add_constraint(
        self,
        table_name: str,
        constraint_type: str,
        column_name: str,
        reference: str = "",
    ) -> str:
        """
        Add a constraint to a table column. Supports UNIQUE and FOREIGN_KEY.

        Args:
            table_name: The table to add the constraint to.
            constraint_type: Type of constraint: 'UNIQUE' or 'FOREIGN_KEY'.
            column_name: The column the constraint applies to.
            reference: For FOREIGN_KEY only: the referenced table and column
                      in format 'table_name.column_name'.

        Returns:
            Success message or error explanation.
        """
        self._check_done()
        args = {
            "table_name": table_name,
            "constraint_type": constraint_type,
            "column_name": column_name,
        }
        if reference:
            args["reference"] = reference

        result = self._env.step(DBSurgeonAction(
            tool_name="add_constraint",
            arguments=args,
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def execute_fix(self, sql: str) -> str:
        """
        Execute a DDL or DML statement to fix the database schema.
        Use this for complex fixes not covered by other tools, such as
        creating missing tables, altering constraints, or restructuring schema.

        Args:
            sql: The SQL DDL/DML statement to execute. Supports ALTER TABLE,
                 CREATE TABLE, CREATE INDEX, INSERT, UPDATE, DELETE.
                 Destructive operations like DROP DATABASE are blocked.

        Returns:
            Success confirmation or detailed error message.
        """
        self._check_done()
        result = self._env.step(DBSurgeonAction(
            tool_name="execute_fix",
            arguments={"sql": sql},
        ))
        self._update_state(result)
        return result.observation.last_action_result

    def submit(self) -> str:
        """
        Submit your fix and end the episode. Call this when you believe
        the database is repaired and the business query should work.
        Your fix will be evaluated against multiple test queries.

        Returns:
            Final evaluation showing how many test queries pass,
            the overall score, and whether the fix was successful.
        """
        self._check_done()
        result = self._env.step(DBSurgeonAction(
            tool_name="submit",
            arguments={},
        ))
        self._update_state(result)
        # Set final reward on submit
        self.reward = self._total_reward
        return result.observation.last_action_result

    # ═══════════════════════════════════════════════════════════
    # PRIVATE METHODS — NOT exposed as tools
    # ═══════════════════════════════════════════════════════════

    def _check_done(self):
        """Raise ValueError if episode is over (TRL convention)."""
        if self.done:
            raise ValueError(
                "Episode is over. The database fix has been submitted "
                "or the step limit was reached."
            )

    def _update_state(self, result):
        """Update internal state from step result."""
        self._total_reward += result.reward
        self.done = result.done
        if result.done:
            self.reward = self._total_reward

    def _format_observation(self, obs) -> str:
        """Format the observation into a readable string for the LLM."""
        return f"""🏥 DATABASE SURGERY REQUIRED

You are a database engineer. A production database has schema issues causing query failures.
Your task: diagnose the problem and fix it using the available tools.

═══ CURRENT SCHEMA ═══
{obs.schema_snapshot}

═══ FAILING BUSINESS QUERY ═══
{obs.failing_query}

═══ ERROR LOG ═══
{obs.error_log}

═══ INSTRUCTIONS ═══
1. Use inspect_schema() to examine tables in detail
2. Use run_query() to test hypotheses
3. Use fix_column(), add_index(), add_constraint(), or execute_fix() to repair the schema
4. Use submit() when you believe the database is fixed
5. You have {obs.max_steps} steps maximum. Be efficient!

Start by inspecting the schema to understand what's wrong."""
