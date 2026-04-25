"""
DB-Surgeon Environment — Core environment logic.

Implements the OpenEnv Environment interface:
- reset() → Create broken DB, return initial observation
- step(action) → Execute action, return (observation, reward, done)
- state() → Return current episode metadata
"""

from __future__ import annotations

import uuid
from typing import Optional

from db_surgeon.models import (
    DBSurgeonAction,
    DBSurgeonObservation,
    DBSurgeonState,
    StepResult,
)
from db_surgeon.server.broken_db_generator import BrokenDBGenerator, BrokenDBScenario
from db_surgeon.server.db_manager import DBManager
from db_surgeon.server.evaluation_oracle import EvaluationOracle
from db_surgeon.server.reward import RewardCalculator


class DBSurgeonEnvironment:
    """
    OpenEnv-compatible RL environment for database debugging.
    
    Each episode:
    1. Generates a fresh broken database scenario
    2. The agent inspects, diagnoses, and fixes the schema
    3. Reward is calculated based on execution results
    4. Episode ends on submit() or step limit
    
    Supports concurrent sessions via the SUPPORTS_CONCURRENT_SESSIONS flag.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 15

    def __init__(self, seed: Optional[int] = None):
        self._generator = BrokenDBGenerator(seed=seed)
        self._db: Optional[DBManager] = None
        self._oracle: Optional[EvaluationOracle] = None
        self._reward_calc: Optional[RewardCalculator] = None
        self._scenario: Optional[BrokenDBScenario] = None
        self._state: Optional[DBSurgeonState] = None
        self._action_history: list[str] = []

    def reset(self) -> DBSurgeonObservation:
        """
        Start a new episode with a fresh broken database.
        
        Returns:
            Initial observation showing the broken schema and failing query.
        """
        # Clean up previous episode
        if self._db:
            self._db.reset()

        # Generate new scenario
        self._scenario = self._generator.generate()

        # Create database with broken schema
        self._db = DBManager()
        self._db.create_database(
            self._scenario.schema_sql,
            self._scenario.seed_data_sql,
        )

        # Set up evaluation oracle
        self._oracle = EvaluationOracle(self._scenario.eval_queries)
        self._oracle.set_baseline(self._db)

        # Set up reward calculator
        self._reward_calc = RewardCalculator(
            db=self._db,
            oracle=self._oracle,
            business_query=self._scenario.business_query,
            root_cause=self._scenario.root_cause,
            involved_tables=self._scenario.involved_tables,
            max_steps=self.MAX_STEPS,
        )

        # Initialize state
        self._state = DBSurgeonState(
            episode_id=str(uuid.uuid4())[:8],
            step_count=0,
            initial_bug_type=self._scenario.bug_type,
            root_cause=self._scenario.root_cause,
            is_fixed=False,
            done=False,
            total_reward=0.0,
        )
        self._action_history = []

        # Get initial error from business query
        _, error_msg = self._db.execute_query(self._scenario.business_query)

        return DBSurgeonObservation(
            schema_snapshot=self._db.get_schema(),
            error_log=error_msg,
            failing_query=self._scenario.business_query,
            last_action_result="Environment reset. A broken database has been loaded. Diagnose the issue and fix it.",
            step_number=0,
            max_steps=self.MAX_STEPS,
            action_history=[],
        )

    def step(self, action: DBSurgeonAction) -> StepResult:
        """
        Execute an agent action and return the result.
        
        Args:
            action: The tool call to execute.
            
        Returns:
            StepResult with updated observation, reward, and done flag.
        """
        if self._state is None or self._state.done:
            return StepResult(
                observation=DBSurgeonObservation(
                    last_action_result="Error: Episode is over. Call reset() to start a new episode.",
                ),
                reward=0.0,
                done=True,
            )

        if self._state.step_count >= self.MAX_STEPS:
            self._state.done = True
            return StepResult(
                observation=DBSurgeonObservation(
                    last_action_result=f"Step limit ({self.MAX_STEPS}) reached. Episode ended.",
                    step_number=self._state.step_count,
                    max_steps=self.MAX_STEPS,
                ),
                reward=-1.0,
                done=True,
            )

        # Execute the action
        is_submit = action.tool_name == "submit"
        action_result = self._execute_action(action)
        action_success, action_output = action_result

        # Record action in history
        action_summary = f"[{self._state.step_count}] {action.tool_name}({self._format_args(action.arguments)})"
        if not action_success:
            action_summary += " → ERROR"
        self._action_history.append(action_summary)

        # Calculate reward
        reward = self._reward_calc.calculate(
            action=action,
            action_result=action_result,
            step_number=self._state.step_count,
            is_submit=is_submit,
        )

        # Update state
        self._state.step_count += 1
        self._state.total_reward += reward

        # Check if done
        if is_submit:
            self._state.done = True
            bq_pass, _ = self._db.execute_query(self._scenario.business_query)
            self._state.is_fixed = bq_pass
        elif self._reward_calc.should_force_done:
            self._state.done = True
            action_output += "\n\nToo many repeated actions. Episode ended."
        elif self._state.step_count >= self.MAX_STEPS:
            self._state.done = True
            action_output += f"\n\nStep limit ({self.MAX_STEPS}) reached. Episode ended."

        # Build observation
        _, error_msg = self._db.execute_query(self._scenario.business_query)
        observation = DBSurgeonObservation(
            schema_snapshot=self._db.get_schema(),
            error_log=error_msg if not self._state.is_fixed else "No errors.",
            failing_query=self._scenario.business_query,
            last_action_result=action_output,
            step_number=self._state.step_count,
            max_steps=self.MAX_STEPS,
            action_history=list(self._action_history),
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=self._state.done,
            info={
                "bug_type": self._scenario.bug_type,
                "bug_variant": self._scenario.bug_variant,
                "is_fixed": self._state.is_fixed,
                "total_reward": self._state.total_reward,
            },
        )

    def state(self) -> DBSurgeonState:
        """Return current episode state metadata."""
        if self._state is None:
            return DBSurgeonState()
        return self._state

    # ─── Action execution ───────────────────────────────────────

    def _execute_action(self, action: DBSurgeonAction) -> tuple[bool, str]:
        """
        Route and execute an action against the database.
        
        Returns:
            (success, result_or_error) tuple.
        """
        tool = action.tool_name
        args = action.arguments

        if tool == "inspect_schema":
            return self._action_inspect_schema(args)
        elif tool == "run_query":
            return self._action_run_query(args)
        elif tool == "fix_column":
            return self._action_fix_column(args)
        elif tool == "add_index":
            return self._action_add_index(args)
        elif tool == "add_constraint":
            return self._action_add_constraint(args)
        elif tool == "execute_fix":
            return self._action_execute_fix(args)
        elif tool == "submit":
            return self._action_submit()
        else:
            return False, f"Unknown tool: '{tool}'. Available tools: inspect_schema, run_query, fix_column, add_index, add_constraint, execute_fix, submit"

    def _action_inspect_schema(self, args: dict) -> tuple[bool, str]:
        table_name = args.get("table_name", "")
        if table_name:
            result = self._db.get_table_info(table_name)
        else:
            result = self._db.get_schema()
            tables = self._db.get_table_names()
            result = f"Tables in database: {', '.join(tables)}\n\n{result}"
        return True, result

    def _action_run_query(self, args: dict) -> tuple[bool, str]:
        sql = args.get("sql", "")
        if not sql:
            return False, "Error: 'sql' argument is required."
        return self._db.execute_query(sql)

    def _action_fix_column(self, args: dict) -> tuple[bool, str]:
        table_name = args.get("table_name", "")
        column_name = args.get("column_name", "")
        new_type = args.get("new_type", "")
        new_name = args.get("new_name", "")

        if not table_name or not column_name:
            return False, "Error: 'table_name' and 'column_name' are required."

        return self._db.fix_column(table_name, column_name, new_type, new_name)

    def _action_add_index(self, args: dict) -> tuple[bool, str]:
        table_name = args.get("table_name", "")
        column_name = args.get("column_name", "")
        if not table_name or not column_name:
            return False, "Error: 'table_name' and 'column_name' are required."
        return self._db.add_index(table_name, column_name)

    def _action_add_constraint(self, args: dict) -> tuple[bool, str]:
        table_name = args.get("table_name", "")
        constraint_type = args.get("constraint_type", "")
        column_name = args.get("column_name", "")
        reference = args.get("reference", "")

        if not table_name or not constraint_type or not column_name:
            return False, "Error: 'table_name', 'constraint_type', and 'column_name' are required."

        # Build and execute the constraint DDL
        if constraint_type.upper() == "FOREIGN_KEY":
            if not reference:
                return False, "Error: 'reference' is required for FOREIGN_KEY (format: table.column)."
            ref_parts = reference.split(".")
            if len(ref_parts) != 2:
                return False, "Error: 'reference' must be in format 'table_name.column_name'."
            # SQLite doesn't support ADD CONSTRAINT on existing tables easily
            # We'd need table recreation — use execute_fix for complex cases
            return False, "Error: Adding FK constraints to existing tables requires table recreation. Use 'execute_fix' with a full DDL statement instead."

        elif constraint_type.upper() == "UNIQUE":
            idx_name = f"uq_{table_name}_{column_name}"
            return self._db.execute_ddl(
                f"CREATE UNIQUE INDEX {idx_name} ON {table_name}({column_name});"
            )

        else:
            return False, f"Error: Unsupported constraint type '{constraint_type}'. Use FOREIGN_KEY or UNIQUE."

    def _action_execute_fix(self, args: dict) -> tuple[bool, str]:
        sql = args.get("sql", "")
        if not sql:
            return False, "Error: 'sql' argument is required."
        return self._db.execute_ddl(sql)

    def _action_submit(self) -> tuple[bool, str]:
        """Score the current database state and end the episode."""
        eval_result = self._oracle.detailed_score(self._db)
        bq_pass, bq_result = self._db.execute_query(self._scenario.business_query)

        lines = ["=== SUBMISSION RESULT ==="]
        lines.append(f"Business Query: {'PASS ✓' if bq_pass else 'FAIL ✗'}")
        lines.append(f"Evaluation Score: {eval_result['passed']}/{eval_result['total']} queries passed")
        lines.append(f"Overall Score: {eval_result['score']:.1%}")
        
        if eval_result['regressions'] > 0:
            lines.append(f"⚠ Regressions: {eval_result['regressions']} previously-passing queries now fail")

        if bq_pass:
            lines.append("\n🎉 Database fixed successfully!")
        else:
            lines.append(f"\n❌ Business query still failing: {bq_result[:200]}")

        return bq_pass, "\n".join(lines)

    # ─── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _format_args(args: dict) -> str:
        """Format action arguments for display."""
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            val_str = str(v)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            parts.append(f"{k}={val_str}")
        return ", ".join(parts)
