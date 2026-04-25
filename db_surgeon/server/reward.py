"""
Reward Calculator — Multi-component reward system for DB-Surgeon.

Implements the reward formula from rule.md §8 with:
- Outcome rewards (query passes, eval score)
- Process rewards (diagnostics, efficiency)
- Penalties (invalid SQL, breaking things, repeated actions)
- Anti-hacking (hidden eval queries, regression detection)
"""

from __future__ import annotations

from typing import Optional

from db_surgeon.models import DBSurgeonAction
from db_surgeon.server.db_manager import DBManager
from db_surgeon.server.evaluation_oracle import EvaluationOracle


class RewardCalculator:
    """
    Calculates multi-component rewards for each step.
    
    The reward is designed to be:
    - Multi-signal: no single action maximizes all components
    - Verifiable: based on SQL execution, not text quality
    - Anti-hackable: hidden eval queries + regression detection
    """

    # Reward constants
    BUSINESS_QUERY_PASS = 5.0
    EVAL_SCORE_MULTIPLIER = 5.0
    CAUSAL_FIX_BONUS = 3.0
    PARTIAL_IMPROVEMENT = 2.0
    GOOD_DIAGNOSTIC = 1.0
    EFFICIENCY_BONUS = 1.0

    INVALID_SQL_PENALTY = -1.0
    REGRESSION_PENALTY = -3.0
    REPEATED_ACTION_PENALTY = -1.0
    STEP_TAX = -0.1

    def __init__(
        self,
        db: DBManager,
        oracle: EvaluationOracle,
        business_query: str,
        root_cause: str,
        involved_tables: list[str],
        max_steps: int = 15,
    ):
        self._db = db
        self._oracle = oracle
        self._business_query = business_query
        self._root_cause = root_cause.lower()
        self._involved_tables = [t.lower() for t in involved_tables]
        self._max_steps = max_steps

        # State tracking
        self._business_query_was_passing = False
        self._prev_eval_score = 0.0
        self._action_history: list[str] = []
        self._consecutive_repeats = 0

        # Initialize baseline
        bq_pass, _ = db.execute_query(business_query)
        self._business_query_was_passing = bq_pass
        self._prev_eval_score = oracle.score(db)

    def calculate(
        self,
        action: DBSurgeonAction,
        action_result: tuple[bool, str],
        step_number: int,
        is_submit: bool = False,
    ) -> float:
        """
        Calculate reward for a single step.
        
        Args:
            action: The action that was executed.
            action_result: (success, result_string) from execution.
            step_number: Current step number (0-indexed).
            is_submit: Whether this is a submit action.
            
        Returns:
            Reward value for this step.
        """
        reward = 0.0
        action_success, action_output = action_result
        action_key = self._action_key(action)

        # ─── OUTCOME REWARDS ───

        # Business query now passes
        bq_pass, _ = self._db.execute_query(self._business_query)
        if bq_pass and not self._business_query_was_passing:
            reward += self.BUSINESS_QUERY_PASS
            self._business_query_was_passing = True

        # On submit: score all eval queries
        if is_submit:
            eval_score = self._oracle.score(self._db)
            reward += eval_score * self.EVAL_SCORE_MULTIPLIER

        # ─── PROCESS REWARDS ───

        # Causal fix — action addresses root cause
        if action_success and self._is_causal_fix(action):
            reward += self.CAUSAL_FIX_BONUS

        # Partial improvement — eval score went up
        current_eval = self._oracle.score(self._db)
        if current_eval > self._prev_eval_score:
            reward += self.PARTIAL_IMPROVEMENT

        # Good diagnostic — inspected a relevant table
        if action.tool_name == "inspect_schema" and self._is_relevant_table(action):
            reward += self.GOOD_DIAGNOSTIC

        # Efficiency bonus — solved early
        if bq_pass and step_number < self._max_steps // 2:
            reward += self.EFFICIENCY_BONUS

        # ─── PENALTIES ───

        # Invalid SQL
        if not action_success and action.tool_name in ("run_query", "execute_fix", "fix_column"):
            reward += self.INVALID_SQL_PENALTY

        # Regression — broke something that was working
        regressions = self._oracle.count_regressions(self._db)
        if regressions > 0:
            reward += self.REGRESSION_PENALTY * regressions

        # Repeated action
        if self._is_repeated(action_key):
            reward += self.REPEATED_ACTION_PENALTY

        # Step tax
        reward += self.STEP_TAX

        # ─── Update tracking state ───
        self._action_history.append(action_key)
        self._prev_eval_score = current_eval
        self._oracle.update_baseline(self._db)

        return reward

    # ─── Private helpers ────────────────────────────────────────

    def _action_key(self, action: DBSurgeonAction) -> str:
        """Create a hashable key for an action."""
        args_str = str(sorted(action.arguments.items()))
        return f"{action.tool_name}:{args_str}"

    def _is_causal_fix(self, action: DBSurgeonAction) -> bool:
        """
        Check if the action directly addresses the root cause.
        
        Uses keyword matching against the root cause description.
        This is a heuristic — the eval oracle provides the ground truth.
        """
        if action.tool_name not in ("fix_column", "execute_fix", "add_constraint"):
            return False

        # Check if tool arguments reference involved tables/columns
        args_str = str(action.arguments).lower()
        for table in self._involved_tables:
            # Strip prefix for matching
            base_name = table.split("_", 2)[-1] if "_" in table else table
            if base_name in args_str or table in args_str:
                return True

        return False

    def _is_relevant_table(self, action: DBSurgeonAction) -> bool:
        """Check if the agent is inspecting a table involved in the bug."""
        table_arg = action.arguments.get("table_name", "").lower()
        if not table_arg:
            # Inspecting all tables is also somewhat useful
            return True
        return any(table_arg in t for t in self._involved_tables)

    def _is_repeated(self, action_key: str) -> bool:
        """Check if this exact action was already taken."""
        if action_key in self._action_history:
            self._consecutive_repeats += 1
            return True
        self._consecutive_repeats = 0
        return False

    @property
    def should_force_done(self) -> bool:
        """Whether to force the episode to end due to too many repeats."""
        return self._consecutive_repeats >= 3
