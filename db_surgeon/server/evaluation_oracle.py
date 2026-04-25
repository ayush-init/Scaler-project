"""
Evaluation Oracle — Hidden query set the agent never sees directly.

Provides objective scoring of the database state by running a set
of evaluation queries that test different aspects of correctness:
- Business query (the one the agent sees)
- JOIN queries across affected tables
- INSERT queries testing constraints
- Aggregate queries testing data integrity
"""

from __future__ import annotations

from db_surgeon.server.db_manager import DBManager


class EvaluationOracle:
    """
    Scores the database state using hidden evaluation queries.
    
    The agent only sees the business query. The oracle runs additional
    queries that the agent doesn't know about, making it much harder
    to game the reward.
    """

    def __init__(self, eval_queries: list[str]):
        """
        Args:
            eval_queries: List of SQL queries to test. The first is
                         typically the visible business query; the rest
                         are hidden from the agent.
        """
        self.eval_queries = eval_queries
        self._baseline_results: list[bool] = []

    def set_baseline(self, db: DBManager) -> None:
        """
        Record which queries pass BEFORE the agent acts.
        
        This lets us detect regressions (queries that were passing
        but now fail after the agent's actions).
        
        Args:
            db: The database manager with the initial (broken) schema.
        """
        self._baseline_results = []
        for q in self.eval_queries:
            success, _ = db.execute_query(q)
            self._baseline_results.append(success)

    def score(self, db: DBManager) -> float:
        """
        Run all eval queries and return fraction that pass.
        
        Args:
            db: The database manager with current (possibly fixed) schema.
            
        Returns:
            Score from 0.0 to 1.0.
        """
        if not self.eval_queries:
            return 0.0

        passed = 0
        for q in self.eval_queries:
            success, _ = db.execute_query(q)
            if success:
                passed += 1

        return passed / len(self.eval_queries)

    def detailed_score(self, db: DBManager) -> dict:
        """
        Run all eval queries and return detailed per-query results.
        
        Returns:
            Dict with 'score', 'passed', 'total', 'details', and 'regressions'.
        """
        if not self.eval_queries:
            return {"score": 0.0, "passed": 0, "total": 0, "details": [], "regressions": 0}

        details = []
        passed = 0
        regressions = 0

        for i, q in enumerate(self.eval_queries):
            success, result = db.execute_query(q)
            was_passing = (
                self._baseline_results[i]
                if i < len(self._baseline_results)
                else False
            )
            
            is_regression = was_passing and not success
            if is_regression:
                regressions += 1

            details.append({
                "query": q[:100],
                "passed": success,
                "was_passing": was_passing,
                "is_regression": is_regression,
                "error": "" if success else result[:200],
            })
            
            if success:
                passed += 1

        return {
            "score": passed / len(self.eval_queries),
            "passed": passed,
            "total": len(self.eval_queries),
            "details": details,
            "regressions": regressions,
        }

    def count_regressions(self, db: DBManager) -> int:
        """
        Count how many previously-passing queries now fail.
        
        This is used to apply the regression penalty (-3.0).
        """
        if not self._baseline_results:
            return 0

        regressions = 0
        for i, q in enumerate(self.eval_queries):
            if i >= len(self._baseline_results):
                break
            if self._baseline_results[i]:
                success, _ = db.execute_query(q)
                if not success:
                    regressions += 1

        return regressions

    def update_baseline(self, db: DBManager) -> None:
        """
        Update the baseline to the current state.
        
        Called after each step so regressions are detected
        relative to the previous state.
        """
        self.set_baseline(db)
