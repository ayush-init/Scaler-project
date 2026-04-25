"""
DB-Surgeon Client — OpenEnv client for connecting to the environment server.

Provides both sync and async interfaces for interacting with
the DB-Surgeon environment via HTTP.
"""

from __future__ import annotations

import requests
from typing import Optional

from db_surgeon.models import (
    DBSurgeonAction,
    DBSurgeonObservation,
    DBSurgeonState,
    StepResult,
)


class DBSurgeonEnv:
    """
    Client for connecting to the DB-Surgeon environment server.
    
    Supports HTTP-based communication for simplicity.
    For production/training, use the WebSocket-based OpenEnv client.
    
    Usage:
        env = DBSurgeonEnv(base_url="http://localhost:7860")
        obs = env.reset()
        result = env.step(DBSurgeonAction(tool_name="inspect_schema", arguments={}))
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self._base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(self) -> StepResult:
        """Reset the environment and get the initial observation."""
        resp = self._session.post(f"{self._base_url}/reset")
        resp.raise_for_status()
        data = resp.json()
        obs = self._parse_observation(data["observation"])
        return StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action: DBSurgeonAction) -> StepResult:
        """Execute an action and get the result."""
        resp = self._session.post(
            f"{self._base_url}/step",
            json={
                "tool_name": action.tool_name,
                "arguments": action.arguments,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        obs = self._parse_observation(data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward", 0.0),
            done=data.get("done", False),
            info=data.get("info", {}),
        )

    def state(self) -> DBSurgeonState:
        """Get the current environment state."""
        resp = self._session.get(f"{self._base_url}/state")
        resp.raise_for_status()
        data = resp.json()["state"]
        return DBSurgeonState(
            episode_id=data.get("episode_id", ""),
            step_count=data.get("step_count", 0),
            initial_bug_type=data.get("initial_bug_type", ""),
            root_cause=data.get("root_cause", ""),
            is_fixed=data.get("is_fixed", False),
            done=data.get("done", False),
            total_reward=data.get("total_reward", 0.0),
        )

    def health(self) -> dict:
        """Check if the server is running."""
        resp = self._session.get(f"{self._base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _parse_observation(data: dict) -> DBSurgeonObservation:
        return DBSurgeonObservation(
            schema_snapshot=data.get("schema_snapshot", ""),
            error_log=data.get("error_log", ""),
            failing_query=data.get("failing_query", ""),
            last_action_result=data.get("last_action_result", ""),
            step_number=data.get("step_number", 0),
            max_steps=data.get("max_steps", 15),
            action_history=data.get("action_history", []),
        )


class DBSurgeonLocalEnv:
    """
    Local (in-process) client that skips HTTP and talks directly
    to the environment. Used for training when the server and
    trainer run in the same process.
    """

    def __init__(self, seed: Optional[int] = None):
        from db_surgeon.server.db_surgeon_environment import DBSurgeonEnvironment
        self._env = DBSurgeonEnvironment(seed=seed)

    def reset(self) -> StepResult:
        obs = self._env.reset()
        return StepResult(observation=obs, reward=0.0, done=False)

    def step(self, action: DBSurgeonAction) -> StepResult:
        return self._env.step(action)

    def state(self) -> DBSurgeonState:
        return self._env.state()

    def close(self):
        if self._env._db:
            self._env._db.reset()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
