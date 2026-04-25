"""
DB-Surgeon Environment Models

Defines the typed data structures for Action, Observation, and State
following the OpenEnv specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DBSurgeonAction:
    """
    Agent action — represents a tool call with arguments.
    
    The agent sends structured actions to the environment via tool calls.
    Each action maps to a specific database operation.
    """
    tool_name: str
    """One of: inspect_schema, run_query, fix_column, add_index,
    add_constraint, execute_fix, submit"""
    
    arguments: dict = field(default_factory=dict)
    """Tool-specific arguments as key-value pairs."""


@dataclass
class DBSurgeonObservation:
    """
    What the agent sees after each action.
    
    Contains the current database state, error information,
    the failing business query, and episode progress.
    """
    schema_snapshot: str = ""
    """Current CREATE TABLE statements for all tables."""
    
    error_log: str = ""
    """Recent SQL errors or diagnostic messages."""
    
    failing_query: str = ""
    """The business query that must execute successfully."""
    
    last_action_result: str = ""
    """Output or error from the most recent tool call."""
    
    step_number: int = 0
    """Current step within the episode (0-indexed)."""
    
    max_steps: int = 15
    """Maximum steps allowed per episode."""
    
    action_history: list[str] = field(default_factory=list)
    """Summary of all actions taken so far in this episode."""


@dataclass
class DBSurgeonState:
    """
    Internal environment state for tracking and debugging.
    
    Contains ground-truth information about the episode that
    the agent should NOT see directly.
    """
    episode_id: str = ""
    """Unique identifier for this episode."""
    
    step_count: int = 0
    """Number of steps taken so far."""
    
    initial_bug_type: str = ""
    """Category of the injected bug (e.g., 'fk_violation')."""
    
    root_cause: str = ""
    """Human-readable description of the exact fix needed."""
    
    is_fixed: bool = False
    """Whether the database has been successfully repaired."""
    
    done: bool = False
    """Whether the episode has ended (submit called or step limit reached)."""
    
    total_reward: float = 0.0
    """Accumulated reward across all steps in this episode."""


@dataclass
class StepResult:
    """
    Result returned by environment.step().
    
    Bundles the new observation, reward, and done flag.
    """
    observation: DBSurgeonObservation
    """The updated observation after executing the action."""
    
    reward: float = 0.0
    """Reward signal for this step."""
    
    done: bool = False
    """Whether the episode has ended."""
    
    info: dict = field(default_factory=dict)
    """Additional diagnostic information."""
