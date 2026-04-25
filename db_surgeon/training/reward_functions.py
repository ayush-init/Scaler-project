"""
Reward Functions — Used by TRL GRPOTrainer to score episodes.

These functions read accumulated reward from the environment instances
after episodes complete.
"""

from __future__ import annotations


def reward_func(prompts, completions, **kwargs) -> list[float]:
    """
    Read accumulated reward from each environment instance.
    
    This is the primary reward function passed to GRPOTrainer.
    It simply reads the reward that was accumulated during the
    multi-turn episode from the environment wrapper.
    
    Args:
        prompts: List of prompt strings/messages from trainer.
        completions: List of completion strings from trainer.
        **kwargs: Additional data from GRPOTrainer including
                  'environments' (list of DBSurgeonToolEnv instances).
        
    Returns:
        List of float rewards, one per environment.
    """
    environments = kwargs.get("environments", [])
    if environments:
        rewards = []
        for env in environments:
            rewards.append(getattr(env, "reward", 0.0))
        return rewards
    
    # Fallback: if no environments, return zeros
    return [0.0] * len(completions)


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Bonus reward for proper tool-calling format.
    
    Gives a small bonus if the completion contains structured
    tool calls (function-calling JSON), penalizes free-text-only
    responses that don't interact with the environment.
    
    Args:
        completions: List of completion strings.
        **kwargs: Additional data.
        
    Returns:
        List of format bonus rewards.
    """
    rewards = []
    for completion in completions:
        content = completion if isinstance(completion, str) else str(completion)
        # Check if completion contains tool call patterns
        if "function" in content.lower() or "tool_call" in content.lower():
            rewards.append(0.5)  # Small bonus for using tools
        elif len(content.strip()) < 10:
            rewards.append(-0.5)  # Penalty for empty/trivial responses
        else:
            rewards.append(0.0)
    return rewards
