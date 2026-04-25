"""
Metrics Visualization — Plots reward curves and training metrics.

Generates publication-quality plots for the hackathon demo:
- Reward vs. training steps (smoothed)
- Success rate vs. training steps
- Tool call distribution
- Before/after comparison

Usage:
    python -m db_surgeon.metrics.plot_rewards
"""

from __future__ import annotations

import json
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 12


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Apply moving average smoothing."""
    if len(values) <= window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2 + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def plot_reward_curve(
    rewards: list[float],
    title: str = "DB-Surgeon: Reward Over Training",
    output_path: str = "metrics/results/reward_curve.png",
    baseline_reward: float | None = None,
):
    """Plot reward vs. training steps with smoothing."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw rewards (transparent)
    ax.plot(rewards, alpha=0.2, color="#6366F1", linewidth=0.5, label="Raw")
    
    # Smoothed rewards
    smoothed = smooth(rewards, window=max(len(rewards) // 20, 5))
    ax.plot(smoothed, color="#6366F1", linewidth=2.5, label="Smoothed (moving avg)")

    # Baseline
    if baseline_reward is not None:
        ax.axhline(
            y=baseline_reward, color="#EF4444", linestyle="--",
            linewidth=1.5, label=f"Random baseline ({baseline_reward:.1f})",
        )

    # Zero line
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Training Episode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_success_rate(
    successes: list[bool],
    title: str = "DB-Surgeon: Success Rate Over Training",
    output_path: str = "metrics/results/success_rate.png",
    window: int = 20,
):
    """Plot rolling success rate."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Rolling success rate
    rates = []
    for i in range(len(successes)):
        start = max(0, i - window + 1)
        window_slice = successes[start:i + 1]
        rates.append(sum(window_slice) / len(window_slice))

    ax.fill_between(range(len(rates)), rates, alpha=0.2, color="#10B981")
    ax.plot(rates, color="#10B981", linewidth=2.5, label=f"Rolling success rate (window={window})")

    ax.set_xlabel("Training Episode", fontsize=14, fontweight="bold")
    ax.set_ylabel("Success Rate", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison(
    baseline_rewards: list[float],
    trained_rewards: list[float],
    title: str = "DB-Surgeon: Baseline vs. Trained Agent",
    output_path: str = "metrics/results/comparison.png",
):
    """Plot side-by-side comparison of baseline vs. trained."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline histogram
    axes[0].hist(baseline_rewards, bins=15, color="#EF4444", alpha=0.7, edgecolor="white")
    axes[0].axvline(
        sum(baseline_rewards) / len(baseline_rewards),
        color="#991B1B", linewidth=2, linestyle="--",
        label=f"Mean: {sum(baseline_rewards)/len(baseline_rewards):.1f}",
    )
    axes[0].set_title("Random Baseline", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Episode Reward")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    # Trained histogram
    axes[1].hist(trained_rewards, bins=15, color="#6366F1", alpha=0.7, edgecolor="white")
    axes[1].axvline(
        sum(trained_rewards) / len(trained_rewards),
        color="#312E81", linewidth=2, linestyle="--",
        label=f"Mean: {sum(trained_rewards)/len(trained_rewards):.1f}",
    )
    axes[1].set_title("Trained Agent", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Episode Reward")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def generate_demo_plots():
    """Generate demo plots with simulated data (for hackathon presentation setup)."""
    print("📊 Generating demo plots with simulated data...")

    # Simulated reward trajectory (showing improvement)
    np.random.seed(42)
    n = 200
    base = np.linspace(-3, 8, n)
    noise = np.random.normal(0, 2, n)
    rewards = (base + noise).tolist()

    plot_reward_curve(
        rewards,
        title="DB-Surgeon: Reward Over Training (Simulated)",
        output_path="metrics/results/demo_reward_curve.png",
        baseline_reward=-2.5,
    )

    # Simulated success rate
    successes = [
        bool(np.random.random() < min(0.8, max(0.0, 0.01 * i - 0.5)))
        for i in range(n)
    ]
    plot_success_rate(
        successes,
        title="DB-Surgeon: Success Rate Over Training (Simulated)",
        output_path="metrics/results/demo_success_rate.png",
    )

    # Comparison
    baseline = np.random.normal(-2.5, 1.5, 50).tolist()
    trained = np.random.normal(6.0, 3.0, 50).tolist()
    plot_comparison(
        baseline, trained,
        title="DB-Surgeon: Random vs. Trained Agent (Simulated)",
        output_path="metrics/results/demo_comparison.png",
    )

    print("✅ Demo plots generated in metrics/results/")


def main():
    parser = argparse.ArgumentParser(description="Plot DB-Surgeon metrics")
    parser.add_argument("--results", help="Path to results JSON file")
    parser.add_argument("--demo", action="store_true", help="Generate demo plots with simulated data")
    args = parser.parse_args()

    if args.demo:
        generate_demo_plots()
    elif args.results:
        with open(args.results) as f:
            data = json.load(f)
        rewards = data.get("rewards", [])
        if rewards:
            plot_reward_curve(rewards)
            successes = [r > 0 for r in rewards]
            plot_success_rate(successes)
    else:
        print("Use --demo for simulated plots or --results <path> for real data")


if __name__ == "__main__":
    main()
