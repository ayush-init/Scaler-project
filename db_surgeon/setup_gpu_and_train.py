"""
DB-Surgeon — Upgrade HF Space to GPU & Trigger Training

This script:
1. Upgrades your HF Space hardware to Nvidia T4 GPU
2. Waits for the Space to boot up
3. Triggers training via the Gradio API
4. Monitors training progress
5. Downgrades back to free CPU when done to save credits
"""

import os
import sys
import time

# ─── Read token from .env ───
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
TOKEN = None
if os.path.exists(ENV_PATH):
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                TOKEN = line.split("=", 1)[1].strip()

if not TOKEN:
    print("ERROR: No HF_TOKEN found in e:\\Scaler\\.env")
    print("Add a line like: HF_TOKEN=hf_xxxxx")
    sys.exit(1)

REPO_ID = "ayush0211/db-surgeon"
SPACE_URL = "https://ayush0211-db-surgeon.hf.space"

# ─── Import huggingface_hub ───
try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import HfApi

api = HfApi(token=TOKEN)


def get_space_status():
    """Get current Space runtime status."""
    try:
        info = api.get_space_runtime(REPO_ID)
        return info
    except Exception as e:
        print(f"  Warning: Could not get status: {e}")
        return None


def upgrade_to_gpu(hardware="t4-small"):
    """Upgrade the Space hardware to GPU."""
    print(f"\n{'='*60}")
    print(f"  STEP 1: Upgrading Space to {hardware.upper()}")
    print(f"{'='*60}")

    current = get_space_status()
    if current:
        print(f"  Current hardware : {current.hardware}")
        print(f"  Current stage    : {current.stage}")

    print(f"\n  Requesting upgrade to: {hardware}")
    try:
        api.request_space_hardware(REPO_ID, hardware)
        print(f"  Hardware upgrade requested!")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Wait for Space to start building/running
    print(f"\n  Waiting for Space to boot with GPU...")
    print(f"  (This takes 2-5 minutes for first GPU boot)\n")

    for i in range(60):  # Wait up to 10 minutes
        time.sleep(10)
        info = get_space_status()
        if info:
            stage = info.stage
            hw = info.hardware
            print(f"  [{i*10:>3}s] Stage: {stage:<15} Hardware: {hw}")
            if stage == "RUNNING":
                print(f"\n  Space is RUNNING on {hw}!")
                return True
            elif stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
                print(f"\n  ERROR: Space entered {stage} state!")
                return False

    print("\n  Timeout waiting for Space to start.")
    return False


def wait_for_gradio():
    """Wait for the Gradio app to be responsive."""
    print(f"\n{'='*60}")
    print(f"  STEP 2: Waiting for Gradio API to respond")
    print(f"{'='*60}")

    try:
        import httpx
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
        import httpx

    for i in range(30):  # Wait up to 5 minutes
        try:
            r = httpx.get(f"{SPACE_URL}/", timeout=15, follow_redirects=True)
            if r.status_code == 200 and ("gradio" in r.text.lower() or "DB-Surgeon" in r.text):
                print(f"  Gradio API is ready!")
                return True
        except Exception:
            pass
        print(f"  [{i*10:>3}s] Waiting for Gradio...")
        time.sleep(10)

    print("  Timeout waiting for Gradio.")
    return False


def trigger_training(episodes=200, model="Qwen/Qwen3-0.6B", lr=5e-6):
    """Trigger training via the Gradio API."""
    print(f"\n{'='*60}")
    print(f"  STEP 3: Triggering GRPO Training")
    print(f"{'='*60}")
    print(f"  Model    : {model}")
    print(f"  Episodes : {episodes}")
    print(f"  LR       : {lr}")

    try:
        from gradio_client import Client
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio_client"])
        from gradio_client import Client

    try:
        client = Client(SPACE_URL)
        print(f"\n  Connected to Gradio API!")

        # Call the training function (fn_index for train_btn.click)
        result = client.predict(
            episodes,   # num_episodes
            model,      # model_name
            lr,         # learning_rate
            api_name="/run_training" if hasattr(client, 'view_api') else None,
            fn_index=5,  # train_btn is the 6th click handler (0-indexed)
        )
        print(f"  Training triggered! Response: {result}")
        return True
    except Exception as e:
        print(f"  Could not trigger via API: {e}")
        print(f"\n  FALLBACK: Please go to the Space UI and click 'Start Training' manually:")
        print(f"  {SPACE_URL}")
        return False


def monitor_training():
    """Monitor training progress by polling status."""
    print(f"\n{'='*60}")
    print(f"  STEP 4: Monitoring Training Progress")
    print(f"{'='*60}")
    print(f"  (Press Ctrl+C to stop monitoring — training continues on HF)\n")

    try:
        from gradio_client import Client
    except ImportError:
        return

    try:
        client = Client(SPACE_URL)
    except Exception:
        print("  Could not connect for monitoring.")
        return

    last_log = ""
    for i in range(720):  # Monitor for up to 4 hours
        try:
            # Call get_training_status (fn_index for status_btn.click)
            status, log = client.predict(
                api_name="/get_training_status" if hasattr(client, 'view_api') else None,
                fn_index=6,
            )
            if log != last_log:
                # Print new log lines
                new_lines = log[len(last_log):].strip()
                if new_lines:
                    for line in new_lines.split("\n"):
                        print(f"  [{time.strftime('%H:%M:%S')}] {line}")
                last_log = log

            if "Training Complete" in status or "complete" in status.lower():
                print(f"\n  TRAINING COMPLETE!")
                return True
            if "Error" in status:
                print(f"\n  TRAINING ERROR: {status}")
                return False

        except KeyboardInterrupt:
            print(f"\n  Monitoring stopped. Training continues on HF.")
            print(f"  Check status at: {SPACE_URL}")
            return None
        except Exception as e:
            print(f"  [{time.strftime('%H:%M:%S')}] Monitor error: {e}")

        time.sleep(20)

    return None


def downgrade_to_cpu():
    """Downgrade Space back to free CPU to save credits."""
    print(f"\n{'='*60}")
    print(f"  STEP 5: Downgrading to Free CPU")
    print(f"{'='*60}")

    try:
        api.request_space_hardware(REPO_ID, "cpu-basic")
        print(f"  Hardware downgraded to cpu-basic (FREE)")
        print(f"  Your credits are safe!")
    except Exception as e:
        print(f"  WARNING: Could not downgrade: {e}")
        print(f"  Go to Settings manually and switch to CPU Basic!")
        print(f"  https://huggingface.co/spaces/{REPO_ID}/settings")


def main():
    print(r"""
    ╔═══════════════════════════════════════════════════════════╗
    ║          🏥 DB-SURGEON — HF Training Pipeline            ║
    ║                                                           ║
    ║  This script will:                                        ║
    ║   1. Upgrade your Space to T4 GPU ($0.40/hr)              ║
    ║   2. Wait for it to boot                                  ║
    ║   3. Trigger GRPO training (200 episodes)                 ║
    ║   4. Monitor progress                                     ║
    ║   5. Downgrade back to FREE CPU when done                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    user = api.whoami()
    print(f"  Authenticated as: {user['name']}")
    print(f"  Space: {REPO_ID}")

    # --- Parse args ---
    episodes = 200
    hardware = "t4-small"
    model = "Qwen/Qwen3-0.6B"

    for arg in sys.argv[1:]:
        if arg.startswith("--episodes="):
            episodes = int(arg.split("=")[1])
        elif arg.startswith("--hardware="):
            hardware = arg.split("=")[1]
        elif arg.startswith("--model="):
            model = arg.split("=")[1]
        elif arg == "--skip-upgrade":
            hardware = None
        elif arg == "--downgrade-only":
            downgrade_to_cpu()
            return

    print(f"  Episodes: {episodes}")
    print(f"  Hardware: {hardware or 'skip'}")
    print(f"  Model: {model}")

    # Step 1: Upgrade GPU
    if hardware:
        ok = upgrade_to_gpu(hardware)
        if not ok:
            print("\n  Failed to upgrade. Aborting.")
            return

    # Step 2: Wait for Gradio
    ok = wait_for_gradio()
    if not ok:
        print("\n  Gradio not responding. Check the Space logs.")
        print(f"  URL: {SPACE_URL}")
        return

    # Step 3: Trigger training
    trigger_training(episodes=episodes, model=model)

    # Step 4: Monitor
    print(f"\n  You can also monitor at: {SPACE_URL}")
    result = monitor_training()

    # Step 5: Downgrade
    if result is True:
        print("\n  Training completed successfully!")
        ans = input("\n  Downgrade to free CPU now? (y/n): ").strip().lower()
        if ans == "y":
            downgrade_to_cpu()
        else:
            print(f"  Remember to downgrade manually to save credits!")
            print(f"  https://huggingface.co/spaces/{REPO_ID}/settings")
    elif result is None:
        print(f"\n  Training is still running on HF.")
        print(f"  When done, run: python setup_gpu_and_train.py --downgrade-only")
    else:
        print(f"\n  Something went wrong. Check the Space logs.")
        ans = input("\n  Downgrade to free CPU now? (y/n): ").strip().lower()
        if ans == "y":
            downgrade_to_cpu()


if __name__ == "__main__":
    main()
