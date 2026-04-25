"""Quick check: Is training done?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
TOKEN = None
with open(ENV_PATH) as f:
    for line in f:
        if line.strip().startswith("HF_TOKEN="):
            TOKEN = line.strip().split("=", 1)[1].strip()

import httpx

print("\n[DB-Surgeon] Training Status Check")
print("=" * 45)

# Check Space runtime
from huggingface_hub import HfApi
api = HfApi(token=TOKEN)
info = api.get_space_runtime("ayush0211/db-surgeon")
print(f"  Hardware : {info.hardware}")
print(f"  Stage    : {info.stage}")

# Check Gradio UI
try:
    r = httpx.get("https://ayush0211-db-surgeon.hf.space/", timeout=15, follow_redirects=True)
    if r.status_code == 200:
        print(f"  Gradio   : OK - Responding")
    else:
        print(f"  Gradio   : FAIL - HTTP {r.status_code}")
except Exception as e:
    print(f"  Gradio   : FAIL - {e}")

print()
print(">> To see training logs: Open the Space -> Training tab -> Refresh Status")
print(">> When done, run: python setup_gpu_and_train.py --downgrade-only")
print()

if info.hardware == "cpu-basic":
    print("INFO: GPU is OFF. Training is either done or was never started.")
elif info.stage == "RUNNING":
    print("ACTIVE: Space is RUNNING on GPU. Training may still be in progress.")
    print("   Check the Training tab in the browser for live logs.")
else:
    print(f"WARNING: Space is in {info.stage} state.")
