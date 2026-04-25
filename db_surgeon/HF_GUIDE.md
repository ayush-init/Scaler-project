# DB-SURGEON — HuggingFace $30 Budget Guide

You have $30 in HuggingFace credits. This gives you plenty of compute! 
Here is exactly how to spend it to train your model and host your project, step-by-step.

---

## The Strategy

We will use two HuggingFace features:
1. **HF Jobs (or Spaces Training Tab)** to run the long Unsloth GRPO training on a GPU.
2. **HF Spaces** to host the final environment GUI for the Hackathon judges.

---

## STEP 1: Push the Application to a HuggingFace Space

First, we need to push your local code up to HuggingFace. We generated a Gradio app that handles both training and the interactive demo.

1. **Log into HuggingFace** via terminal on your local machine:
   ```bash
   huggingface-cli login
   ```
   *(Paste your token from https://huggingface.co/settings/tokens. Make sure it has "Write" permissions).*

2. **Initialize a Git repository and push** to a new HF Space. 
   Go to **[HuggingFace Spaces](https://huggingface.co/spaces)** and click **Create new Space**.
   - **Name:** `db-surgeon`
   - **License:** MIT (or whatever you prefer)
   - **SDK:** Gradio
   - **Hardware:** Blank (CPU basic) to start.

3. **Push the `hf_space` files and project code** to your new Space.
   To make it easy, we will just use the browser UI:
   - In your newly created Space, go to the **Files** tab.
   - Look at the `hf_space/` folder we just created locally (`e:\Scaler\db_surgeon\hf_space\`).
   - Upload the local `app.py` and `requirements.txt` from the `hf_space/` folder to the root of your HF Space.
   - **IMPORTANT:** Also upload the `db_surgeon/` package folder (the one containing `models.py`, `client.py`, `server/`, `training/`). The `app.py` needs to import from it.

---

## STEP 2: Train the Agent using your $30 Credits

We built a **Training Tab** directly into the Gradio UI we just pushed. Here is how to use it:

1. **Upgrade the Space Hardware (Spends Credits):**
   - Go to the **Settings** tab of your Space.
   - Scroll down to **Space Hardware**.
   - Change it from CPU to **Nvidia T4 (small)** ($0.40/hr) or **Nvidia A10G (small)** ($1.00/hr).
   - *Recommendation:* Start with the **T4 (small)**. Your $30 will last 75 hours!

2. **Run the Training:**
   - Go back to the **App** tab of your Space. It will take a few minutes to boot up the GPU.
   - Once it loads, click the **🚀 Training** tab.
   - Set episodes to **200**.
   - Click **Start Training**.
   - The training will now run in the background. You can click **Refresh Status** to see the live logs.

3. **Wait & Download:**
   - Training 200 episodes on a T4 will take about 2-3 hours (costing ~$1.20).
   - Once it says ✅ **Training Complete**, the model weights will be saved to the `/tmp/db_surgeon_output` folder in the Space.
   - *Wait, how do I get the model?* The `app.py` snippet we wrote automatically attempts to save and optionally push the model back to HF! If it doesn't push, you can modify the script to upload the resulting LoRA adapter. 

---

## STEP 3: Scale Down to Save Money

Once training is complete and you have your model weights / plots:

1. Go back to your Space **Settings**.
2. Change the Space Hardware back downwards to **CPU Basic (Free)** OR set it to **Pause after 15 minutes of inactivity**.
3. **DO NOT forget to downgrade or pause it**, otherwise it will drain your $30 over the weekend!

---

## Alternative: Using HF ZeroGPU or HF Jobs (Headless)

If the Space background task gets interrupted (Spaces sometimes restart), you can train purely headless using HuggingFace Jobs:

1. Create a script `run_hf_job.sh` locally:
   ```bash
   pip install unsloth trl datasets accelerate peft
   pip install -e ./db_surgeon
   python -m db_surgeon.training.train_unsloth
   ```
2. Make sure your local terminal has `huggingface_hub` installed (`pip install huggingface_hub`).
3. Run the job from your terminal:
   ```bash
   huggingface-cli login
   huggingface-cli run-job - hardware t4-small run_hf_job.sh
   ```
   *(This requires packaging the repo into a Docker image or using a specific HF compute script, which is slightly more advanced).*

**Best Path:** Use the Gradio App UI (Tab 2) we generated in the `hf_space/` folder. It is designed to run the Unsloth GRPO directly in the background thread of the Space while giving you live UI feedback.

---

## RECAP

1. Create Gradio Space.
2. Upload `hf_space/app.py`, `requirements.txt`, and the `db_surgeon` core folder to it.
3. Switch Settings to **T4 GPU**, wait for boot.
4. Go to **Training Tab**, start it, and watch the logs.
5. Review plots, save model.
6. Switch back to **Free CPU** or pause.
