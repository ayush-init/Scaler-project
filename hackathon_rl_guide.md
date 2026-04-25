# Hackathon Self-Serve Guide  
### Build an RL Environment, Train an LLM, Ship a Demo

---

## 0) What you are building

The core idea is not just to fine-tune a text model, but to build a specialized LLM system that can:

- Act inside an environment  
- Get feedback  
- Improve through reinforcement learning  

**Stack:**

Environment → Verifier/Reward → TRL Trainer → Unsloth → Deployment (OpenEnv / Spaces)

---

## 1) Start with the right project idea

Pick a task that:

- Supports step-by-step actions  
- Has programmatic verification  
- Is challenging but solvable  

⚠️ RL fails if success probability = 0.

---

## 2) Minimum RL Loop

Prompt → Model Output → Execute → Reward → Update Model

---

## 3) SFT vs RL

- Lots of labeled data → SFT  
- No data but verifiable outputs → RL  
- Practical setup → SFT + RL  

---

## 4) Design the Environment First

Define:

- reset()  
- step(action)  
- state()  
- reward  

---

## 5) Build with OpenEnv

Typical components:

- Action dataclass  
- Observation dataclass  
- State  
- reset / step  
- FastAPI interface  

---

## 6) Start Simple

Progression:

1. Easy  
2. Medium  
3. Hard  

---

## 7) Reward Design

Include:

- Execution success  
- Correctness  
- Format  
- Timeouts  
- Safety  

---

## 8) Prevent Reward Hacking

- Multiple reward checks  
- Restricted execution  
- No global state  
- Inspect outputs  

---

## 9) Process Feedback

- Step-level checks  
- Trace analysis  

---

## 10) Training Stack

- TRL  
- Unsloth  
- OpenEnv  

---

## 11) Verifiable RL

Use:

- Tests  
- Regex  
- Executors  

---

## 12) Keep Inference Fast

Optimize:

- Sampling  
- Environment  
- Runtime  

---

## 13) Deploy Early

Use Hugging Face Spaces

---

## 14) Scale Later

Ensure everything works before scaling

---

## 15) Monitor Training

Track:

- Reward  
- Success rate  
- Outputs  

---

## 16) Save Models Properly

Avoid incorrect LoRA merging

---

## 17) Team Structure

- Environment  
- Rewards  
- Training  
- Demo  

---

## 18) 1-Day Plan

1. Task  
2. Env  
3. Rewards  
4. Deploy  
5. Train  
6. Inspect  
7. Improve  
8. Scale  
9. Demo  

---

## 19) Judges Expect

- Clear system  
- Good rewards  
- Improvement proof  
- Demo  

---

## 20) Common Mistakes

- Too hard task  
- Single reward  
- No checks  
- Early training  

---

## Final Advice

Build simple → verify → improve with RL
