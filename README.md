# recursive-self-improvement

**A Recursive Self-Improving AI that fine-tunes its own weights from failures — meta agent, target agent, LoRA, repeat.**

Built for the [AIEWF Hackathon 2026](https://cerebralvalley.ai/e/aiewf-hackathon-2026/details).

---

## What it does

Each generation, the loop:

1. **Meta agent** (Kimi 2.6 on DO Model Studio) writes a target agent strategy
2. **Target agent** (Nemotron on DO Model Studio) writes a Triton GPU kernel (`sol.py`)
3. **Tensara** evaluates correctness and benchmarks on H100 — returns GFLOPS
4. **Curator** (small LLM) converts failure logs into structured training pairs → `train.jsonl`
5. **llama-finetune** runs a LoRA pass on Gemma4-1B (CPU) — updates raw weights
6. Repeat with the improved model

The training hyperparameters (LoRA rank, learning rate, epochs) self-adjust each generation based on `ΔPerformance`.

```
Gen 0:  Kimi writes strategy → Nemotron writes kernel → WRONG_ANSWER → curate → train Gemma4
Gen 1:  improved strategy    → better kernel          → ACCEPTED     → 41 GFLOPS
Gen 2:  ...                                                           → 56 GFLOPS (+36%)
```

---

## Architecture

```
providers/do.json          ← DO Model Studio (OpenAI-compat endpoint)
profiles/kimi26-do.json    ← meta agent
profiles/nemotron-do.json  ← target agent

rsi/loop.py                ← main orchestrator (5-step RSI cycle)
rsi/agent.py               ← OpenAI-compat runner
rsi/curate.py              ← failure logs → train.jsonl
rsi/train.py               ← llama-finetune wrapper + hyperparameter tracker

tasks/gpu_kernel_task/     ← matrix-vector problem + Tensara evaluator
setup/droplet_setup.sh     ← DO CPU droplet initialization
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/<you>/recursive-self-improvement
cd recursive-self-improvement
cp .env.example .env
# fill in DO_API_KEY and TENSARA_API_KEY
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Run (API-only, no local training)

```bash
python -m rsi run \
  --problem matrix-vector \
  --meta-agent-profile profiles/kimi26-do.json \
  --target-agent-profile profiles/nemotron-do.json \
  --max-gen 5 \
  --run-id 001
```

### 4. Run with RSI (LoRA training on CPU)

```bash
# First provision a DO CPU droplet:
bash setup/droplet_setup.sh

# Then run with the local model:
python -m rsi run \
  --problem matrix-vector \
  --base-model /opt/models/google_gemma-4-1b-it-Q4_K_M.gguf \
  --llama-bin-dir /opt/llama.cpp/build/bin \
  --threads 8 \
  --max-gen 10 \
  --run-id 001
```

---

## Requirements

- Python 3.10+
- `openai` Python package
- DigitalOcean Model Studio API key
- Tensara API key
- *(for LoRA training)* DO CPU droplet with llama.cpp built

---

## RSI — what makes it recursive?

The loop is recursive because:
- The model's own failures generate the next training batch
- The training config adjusts based on the performance delta each generation
- There is no human in the loop after generation 0

This is an early-stage implementation of [Recursive Self-Improvement](https://en.wikipedia.org/wiki/Recursive_self-improvement) — models that bootstrap their own capabilities from their own mistakes.

---

## License

MIT
