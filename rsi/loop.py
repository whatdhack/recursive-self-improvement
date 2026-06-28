"""
RSI Orchestrator — Recursive Self-Improvement loop.

5-step cycle per generation:
  1. Meta agent  (Kimi 2.6)    writes target_agent.py
  2. Target agent (Nemotron)   runs target_agent.py → sol.py
  3. Evaluate    (Tensara)     checks correctness + benchmarks on H100
  4. Curate                    failure logs → train.jsonl entry
  5. Train       (llama-finetune)  LoRA updates Gemma4-1B weights

Usage:
  python -m rsi run \\
    --problem matrix-vector \\
    --meta-agent-profile profiles/kimi26-do.json \\
    --target-agent-profile profiles/nemotron-do.json \\
    --max-gen 10 \\
    --run-id 001 \\
    --base-model ./models/gemma4-1b-it-q4_k_m.gguf \\
    --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "gpu_kernel_task"
PROVIDER = ROOT / "providers" / "do.json"

sys.path.insert(0, str(TASK_DIR))
from tensara_client import TensaraClient  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def gen_dir(run_dir: Path, gen: int) -> Path:
    d = run_dir / f"gen-{gen}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Step 1: Meta agent writes target_agent.py ────────────────────────────────

META_SYSTEM = """You are an expert AI engineer. Your job is to write a Python
target agent script that will call an LLM to solve a GPU kernel coding task.

The script must:
1. Read the task description from the file path given in the environment variable TASK_MD.
2. Call the LLM using the openai Python package with:
   - base_url from env var OPENAI_BASE_URL
   - api_key from env var OPENAI_API_KEY
   - model name from env var MODEL_NAME
   - max_tokens=8000 (required — the model needs space to think and then produce code)
3. Extract the response text: use resp.choices[0].message.content if it is not None,
   otherwise fall back to resp.choices[0].message.reasoning_content (some models are
   thinking models that put the final answer in reasoning_content when content is None).
4. Write the solution to sol.py in the directory given by OUTPUT_DIR env var.
5. The solution must be a valid Triton kernel implementing the `solution` function.

Output ONLY the Python script, no explanation."""


def run_meta_agent(meta_profile: Path, gdir: Path) -> Path:
    from rsi.agent import run_agent

    task_md = (TASK_DIR / "task.md").read_text()
    user_prompt = (
        f"Write the target agent script for this task:\n\n{task_md}\n\n"
        "The script should instruct the LLM to write a Triton kernel sol.py."
    )
    log("  [meta] calling Kimi 2.6...")
    code = run_agent(meta_profile, PROVIDER, META_SYSTEM, user_prompt)

    # Strip markdown fences if present
    import re
    code = re.sub(r"^```python\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code)

    target_agent_path = gdir / "target_agent.py"
    target_agent_path.write_text(code)
    log(f"  [meta] wrote target_agent.py ({len(code)} chars)")
    return target_agent_path


# ── Step 2: Target agent runs and writes sol.py ───────────────────────────────

def run_target_agent(target_profile: Path, target_agent_path: Path, gdir: Path) -> Path:
    import json as _json
    target_profile_data = _json.loads(target_profile.read_text())
    provider_data = _json.loads(PROVIDER.read_text())

    api_key = os.environ.get(provider_data["api_key_env"], "")
    env = os.environ.copy()
    env["OPENAI_BASE_URL"] = provider_data["base_url"]
    env["OPENAI_API_KEY"] = api_key
    env["MODEL_NAME"] = target_profile_data["model"]
    env["TASK_MD"] = str(TASK_DIR / "task.md")
    env["OUTPUT_DIR"] = str(gdir)

    log(f"  [target] running target_agent.py (model: {target_profile_data['model']})...")
    result = subprocess.run(
        [sys.executable, str(target_agent_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    sol_path = gdir / "sol.py"
    if not sol_path.exists():
        log("  [target] WARNING: sol.py not written by target agent")
    else:
        log(f"  [target] sol.py written ({sol_path.stat().st_size} bytes)")
    return sol_path


# ── Step 3: Evaluate via Tensara ──────────────────────────────────────────────

def run_evaluation(gdir: Path, problem: str, gpu_type: str) -> dict:
    evaluate_script = TASK_DIR / "evaluate.py"
    result = subprocess.run(
        [sys.executable, str(evaluate_script), "--gen-dir", str(gdir),
         "--problem", problem, "--gpu-type", gpu_type],
        capture_output=False,
        text=True,
    )
    results_path = gdir / "results.json"
    if results_path.exists():
        return read_json(results_path)
    return {"status": "NO_RESULTS", "accuracy": 0.0}


# ── Step 4+5: Curate failures + train ─────────────────────────────────────────

def run_curate_and_train(
    gdir: Path,
    run_dir: Path,
    results: dict,
    gen: int,
    curator_profile: Path,
    cfg,
    tracker,
    args,
) -> tuple:
    from rsi.curate import append_to_jsonl, curate_via_api
    from rsi.train import LoraConfig, merge_lora, run_lora_training, save_hp_log

    train_jsonl = run_dir / "train.jsonl"
    sol_path = gdir / "sol.py"
    status = results.get("status", "UNKNOWN")

    # Curate: only on failure
    if status not in ("ACCEPTED", "CHECKED", "SUCCESS") and sol_path.exists():
        log("  [curate] generating training pair from failure...")
        failed_code = sol_path.read_text()
        pair = curate_via_api(failed_code, results, curator_profile, PROVIDER)
        if pair:
            append_to_jsonl(pair, train_jsonl)
        else:
            log("  [curate] WARNING: could not parse training pair from LLM response")

    # Train: only if we have a base model and training data
    if not args.base_model or not Path(args.base_model).exists():
        log("  [train] skipping (--base-model not set or not found)")
        return cfg, tracker

    if not train_jsonl.exists() or train_jsonl.stat().st_size == 0:
        log("  [train] skipping (train.jsonl is empty)")
        return cfg, tracker

    llama_finetune = Path(args.llama_bin_dir) / "llama-finetune"
    llama_export = Path(args.llama_bin_dir) / "llama-export-lora"

    if not llama_finetune.exists():
        log(f"  [train] skipping (llama-finetune not found at {llama_finetune})")
        return cfg, tracker

    base_model = Path(args.base_model)
    lora_in = run_dir / f"gen-{gen - 1}" / "lora.gguf" if gen > 0 else None
    lora_out = gdir / "lora.gguf"
    merged_out = gdir / f"gemma4-gen{gen}.gguf"

    # Adjust hyperparams based on previous performance
    gflops = results.get("average_gflops")
    tracker.record(gflops)
    cfg = tracker.adjust(cfg)
    log(f"  [train] config: rank={cfg.rank} lr={cfg.lr} epochs={cfg.epochs}")

    elapsed = run_lora_training(llama_finetune, base_model, lora_in, lora_out, train_jsonl, cfg)
    merge_lora(llama_export, base_model, lora_out, merged_out)
    save_hp_log(run_dir, gen, cfg, elapsed)

    return cfg, tracker


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI loop — Recursive Self-Improvement")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--problem", default="matrix-vector")
    parser.add_argument("--meta-agent-profile", default="profiles/kimi26-do.json")
    parser.add_argument("--target-agent-profile", default="profiles/nemotron-do.json")
    parser.add_argument("--curator-profile", default="profiles/curator-do.json")
    parser.add_argument("--max-gen", type=int, default=10)
    parser.add_argument("--run-id", default="001")
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--base-model", default=None,
                        help="Path to Gemma4-1B GGUF for LoRA training")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--llama-bin-dir", default="./llama.cpp/build/bin",
                        help="Directory containing llama-finetune and llama-export-lora")
    parser.add_argument("--run-dir", default="./runs")
    args = parser.parse_args()

    meta_profile = ROOT / args.meta_agent_profile
    target_profile = ROOT / args.target_agent_profile
    curator_profile = ROOT / args.curator_profile
    run_dir = Path(args.run_dir) / f"run-{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    from rsi.train import HyperparamTracker, LoraConfig
    cfg = LoraConfig(rank=args.lora_rank, threads=args.threads)
    tracker = HyperparamTracker()

    print(f"\nRSI Loop — {args.problem}  (GPU: {args.gpu_type})")
    print("─" * 60)

    for gen in range(args.max_gen):
        gdir = gen_dir(run_dir, gen)
        log(f"\n[Gen {gen}] starting")

        # Step 1: Meta agent
        try:
            target_agent_path = run_meta_agent(meta_profile, gdir)
        except Exception as e:
            log(f"[Gen {gen}] meta agent failed: {e}")
            continue

        # Step 2: Target agent
        try:
            sol_path = run_target_agent(target_profile, target_agent_path, gdir)
        except Exception as e:
            log(f"[Gen {gen}] target agent failed: {e}")
            continue

        # Step 3: Evaluate
        log(f"[Gen {gen}] evaluating via Tensara ({args.gpu_type})...")
        results = run_evaluation(gdir, args.problem, args.gpu_type)
        status = results.get("status", "UNKNOWN")
        gflops = results.get("average_gflops")
        latency = results.get("average_latency_ms")

        perf_str = f"{gflops:.1f} GFLOPS  {latency:.2f} ms" if gflops else "N/A"
        log(f"[Gen {gen}] Tensara: {status}  |  {perf_str}")
        write_json(gdir / "results.json", results)

        # Steps 4+5: Curate + Train
        cfg, tracker = run_curate_and_train(
            gdir, run_dir, results, gen, curator_profile, cfg, tracker, args
        )

        print("─" * 60)

    log("RSI loop complete.")


if __name__ == "__main__":
    main()
