"""
RSI Orchestrator — Recursive Self-Improvement loop.

5-step cycle per generation:
  1. Meta agent  (Kimi 2.6)    writes target_agent.py  [gen 0 only]
     Feedback agent (Kimi 2.6) rewrites target_agent.py from prior script+errors [gen 1+]
  2. Target agent (Kimi 2.5)   runs target_agent.py → sol.py
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

class _Tee:
    """Write to both a stream and a log file — used to tee stdout+stderr."""
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._path = path

    def write(self, data: str) -> int:
        self._stream.write(data)
        with open(self._path, "a") as f:
            f.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def fileno(self) -> int:
        return self._stream.fileno()

    def isatty(self) -> bool:
        return False


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

CRITICAL instructions for the system prompt you pass to the target LLM:
- Tell it to output ONLY raw Python code — no markdown fences, no docstrings, no comments,
  no explanations, no task description echoed back. Code only.
- Tell it the output must start with import statements and nothing else.
- This is essential: large comment blocks or docstrings cause the file to be truncated
  mid-string when the model hits the token limit, producing a SyntaxError.
- Tell it the code must be complete and fully implemented — no `pass` placeholders,
  no pseudo-code, no TODO comments. Every function body must contain real code.

Triton kernel rules the system prompt MUST include verbatim:
1. NEVER use Python `if` inside an @triton.jit kernel. Use masking:
     mask = offsets < N; tl.load(ptr + offsets, mask=mask, other=0.0)
2. Accumulators must be tl.zeros tensors, NOT Python scalars:
     acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  # correct
     acc = 0.0  # WRONG — will not compile
3. Use vectorized tl.load with tl.arange() offsets, never scalar element-wise loads.
4. BLOCK_SIZE must be a tl.constexpr parameter and a power of 2 (e.g. 128 or 256).
5. Compute the grid in the Python launcher (triton.cdiv), not inside the kernel.
6. Always import: import triton; import triton.language as tl; import torch

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


# ── Step 1b: Feedback agent rewrites target_agent.py (gen 1+) ────────────────

LEADERBOARD_TARGET_GFLOPS = 644.5  # H100 leaderboard #30 as of 2026-06-28

FEEDBACK_SYSTEM = """You are an expert AI engineer improving a target agent script.

The target agent calls an LLM to write a Triton GPU kernel (sol.py) that implements
matrix-vector multiplication on an H100 GPU. Your job: study the generation history,
the previous target_agent.py, the sol.py it produced, and the eval result — then
write an IMPROVED target_agent.py that either fixes errors or boosts performance.

The script must:
1. Read the task from the TASK_MD env var.
2. Call the LLM via openai (OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME env vars).
3. Use max_tokens=8000.
4. Extract content: resp.choices[0].message.content if not None, else reasoning_content.
5. Write the solution to OUTPUT_DIR/sol.py.

── MODE A: CORRECTNESS (status is not ACCEPTED) ──────────────────────────────
Fix the specific error in the system prompt or add post-processing. Do NOT repeat
approaches that already failed (check the history). Triton rules to enforce verbatim:
1. NEVER use Python `if` inside @triton.jit — use masking:
     mask = offsets < N; tl.load(ptr + offsets, mask=mask, other=0.0)
2. Accumulators: tl.zeros((BLOCK_SIZE,), dtype=tl.float32) NOT acc = 0.0
3. Vectorized tl.load with tl.arange() offsets only — no scalar element-wise loads.
4. BLOCK_SIZE must be tl.constexpr and a power of 2 (128 or 256).
5. Grid computed in Python launcher: grid = (triton.cdiv(M, BLOCK_SIZE),)
6. Always import: import triton; import triton.language as tl; import torch
7. Output ONLY raw Python code — no markdown fences, starts with imports, no pass.

── MODE B: PERFORMANCE (status is ACCEPTED, GFLOPS below target) ─────────────
The kernel is correct. Push for higher GFLOPS. Instruct the LLM to try:
1. Use tl.dot() for the inner product — it maps to tensor core instructions on H100.
   Pattern: reshape row block to (1, N) and x to (N, 1), use tl.dot.
2. Increase BLOCK_SIZE to 256 or 512 for better memory throughput.
3. Use multiple warps (num_warps=8) and stages (num_stages=4) in the triton.jit decorator.
4. Process multiple rows per program using a 2D tile (BLOCK_ROWS × BLOCK_COLS).
5. Use tl.load with eviction_policy='evict_last' for streaming access patterns.
6. Align pointer arithmetic to 128-byte boundaries using tl.multiple_of().
The leaderboard target is {leaderboard_target:.1f} GFLOPS. Show the current GFLOPS
and target in the system prompt so the LLM knows what to beat.

Output ONLY the Python script for target_agent.py, no explanation.""".format(
    leaderboard_target=LEADERBOARD_TARGET_GFLOPS
)


def _perf_summary(results: dict) -> str:
    """Format a compact performance summary from results.json for the feedback prompt."""
    status = results.get("status", "UNKNOWN")
    gflops = results.get("average_gflops")
    latency = results.get("average_latency_ms")
    error_msg = results.get("error_message", "")

    lines = [f"Status: {status}"]
    if gflops:
        pct = gflops / LEADERBOARD_TARGET_GFLOPS * 100
        lines.append(f"Performance: {gflops:.1f} GFLOPS  {latency:.3f} ms  "
                     f"({pct:.1f}% of leaderboard target {LEADERBOARD_TARGET_GFLOPS:.1f} GFLOPS)")
    if error_msg:
        lines.append(f"Error: {error_msg}")

    # Per-shape breakdown (when ACCEPTED)
    details = results.get("details", [])
    if details:
        lines.append("Per-shape breakdown:")
        for r in details:
            name = r.get("name", "?")
            g = r.get("gflops")
            t = r.get("runtime_ms")
            s = r.get("status", "?")
            lines.append(f"  {name}: {s}  {g:.1f} GFLOPS  {t:.3f} ms" if g else f"  {name}: {s}")

    return "\n".join(lines)


def run_feedback_agent(
    meta_profile: Path,
    gdir: Path,
    prev_target_agent: Path,
    prev_sol: Path,
    prev_results: dict,
    context_path: Path,
) -> Path:
    from rsi.agent import run_agent

    prev_script = prev_target_agent.read_text() if prev_target_agent and prev_target_agent.exists() else "(not available)"
    prev_code = prev_sol.read_text() if prev_sol and prev_sol.exists() else "(not available)"
    history = context_path.read_text() if context_path.exists() else "(no history yet)"
    perf = _perf_summary(prev_results)

    is_accepted = prev_results.get("status") in ("ACCEPTED", "CHECKED", "SUCCESS")
    mode = "B (performance)" if is_accepted else "A (correctness)"

    user_prompt = (
        f"GENERATION HISTORY (one line per gen):\n{history}\n\n"
        f"EVAL RESULT — mode {mode}:\n{perf}\n\n"
        f"PREVIOUS target_agent.py:\n```python\n{prev_script}\n```\n\n"
        f"WHAT IT PRODUCED (sol.py):\n```python\n{prev_code}\n```\n\n"
        + (
            f"The kernel is correct at {prev_results.get('average_gflops', 0):.1f} GFLOPS. "
            f"Target is {LEADERBOARD_TARGET_GFLOPS:.1f} GFLOPS. "
            "Rewrite target_agent.py to instruct the LLM to optimise for higher GFLOPS."
            if is_accepted else
            "Rewrite target_agent.py to fix the error and produce a correct Triton kernel. "
            "Do NOT repeat approaches already tried in the history above."
        )
    )

    log(f"  [feedback] calling Kimi 2.6 (mode {mode})...")
    code = run_agent(meta_profile, PROVIDER, FEEDBACK_SYSTEM, user_prompt)

    import re
    code = re.sub(r"^```python\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code)

    target_agent_path = gdir / "target_agent.py"
    target_agent_path.write_text(code)
    log(f"  [feedback] wrote target_agent.py ({len(code)} chars)")
    return target_agent_path


def update_context(context_path: Path, gen: int, status: str, results: dict, prev_gflops: float | None = None) -> None:
    gflops = results.get("average_gflops")
    latency = results.get("average_latency_ms")
    error_msg = results.get("error_message", "")

    line = f"Gen {gen}: {status}"
    if gflops:
        pct = gflops / LEADERBOARD_TARGET_GFLOPS * 100
        line += f" — {gflops:.1f} GFLOPS  {latency:.3f} ms  ({pct:.1f}% of target)"
        if prev_gflops:
            delta = gflops - prev_gflops
            line += f"  Δ{delta:+.1f} GFLOPS"
    if error_msg:
        # Keep it to one line — first line of the error is most useful
        first_line = error_msg.splitlines()[0][:120]
        line += f" — {first_line}"

    with open(context_path, "a") as f:
        f.write(line + "\n")


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
        sol_path = _clean_sol(sol_path)
        log(f"  [target] sol.py written ({sol_path.stat().st_size} bytes)")
    return sol_path


def _clean_sol(sol_path: Path) -> Path:
    """
    Strip reasoning/prose from sol.py — keep only Python code.

    Kimi K2.5 (a thinking model) sometimes dumps its chain-of-thought
    as plain text before or around the code block, producing an unterminated
    string when the output is truncated at the token limit.

    Strategy:
    1. If a ```python ... ``` block exists, extract it.
    2. Otherwise find the first line that looks like Python (import / def / @)
       and keep everything from there on.
    3. If neither, leave the file as-is (will fail static check with a clear error).
    """
    import re
    raw = sol_path.read_text(encoding="utf-8", errors="replace")

    # 1. Prefer an explicit ```python ... ``` fence
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        code = m.group(1).strip()
        sol_path.write_text(code + "\n", encoding="utf-8")
        return sol_path

    # 2. Find the first Python-looking line
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(import |from |def |@|class )", line.strip()):
            code = "\n".join(lines[i:]).strip()
            sol_path.write_text(code + "\n", encoding="utf-8")
            return sol_path

    # 3. Leave unchanged
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

    log_path = run_dir / "run.log"
    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)

    from rsi.train import HyperparamTracker, LoraConfig
    cfg = LoraConfig(rank=args.lora_rank, threads=args.threads)
    tracker = HyperparamTracker()

    context_path = run_dir / "context.md"
    prev_target_agent: Path | None = None
    prev_sol: Path | None = None
    prev_results: dict = {}

    print(f"\nRSI Loop — {args.problem}  (GPU: {args.gpu_type})")
    print("─" * 60)

    for gen in range(args.max_gen):
        gdir = gen_dir(run_dir, gen)
        log(f"\n[Gen {gen}] starting")

        # Step 1: Meta agent bootstraps gen 0; feedback agent rewrites for gen 1+
        try:
            if gen == 0:
                target_agent_path = run_meta_agent(meta_profile, gdir)
            else:
                target_agent_path = run_feedback_agent(
                    meta_profile, gdir,
                    prev_target_agent, prev_sol, prev_results, context_path,
                )
        except Exception as e:
            log(f"[Gen {gen}] agent failed: {e}")
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

        # Update context and carry state forward for the feedback agent
        prev_gflops = prev_results.get("average_gflops")
        update_context(context_path, gen, status, results, prev_gflops)
        prev_target_agent = target_agent_path
        prev_sol = sol_path
        prev_results = results

        # Steps 4+5: Curate + Train
        cfg, tracker = run_curate_and_train(
            gdir, run_dir, results, gen, curator_profile, cfg, tracker, args
        )

        print("─" * 60)

    log("RSI loop complete.")


if __name__ == "__main__":
    main()
