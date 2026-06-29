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

# A minimal Triton matvec that compiles and runs correctly on H100.
# Included verbatim in prompts so the LLM has a correct starting point.
REFERENCE_KERNEL = '''\
import triton
import triton.language as tl

@triton.jit
def matvec_kernel(A_ptr, x_ptr, y_ptr, M, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(A_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += a * x
    tl.store(y_ptr + row, tl.sum(acc, axis=0))

def solution(A, x, y, M, N):
    BLOCK_N = 256
    matvec_kernel[(M,)](A, x, y, M, N, BLOCK_N=BLOCK_N)
'''

# ── Shared agentic loop (used by meta AND feedback agents) ────────────────────

LEADERBOARD_TARGET_GFLOPS = 644.5  # H100 leaderboard #30 as of 2026-06-28


def _run_agentic_loop(
    model: str,
    api_key: str,
    base_url: str,
    system_prompt: str,
    user_message: str,
    max_turns: int = 8,
) -> list:
    """Agentic loop with write_file, read_file, bash tools. Returns message history."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file on disk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path":    {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read and return the full contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command, return stdout + stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                    },
                    "required": ["command"],
                },
            },
        },
    ]

    def _write(path: str, content: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}."

    def _read(path: str) -> str:
        try:
            return open(path, encoding="utf-8").read()
        except FileNotFoundError:
            return f"Error: {path} not found."
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _bash(cmd: str) -> str:
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            out = r.stdout
            if r.stderr:
                out += "\n[stderr]\n" + r.stderr
            return out.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: timed out after 60s"
        except Exception as e:
            return f"Error: {e}"

    def _dispatch(name: str, args: dict) -> str:
        if name == "write_file":
            return _write(**args)
        elif name == "read_file":
            return _read(**args)
        elif name == "bash":
            cmd = args.get("command", args.get("cmd", ""))
            return _bash(cmd)
        return f"Unknown tool: {name}"

    messages: list[dict] = [{"role": "user", "content": user_message}]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=4000,
        )
        msg = response.choices[0].message

        if msg.content:
            print(f"    [turn {turn}] {msg.content[:200]}")

        if not msg.tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _dispatch(tc.function.name, args)
            print(f"    [{tc.function.name}] → {result[:300]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return messages


# ── Step 1: Meta agent (agentic loop, like siawdh) ────────────────────────────

META_SYSTEM = """You are an expert AI engineer. Your task: write target_agent.py for a Triton GPU kernel task.

WHAT YOU MUST DO:
1. read_file the reference agent (path given in the user message) — study its structure
2. read_file the task description (path given in the user message)
3. write_file a custom target_agent.py at the output path given in the user message

The target_agent.py will be run with env vars:
  OPENAI_BASE_URL, OPENAI_API_KEY, MODEL_NAME  — LLM connection
  TASK_MD                                       — path to task.md
  OUTPUT_DIR                                    — directory to write sol.py, results.json

It must have write_file, read_file, bash tools and an agent loop that:
  1. Writes a Triton kernel to OUTPUT_DIR/sol.py (start from the REFERENCE KERNEL below)
  2. Validates: python -m py_compile <sol_path>
  3. Evaluates: python <evaluate_py> --gen-dir <OUTPUT_DIR> --problem matrix-vector
  4. Reads results.json, fixes errors, iterates until ACCEPTED
  5. After ACCEPTED: tries to improve GFLOPS (bigger BLOCK_N, num_warps=8)

The SYSTEM_PROMPT inside target_agent.py MUST include:

REFERENCE KERNEL (agent starts here — copy verbatim):
{reference_kernel}

TRITON RULES (include verbatim in SYSTEM_PROMPT):
0. CRITICAL: do NOT `import torch` — Tensara forbids it. Use A.new_empty(M) instead of torch.empty().
1. tl.zeros((BLOCK_N,), dtype=tl.float32) — NEVER tl.zeros(())
2. NO tl.expand_dims
3. NO Python `if` inside @triton.jit — use mask= on tl.load/tl.store
4. BLOCK_N must be tl.constexpr and a power of 2 (256 or 512)
5. One program per row: row = tl.program_id(0), grid = (M,)
6. Explicit type cast on loads: .to(tl.float32)

Write ONLY target_agent.py using write_file. Do not print the code.""".format(
    reference_kernel=REFERENCE_KERNEL,
)


def run_meta_agent(meta_profile: Path, gdir: Path) -> Path:
    provider_data = json.loads(PROVIDER.read_text())
    meta_data = json.loads(meta_profile.read_text())
    api_key = os.environ.get(provider_data["api_key_env"], "")

    reference_path = TASK_DIR / "reference_target_agent.py"
    task_md_path   = TASK_DIR / "task.md"
    target_agent_path = gdir / "target_agent.py"
    evaluate_py = TASK_DIR / "evaluate.py"

    user_message = (
        f"Reference agent to study: {reference_path}\n"
        f"Task description: {task_md_path}\n"
        f"evaluate.py path: {evaluate_py}\n"
        f"Write target_agent.py to: {target_agent_path}\n\n"
        "Read the reference agent first, then write the adapted target_agent.py."
    )

    log("  [meta] agentic loop (Kimi 2.6)...")
    _run_agentic_loop(
        model=meta_data["model"],
        api_key=api_key,
        base_url=provider_data["base_url"],
        system_prompt=META_SYSTEM,
        user_message=user_message,
        max_turns=6,
    )

    if not target_agent_path.exists():
        log("  [meta] WARNING: target_agent.py not written — falling back to reference scaffold")
        import shutil
        shutil.copy(reference_path, target_agent_path)

    size = target_agent_path.stat().st_size
    log(f"  [meta] wrote target_agent.py ({size} bytes)")
    return target_agent_path


# ── Step 1b: Feedback agent (agentic loop, like siawdh) ──────────────────────

FEEDBACK_SYSTEM = """You are an expert AI engineer improving a GPU kernel agent across generations.

WHAT YOU MUST DO:
1. read_file context.md (path given) — understand the full history of what was tried
2. read_file results.json from the previous generation (path given) — see what failed
3. read_file the previous target_agent.py (path given) — see the current implementation
4. read_file sol.py from the previous generation (path given) — see the kernel produced
5. read_file agent_execution.json (path given) — see the full tool call trace of what the agent tried
6. (optional) read_file earlier improvement.md files to avoid repeating past mistakes
7. write_file improvement.md — analysis: what went wrong, what to fix, what to try next
8. write_file target_agent.py — improved agent that addresses the issues found

── MODE A: CORRECTNESS (status is not ACCEPTED) ──────────────────────────────
The kernel is failing. The SYSTEM_PROMPT inside target_agent.py must include:

REFERENCE KERNEL (agent MUST start from this — embed it verbatim):
{reference_kernel}

TRITON RULES (always include all of these):
0. CRITICAL: do NOT `import torch` — Tensara forbids it. Use A.new_empty(M) to allocate output.
1. tl.zeros((BLOCK_N,), dtype=tl.float32) — NEVER tl.zeros(())
2. NO tl.expand_dims
3. NO Python `if` inside @triton.jit — use mask=
4. BLOCK_N must be tl.constexpr and a power of 2
5. One program per row: row = tl.program_id(0), grid = (M,)
6. Explicit type cast: a = tl.load(...).to(tl.float32)

── MODE B: PERFORMANCE (status is ACCEPTED) ──────────────────────────────────
The kernel is correct. Push for higher GFLOPS in the SYSTEM_PROMPT:
- Larger BLOCK_N (512, 1024)
- tl.dot() for tensor cores
- num_warps=8, num_stages=4
- Multi-row tiles (BLOCK_ROWS × BLOCK_N)
- Leaderboard target: {leaderboard_target:.1f} GFLOPS

Write BOTH files using write_file — improvement.md first, then target_agent.py.""".format(
    reference_kernel=REFERENCE_KERNEL,
    leaderboard_target=LEADERBOARD_TARGET_GFLOPS,
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
    provider_data = json.loads(PROVIDER.read_text())
    meta_data = json.loads(meta_profile.read_text())
    api_key = os.environ.get(provider_data["api_key_env"], "")

    is_accepted = prev_results.get("status") in ("ACCEPTED", "CHECKED", "SUCCESS")
    mode = "B (performance)" if is_accepted else "A (correctness)"
    perf = _perf_summary(prev_results)

    target_agent_path = gdir / "target_agent.py"
    improvement_path  = gdir / "improvement.md"
    prev_results_path   = prev_target_agent.parent / "results.json"        if prev_target_agent else None
    prev_execution_path = prev_target_agent.parent / "agent_execution.json" if prev_target_agent else None
    evaluate_py = TASK_DIR / "evaluate.py"

    # Build list of previous improvement.md files for context
    prev_improvements = []
    if prev_target_agent:
        for g in sorted(prev_target_agent.parent.parent.iterdir()):
            imp = g / "improvement.md"
            if imp.exists() and imp != improvement_path:
                prev_improvements.append(str(imp))

    user_message = (
        f"Mode: {mode}\n"
        f"Latest eval result:\n{perf}\n\n"
        f"Files to read:\n"
        f"  context.md:            {context_path}\n"
        f"  prev target_agent.py:  {prev_target_agent or 'N/A'}\n"
        f"  prev sol.py:           {prev_sol or 'N/A'}\n"
        f"  prev results.json:     {prev_results_path or 'N/A'}\n"
        f"  prev agent_execution.json: {prev_execution_path or 'N/A'}\n"
        + (f"  earlier improvements:  {', '.join(prev_improvements)}\n" if prev_improvements else "")
        + f"\nFiles to write:\n"
        f"  improvement.md:        {improvement_path}\n"
        f"  target_agent.py:       {target_agent_path}\n"
        f"  evaluate.py path:      {evaluate_py}\n\n"
        "Read all context files first, then write improvement.md and target_agent.py."
    )

    log(f"  [feedback] agentic loop (Kimi 2.6, mode {mode})...")
    _run_agentic_loop(
        model=meta_data["model"],
        api_key=api_key,
        base_url=provider_data["base_url"],
        system_prompt=FEEDBACK_SYSTEM,
        user_message=user_message,
        max_turns=8,
    )

    if not target_agent_path.exists():
        log("  [feedback] WARNING: target_agent.py not written — falling back to reference scaffold")
        import shutil
        shutil.copy(TASK_DIR / "reference_target_agent.py", target_agent_path)

    size = target_agent_path.stat().st_size
    log(f"  [feedback] wrote target_agent.py ({size} bytes)")
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
        timeout=900,
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

def run_evaluation(gdir: Path, problem: str, gpu_type: str, submit: bool = False) -> dict:
    evaluate_script = TASK_DIR / "evaluate.py"
    cmd = [sys.executable, str(evaluate_script), "--gen-dir", str(gdir),
           "--problem", problem, "--gpu-type", gpu_type]
    if submit:
        cmd.append("--submit")
    subprocess.run(cmd, capture_output=False, text=True)
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
    parser.add_argument("--submit", action="store_true", default=False,
                        help="Submit to Tensara leaderboard when ACCEPTED and beats personal best")
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
        results = run_evaluation(gdir, args.problem, args.gpu_type, submit=args.submit)
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
