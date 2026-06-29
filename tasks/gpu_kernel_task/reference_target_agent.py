#!/usr/bin/env python3
"""
SIA-style target agent for Triton GPU kernel optimization.

Fixed scaffold: tools (write_file, read_file, bash) + agentic loop.
SYSTEM_PROMPT is the only thing that varies across generations — it is
written/improved by the meta/feedback agent and injected by loop.py.
"""
import json
import os
import subprocess
from openai import OpenAI

# ── Client (all config from env vars set by loop.py) ─────────────────────────
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)
MODEL       = os.environ["MODEL_NAME"]
TASK_MD     = os.environ["TASK_MD"]
OUTPUT_DIR  = os.environ["OUTPUT_DIR"]
SOL_PATH    = os.path.join(OUTPUT_DIR, "sol.py")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")
EVALUATE_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py")
MAX_TURNS   = 8

# ── Tools ─────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Absolute file path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read and return the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command and return stdout + stderr. Use this to validate sol.py and run evaluate.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                },
                "required": ["command"],
            },
        },
    },
]

# ── Tool implementations ──────────────────────────────────────────────────────
def _write_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}."

def _read_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: {path} not found."
    except Exception as e:
        return f"Error: {e}"

def _bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        out = r.stdout
        if r.stderr:
            out += "\n[stderr]\n" + r.stderr
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 120s"
    except Exception as e:
        return f"Error: {e}"

def _dispatch(name: str, args: dict) -> str:
    if name == "write_file":
        return _write_file(**args)
    elif name == "read_file":
        return _read_file(**args)
    elif name == "bash":
        cmd = args.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        return _bash(cmd)
    return f"Unknown tool: {name}"

# ── System prompt (injected per generation by loop.py) ───────────────────────
SYSTEM_PROMPT = """<<<SYSTEM_PROMPT>>>"""

# ── Agent loop ────────────────────────────────────────────────────────────────
task = open(TASK_MD).read()
messages = [{"role": "user", "content": (
    f"Task:\n{task}\n\n"
    f"Write your Triton kernel to {SOL_PATH} using write_file, then validate it:\n"
    f"  python -m py_compile {SOL_PATH}  # syntax check\n"
    f"  python {EVALUATE_PY} --gen-dir {OUTPUT_DIR} --problem matrix-vector  # run on H100\n"
    "Iterate until the eval returns ACCEPTED, then try to improve GFLOPS further."
)}]

best_latency = float("inf")
best_code = None

for turn in range(MAX_TURNS):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=4000,
    )

    msg    = response.choices[0].message
    finish = response.choices[0].finish_reason

    if msg.content:
        print(f"[turn {turn}] {msg.content[:300]}")

    if not msg.tool_calls:
        break

    # Append assistant turn
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

    # Execute tools
    for tc in msg.tool_calls:
        args   = json.loads(tc.function.arguments)
        result = _dispatch(tc.function.name, args)
        print(f"  [{tc.function.name}] → {result[:400]}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # After each turn, check results.json for ACCEPTED status
    try:
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        status  = results.get("status", "")
        latency = results.get("average_latency_ms", float("inf"))
        gflops  = results.get("average_gflops", 0.0)
    except (FileNotFoundError, json.JSONDecodeError):
        continue

    if status in ("ACCEPTED", "CHECKED", "SUCCESS"):
        if latency < best_latency:
            best_latency = latency
            try:
                best_code = open(SOL_PATH).read()
            except Exception:
                pass
            print(f"[best] {best_latency:.4f} ms  {gflops:.1f} GFLOPS")

        remaining = MAX_TURNS - turn - 1
        if remaining > 0:
            messages.append({"role": "user", "content": (
                f"ACCEPTED: {latency:.4f} ms ({gflops:.1f} GFLOPS). "
                f"Best so far: {best_latency:.4f} ms. "
                f"Leaderboard target: 0.0853 ms (644.5 GFLOPS). "
                f"You need {latency/0.0853:.1f}x speedup to reach the top. "
                f"{remaining} turn(s) left — try a faster approach: "
                "larger tiles, tl.dot for tensor cores, num_warps=8, num_stages=4."
            )})

# Restore best solution if a later attempt was worse
if best_code:
    try:
        final = json.load(open(RESULTS_PATH))
        if final.get("average_latency_ms", float("inf")) > best_latency:
            open(SOL_PATH, "w").write(best_code)
            print(f"[restore] restored best solution ({best_latency:.4f} ms)")
    except Exception:
        pass

if not os.path.exists(SOL_PATH):
    print(f"WARNING: agent finished without writing {SOL_PATH}")

# Save tool-call trace so the feedback agent can read it
execution_log_path = os.path.join(OUTPUT_DIR, "agent_execution.json")
try:
    with open(execution_log_path, "w") as f:
        json.dump(messages, f, indent=2)
except Exception as e:
    print(f"WARNING: could not save agent_execution.json: {e}")
