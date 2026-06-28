import os
import json
import subprocess
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
MODEL = "gpt-oss-120b-fast"

# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write (overwrite) a file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path to write"},
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
            "description": "Run a bash command and return stdout + stderr.",
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

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} characters to '{path}'."
    except Exception as e:
        return f"Error writing file: {e}"


def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{path}' not found."
    except Exception as e:
        return f"Error reading file: {e}"


def bash(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def dispatch_tool(name: str, inputs: dict) -> str:
    if name == "write_file":
        return write_file(**inputs)
    elif name == "read_file":
        return read_file(**inputs)
    elif name == "bash":
        # Some models (e.g. gpt-oss) call bash with "cmd" instead of "command",
        # or pass a list instead of a plain string, or pass extra keys like "path".
        # Normalise before dispatch.
        if "cmd" in inputs and "command" not in inputs:
            inputs["command"] = inputs.pop("cmd")
        cmd = inputs.get("command", "")
        if isinstance(cmd, list):
            cmd = " ".join(str(c) for c in cmd)
        # If the model passed {"command": "cat", "path": "..."}, compose the command
        if "path" in inputs and str(inputs["path"]) not in cmd:
            cmd = f"{cmd} {inputs['path']}"
        return bash(cmd)
    else:
        return f"Unknown tool: {name}"


# ── Multi-Trajectory Logger ───────────────────────────────────────────────────

class MultiTrajectoryLogger:
    """
    Logger for tasks with multiple independent samples (e.g., GPQA with multiple questions).

    For tasks where you need to process multiple independent items (questions, test cases,
    samples), this logger saves each trajectory separately instead of one large file.

    Usage:
        logger = MultiTrajectoryLogger(working_dir)

        for idx, question in enumerate(questions):
            messages = []
            messages.append({"role": "user", "content": question_prompt})

            response = client.messages.create(...)
            messages.append({"role": "assistant", "content": response.content})

            # Save this trajectory
            logger.log_trajectory(idx, messages)

        logger.finalize(len(questions))
    """

    def __init__(self, working_dir: str):
        """
        Initialize the multi-trajectory logger.

        Args:
            working_dir: Path to the working directory where agent_execution/ will be created
        """
        import os
        self.working_dir = working_dir
        self.execution_folder = os.path.join(working_dir, "agent_execution")
        os.makedirs(self.execution_folder, exist_ok=True)
        print(f"Initialized multi-trajectory logger at: {self.execution_folder}")

    def log_trajectory(self, trajectory_id: int, messages: list):
        """
        Save a complete trajectory for one sample.

        Args:
            trajectory_id: Index of this trajectory (0-based)
            messages: List of message dicts (same format as Anthropic API messages)
        """
        import os
        filename = f"execution_q{trajectory_id}.json"
        filepath = os.path.join(self.execution_folder, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved trajectory {trajectory_id} to {filename}")
        except Exception as e:
            print(f"  ✗ Error saving trajectory {trajectory_id}: {e}")

    def finalize(self, total_count: int):
        """
        Log completion message.

        Args:
            total_count: Total number of trajectories saved
        """
        print(f"\n{'='*60}")
        print(f"✓ Multi-trajectory logging complete:")
        print(f"  - Total trajectories: {total_count}")
        print(f"  - Saved to: {self.execution_folder}/")
        print(f"  - Files: execution_q0.json to execution_q{total_count-1}.json")
        print(f"{'='*60}\n")


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(task: str, results_path: str, solution_path: str, dataset_dir: str = "", max_iterations: int = 10) -> None:
    """
    Agent loop that drives kernel optimization.

    After correctness is confirmed (ACCEPTED), the loop injects a continuation
    prompt to force the LLM to keep optimizing for lower latency.  The best
    solution seen so far is saved to <solution_path>.best and restored at the
    end if it outperforms the final submission.

    Args:
        task:           Initial user prompt describing the task.
        results_path:   Absolute path to results.json written by evaluate.py.
        solution_path:  Absolute path to sol.py (the solution file).
        dataset_dir:    Absolute path to data/public/ (contains tensara_client.py).
        max_iterations: Hard cap on LLM round-trips (default 10).
    """
    print(f"\n{'='*60}\nTask: {task}\nMax iterations: {max_iterations}\n{'='*60}")

    # Fetch performance target: leaderboard best (public, no auth), then baseline fallback
    import sys as _sys
    _sys.path.insert(0, dataset_dir or os.path.dirname(os.path.abspath(__file__)))
    _problem_slug = None
    target_latency = float("inf")
    target_label = "baseline"
    target_same_language = False
    try:
        from tensara_client import TensaraClient as _TC
        _problem_slug = os.path.basename(os.path.dirname(solution_path)) if os.path.basename(solution_path) == "sol.py" else None
        if not _problem_slug:
            _problem_slug = os.path.basename(os.path.dirname(results_path))
        _tc = _TC(api_key=os.environ.get("TENSARA_API_KEY", ""))

        # Determine language from solution path (default triton)
        _language = "triton"

        # Try leaderboard first (public endpoint)
        _lb = _tc.get_leaderboard_best(_problem_slug or "", language=_language)
        if _lb:
            target_latency = _lb["avg_latency_ms"]
            _lang_str = _lb["language"]
            _same = _lb.get("same_language", False)
            target_same_language = _same
            if _same:
                target_label = f"leaderboard #{_lb['rank']} ({_lang_str}, {_lb['username']})"
            else:
                target_label = f"leaderboard #1 overall ({_lang_str}, {_lb['username']}) — no {_language} entries yet"
            print(f"[Target] Beat {target_label}: {target_latency:.4f} ms / {_lb.get('avg_gflops', 0):.1f} GFLOPS")
        else:
            # Fall back to baseline benchmarks
            _baseline = _tc.get_baseline_best(_problem_slug or "")
            if _baseline:
                target_latency = _baseline["avg_latency_ms"]
                target_label = f"baseline ({_baseline.get('framework', 'torch')})"
                print(f"[Target] Beat {target_label}: {target_latency:.4f} ms")
    except Exception as _e:
        print(f"[Target] Could not fetch target: {_e}")

    messages = [{"role": "user", "content": task}]
    best_latency = float("inf")
    best_solution_code = None

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        message = response.choices[0].message

        if message.content:
            print(f"\nAssistant: {message.content}")

        # Append assistant turn (must be before tool results)
        messages.append(message.model_dump(exclude_none=True))

        # Execute tool calls if any
        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                print(f"\n[Tool] {name}({json.dumps(args, ensure_ascii=False)})")
                result = dispatch_tool(name, args)
                print(f"[Result] {result[:300]}{'...' if len(result) > 300 else ''}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # After each iteration, check whether results.json reflects an ACCEPTED run
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            results = {}

        status = results.get("status", "")
        latency = results.get("average_latency_ms", float("inf"))
        gflops = results.get("average_gflops", 0.0)

        if status == "ACCEPTED":
            # Track the best solution seen so far
            if latency < best_latency:
                best_latency = latency
                try:
                    with open(solution_path, "r", encoding="utf-8") as f:
                        best_solution_code = f.read()
                except Exception:
                    pass
                print(f"\n[Best] New best: {best_latency:.4f} ms  ({gflops:.1f} GFLOPS)")

            remaining = max_iterations - iteration - 1
            if remaining > 0:
                # Inject a forced continuation — the LLM must now respond, so it
                # cannot simply "decide it is done" by emitting no tool calls.
                target_str = (
                    f"  Target to beat: {target_latency:.4f} ms ({target_label}).\n"
                    if target_latency < float("inf") else ""
                )
                beats = latency < target_latency if target_latency < float("inf") else False
                beats_str = (
                    f"  You ARE beating the target ({latency:.4f} ms < {target_latency:.4f} ms). "
                    f"Keep going — aim even lower.\n"
                    if beats else
                    f"  You are NOT yet beating the target ({latency:.4f} ms vs {target_latency:.4f} ms target). "
                    f"You need to go {latency/target_latency:.2f}x faster.\n"
                    if target_latency < float("inf") else ""
                )
                continuation = (
                    f"Your kernel is ACCEPTED with average latency {latency:.4f} ms "
                    f"({gflops:.1f} GFLOPS). "
                    f"Best so far: {best_latency:.4f} ms.\n"
                    f"{target_str}"
                    f"{beats_str}"
                    f"You have {remaining} iteration(s) remaining. "
                    f"Try to reduce latency further. Ideas to explore:\n"
                    f"  • Increase BLOCK_SIZE (try 512 or 1024) to improve memory throughput\n"
                    f"  • Use vectorized loads: load 4 fp16 values per thread using tl.load with stride\n"
                    f"  • Tune num_warps (try 4 or 8)\n"
                    f"  • Reduce grid overhead by processing more elements per block\n"
                    f"Write a new kernel variant, run evaluate.py, and report the result."
                )
                messages.append({"role": "user", "content": continuation})
                print(f"\n[Loop] Injecting optimization prompt ({remaining} iterations left).")
                continue

        # If the LLM made no tool calls and we are not in the middle of an
        # optimization loop, treat this as natural completion.
        if not message.tool_calls:
            print("\nNo tool calls — finishing.")
            break

    # Restore best solution if a later attempt was worse
    if best_solution_code is not None:
        try:
            with open(solution_path, "r", encoding="utf-8") as f:
                current_code = f.read()
        except Exception:
            current_code = ""

        try:
            with open(results_path, "r", encoding="utf-8") as f:
                final_results = json.load(f)
            final_latency = final_results.get("average_latency_ms", float("inf"))
        except Exception:
            final_latency = float("inf")

        if best_latency < final_latency and current_code != best_solution_code:
            with open(solution_path, "w", encoding="utf-8") as f:
                f.write(best_solution_code)
            print(f"\n[Best] Restored best solution ({best_latency:.4f} ms) "
                  f"— final attempt was slower ({final_latency:.4f} ms).")

    # Submit best solution if it beats our own personal best (checked via local cache).
    if best_latency < float("inf") and best_solution_code is not None and _problem_slug:
        my_best = _tc.get_my_best_submission(_problem_slug, gpu_type="H100", language=_language)
        if best_latency < my_best:
            prev_str = f"{my_best:.4f} ms" if my_best < float("inf") else "none"
            print(f"\n[Submit] New personal best ({best_latency:.4f} ms, prev: {prev_str}) — submitting to leaderboard...")
            submit_cmd = (
                f"python {os.path.join(os.path.dirname(results_path.replace('results.json', '')), '')}/../data/public/evaluate.py"
                f" --gen-dir {os.path.dirname(results_path)}"
                f" --problem {_problem_slug}"
                f" --language triton"
                f" --submit"
            )
            bash(submit_cmd)
        else:
            print(f"\n[Submit] {best_latency:.4f} ms does not beat personal best ({my_best:.4f} ms) — skipping.")

    print(f"\n{'='*60}\nDone. Best latency: {best_latency:.4f} ms\n{'='*60}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_dir", required=True)
    args = parser.parse_args()

    solution_path = os.path.join(args.working_dir, "sol.py")
    results_path  = os.path.join(args.working_dir, "results.json")

    run_agent(
        task=(
            f"Write an optimized Triton kernel for the task described in "
            f"{args.dataset_dir}. Save your solution to {solution_path} and "
            f"evaluate it with evaluate.py."
        ),
        results_path=results_path,
        solution_path=solution_path,
        dataset_dir=args.dataset_dir,
        max_iterations=10,
    )


# ── Multi-Trajectory Usage Example ────────────────────────────────────────────
"""
USAGE EXAMPLE: Multi-Trajectory Logging for GPQA-style tasks

For tasks with multiple independent questions/samples (like GPQA with 198 questions),
use MultiTrajectoryLogger instead of saving to a single agent_execution.json file.

Example implementation:

    import argparse
    import os

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--working_dir', required=True)
    args = parser.parse_args()

    # Initialize multi-trajectory logger
    logger = MultiTrajectoryLogger(args.working_dir)

    # Load dataset (e.g., GPQA questions)
    questions_file = os.path.join(args.dataset_dir, "diamond_qna.json")
    with open(questions_file) as f:
        questions = json.load(f)

    # Process each question independently
    for idx, question_data in enumerate(questions):
        print(f"\\nProcessing question {idx+1}/{len(questions)}...")

        # Build conversation for this question
        messages = []
        messages.append({
            "role": "user",
            "content": f"Question: {question_data['question']}\\nChoices: {question_data['choices']}"
        })

        # Get response from Claude
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
        )

        # Add response to messages
        messages.append({
            "role": "assistant",
            "content": response.choices[0].message.content,
        })

        # Save this trajectory
        logger.log_trajectory(idx, messages)

    # Finalize logging
    logger.finalize(len(questions))

This creates:
    working_dir/agent_execution/execution_q0.json
    working_dir/agent_execution/execution_q1.json
    ...
    working_dir/agent_execution/execution_q197.json

Instead of a single large:
    working_dir/agent_execution.json

Benefits:
- Each trajectory is isolated and independently parseable
- Easier to debug specific questions
- Better for large datasets (no single huge file)
- Feedback agent can analyze patterns across trajectories
"""