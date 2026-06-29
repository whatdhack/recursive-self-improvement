#!/usr/bin/env python3
"""
Evaluate Triton/CUDA solutions using Tensara benchmark times.

This script:
1. Locates the solution file (sol.py or sol.cu) in the generation directory.
2. Identifies the problem slug and solution programming language.
3. Submits the code to Tensara for a remote H100 GPU performance benchmark.
4. Parses the Server-Sent Events (SSE) streaming benchmark responses.
5. Saves results and latency/GFLOPS metrics per shape to results.json.

Usage:
    python evaluate.py --gen-dir path/to/generation/directory
"""

import sys
import os
import ast
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from datetime import datetime

# Add script directory to sys.path so we can import tensara_client
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from tensara_client import TensaraClient


def find_solution_file(gen_dir: Path) -> Optional[Path]:
    """Find a solution python or CUDA file in the generation directory."""
    patterns = ["**/sol.py", "**/sol.cu", "**/solution.py", "**/solution.cu"]
    for pattern in patterns:
        for p in gen_dir.rglob(pattern):
            # Exclude virtual environments and cache directories
            if any(x in p.parts for x in [".venv", "venv", "__pycache__", ".git"]):
                continue
            # Exclude driver/eval files
            if p.name in ["evaluate.py", "run_triton.py", "tensara_client.py"]:
                continue
            return p
    return None


def check_triton_sol(sol_path: Path) -> List[str]:
    """
    Static AST analysis of a Triton solution file.

    Catches common errors that would cause COMPILE_ERROR or NameError on the
    remote runner, before wasting a Tensara API round-trip.

    Returns a list of issue strings (empty = no issues found).
    """
    issues: List[str] = []

    try:
        src = sol_path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Could not read solution file: {e}"]

    # 1. Python syntax check
    try:
        tree = ast.parse(src, filename=str(sol_path))
    except SyntaxError as e:
        return [f"SyntaxError at line {e.lineno}: {e.msg}"]

    # Collect top-level imported names
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)

    # 2. Per-kernel checks
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        # Only inspect @triton.jit decorated functions
        is_jit = any(
            (isinstance(d, ast.Attribute) and d.attr == "jit")
            or (isinstance(d, ast.Name) and d.id == "jit")
            for d in node.decorator_list
        )
        if not is_jit:
            continue

        # Collect constexpr params
        constexpr_params = [
            arg.arg
            for arg in node.args.args
            if arg.annotation
            and isinstance(arg.annotation, ast.Attribute)
            and arg.annotation.attr == "constexpr"
        ]

        # All Name nodes used anywhere in the kernel body
        body_module = ast.Module(body=node.body, type_ignores=[])
        used_names = {n.id for n in ast.walk(body_module) if isinstance(n, ast.Name)}

        for param in constexpr_params:
            if param not in used_names:
                issues.append(
                    f"kernel '{node.name}': tl.constexpr param '{param}' is declared "
                    f"but never used inside the kernel body — this causes COMPILE_ERROR on Tensara. "
                    f"Either use it or remove it."
                )

        # Check for .ravel() — not a valid Triton tensor method
        for child in ast.walk(body_module):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "ravel"
            ):
                issues.append(
                    f"kernel '{node.name}': .ravel() is not a valid Triton tensor "
                    f"operation and will cause COMPILE_ERROR. Use 1-D indexing directly."
                )
                break

        # Check for tl.cdiv with non-constexpr (runtime) first argument
        for child in ast.walk(body_module):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "cdiv"
                and child.args
            ):
                first_arg = child.args[0]
                # If the first arg is a plain Name that is NOT a constexpr param, warn
                if isinstance(first_arg, ast.Name) and first_arg.id not in constexpr_params:
                    issues.append(
                        f"kernel '{node.name}': tl.cdiv('{first_arg.id}', ...) — "
                        f"'{first_arg.id}' is a runtime argument, not a constexpr. "
                        f"Use Python integer arithmetic in the launch wrapper instead "
                        f"(e.g. (n + B - 1) // B)."
                    )

    # 3. Module-level checks
    # torch used without import
    if "torch" not in imported:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                issues.append(
                    "'torch' is used (e.g. torch.float32) but 'import torch' is missing — "
                    "this causes NameError at runtime."
                )
                break

    return issues


def get_problem_slug(solution_path: Path, client: TensaraClient) -> str:
    """Identify the problem slug based on the path and known problems from Tensara."""
    # Path-based detection first — no API key required.
    # Works when sol.py lives inside a directory named after the problem
    # (e.g. .../data/public/relu/sol.py).
    for part in solution_path.parts:
        if part and part not in {".", "..", "sol.py", "sol.cu"}:
            # Heuristic: problem slugs are lowercase, hyphen-separated
            if part.replace("-", "").replace("_", "").isalpha() and part.islower() and len(part) > 2:
                # Try to validate against the API if available
                try:
                    problems = client.list_problems()
                    slugs = {p.get("slug") for p in problems if p.get("slug")}
                    if part in slugs:
                        return part
                except Exception:
                    # API unavailable — trust the path
                    return part

    # API-assisted detection as a fallback
    try:
        problems = client.list_problems()
        slugs = [p.get("slug") for p in problems if p.get("slug")]
        for part in solution_path.parts:
            if part in slugs:
                return part
        for part in solution_path.parts:
            for slug in slugs:
                if slug.lower() in part.lower():
                    return slug
    except Exception as e:
        print(f"Warning: could not fetch problems list from Tensara: {e}")

    print("Warning: could not detect problem slug from path or API — pass --problem explicitly.")
    return "vector-addition"


def _parse_sse_stream(stream) -> Dict[str, Any]:
    """Shared SSE parser for checker and benchmark responses."""
    results_dict = {}
    status = "UNKNOWN"
    error_message = None
    error_details = None
    passed_tests = 0
    total_tests = 0

    for event, data in stream:
        if not isinstance(data, dict):
            continue
        cur_status = data.get("status")
        if cur_status:
            status = cur_status
        if "errorMessage" in data and data["errorMessage"]:
            error_message = data["errorMessage"]
        if "errorDetails" in data and data["errorDetails"]:
            error_details = data["errorDetails"]
        passed = data.get("passedTests") if data.get("passedTests") is not None else data.get("passed_tests")
        if passed is not None:
            passed_tests = passed
        total = data.get("totalTests") if data.get("totalTests") is not None else data.get("total_tests")
        if total is not None:
            total_tests = total
        if cur_status == "BENCHMARK_RESULT":
            res = data.get("result")
            if res:
                name = res.get("name") or f"Test {res.get('test_id')}"
                results_dict[name] = {
                    "name": name,
                    "status": "PASSED",
                    "runtime_ms": res.get("runtime_ms"),
                    "gflops": res.get("gflops"),
                    "speedup": res.get("speedup"),
                }
        tr_list = data.get("test_results") or data.get("results")
        if tr_list and isinstance(tr_list, list):
            for tr in tr_list:
                name = tr.get("name") or f"Test {tr.get('test_id') or tr.get('id')}"
                t_status = tr.get("status") or tr.get("state") or "UNKNOWN"
                results_dict[name] = {
                    "name": name,
                    "status": t_status,
                    "runtime_ms": tr.get("runtime_ms"),
                    "gflops": tr.get("gflops"),
                    "speedup": tr.get("speedup"),
                }
                if "errorMessage" in tr and tr["errorMessage"]:
                    results_dict[name]["error"] = tr["errorMessage"]

    return {
        "status": status,
        "errorMessage": error_message,
        "errorDetails": error_details,
        "passedTests": passed_tests,
        "totalTests": total_tests,
        "test_results": list(results_dict.values()),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception is an HTTP 429 / rate limit."""
    msg = str(exc)
    return "429" in msg or "rate limit" in msg.lower()


def run_tensara_checker(
    client: TensaraClient, slug: str, code: str, language: str, gpu_type: str = "H100"
) -> Dict[str, Any]:
    """Check correctness against the reference implementation before benchmarking.
    Retries automatically on HTTP 429 with exponential backoff."""
    print(f"Checking correctness on Tensara for problem '{slug}' (language: {language})...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            stream = client.run_checker(slug, code, dtype="float32", language=language, gpu_type=gpu_type)
            return _parse_sse_stream(stream)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = 60 * (2 ** (attempt - 1))  # 60, 120, 240 ...
                print(f"  Rate limited ({e}). Waiting {wait}s then retrying ({attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                return {"status": "CLIENT_ERROR", "errorMessage": str(e), "test_results": []}


def run_tensara_benchmark(
    client: TensaraClient, slug: str, code: str, language: str, gpu_type: str = "H100"
) -> Dict[str, Any]:
    """Run benchmark on Tensara and parse streaming SSE results.
    Retries automatically on HTTP 429 with exponential backoff."""
    print(f"Running benchmark on Tensara for problem '{slug}' (language: {language})...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            stream = client.run_benchmark(slug, code, dtype="float32", language=language, gpu_type=gpu_type)
            return _parse_sse_stream(stream)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                wait = 60 * (2 ** (attempt - 1))  # 60, 120, 240 ...
                print(f"  Rate limited ({e}). Waiting {wait}s then retrying ({attempt}/{max_retries})...")
                time.sleep(wait)
            else:
                return {"status": "CLIENT_ERROR", "errorMessage": str(e), "test_results": []}


def _ownbest_path(task_dir: Path, slug: str, language: str, gpu_type: str) -> Path:
    """Return the ownbest file path for a given problem, language, and GPU."""
    ext = ".cu" if language == "cuda" else ".py"
    return task_dir / "ownbest" / "tensara" / f"{slug}-{language}-{gpu_type}{ext}"


def save_ownbest(
    task_dir: Path,
    slug: str,
    language: str,
    gpu_type: str,
    code: str,
    eval_results: dict,
    run_id: str = "",
) -> bool:
    """Save solution to ownbest/tensara/ if it beats the current best on file.

    Returns True if the file was written (new best), False otherwise.
    """
    dest = _ownbest_path(task_dir, slug, language, gpu_type)
    avg_latency = eval_results.get("average_latency_ms")
    avg_gflops = eval_results.get("average_gflops")

    if avg_latency is None:
        return False

    # Check existing best
    current_best = float("inf")
    if dest.exists():
        for line in dest.read_text(encoding="utf-8").splitlines():
            if line.startswith("# latency_ms:") or line.startswith("// latency_ms:"):
                try:
                    current_best = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

    if avg_latency >= current_best:
        return False

    # Build header comment
    ts = datetime.now().strftime("%Y-%m-%d")
    comment = "#" if language != "cuda" else "//"
    lines = [
        f"{comment} problem:    {slug}",
        f"{comment} gpu:        {gpu_type}",
        f"{comment} language:   {language}",
        f"{comment} gflops:     {avg_gflops:.2f}" if avg_gflops is not None else f"{comment} gflops:     N/A",
        f"{comment} latency_ms: {avg_latency:.4f}",
        f"{comment} date:       {ts}",
    ]
    if run_id:
        lines.append(f"{comment} run:        {run_id}")
    header = "\n".join(lines) + "\n\n"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + code, encoding="utf-8")
    prev_str = f"{current_best:.4f} ms" if current_best < float("inf") else "none"
    print(f"\n  New personal best saved → {dest}  ({avg_latency:.4f} ms, prev: {prev_str})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Triton/CUDA solutions using Tensara benchmark times"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--gen-dir",
        type=Path,
        help="Generation directory containing the solution file (sol.py or sol.cu)",
    )
    group.add_argument(
        "--submission", type=Path, help="Direct path to solution file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results (default: gen-dir/results.json)",
    )
    parser.add_argument(
        "--gpu-type",
        dest="gpu_type",
        type=str,
        default="H100",
        choices=["T4", "H100", "H200", "B200", "A100-80GB", "A10G", "L40S", "L4"],
        help="GPU type for Tensara benchmarking (default: H100).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default=None,
        help="Problem slug (e.g. 'relu', 'matrix-multiplication'). Overrides auto-detection from path.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=["python", "cuda", "triton", "cutedsl", "cutile"],
        help="Override the language sent to Tensara (default: inferred from file extension).",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        default=False,
        help="If set, submit the solution to the Tensara leaderboard when ACCEPTED.",
    )

    args = parser.parse_args()

    # Determine generation directory and output path
    if args.submission:
        sol_file = args.submission
        gen_dir = sol_file.parent
        output_path = args.output or gen_dir / "results.json"
    else:
        gen_dir = args.gen_dir
        sol_file = find_solution_file(gen_dir)
        output_path = args.output or gen_dir / "results.json"

    # Handle missing solution file
    if not sol_file or not sol_file.exists():
        error_text = f"No solution file (sol.py or sol.cu) found in {gen_dir}."
        print(f"Error: {error_text}")
        eval_results = {
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "status": "NO_SOLUTION_FILE",
            "error_message": error_text,
            "timestamp": datetime.now().isoformat(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        sys.exit(1)

    print(f"Found solution file: {sol_file}")

    # Read solution code
    with open(sol_file, "r", encoding="utf-8") as f:
        code = f.read()

    # Determine language (explicit flag wins; otherwise infer from file extension)
    language = args.language or ("cuda" if sol_file.suffix == ".cu" else "triton")

    # Static pre-check for Triton solutions before hitting Tensara
    if language in ("triton", "python") and sol_file.suffix == ".py":
        static_issues = check_triton_sol(sol_file)
        if static_issues:
            issue_text = "\n".join(f"  • {i}" for i in static_issues)
            print(f"\nStatic analysis found issues in {sol_file.name}:\n{issue_text}\n")
            print("Fix the above issues before submitting to Tensara.\n")
            eval_results = {
                "accuracy": 0.0,
                "correct": 0,
                "total": 0,
                "status": "STATIC_CHECK_FAILED",
                "timestamp": datetime.now().isoformat(),
                "error_message": f"Static analysis failed:\n{issue_text}",
                "details": [],
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            sys.exit(1)

    # Setup client
    api_key = os.environ.get("TENSARA_API_KEY")
    client = TensaraClient(api_key=api_key)

    # Determine problem slug (explicit flag wins over path-based detection)
    slug = args.problem or get_problem_slug(sol_file, client)

    # Fetch performance target: leaderboard best (public), then baseline fallback
    target = client.get_leaderboard_best(slug, gpu_type=args.gpu_type, language=language)
    if target:
        same_lang = target.get("same_language", False)
        lang_note = f"{target['language']}" if same_lang else f"{target['language']} — no {language} entries yet"
        print(
            f"Leaderboard target (#{target['rank']}, {lang_note}, {target['username']} / {args.gpu_type}): "
            f"{target['avg_latency_ms']:.4f} ms  "
            f"({target['avg_gflops']:.1f} GFLOPS)  ← beat this to reach the top"
        )
    else:
        target = client.get_baseline_best(slug, gpu_type=args.gpu_type)
        if target:
            print(
                f"Baseline target ({target.get('framework', 'torch')} / {args.gpu_type}): "
                f"{target['avg_latency_ms']:.4f} ms  "
                f"({target['avg_gflops']:.1f} GFLOPS)  ← beat this to lead the board"
            )

    # Step 1: Check correctness first
    checker_data = run_tensara_checker(client, slug, code, language, gpu_type=args.gpu_type)
    checker_status = checker_data.get("status", "UNKNOWN")

    ACCEPTED_STATUSES = {"ACCEPTED", "CHECKED", "SUCCESS", "CORRECT"}
    checker_passed = checker_status in ACCEPTED_STATUSES

    if not checker_passed:
        # Save checker failure and exit without benchmarking
        checker_results_list = checker_data.get("test_results", [])
        checker_passed_count = checker_data.get("passedTests", 0)
        checker_total_count = checker_data.get("totalTests", 0)
        if checker_results_list:
            checker_total_count = len(checker_results_list)
            checker_passed_count = sum(
                1 for r in checker_results_list
                if r.get("status") in ["PASSED", "SUCCESS", "CORRECT"]
            )
        checker_accuracy = (
            (checker_passed_count / checker_total_count * 100.0)
            if checker_total_count > 0 else 0.0
        )
        eval_results = {
            "accuracy": checker_accuracy,
            "correct": checker_passed_count,
            "total": checker_total_count,
            "status": checker_status,
            "timestamp": datetime.now().isoformat(),
            "details": checker_results_list,
        }
        if checker_data.get("errorMessage"):
            eval_results["error_message"] = checker_data["errorMessage"]
        if checker_data.get("errorDetails"):
            eval_results["error_details"] = checker_data["errorDetails"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print("\n" + "=" * 80)
        print("Tensara Correctness Check Results")
        print("=" * 80)
        print(f"Problem:            {slug}")
        print(f"Language:           {language}")
        print(f"Status:             {checker_status}  ← FAILED — skipping benchmark")
        print(f"Accuracy:           {checker_accuracy:.2f}% ({checker_passed_count}/{checker_total_count} passed)")
        if checker_data.get("errorMessage"):
            print(f"Error:              {checker_data['errorMessage']}")
        print("=" * 80)
        sys.exit(1)

    print(f"Correctness check PASSED ({checker_status}) — proceeding to benchmark.")

    # Step 2: Run benchmark only after checker passes
    benchmark_data = run_tensara_benchmark(client, slug, code, language, gpu_type=args.gpu_type)

    # Extract parsed results
    results_list = benchmark_data.get("test_results", [])
    passed_count = benchmark_data.get("passedTests", 0)
    total_count = benchmark_data.get("totalTests", 0)
    status = benchmark_data.get("status", "UNKNOWN")
    error_msg = benchmark_data.get("errorMessage")
    error_det = benchmark_data.get("errorDetails")

    # Always compute summary counts from the test results breakdown if populated
    if results_list:
        total_count = len(results_list)
        passed_count = sum(1 for r in results_list if r.get("status") in ["PASSED", "SUCCESS", "CORRECT"])

    if status in ["UNKNOWN", "BENCHMARK_RESULT"] and passed_count == total_count and total_count > 0:
        status = "ACCEPTED"

    # Calculate accuracy
    if total_count > 0:
        accuracy = (passed_count / total_count) * 100.0
    else:
        accuracy = 100.0 if status in ["ACCEPTED", "CHECKED", "SUCCESS"] else 0.0

    # Build evaluation results dict
    eval_results = {
        "accuracy": accuracy,
        "correct": passed_count,
        "total": total_count,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }

    if error_msg:
        eval_results["error_message"] = error_msg
    if error_det:
        eval_results["error_details"] = error_det

    # Calculate average latency and GFLOPS
    latencies = [
        r["runtime_ms"]
        for r in results_list
        if r.get("runtime_ms") is not None
    ]
    gflops_list = [
        r["gflops"] for r in results_list if r.get("gflops") is not None
    ]

    if latencies:
        eval_results["average_latency_ms"] = sum(latencies) / len(latencies)
    if gflops_list:
        eval_results["average_gflops"] = sum(gflops_list) / len(gflops_list)

    # Add flat keys for each shape for SIA context_manager to extract
    for r in results_list:
        name = r["name"]
        name_clean = (
            name.replace(" = ", "_").replace("^", "_").replace(" ", "")
        )

        if r.get("runtime_ms") is not None:
            eval_results[f"{name_clean}_latency_ms"] = r["runtime_ms"]
        if r.get("gflops") is not None:
            eval_results[f"{name_clean}_gflops"] = r["gflops"]

    # Include raw details list
    eval_results["details"] = results_list

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Tensara Triton/CUDA Benchmark Evaluation Results")
    print("=" * 80)
    print(f"Problem:            {slug}")
    print(f"Language:           {language}")
    print(f"Status:             {status}")
    print(f"Accuracy:           {accuracy:.2f}% ({passed_count}/{total_count} passed)")
    if "average_latency_ms" in eval_results:
        print(f"Average Latency:    {eval_results['average_latency_ms']:.4f} ms")
    if "average_gflops" in eval_results:
        print(f"Average GFLOPS:     {eval_results['average_gflops']:.2f} GFLOPS")
    print("=" * 80)

    if results_list:
        print("\nTest Case Breakdown:")
        print("-" * 80)
        for r in results_list:
            name = r["name"]
            t_status = r["status"]
            runtime = r.get("runtime_ms")
            gflops = r.get("gflops")
            runtime_str = f"{runtime:.4f} ms" if runtime is not None else "N/A"
            gflops_str = f"{gflops:.2f} GFLOPS" if gflops is not None else "N/A"
            print(f"  - {name:15s} : {t_status:8s} | Latency: {runtime_str:10s} | Perf: {gflops_str}")
            if "error" in r:
                print(f"    Error: {r['error']}")
        print("-" * 80)

    # Compare against target (leaderboard best or baseline) and report
    avg_latency = eval_results.get("average_latency_ms")
    target_latency = target.get("avg_latency_ms") if target else None
    if target and avg_latency is not None and target_latency is not None:
        target_label = (
            target.get("username", target.get("framework", "target"))
        )
        if avg_latency < target_latency:
            print(f"\n  BEATS target ({target_label}): {avg_latency:.4f} ms vs {target_latency:.4f} ms  ({target_latency/avg_latency:.2f}x faster)")
        else:
            print(f"\n  Does NOT beat target ({target_label}): {avg_latency:.4f} ms vs {target_latency:.4f} ms — need to go {avg_latency/target_latency:.2f}x faster")

    # Save to ownbest/tensara/ if ACCEPTED and beats current best on file.
    if status == "ACCEPTED":
        run_id = os.environ.get("SIA_RUN_ID", "")
        save_ownbest(script_dir.parent.parent, slug, language, args.gpu_type, code, eval_results, run_id=run_id)

    # Submit to leaderboard if --submit and ACCEPTED and beats our own previous best.
    if args.submit and status == "ACCEPTED" and avg_latency is not None:
        my_best = client.get_my_best_submission(slug, gpu_type=args.gpu_type, language=language)
        if avg_latency < my_best:
            prev_str = f"{my_best:.4f} ms" if my_best < float("inf") else "none"
            print(f"\nSubmitting to Tensara leaderboard (new personal best: {avg_latency:.4f} ms, prev: {prev_str})...")
            try:
                submit_stream = client.submit_solution(slug, code, language=language, gpu_type=args.gpu_type)
                submit_result = _parse_sse_stream(submit_stream)
                submit_status = submit_result.get("status", "UNKNOWN")
                print(f"Submission status: {submit_status}")
                eval_results["submission_status"] = submit_status
                client.record_submission(slug, args.gpu_type, language, avg_latency)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(eval_results, f, indent=2)
            except Exception as e:
                print(f"Submission failed: {e}")
        else:
            print(f"\nSkipping submission — {avg_latency:.4f} ms does not beat personal best ({my_best:.4f} ms).")

    if status in ["COMPILATION_ERROR", "RUNTIME_ERROR", "CLIENT_ERROR", "STREAM_ERROR"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
