import json
import os
import urllib.request
import urllib.parse
import urllib.error

class TensaraClient:
    """
    A lightweight Python client for the Tensara platform.
    Replicates the functionality of the official Rust CLI using standard library web calls.
    Supports real-time Server-Sent Events (SSE) streaming for compilation and execution.
    """
    def __init__(self, api_key: str, api_base_url: str = "https://tensara.org"):
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")

    def _post_stream(self, path: str, payload: dict):
        url = f"{self.api_base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "tensara-cli",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req) as response:
                current_event = None
                for line in response:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str.startswith("event:"):
                        current_event = line_str[6:].strip()
                    elif line_str.startswith("data:"):
                        data_str = line_str[5:].strip()
                        try:
                            data_json = json.loads(data_str)
                        except json.JSONDecodeError:
                            data_json = data_str
                        yield current_event, data_json
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_data = json.loads(err_body)
                msg = err_data.get("message") or err_data.get("error") or err_data.get("status") or err_body
                raise Exception(f"HTTP {e.code}: {msg}")
            except json.JSONDecodeError:
                raise Exception(f"HTTP {e.code}: {err_body}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    def _get_trpc(self, procedure: str, input_data: dict = None) -> dict:
        if input_data:
            input_json = json.dumps({"json": input_data})
            encoded_input = urllib.parse.quote(input_json)
            url = f"{self.api_base_url}/api/trpc/{procedure}?input={encoded_input}"
        else:
            url = f"{self.api_base_url}/api/trpc/{procedure}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "tensara-cli"
            },
            method="GET"
        )
        try:
            with urllib.request.urlopen(req) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                return parsed["result"]["data"]["json"]
        except Exception as e:
            raise Exception(f"Failed to fetch {procedure}: {str(e)}")

    def list_problems(self) -> list:
        """List all available problems on Tensara."""
        return self._get_trpc("problems.getAll")

    def get_problem(self, slug: str) -> dict:
        """Get details for a specific problem (slug, description, parameters, starters)."""
        return self._get_trpc("problems.getById", {"slug": slug})

    def run_checker(self, problem_slug: str, code: str, dtype: str = "float16", language: str = "python", gpu_type: str = "H100"):
        """Validate solution correctness against the reference implementation."""
        payload = {
            "problemSlug": problem_slug,
            "code": code,
            "dtype": dtype,
            "language": language,
            "gpuType": gpu_type
        }
        return self._post_stream("/api/submissions/checker", payload)

    def run_benchmark(self, problem_slug: str, code: str, dtype: str = "float16", language: str = "python", gpu_type: str = "H100"):
        """Measure performance (GFLOPS, latency) on Tensara hardware."""
        payload = {
            "problemSlug": problem_slug,
            "code": code,
            "dtype": dtype,
            "language": language,
            "gpuType": gpu_type
        }
        return self._post_stream("/api/submissions/benchmark", payload)

    def submit_solution(self, problem_slug: str, code: str, dtype: str = "float16", language: str = "python", gpu_type: str = "H100"):
        """Submit the solution officially to the Tensara leaderboard."""
        payload = {
            "problemSlug": problem_slug,
            "code": code,
            "dtype": dtype,
            "language": language,
            "gpuType": gpu_type
        }
        return self._post_stream("/api/submissions/direct-submit", payload)

    def get_baseline_best(self, problem_slug: str, gpu_type: str = "H100") -> dict:
        """
        Return the best baseline latency/GFLOPS for a problem+GPU combination.

        Uses the baselineBenchmarks embedded in the problem definition, which are
        always public and need no user session.  Falls back gracefully if the
        problem or GPU is not found.

        Returns a dict with keys: avg_latency_ms, avg_gflops, framework, or
        an empty dict if no baseline is available.
        """
        try:
            problem = self.get_problem(problem_slug)
        except Exception:
            return {}

        baselines = problem.get("baselineBenchmarks", {})
        if not baselines:
            return {}

        # Rank frameworks: torch_vanilla is the most relevant PyTorch reference
        preference = ["torch_vanilla", "torch_compile", "tinygrad"]
        ordered = [fw for fw in preference if fw in baselines] + \
                  [fw for fw in baselines if fw not in preference]

        best = {}
        for framework in ordered:
            gpu_data = baselines.get(framework, {}).get(gpu_type, {})
            results = gpu_data.get("results", [])
            if not results:
                continue
            latencies = [r["runtime_ms"] for r in results if r.get("runtime_ms") is not None]
            gflops_list = [r["gflops"] for r in results if r.get("gflops") is not None]
            if latencies:
                best = {
                    "framework": framework,
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "avg_gflops": sum(gflops_list) / len(gflops_list) if gflops_list else 0.0,
                }
                break  # use highest-preference framework that has data

        return best

    def get_leaderboard_best(self, problem_slug: str, gpu_type: str = "H100", language: str = None) -> dict:
        """
        Return the best leaderboard entry for a problem+GPU combination.

        Calls the public tRPC endpoint (no authentication required).
        If `language` is given, prefers entries for that language but falls back
        to the overall best if none exist for that language.

        Returns a dict with keys: username, avg_latency_ms, avg_gflops, language, rank
        or an empty dict if the leaderboard is empty / unreachable.
        """
        try:
            entries = self._get_trpc(
                "submissions.getProblemLeaderboard",
                {"slug": problem_slug, "gpuType": gpu_type},
            )
        except Exception:
            return {}

        if not entries:
            return {}

        # Entries are already sorted best-first (lowest runtime).
        # The API stores Triton submissions as language='python' (Triton is Python-based).
        # Map 'triton' -> 'python' for filtering purposes.
        api_language = "python" if language == "triton" else language
        if api_language:
            lang_entries = [e for e in entries if e.get("language") == api_language]
            if lang_entries:
                best = lang_entries[0]
                return {
                    "username": best.get("username", ""),
                    "avg_latency_ms": best["runtime"],
                    "avg_gflops": best.get("gflops", 0.0),
                    "language": best.get("language", language),
                    "rank": entries.index(best) + 1,
                    "same_language": True,
                }

        # Fall back to overall best (any language)
        best = entries[0]
        return {
            "username": best.get("username", ""),
            "avg_latency_ms": best["runtime"],
            "avg_gflops": best.get("gflops", 0.0),
            "language": best.get("language", ""),
            "rank": 1,
            "same_language": False,
        }

    # ── Local submission cache ─────────────────────────────────────────────────
    # Tracks the best latency we have ever successfully submitted, keyed by
    # "{user_id}_{slug}_{gpu}_{language}" so different API keys stay separate.

    @staticmethod
    def _cache_path() -> str:
        return os.path.join(os.path.expanduser("~"), ".tensara_cache.json")

    def _cache_key(self, problem_slug: str, gpu_type: str, language: str) -> str:
        # TENSARA_USERID env var (set explicitly by user) takes priority.
        # Fall back to the user-ID fragment embedded in the API key.
        user_id = (
            os.environ.get("TENSARA_USERID")
            or (self.api_key.split("_")[1] if self.api_key and self.api_key.startswith("tsra_") else "anon")
        )
        return f"{user_id}_{problem_slug}_{gpu_type}_{language}"

    def get_my_best_submission(self, problem_slug: str, gpu_type: str = "H100", language: str = "python") -> float:
        """Return the best latency (ms) we have previously submitted, or inf if none."""
        try:
            with open(self._cache_path(), "r") as f:
                cache = json.load(f)
            return float(cache.get(self._cache_key(problem_slug, gpu_type, language), float("inf")))
        except Exception:
            return float("inf")

    def record_submission(self, problem_slug: str, gpu_type: str, language: str, latency_ms: float) -> None:
        """Persist a successful submission's latency to the local cache."""
        path = self._cache_path()
        try:
            with open(path, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
        key = self._cache_key(problem_slug, gpu_type, language)
        if latency_ms < float(cache.get(key, float("inf"))):
            cache[key] = latency_ms
            with open(path, "w") as f:
                json.dump(cache, f, indent=2)

    def run_sample(self, problem_slug: str, code: str, dtype: str = "float16", language: str = "python", gpu_type: str = "H100"):
        """Run a simple validation check (subset of checker)."""
        payload = {
            "problemSlug": problem_slug,
            "code": code,
            "dtype": dtype,
            "language": language,
            "gpuType": gpu_type
        }
        return self._post_stream("/api/submissions/sample", payload)
