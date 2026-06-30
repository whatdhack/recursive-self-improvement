"""Lifecycle manager for a llama-server subprocess (OpenAI-compat inference)."""
import subprocess
import time
import urllib.request
from pathlib import Path


class LlamaServer:
    def __init__(
        self,
        server_bin: Path,
        model_path: Path,
        port: int = 8080,
        threads: int = 8,
        ctx_size: int = 2048,
    ):
        self.server_bin = Path(server_bin)
        self.model_path = Path(model_path)
        self.port = port
        self.threads = threads
        self.ctx_size = ctx_size
        self._proc: subprocess.Popen | None = None

    def start(self, timeout: int = 120) -> None:
        """Start llama-server and block until the /health endpoint responds."""
        cmd = [
            str(self.server_bin),
            "--model", str(self.model_path),
            "--port", str(self.port),
            "--threads", str(self.threads),
            "--ctx-size", str(self.ctx_size),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                urllib.request.urlopen(
                    f"http://localhost:{self.port}/health", timeout=2
                )
                return  # server ready
            except Exception:
                if self._proc.poll() is not None:
                    raise RuntimeError(
                        f"llama-server exited early with code {self._proc.returncode}"
                    )
                time.sleep(2)
        self.stop()
        raise RuntimeError(f"llama-server did not become ready within {timeout}s")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
