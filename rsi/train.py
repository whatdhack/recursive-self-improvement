"""
CPU LoRA trainer — wraps llama.cpp's llama-finetune binary.

Each generation:
  1. Reads the current GGUF checkpoint (or base model on Gen 0)
  2. Runs llama-finetune with train.jsonl
  3. Exports the merged next-gen GGUF

Hyperparameters self-adjust based on ΔPerformance from the previous generation.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoraConfig:
    rank: int = 16
    alpha: float = 32.0
    lr: float = 2e-4
    epochs: int = 3
    batch: int = 4
    threads: int = 8

    def as_args(self) -> list[str]:
        return [
            "--lora-r", str(self.rank),
            "--lora-alpha", str(self.alpha),
            "--lr", str(self.lr),
            "--epochs", str(self.epochs),
            "--batch", str(self.batch),
            "--threads", str(self.threads),
        ]


@dataclass
class HyperparamTracker:
    """
    Tracks GFLOPS per generation and adjusts LoRA config.

    Rules:
      - If ΔPerformance > 0: keep config (it's working)
      - If ΔPerformance == 0 or stalled 2 gens: bump rank and LR
      - If ΔPerformance < 0 (regression): reduce LR, increase epochs
    """
    history: list[float] = field(default_factory=list)

    def record(self, gflops: float | None) -> None:
        self.history.append(gflops or 0.0)

    def adjust(self, cfg: LoraConfig) -> LoraConfig:
        if len(self.history) < 2:
            return cfg

        delta = self.history[-1] - self.history[-2]

        if delta > 0:
            # Improving — hold config
            return cfg
        elif delta < 0:
            # Regression — pull back LR, train longer
            return LoraConfig(
                rank=cfg.rank,
                alpha=cfg.alpha,
                lr=max(cfg.lr * 0.5, 5e-5),
                epochs=min(cfg.epochs + 2, 10),
                batch=cfg.batch,
                threads=cfg.threads,
            )
        else:
            # Stalled — push rank and LR up
            return LoraConfig(
                rank=min(cfg.rank * 2, 64),
                alpha=cfg.alpha * 2,
                lr=min(cfg.lr * 1.5, 5e-4),
                epochs=cfg.epochs,
                batch=cfg.batch,
                threads=cfg.threads,
            )


def run_lora_training(
    llama_finetune_bin: Path,
    base_model: Path,
    lora_in: Path | None,
    lora_out: Path,
    train_jsonl: Path,
    cfg: LoraConfig,
) -> float:
    """
    Run llama-finetune and return wall-clock seconds elapsed.
    lora_in=None means Gen 0 (no prior adapter).
    """
    cmd = [
        str(llama_finetune_bin),
        "--model-base", str(base_model),
        "--train-data", str(train_jsonl),
        "--checkpoint-out", str(lora_out),
    ] + cfg.as_args()

    if lora_in and lora_in.exists():
        cmd += ["--checkpoint-in", str(lora_in)]

    print(f"  [train] {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"llama-finetune failed (exit {result.returncode})")

    print(f"  [train] LoRA pass complete in {elapsed:.0f}s → {lora_out}")
    return elapsed


def merge_lora(
    llama_export_bin: Path,
    base_model: Path,
    lora_path: Path,
    merged_out: Path,
) -> None:
    """Bake LoRA adapter into a standalone GGUF (llama-export-lora)."""
    cmd = [
        str(llama_export_bin),
        "--model-base", str(base_model),
        "--lora", str(lora_path),
        "--output", str(merged_out),
    ]
    print(f"  [merge] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"llama-export-lora failed (exit {result.returncode})")
    print(f"  [merge] merged model → {merged_out}")


def save_hp_log(run_dir: Path, gen: int, cfg: LoraConfig, elapsed: float) -> None:
    log_path = run_dir / "hyperparam_log.jsonl"
    entry = {
        "gen": gen,
        "rank": cfg.rank,
        "alpha": cfg.alpha,
        "lr": cfg.lr,
        "epochs": cfg.epochs,
        "batch": cfg.batch,
        "elapsed_s": round(elapsed, 1),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
