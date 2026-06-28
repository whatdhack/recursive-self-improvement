"""
Generate an animated explainer video for the RSI loop.
Produces: rsi_explainer.mp4
"""

import os
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

OUT_DIR = Path(tempfile.mkdtemp(prefix="rsi_frames_"))
FPS = 24
W, H = 1920, 1080
DPI = 96


# ── colour palette ────────────────────────────────────────────────────────────
BG      = "#0f1117"
BLUE    = "#3b82f6"
YELLOW  = "#f59e0b"
GREEN   = "#22c55e"
RED     = "#ef4444"
PURPLE  = "#a855f7"
TEAL    = "#06b6d4"
WHITE   = "#f1f5f9"
GREY    = "#475569"
LGREY   = "#94a3b8"


def fig():
    f = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return f


def save_frame(f, idx):
    path = OUT_DIR / f"frame_{idx:05d}.png"
    f.savefig(path, facecolor=BG, dpi=DPI)
    plt.close(f)
    return path


def add_title(ax, text, y=0.93, size=44, color=WHITE):
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=size,
            color=color, fontweight="bold",
            fontfamily="monospace")


def add_subtitle(ax, text, y=0.84, size=22, color=LGREY):
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=size,
            color=color, fontfamily="monospace")


def base_ax(f):
    ax = f.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG)
    return ax


def draw_box(ax, x, y, w, h, color, label, sublabel="", alpha=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.01",
                         linewidth=2, edgecolor=color,
                         facecolor=color + "22", alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y + 0.012, label, ha="center", va="center",
            fontsize=16, color=color, fontweight="bold",
            fontfamily="monospace")
    if sublabel:
        ax.text(x, y - 0.025, sublabel, ha="center", va="center",
                fontsize=12, color=LGREY, fontfamily="monospace")


def arrow(ax, x0, y0, x1, y1, color=GREY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2, mutation_scale=18))


# ── SLIDE HELPERS ─────────────────────────────────────────────────────────────

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def render_frames(draw_fn, n_frames, start_idx):
    for i in range(n_frames):
        f = fig()
        ax = base_ax(f)
        draw_fn(ax, i, n_frames)
        save_frame(f, start_idx + i)
    return start_idx + n_frames


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title card  (3 s)
# ══════════════════════════════════════════════════════════════════════════════

def slide_title(ax, i, n):
    progress = i / max(n - 1, 1)

    # animated glow line
    ax.axhline(0.5, color=BLUE, lw=1, alpha=0.15)

    alpha = clamp(progress * 3)
    ax.text(0.5, 0.60,
            "Recursive Self-Improvement",
            ha="center", va="center", fontsize=58,
            color=WHITE, fontweight="bold",
            fontfamily="monospace", alpha=alpha)

    ax.text(0.5, 0.50,
            "An AI that fine-tunes its own weights from failures",
            ha="center", va="center", fontsize=26,
            color=LGREY, fontfamily="monospace",
            alpha=clamp((progress - 0.3) * 2))

    ax.text(0.5, 0.40,
            "Meta Agent  →  Target Agent  →  GPU Eval  →  Curator  →  LoRA  →  repeat",
            ha="center", va="center", fontsize=18,
            color=TEAL, fontfamily="monospace",
            alpha=clamp((progress - 0.6) * 3))

    ax.text(0.5, 0.10,
            "AIEWF Hackathon 2026",
            ha="center", va="center", fontsize=16,
            color=GREY, fontfamily="monospace",
            alpha=clamp((progress - 0.7) * 3))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — The task  (3 s)
# ══════════════════════════════════════════════════════════════════════════════

def slide_task(ax, i, n):
    progress = i / max(n - 1, 1)

    add_title(ax, "The Task", size=40)
    add_subtitle(ax, "Write a Triton GPU kernel that computes  y = A @ x  as fast as possible", size=20)

    # formula box
    ax.text(0.5, 0.62,
            "y  =  A  @  x",
            ha="center", va="center", fontsize=48,
            color=GREEN, fontweight="bold", fontfamily="monospace",
            alpha=clamp(progress * 4))

    details = [
        ("Matrix A", "M × N  float16"),
        ("Vector x", "N      float16"),
        ("Output y", "M      float32"),
        ("Hardware", "H100 GPU (Tensara)"),
        ("Target",   "644 GFLOPS  (leaderboard best)"),
    ]
    for j, (k, v) in enumerate(details):
        a = clamp((progress - 0.2 - j * 0.08) * 6)
        ax.text(0.32, 0.46 - j * 0.065, k + " :", ha="right",
                fontsize=17, color=LGREY, fontfamily="monospace", alpha=a)
        ax.text(0.34, 0.46 - j * 0.065, v, ha="left",
                fontsize=17, color=WHITE, fontfamily="monospace", alpha=a)

    ax.text(0.5, 0.10,
            "Language: Triton (Python GPU programming)",
            ha="center", fontsize=15, color=GREY,
            fontfamily="monospace",
            alpha=clamp((progress - 0.8) * 5))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — 5-step loop diagram  (5 s, animated step-by-step)
# ══════════════════════════════════════════════════════════════════════════════

STEPS = [
    (BLUE,   "① META AGENT",    "Kimi K2.6",         "writes target_agent.py"),
    (YELLOW, "② TARGET AGENT",  "Kimi K2.5",         "writes sol.py"),
    (GREEN,  "③ TENSARA EVAL",  "H100 GPU",          "correctness + benchmark"),
    (RED,    "④ CURATOR",       "Gemma 4 31B",       "failure → train.jsonl"),
    (PURPLE, "⑤ LORA TRAINER",  "llama-finetune CPU","updates Gemma4-1B weights"),
]

XS = [0.12, 0.30, 0.50, 0.68, 0.87]
YB = 0.46


def slide_loop(ax, i, n):
    progress = i / max(n - 1, 1)
    n_visible = math.ceil(progress * (len(STEPS) + 1))

    add_title(ax, "The RSI Loop — 5 Steps per Generation", size=34)

    for k, (color, label, model, sub) in enumerate(STEPS):
        if k >= n_visible:
            break
        a = clamp((progress - k / (len(STEPS) + 1)) * (len(STEPS) + 1) * 1.5)
        draw_box(ax, XS[k], YB, 0.15, 0.22, color, label, model + "\n" + sub, alpha=a)
        ax.text(XS[k], YB - 0.17, sub, ha="center", fontsize=11,
                color=LGREY, fontfamily="monospace", alpha=a)

        if k > 0 and k < n_visible:
            arrow(ax, XS[k-1] + 0.08, YB, XS[k] - 0.08, YB, color=GREY)

    # loop-back arrow
    if n_visible > len(STEPS):
        a = clamp((progress - len(STEPS) / (len(STEPS) + 1)) * 8)
        ax.annotate("", xy=(XS[0], YB - 0.13), xytext=(XS[-1], YB - 0.13),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL,
                                    lw=2.5, mutation_scale=20,
                                    connectionstyle="arc3,rad=-0.35"),
                    alpha=a)
        ax.text(0.5, 0.20, "← repeat with improved weights →",
                ha="center", fontsize=16, color=TEAL,
                fontfamily="monospace", alpha=a)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Step deep-dives  (one per step, 2 s each)
# ══════════════════════════════════════════════════════════════════════════════

DEEPDIVES = [
    (BLUE,   "① META AGENT — Kimi K2.6",
     ["Reads the task description (task.md)",
      "Writes target_agent.py — a Python script",
      "Instructs the target LLM: prompt, model, max_tokens",
      "Strips markdown fences, saves to gen-N/target_agent.py",
      "Takes ~30-50 seconds per generation"]),

    (YELLOW, "② TARGET AGENT — Kimi K2.5",
     ["Executes target_agent.py as a subprocess",
      "Calls Kimi K2.5 via DO Model Studio API",
      "Generates the Triton kernel code",
      "Writes sol.py to the generation directory",
      "Takes ~2 minutes per generation"]),

    (GREEN,  "③ TENSARA EVALUATION — H100",
     ["Static AST analysis first (catch obvious bugs)",
      "Correctness check vs reference implementation",
      "If correct → full benchmark for GFLOPS + latency",
      "Streams results via SSE from tensara.org",
      "Saves results.json with status + per-shape metrics"]),

    (RED,    "④ CURATOR — Gemma 4 31B",
     ["Runs only on FAILED kernels",
      "Reads the error message from results.json",
      "Reads the failed sol.py code",
      "Produces a (prompt, completion) JSON training pair",
      "Appends to train.jsonl — the self-generated dataset"]),

    (PURPLE, "⑤ LORA TRAINER — llama-finetune on CPU",
     ["Fine-tunes Gemma4-1B on train.jsonl",
      "Applies LoRA adapter (rank, alpha, LR, epochs)",
      "Hyperparams self-adjust based on ΔPerformance:",
      "  improving → hold  |  stalled → boost rank",
      "Merges adapter into gemma4-genN.gguf for next gen"]),
]


def slide_deepdive(step_idx):
    color, title, bullets = DEEPDIVES[step_idx]

    def draw(ax, i, n):
        progress = i / max(n - 1, 1)
        add_title(ax, title, size=32, color=color)

        for j, b in enumerate(bullets):
            a = min(1.0, max(0, (progress - j * 0.15) * 6))
            ax.text(0.15, 0.64 - j * 0.10, "▸  " + b,
                    fontsize=20, color=WHITE,
                    fontfamily="monospace", alpha=a)

    return draw


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Live output  (4 s)
# ══════════════════════════════════════════════════════════════════════════════

TERMINAL_LINES = [
    ("dim",    "RSI Loop — matrix-vector  (GPU: H100)"),
    ("dim",    "────────────────────────────────────────────────────"),
    ("cyan",   "[08:13:58] [Gen 0] starting"),
    ("blue",   "[08:13:59]   [meta] calling Kimi K2.6..."),
    ("blue",   "[08:14:42]   [meta] wrote target_agent.py (1799 chars)"),
    ("yellow", "[08:14:42]   [target] running target_agent.py (model: kimi-k2.5)..."),
    ("yellow", "[08:16:35]   [target] sol.py written (824 bytes)"),
    ("green",  "[08:16:35] [Gen 0] evaluating via Tensara (H100)..."),
    ("green",  "Leaderboard target: 0.0853 ms  (644.5 GFLOPS)"),
    ("red",    "Status: COMPILE_ERROR  ← FAILED"),
    ("purple", "[08:16:51]   [curate] generating training pair from failure..."),
    ("purple", "  [curate] appended training pair → train.jsonl (1 total)"),
    ("dim",    "────────────────────────────────────────────────────"),
    ("cyan",   "[08:16:55] [Gen 1] starting"),
    ("blue",   "[08:16:55]   [meta] calling Kimi K2.6..."),
    ("yellow", "[08:18:44]   [target] running target_agent.py (model: kimi-k2.5)..."),
    ("yellow", "[08:20:27]   [target] sol.py written (794 bytes)"),
    ("purple", "  [curate] appended training pair → train.jsonl (2 total)"),
]

TERM_COLORS = {
    "dim":    GREY,
    "cyan":   TEAL,
    "blue":   BLUE,
    "yellow": YELLOW,
    "green":  GREEN,
    "red":    RED,
    "purple": PURPLE,
}


def slide_terminal(ax, i, n):
    progress = i / max(n - 1, 1)
    add_title(ax, "Live Run Output", size=34)

    # terminal bg
    box = FancyBboxPatch((0.04, 0.08), 0.92, 0.70,
                         boxstyle="round,pad=0.01",
                         facecolor="#1e2130", edgecolor=GREY, lw=1)
    ax.add_patch(box)

    n_lines = int(progress * len(TERMINAL_LINES)) + 1
    for j, (style, line) in enumerate(TERMINAL_LINES[:n_lines]):
        ax.text(0.06, 0.74 - j * 0.038, line,
                fontsize=13, color=TERM_COLORS[style],
                fontfamily="monospace")


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Why it's recursive  (3 s)
# ══════════════════════════════════════════════════════════════════════════════

RECURSIVE_POINTS = [
    (YELLOW, "Model failures  →  generate training data  (automatically)"),
    (GREEN,  "Training data   →  improves model weights  (LoRA)"),
    (BLUE,   "Better weights  →  write better kernels    (next gen)"),
    (TEAL,   "Better kernels  →  fewer failures          (loop improves)"),
    (WHITE,  "No human labels. No human feedback. Fully autonomous."),
]


def slide_recursive(ax, i, n):
    progress = i / max(n - 1, 1)
    add_title(ax, "What Makes It Recursive?", size=38)

    for j, (color, text) in enumerate(RECURSIVE_POINTS):
        a = min(1.0, max(0, (progress - j * 0.15) * 5))
        ax.text(0.5, 0.68 - j * 0.10, text,
                ha="center", fontsize=20,
                color=color, fontfamily="monospace",
                fontweight="bold" if j == len(RECURSIVE_POINTS) - 1 else "normal",
                alpha=a)

    # feedback loop visual
    if progress > 0.8:
        a = (progress - 0.8) * 5
        for k, color in enumerate([YELLOW, GREEN, BLUE, TEAL]):
            angle = k * 90
            x = 0.5 + 0.12 * math.cos(math.radians(angle + 45))
            y = 0.20 + 0.06 * math.sin(math.radians(angle + 45))
            ax.plot(x, y, "o", color=color, markersize=12, alpha=a)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Outro  (3 s)
# ══════════════════════════════════════════════════════════════════════════════

def slide_outro(ax, i, n):
    progress = i / max(n - 1, 1)

    ax.text(0.5, 0.65, "recursive-self-improvement",
            ha="center", va="center", fontsize=48,
            color=WHITE, fontweight="bold", fontfamily="monospace",
            alpha=clamp(progress * 3))

    ax.text(0.5, 0.52, "github.com/whatdhack/recursive-self-improvement",
            ha="center", va="center", fontsize=22,
            color=TEAL, fontfamily="monospace",
            alpha=clamp((progress - 0.3) * 4))

    items = [
        "Kimi K2.6  (meta)   ·  Kimi K2.5  (target)",
        "Gemma 4 31B  (curator)  ·  Gemma4-1B  (LoRA target)",
        "Tensara H100  ·  DigitalOcean Model Studio",
    ]
    for j, t in enumerate(items):
        ax.text(0.5, 0.38 - j * 0.08, t,
                ha="center", fontsize=16, color=LGREY,
                fontfamily="monospace",
                alpha=clamp((progress - 0.5 - j * 0.1) * 5))

    ax.text(0.5, 0.10, "AIEWF Hackathon 2026",
            ha="center", fontsize=18, color=GREY,
            fontfamily="monospace",
            alpha=clamp((progress - 0.8) * 5))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    idx = 0
    slides = [
        (slide_title,      3),
        (slide_task,       3),
        (slide_loop,       5),
    ]
    for step_i in range(5):
        slides.append((slide_deepdive(step_i), 2))
    slides += [
        (slide_terminal,   5),
        (slide_recursive,  3),
        (slide_outro,      3),
    ]

    total = sum(s * FPS for _, s in slides)
    print(f"Rendering {total} frames across {len(slides)} slides...")

    for draw_fn, duration_s in slides:
        n_frames = duration_s * FPS
        idx = render_frames(draw_fn, n_frames, idx)
        print(f"  rendered {n_frames} frames  (total so far: {idx})")

    out = Path(__file__).parent / "rsi_explainer.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(OUT_DIR / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out),
    ]
    print(f"\nStitching video → {out}")
    subprocess.run(cmd, check=True)
    print(f"\nDone!  {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
