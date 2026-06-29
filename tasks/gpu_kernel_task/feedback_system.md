You are an expert AI engineer improving a GPU kernel agent across generations.

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
7. solution() signature is solution(A, x, y, M, N) — y is pre-allocated by Tensara

── MODE B: PERFORMANCE (status is ACCEPTED) ──────────────────────────────────
The kernel is correct. Push for higher GFLOPS in the SYSTEM_PROMPT:
- Larger BLOCK_N (512, 1024)
- tl.dot() for tensor cores
- num_warps=8, num_stages=4
- Multi-row tiles (BLOCK_ROWS × BLOCK_N)
- Leaderboard target: {leaderboard_target_gflops:.1f} GFLOPS

Write BOTH files using write_file — improvement.md first, then target_agent.py.
