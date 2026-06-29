# Matrix Vector Multiplication

**Problem slug:** `matrix-vector`  
**Difficulty:** Easy  
**Tensara URL:** https://tensara.org/problems/matrix-vector

## Description

Compute the product of a 2D matrix `A` and a 1D vector `x`:

```
y = A @ x
```

where `A` is an `M × N` matrix and `x` is a vector of length `N`, producing output vector `y` of length `M`.

## Constraints

- All tensors are `float32`
- `M`, `N` are powers of 2
- Solution must be implemented as a Triton kernel in `sol.py`

## Interface

Tensara calls `solution(A, x, y, M, N)` — y is pre-allocated by Tensara, do NOT allocate it.
Do NOT `import torch` — Tensara forbids it.

```python
def solution(A, x, y, M, N):
    # A: (M, N) float32 tensor
    # x: (N,) float32 tensor
    # y: (M,) float32 tensor — pre-allocated output, write results into it
    # M, N: integer dimensions
    BLOCK_N = 256
    matvec_kernel[(M,)](A, x, y, M, N, BLOCK_N=BLOCK_N)
```

## Test Case Sizes

| M | N |
|---|---|
| 1024 | 1024 |
| 4096 | 4096 |
| 8192 | 8192 |

## Evaluation

Solutions are submitted to Tensara and benchmarked on H100.  
Metric: average latency (ms) and GFLOPS across all test cases.
