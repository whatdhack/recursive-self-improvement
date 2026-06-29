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
