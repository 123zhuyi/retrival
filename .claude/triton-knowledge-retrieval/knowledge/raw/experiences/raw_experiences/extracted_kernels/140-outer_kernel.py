@libentry()
@triton.autotune(
    configs=[
        triton.Config({"tile_size": 1024}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 2048}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 4096}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 8192}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 16384}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 21760}, num_stages=3, num_warps=1),
        triton.Config({"tile_size": 32768}, num_stages=3, num_warps=1),
    ],
    key=["M", "N"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def outer_kernel(
    lhs,
    rhs,
    res,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_LOOP_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)

    m_tasks_num = tl.cdiv(M, BLOCK_M)
    n_tasks_num = tl.cdiv(N, BLOCK_N)
    total_tasks_num = m_tasks_num * n_tasks_num

    if NEED_LOOP_N:
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num
            start_n = task_id % n_tasks_num

            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)

            offset_n = tl.arange(0, BLOCK_N) + start_n * BLOCK_N
            rhs_val = tl.load(rhs + offset_n, mask=offset_n < N)

            res_val = lhs_val[:, None] * rhs_val[None, :]

            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(
                res + offset_r,
                res_val,
                mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
            )
    else:
        offset_n = tl.arange(0, BLOCK_N)
        rhs_val = tl.load(rhs + offset_n)
        for task_id in range(pid, total_tasks_num, num_jobs):
            start_m = task_id // n_tasks_num

            offset_m = tl.arange(0, BLOCK_M) + start_m * BLOCK_M
            lhs_val = tl.load(lhs + offset_m, mask=offset_m < M)

            res_val = lhs_val[:, None] * rhs_val[None, :]

            offset_r = offset_m[:, None] * N + offset_n[None, :]
            tl.store(
                res + offset_r,
                res_val,
                mask=(offset_m[:, None] < M) & (offset_n[None, :] < N),
            )