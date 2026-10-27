@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_THRESHOLD": 8, "NUM_CORES": 24}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 128, "BLOCK_THRESHOLD": 8, "NUM_CORES": 24}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 256, "BLOCK_THRESHOLD": 8, "NUM_CORES": 24}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_THRESHOLD": 8, "NUM_CORES": 24}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_THRESHOLD: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # When the workload is large, the diagonal partitioning strategy can be enabled for optimization  
    # For the case where M=32 and BLOCK_M=32, NUM_BLOCKS_M=1, it usually enters the else branch (sequential partitioning)  
    if NUM_BLOCKS_M >= BLOCK_THRESHOLD and NUM_BLOCKS_N >= BLOCK_THRESHOLD:
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
            # Implementation of 8 * 8 diagonal partitioning strategy  
            curThresholdM = BLOCK_THRESHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_THRESHOLD * BLOCK_THRESHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_THRESHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_THRESHOLD
            
            cond_val = (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN
            if block_idx % (NUM_BLOCKS_N * BLOCK_THRESHOLD) < cond_val:
                curThresholdN = BLOCK_THRESHOLD
            else:
                curThresholdN = NUM_BLOCKS_N % BLOCK_THRESHOLD
            
            localRelativeBlock = block_idx % (BLOCK_THRESHOLD * NUM_BLOCKS_N) % (BLOCK_THRESHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_THRESHOLD * NUM_BLOCKS_N) * BLOCK_THRESHOLD
            
            # Calculate the least common multiple to determine the base block coordinates  
            # Use a for loop to simulate GCD calculation and avoid using while  
            x = curThresholdM
            y = curThresholdN if curThresholdM > curThresholdN else curThresholdN
            for _ in range(8): # Assuming the threshold is small, a limited number of iterations will suffice

                if y != 0:
                    temp = x % y
                    x = y
                    y = temp
            lcm = curThresholdM * curThresholdN // x
            
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_THRESHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_THRESHOLD
            
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            # Computation logic
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                # Load A
                offs_am = m_start + tl.arange(0, BLOCK_M)
                offs_ak = k_start + tl.arange(0, BLOCK_K)
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
                mask_a = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                tl.compile_hint(a, "dot_pad_only_k")
                
                # Load B
                offs_bk = k_start + tl.arange(0, BLOCK_K)
                offs_bn = n_start + tl.arange(0, BLOCK_N)
                b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
                mask_b = (offs_bk[:, None] < K) & (offs_bn[None, :] < N)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                tl.compile_hint(b, "dot_pad_only_k")
                
                accumulator = tl.dot(a, b, accumulator)
            
            # Store C
            offs_cm = m_start + tl.arange(0, BLOCK_M)
            offs_cn = n_start + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
            mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_c)

    else:
        # Traditional sequential block distribution
        for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                # Load A
                offs_am = m_start + tl.arange(0, BLOCK_M)
                offs_ak = k_start + tl.arange(0, BLOCK_K)
                a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
                mask_a = (offs_am[:, None] < M) & (offs_ak[None, :] < K)
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)
                tl.compile_hint(a, "dot_pad_only_k")
                
                # Load B
                offs_bk = k_start + tl.arange(0, BLOCK_K)
                offs_bn = n_start + tl.arange(0, BLOCK_N)
                b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
                mask_b = (offs_bk[:, None] < K) & (offs_bn[None, :] < N)
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)
                tl.compile_hint(b, "dot_pad_only_k")
                
                accumulator = tl.dot(a, b, accumulator)
            
            # Store C
            offs_cm = m_start + tl.arange(0, BLOCK_M)
            offs_cn = n_start + tl.arange(0, BLOCK_N)
            c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
            mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator.to(tl.float16), mask=mask_c)