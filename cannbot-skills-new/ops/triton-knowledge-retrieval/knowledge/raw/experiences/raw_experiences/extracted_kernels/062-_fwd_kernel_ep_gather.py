@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    start_cur_token = tl.program_id(1)
    grid_num = tl.num_programs(1)
    for cur_token in range(start_cur_token, total_token_num, grid_num):
        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_index in range(0, topk_num):
            expert_id = tl.load(recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index)
            if expert_id >= 0:
                source_token_index = tl.load(input_index + cur_token * input_index_stride0 + topk_index)
                acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + topk_index)
                tmp = tl.load(input_tensor + source_token_index * input_tensor_stride0 + cur_block * BLOCK_D + off_d)
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )