# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import math
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    0: torch.uint8,
    1: torch.int8,
    2: torch.int16,
    3: torch.int32,
    4: torch.int64,
    5: torch.float16,
    6: torch.float32,
    7: torch.float64,
    8: torch.complex32,
    9: torch.complex64,
    10: torch.complex128,
    11: torch.bool,
    12: torch.qint8,
    13: torch.quint8,
    14: torch.qint32,
    15: torch.bfloat16,
    16: torch.quint4x2,
    21: torch.bits8,
    23: torch.float8_e5m2,
    24: torch.float8_e4m3fn,
    285: torch.uint8,  # torch_npu.int4 use torch.uint8
    290: torch.uint8,  # torch_npu.hifloat8 use torch.uint8
    291: torch.float8_e5m2,
    292: torch.float8_e4m3fn,
    296: torch.uint8,  # torch_npu.float4_e2m1fn_x2 use torch.uint8
    297: torch.uint8,  # torch_npu.float4_e1m2fn_x2 use torch.uint8
}


class MoeDistributeDispatchV3OpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeDispatchV3OpBuilder, self).__init__("npu_moe_distribute_dispatch_v3")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/moe_distribute_dispatch_v3.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_moe_distribute_dispatch_v3(Tensor context, Tensor x, Tensor expert_ids, " \
            "int ep_world_size, int ep_rank_id, int moe_expert_num, int ccl_buffer_size, " \
            "*, Tensor? scales=None, Tensor? x_active_mask=None, " \
            "Tensor? expert_scales=None, Tensor? elastic_info=None, Tensor? performance_info=None, "\
            "int tp_world_size=0, int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, " \
            "int shared_expert_rank_num=0, int quant_mode=0, int global_bs=0, int expert_token_nums_type=1, " \
            "str comm_alg=\"\", int zero_expert_num=0, int copy_expert_num=0, int const_expert_num=0, " \
            "int? y_dtype=None, int? x_dtype=None, int? scales_dtype=None) " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        def get_dispatch_dynamic_scales_dtype(x, scales, quant_mode):
            dynamic_scales_dtype = torch.float32
            if quant_mode == 0:
                if x.dtype != torch.bfloat16 and x.dtype != torch.float16 and scales is not None:
                    dynamic_scales_dtype = scales.dtype
            elif quant_mode == 4:
                dynamic_scales_dtype = torch.uint8  # float8_e8m0
            return dynamic_scales_dtype

        def get_dispatch_dynamic_shape(scales, quant_mode, a, h):
            shape = tuple([a])
            if quant_mode == 0 and scales is not None:
                if scales.dim() < 2:
                    raise RuntimeError(f"Expected scales to be at least 2-d, but got {scales.dim()}-d.")
                shape = tuple([a * scales.shape[1]])
            elif quant_mode == 2:
                shape = tuple([a])
            elif quant_mode == 3:
                shape = tuple([a, math.ceil(h / 128)])
            elif quant_mode == 4:
                shape = tuple([a, (math.ceil(h / 32) + 1) // 2 * 2])
            return shape

        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_distribute_dispatch_v3_meta(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num,
                                                ccl_buffer_size, scales=None, x_active_mask=None, expert_scales=None,
                                                elastic_info=None, performance_info=None, tp_world_size=0, tp_rank_id=0,
                                                expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                                                quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                                zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None,
                                                x_dtype=None, scales_dtype=None):
            torch._check(
                (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
                lambda: (
                    f"ep_rank_id should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {ep_rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (shared_expert_rank_num >= 0) and (shared_expert_rank_num < ep_world_size),
                lambda: (
                    f"shared_expert_rank_num should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            is_shared_default = ((shared_expert_num == 1) and (shared_expert_rank_num == 0))
            is_no_shared = ((shared_expert_num == 0) and (shared_expert_rank_num == 0))
            is_valid_shared = (
                (shared_expert_num > 0)
                and ((shared_expert_rank_num // shared_expert_num) > 0)
                and ((shared_expert_rank_num % shared_expert_num) == 0)
            )
            torch._check(
                is_shared_default or is_no_shared or is_valid_shared,
                lambda: (
                    f"shared expert setting invalid, "
                    f"got {shared_expert_num=}, {shared_expert_rank_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                expert_token_nums_type in [0, 1],
                lambda: "the expert_token_nums_type should be 0 or 1" + ops_error(ErrCode.VALUE)
            )

            bs = x.size(0)
            h = x.size(1)
            k = expert_ids.size(1)
            shared_front = (expert_shard_type == 0)
            out_dtype = torch.int8
            local_moe_expert_num = 1
            global_bs_real = 0
            if global_bs == 0:
                global_bs_real = bs * ep_world_size
            else:
                global_bs_real = global_bs
            a = 0
            if shared_front:
                if ep_rank_id < shared_expert_rank_num:
                    local_moe_expert_num = 1
                    max_bs = global_bs_real // ep_world_size
                    rank_num_per_shared_expert = shared_expert_rank_num // shared_expert_num
                    max_shared_group_num = ((ep_world_size + rank_num_per_shared_expert - 1)
                                            // rank_num_per_shared_expert)
                    a = max_bs * max_shared_group_num
                else:
                    local_moe_expert_num = moe_expert_num // (ep_world_size - shared_expert_rank_num)
                    a = global_bs_real * min(local_moe_expert_num, k)
                if elastic_info is not None:
                    if ((is_shared_default) or (is_no_shared)):
                        local_moe_expert_num = max(local_moe_expert_num, 
                                                moe_expert_num // (ep_world_size - shared_expert_rank_num))
                        a = global_bs_real * min(local_moe_expert_num, k)
                    else:
                        max_bs = global_bs_real // ep_world_size
                        rank_num_per_shared_expert = shared_expert_rank_num // shared_expert_num
                        max_shared_group_num = ((ep_world_size + rank_num_per_shared_expert - 1)
                                                // rank_num_per_shared_expert)
                        a = max(max_bs * max_shared_group_num, 
                                global_bs_real * min(moe_expert_num // (ep_world_size - shared_expert_rank_num), k))
                        local_moe_expert_num = max(local_moe_expert_num, 
                                                moe_expert_num // (ep_world_size - shared_expert_rank_num))

            ep_recv_cnt_num = 0
            if tp_world_size == 2:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num * tp_world_size
            else:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num
            if quant_mode == 0:
                out_dtype = x.dtype

            assist_info_for_combine_shape = max(bs * k, a * 128)
            expand_x = x.new_empty(tuple([max(a, a * tp_world_size), h]), dtype=out_dtype)
            dynamic_scales_dtype = get_dispatch_dynamic_scales_dtype(x, scales, quant_mode)
            if tp_world_size == 0:
                dynamic_scales = x.new_empty((a), dtype=dynamic_scales_dtype)
            elif tp_world_size == 1:
                dynamic_scales_shape = get_dispatch_dynamic_shape(scales, quant_mode, a, h)
                dynamic_scales = x.new_empty(dynamic_scales_shape, dtype=dynamic_scales_dtype)
            else:
                dynamic_scales = x.new_empty((a * tp_world_size), dtype=dynamic_scales_dtype)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int64)
            ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
            tp_recv_counts = x.new_empty((tp_world_size), dtype=torch.int32)
            expand_scales = x.new_empty((0), dtype=torch.float32)
            if expert_scales is not None:
                ep_recv_cnt_num = ep_world_size * local_moe_expert_num + global_bs_real * 2 * k * (ep_world_size // 8)
                ep_recv_counts = x.new_empty((ep_recv_cnt_num), dtype=torch.int32)
                expand_scales = x.new_empty((a), dtype=torch.float32)
                assist_info_for_combine_shape = max(assist_info_for_combine_shape, 
                                                    global_bs_real * 2 * k * (ep_world_size // 8))
            expand_idx = x.new_empty(assist_info_for_combine_shape, dtype=torch.int32)
            return (expand_x, dynamic_scales, expand_idx, expert_token_nums,
                    ep_recv_counts, tp_recv_counts, expand_scales)

# Instantiate the builder
moe_distribute_dispatch_v3_op_builder = MoeDistributeDispatchV3OpBuilder()
op_module = moe_distribute_dispatch_v3_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, moe_distribute_dispatch_v3_op_builder.name, "PrivateUse1")
def npu_moe_distribute_dispatch_v3(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, 
                                   ccl_buffer_size, scales=None, x_active_mask=None, expert_scales=None,
                                   elastic_info=None, performance_info=None, tp_world_size=0, tp_rank_id=0,
                                   expert_shard_type=0, shared_expert_num=1, shared_expert_rank_num=0,
                                   quant_mode=0, global_bs=0, expert_token_nums_type=1, comm_alg="",
                                   zero_expert_num=0, copy_expert_num=0, const_expert_num=0, y_dtype=None,
                                   x_dtype=None, scales_dtype=None):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_moe_distribute_dispatch_v3(context, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num,
                                                    ccl_buffer_size, scales, x_active_mask, expert_scales, elastic_info,
                                                    performance_info, tp_world_size, tp_rank_id,
                                                    expert_shard_type, shared_expert_num, shared_expert_rank_num,
                                                    quant_mode, global_bs, expert_token_nums_type, comm_alg,
                                                    zero_expert_num, copy_expert_num, const_expert_num, y_dtype,
                                                    x_dtype, scales_dtype)
