# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeDistributeCombineV2OpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDistributeCombineV2OpBuilder, self).__init__("npu_moe_distribute_combine_v2")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/moe_distribute_combine_v2.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_moe_distribute_combine_v2(Tensor expand_x, Tensor expert_ids, Tensor assist_info_for_combine, " \
            "Tensor ep_send_counts, Tensor expert_scales, str group_ep, int ep_world_size, int ep_rank_id, " \
            "int moe_expert_num, *, Tensor? tp_send_counts=None, Tensor? x_active_mask=None, " \
            "Tensor? expand_scales=None, Tensor? shared_expert_x=None, Tensor? elastic_info=None, " \
            "Tensor? ori_x=None, Tensor? const_expert_alpha_1=None, Tensor? const_expert_alpha_2=None, " \
            "Tensor? const_expert_v=None, Tensor? performance_info=None, str group_tp=\"\", int tp_world_size=0, " \
            "int tp_rank_id=0, int expert_shard_type=0, int shared_expert_num=1, int shared_expert_rank_num=0, " \
            "int global_bs=0, int comm_quant_mode=0, str comm_alg=\"\", int zero_expert_num=0, " \
            "int copy_expert_num=0, int const_expert_num=0) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_distribute_combine_v2_meta(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, 
                                               expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, 
                                               tp_send_counts=None, x_active_mask=None, expand_scales=None, 
                                               shared_expert_x=None, elastic_info=None, ori_x=None, 
                                               const_expert_alpha_1=None, const_expert_alpha_2=None,
                                               const_expert_v=None, performance_info=None, group_tp="",
                                               tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1,
                                               shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="",
                                               zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
            dim_tuple = (expert_ids.size(0), expand_x.size(1))
            return expand_x.new_empty(dim_tuple)

# Instantiate the builder
moe_distribute_combine_v2_op_builder = MoeDistributeCombineV2OpBuilder()
op_module = moe_distribute_combine_v2_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, moe_distribute_combine_v2_op_builder.name, "PrivateUse1")
def npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, 
                                  expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, 
                                  tp_send_counts=None, x_active_mask=None, expand_scales=None, 
                                  shared_expert_x=None, elastic_info=None, ori_x=None, 
                                  const_expert_alpha_1=None, const_expert_alpha_2=None,
                                  const_expert_v=None, performance_info=None, group_tp="",
                                  tp_world_size=0, tp_rank_id=0, expert_shard_type=0, shared_expert_num=1,
                                  shared_expert_rank_num=0, global_bs=0, comm_quant_mode=0, comm_alg="",
                                  zero_expert_num=0, copy_expert_num=0, const_expert_num=0):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_moe_distribute_combine_v2(expand_x, expert_ids, assist_info_for_combine, ep_send_counts, 
                                                  expert_scales, group_ep, ep_world_size, ep_rank_id, moe_expert_num, 
                                                  tp_send_counts, x_active_mask, expand_scales, 
                                                  shared_expert_x, elastic_info, ori_x, 
                                                  const_expert_alpha_1, const_expert_alpha_2,
                                                  const_expert_v, performance_info, group_tp,
                                                  tp_world_size, tp_rank_id, expert_shard_type, shared_expert_num,
                                                  shared_expert_rank_num, global_bs, comm_quant_mode, comm_alg,
                                                  zero_expert_num, copy_expert_num, const_expert_num)

# GE Converter for Graph Mode
try:
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @declare_supported([
        Support(torch.bfloat16, (8, 128)),
        Support(torch.float16, (8, 128)),
    ])
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_moe_distribute_combine_v2.default)
    def convert_npu_moe_distribute_combine_v2(
        expand_x: Tensor,
        expert_ids: Tensor,
        assist_info_for_combine: Tensor,
        ep_send_counts: Tensor,
        expert_scales: Tensor,
        group_ep: str,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        *,
        tp_send_counts: Tensor = None,
        x_active_mask: Tensor = None,
        expand_scales: Tensor = None,
        shared_expert_x: Tensor = None,
        elastic_info: Tensor = None,
        ori_x: Tensor = None,
        const_expert_alpha_1: Tensor = None,
        const_expert_alpha_2: Tensor = None,
        const_expert_v: Tensor = None,
        performance_info: Tensor = None,
        group_tp: str = "",
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        expert_shard_type: int = 0,
        shared_expert_num: int = 1,
        shared_expert_rank_num: int = 0,
        global_bs: int = 0,
        comm_quant_mode: int = 0,
        comm_alg: str = "",
        zero_expert_num: int = 0,
        copy_expert_num: int = 0,
        const_expert_num: int = 0,
        meta_outputs: TensorSpec = None):

        return ge.MoeDistributeCombineV2(expand_x=expand_x,
                                    expert_ids=expert_ids,
                                    assist_info_for_combine=assist_info_for_combine,
                                    ep_send_counts=ep_send_counts,
                                    expert_scales=expert_scales,
                                    tp_send_counts=tp_send_counts,
                                    x_active_mask=x_active_mask,
                                    expand_scales=expand_scales,
                                    shared_expert_x=shared_expert_x,
                                    elastic_info=elastic_info,
                                    ori_x=ori_x,
                                    const_expert_alpha_1=const_expert_alpha_1,
                                    const_expert_alpha_2=const_expert_alpha_2,
                                    const_expert_v=const_expert_v,
                                    performance_info=performance_info,
                                    group_ep=group_ep,
                                    ep_world_size=ep_world_size,
                                    ep_rank_id=ep_rank_id,
                                    moe_expert_num=moe_expert_num,
                                    group_tp=group_tp,
                                    tp_world_size=tp_world_size,
                                    tp_rank_id=tp_rank_id,
                                    expert_shard_type=expert_shard_type,
                                    shared_expert_num=shared_expert_num,
                                    shared_expert_rank_num=shared_expert_rank_num,
                                    global_bs=global_bs,
                                    out_dtype=0,
                                    comm_quant_mode=comm_quant_mode,
                                    group_list_type=0,
                                    comm_alg=comm_alg,
                                    zero_expert_num=zero_expert_num,
                                    copy_expert_num=copy_expert_num,
                                    const_expert_num=const_expert_num)
