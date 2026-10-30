# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @auto_convert_to_tensor([False, False, False, False, False, False, False, False],
        [False, False, False, True, True, True, True, True])
    def MoeDistributeDispatchV3(context: Tensor, x: Tensor, expert_ids: Tensor, scales: Optional[Tensor],
                                x_active_mask: Optional[Tensor], expert_scales: Optional[Tensor], 
                                elastic_info: Optional[Tensor], performance_info: Optional[Tensor], *, 
                                ep_world_size: int, ep_rank_id: int, moe_expert_num: int, ccl_buffer_size: int,
                                tp_world_size: int = 0, tp_rank_id: int = 0, expert_shard_type: int = 0,
                                shared_expert_num: int = 1, shared_expert_rank_num: int = 0, 
                                quant_mode: int = 0, global_bs: int = 0, expert_token_nums_type: int = 1,
                                comm_alg: str = "", zero_expert_num: int = 0, copy_expert_num: int = 0,
                                const_expert_num: int = 0, y_dtype: int = 28, dependencies=[], node_name=None):
        """REG_OP(MoeDistributeDispatchV3)\n
        .INPUT(context, TensorType({DT_INT32}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(expert_ids, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(x_active_mask, TensorType({DT_BOOL}))\n
        .OPTIONAL_INPUT(expert_scales, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(elastic_info, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(performance_info, TensorType({DT_INT64}))\n
        .OUTPUT(expand_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}))\n
        .OUTPUT(dynamic_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))\n
        .OUTPUT(expert_token_nums, TensorType({DT_INT64}))\n
        .OUTPUT(ep_recv_count, TensorType({DT_INT32}))\n
        .OUTPUT(tp_recv_count, TensorType({DT_INT32}))\n
        .OUTPUT(expand_scales, TensorType({DT_FLOAT}))\n
        .REQUIRED_ATTR(ep_world_size, Int)\n
        .REQUIRED_ATTR(ep_rank_id, Int)\n
        .REQUIRED_ATTR(moe_expert_num, Int)\n
        .REQUIRED_ATTR(ccl_buffer_size, Int)\n
        .ATTR(tp_world_size, Int, 0)\n
        .ATTR(tp_rank_id, Int, 0)\n
        .ATTR(expert_shard_type, Int, 0)\n
        .ATTR(shared_expert_num, Int, 1)\n
        .ATTR(shared_expert_rank_num, Int, 0)\n
        .ATTR(quant_mode, Int, 0)\n
        .ATTR(global_bs, Int, 0)\n
        .ATTR(expert_token_nums_type, Int, 1)\n
        .ATTR(comm_alg, String, "")\n
        .ATTR(zero_expert_num, Int, 0)\n
        .ATTR(copy_expert_num, Int, 0)\n
        .ATTR(const_expert_num, Int, 0)\n
        .ATTR(y_dtype, Int, 28)\n
        """
        # process inputs
        inputs = {
            "context": context,
            "x": x,
            "expert_ids": expert_ids,
            "scales": scales,
            "x_active_mask": x_active_mask,
            "expert_scales": expert_scales,
            "elastic_info": elastic_info,
            "performance_info": performance_info,
        }

        # process attrs
        attrs = {
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "ccl_buffer_size": attr.Int(ccl_buffer_size),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "expert_shard_type": attr.Int(expert_shard_type),
            "shared_expert_num": attr.Int(shared_expert_num),
            "shared_expert_rank_num": attr.Int(shared_expert_rank_num),
            "quant_mode": attr.Int(quant_mode),
            "global_bs": attr.Int(global_bs),
            "expert_token_nums_type": attr.Int(expert_token_nums_type),
            "comm_alg": attr.Str(comm_alg),
            "zero_expert_num": attr.Int(zero_expert_num),
            "copy_expert_num": attr.Int(copy_expert_num),
            "const_expert_num": attr.Int(const_expert_num),
            "y_dtype": attr.Int(y_dtype),
        }

        # process outputs
        outputs = [
            "expand_x",
            "dynamic_scales",
            "assist_info_for_combine",
            "expert_token_nums",
            "ep_recv_count",
            "tp_recv_count",
            "expand_scales",
        ]

        return ge_op(
            op_type="MoeDistributeDispatchV3",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeDistributeDispatchV3") \
            .input("context", "DT_INT32") \
            .input("x", "DT_BF16, DT_FLOAT16") \
            .input("expert_ids", "DT_INT32") \
            .optional_input("scales", "DT_FLOAT") \
            .optional_input("x_active_mask", "DT_BOOL") \
            .optional_input("expert_scales", "DT_FLOAT") \
            .optional_input("elastic_info", "DT_INT32") \
            .optional_input("performance_info", "DT_INT64") \
            .required_attr("ep_world_size", attr.Int) \
            .required_attr("ep_rank_id", attr.Int) \
            .required_attr("moe_expert_num", attr.Int) \
            .required_attr("ccl_buffer_size", attr.Int) \
            .attr("tp_world_size", attr.Int(0)) \
            .attr("tp_rank_id", attr.Int(0)) \
            .attr("expert_shard_type", attr.Int(0)) \
            .attr("shared_expert_num", attr.Int(1)) \
            .attr("shared_expert_rank_num", attr.Int(0)) \
            .attr("quant_mode", attr.Int(0)) \
            .attr("global_bs", attr.Int(0)) \
            .attr("expert_token_nums_type", attr.Int(1)) \
            .attr("comm_alg", attr.Str("")) \
            .attr("zero_expert_num", attr.Int(0)) \
            .attr("copy_expert_num", attr.Int(0)) \
            .attr("const_expert_num", attr.Int(0)) \
            .attr("y_dtype", attr.Int(28)) \
            .output("expand_x", "DT_BF16, DT_INT8, DT_FLOAT16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8") \
            .output("dynamic_scales", "DT_FLOAT, DT_FLOAT8_E8M0") \
            .output("assist_info_for_combine", "DT_INT32") \
            .output("expert_token_nums", "DT_INT64") \
            .output("ep_recv_count", "DT_INT32") \
            .output("tp_recv_count", "DT_INT32") \
            .output("expand_scales", "DT_FLOAT")
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_moe_distribute_dispatch_v3.default)
    def converter_moe_distribute_dispatch_v3(
        context: Tensor,
        x: Tensor,
        expert_ids: Tensor,
        ep_world_size: int,
        ep_rank_id: int,
        moe_expert_num: int,
        ccl_buffer_size: int,
        *,
        scales: Tensor = None,
        x_active_mask: Tensor = None,
        expert_scales: Tensor = None,
        elastic_info: Tensor = None,
        performance_info: Tensor = None,
        tp_world_size: int = 0,
        tp_rank_id: int = 0,
        expert_shard_type: int = 0,
        shared_expert_num: int = 1,
        shared_expert_rank_num: int = 0,
        quant_mode: int = 0,
        global_bs: int = 0,
        expert_token_nums_type: int = 1,
        comm_alg: str = "",
        zero_expert_num: int = 0,
        copy_expert_num: int = 0,
        const_expert_num: int = 0,
        meta_outputs: TensorSpec = None):
        from torchair.ge._ge_graph import DataType

        expand_x_dtype = DataType.DT_INT8
        if quant_mode == 0:
            expand_x_dtype = x.dtype

        return MoeDistributeDispatchV3(context=context,
                                        x=x,
                                        expert_ids=expert_ids,
                                        scales=scales,
                                        x_active_mask=x_active_mask,
                                        expert_scales=expert_scales,
                                        elastic_info=elastic_info,
                                        performance_info=performance_info,
                                        ep_world_size=ep_world_size,
                                        ep_rank_id=ep_rank_id,
                                        moe_expert_num=moe_expert_num,
                                        ccl_buffer_size=ccl_buffer_size,
                                        tp_world_size=tp_world_size,
                                        tp_rank_id=tp_rank_id,
                                        expert_shard_type=expert_shard_type,
                                        shared_expert_num=shared_expert_num,
                                        shared_expert_rank_num=shared_expert_rank_num,
                                        quant_mode=quant_mode,
                                        global_bs=global_bs,
                                        expert_token_nums_type=expert_token_nums_type,
                                        comm_alg=comm_alg,
                                        zero_expert_num=zero_expert_num,
                                        copy_expert_num=copy_expert_num,
                                        const_expert_num=const_expert_num)

