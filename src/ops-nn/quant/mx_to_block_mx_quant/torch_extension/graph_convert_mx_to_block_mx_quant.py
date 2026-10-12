# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        DataType,
        Tensor,
        TensorSpec,
        _ge_dtype_to_ge_proto_dtype,
        torch_dtype_value_to_ge_type,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    def MxToBlockMxQuant(x: Tensor, mxscale: Tensor, *, dst_type: int = 292):
        inputs = {"x": x, "mxscale": mxscale}

        attrs = {
            "dst_type": attr.Int(dst_type),
        }

        outputs = ["y", "scale1", "scale2"]

        return ge_op(
            op_type="MxToBlockMxQuant",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            ir=IrDef("MxToBlockMxQuant")
            .input("x", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2")
            .input("mxscale", "DT_FLOAT8_E8M0")
            .attr("dst_type", attr.Int(292))
            .output("y", "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN")
            .output("scale1", "DT_FLOAT8_E8M0")
            .output("scale2", "DT_FLOAT8_E8M0"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_nn.mx_to_block_mx_quant.default)
    def convert_mx_to_block_mx_quant(
        x: Tensor,
        mxscale: Tensor,
        *,
        dst_type: int = 292,
        x_type: int = 296,
        meta_outputs: TensorSpec = None,
    ):
        ge_dst_type = torch_dtype_value_to_ge_type(dst_type)
        ge_x_type = torch_dtype_value_to_ge_type(x_type)
        if x.dtype == DataType.DT_UINT8:
            dim_num = x.rank
            bit_shape = [
                1,
            ] * (dim_num - 1) + [
                2,
            ]
            const = ge.Const(bit_shape)
            x_shape_float8 = ge.Shape(x)
            x_shape_float4 = ge.Mul(x_shape_float8, const)
            x = ge.Bitcast(x, type=ge_x_type)
            x = ge.Reshape(x, x_shape_float4)
        y, scale1, scale2 = MxToBlockMxQuant(x, mxscale, dst_type=ge_dst_type)
        scale1.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT8_E8M0)
        scale2.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT8_E8M0)
        return y, scale1, scale2
else:

    def convert_mx_to_block_mx_quant(*args, **kwargs):
        raise RuntimeError(
            "MxToBlockMxQuant graph converter: torchair is not available."
        )
