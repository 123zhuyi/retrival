/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_grouped_mat_mul.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class AlltoAllvGroupedMatMul : public OpDef {
public:
    explicit AlltoAllvGroupedMatMul(const char *name) : OpDef(name)
    {
        this->Input("gmm_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("gmm_weight")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("send_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                ge::DT_INT32, ge::DT_INT64
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("recv_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                ge::DT_INT32, ge::DT_INT64
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("mm_x")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("mm_weight")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("gmm_x_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("gmm_weight_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("gmm_x_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        this->Input("gmm_weight_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        this->Input("mm_x_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("mm_weight_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        this->Input("mm_x_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("mm_weight_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        this->Output("gmm_y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_FLOAT16, ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Output("mm_y")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16,ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_FLOAT16, ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Output("permute_out")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            });

        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("send_counts").AttrType(REQUIRED).ListInt();
        this->Attr("recv_counts").AttrType(REQUIRED).ListInt();
        this->Attr("trans_gmm_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("trans_mm_weight").AttrType(OPTIONAL).Bool(false);
        this->Attr("permute_out_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("gmm_x_quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("gmm_weight_quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("mm_x_quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("mm_weight_quant_mode").AttrType(OPTIONAL).Int();
        this->Attr("group_size").AttrType(OPTIONAL).Int();
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
        this->Attr("mm_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));

        OpAICoreConfig aicore_config_910_93;
        aicore_config_910_93.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910_93", aicore_config_910_93);
        this->MC2().HcclGroup({"group"});
        
        // 910_95
        OpAICoreConfig aicore_config_950;
        aicore_config_950.Input("gmm_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("gmm_weight")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8})
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("send_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                ge::DT_INT32, ge::DT_INT64
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("recv_counts_tensor")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                ge::DT_INT32, ge::DT_INT64
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("mm_x")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("mm_weight")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

       aicore_config_950.Input("gmm_x_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("gmm_weight_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("gmm_x_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
       aicore_config_950.Input("gmm_weight_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
       aicore_config_950.Input("mm_x_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("mm_weight_scale")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        aicore_config_950.Input("mm_x_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Input("mm_weight_offset")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                ge::DT_FLOAT, ge::DT_FLOAT,
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();
        
        aicore_config_950.Output("gmm_y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_FLOAT16, ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Output("mm_y")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16,ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_FLOAT16, ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        aicore_config_950.Output("permute_out")
            .ParamType(OPTIONAL)
            .DataType({
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
                ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND
            });

        aicore_config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "allto_allv_grouped_mat_mul_apt");
        this->AICore().AddConfig("ascend950", aicore_config_950);
        this->MC2().HcclGroup({"group"});
    }
};

OP_ADD(AlltoAllvGroupedMatMul);
} // namespace ops