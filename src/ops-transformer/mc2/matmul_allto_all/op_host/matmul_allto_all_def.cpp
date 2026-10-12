/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_allto_all_def.cpp
 * \brief 算子信息库定义
 */
#include <register/op_def_registry.h>

namespace ops {
class MatmulAlltoAll : public OpDef {
public:
    explicit MatmulAlltoAll(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
                       ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("comm_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x1_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("group").AttrType(REQUIRED).String();
        this->Attr("world_size").AttrType(REQUIRED).Int();
        this->Attr("all2all_axes").AttrType(OPTIONAL).ListInt({-1, -2});
        this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
        this->Attr("x1_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("x2_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("comm_quant_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
        this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
        this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
        this->Attr("group_size").AttrType(OPTIONAL).Int(0);

        // ascend950 AI处理器定义OpAICoreConfig变量，定制化配置参数
        OpAICoreConfig aicoreConfig_950;
        aicoreConfig_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape，复用二进制，后续图支持后修改
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
            .ExtendCfgInfo("opFile.value", "matmul_allto_all_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig_950);

        // 将group配置为该算子的通信域
        this->MC2().HcclGroup("group");

      // ascend910_93 AI处理器定义OpAICoreConfig变量，定制化配置参数
 	    OpAICoreConfig aicoreConfig_910_93;
 	    aicoreConfig_910_93.Input("x1")
 	        .ParamType(REQUIRED)
 	        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("x2")
 	        .ParamType(REQUIRED)
 	        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("bias")
 	        .ParamType(OPTIONAL)
 	        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("x1_scale")
 	        .ParamType(OPTIONAL)
 	        .DataTypeList({ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("x2_scale")
 	        .ParamType(OPTIONAL)
 	        .DataTypeList({ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("comm_scale")
 	        .ParamType(OPTIONAL)
 	        .DataTypeList({ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("x1_offset")
 	        .ParamType(OPTIONAL)
 	        .DataTypeList({ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Input("x2_offset")
 	        .ParamType(OPTIONAL)
 	        .DataTypeList({ge::DT_FLOAT})
 	        .FormatList({ge::FORMAT_ND})
 	        .AutoContiguous();
 	    aicoreConfig_910_93.Output("y")
 	        .ParamType(REQUIRED)
 	        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
 	        .FormatList({ge::FORMAT_ND});
 	    this->Attr("group").AttrType(REQUIRED).String();
 	    this->Attr("world_size").AttrType(REQUIRED).Int();
 	    this->Attr("all2all_axes").AttrType(OPTIONAL).ListInt({-2, -1});
 	    this->Attr("y_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
 	    this->Attr("x1_quant_mode").AttrType(OPTIONAL).Int(0);
 	    this->Attr("x2_quant_mode").AttrType(OPTIONAL).Int(0);
 	    this->Attr("comm_quant_mode").AttrType(OPTIONAL).Int(0);
 	    this->Attr("comm_quant_dtype").AttrType(OPTIONAL).Int(static_cast<int64_t>(ge::DT_UNDEFINED));
 	    this->Attr("transpose_x1").AttrType(OPTIONAL).Bool(false);
 	    this->Attr("transpose_x2").AttrType(OPTIONAL).Bool(false);
 	    this->Attr("group_size").AttrType(OPTIONAL).Int(0);
 	 
 	    aicoreConfig_910_93.DynamicCompileStaticFlag(true)
 	        .DynamicFormatFlag(true)
 	        .DynamicRankSupportFlag(true)
 	        .DynamicShapeSupportFlag(true)
 	        .NeedCheckSupportFlag(false)
 	        .PrecisionReduceFlag(true)
 	        .ExtendCfgInfo("jitCompile.flag", "static_false") // 动态shape，复用二进制，后续图支持后修改
 	        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
 	        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel")
 	        .ExtendCfgInfo("opFile.value", "matmul_allto_all_apt");
 	    this->AICore().AddConfig("ascend910_93", aicoreConfig_910_93);
 	    this->MC2().HcclGroup("group");

        // 910B
        OpAICoreConfig aicore_config_910b;
        aicore_config_910b.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("x1_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("x2_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("comm_scale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("x1_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicore_config_910b.Input("x2_offset")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        aicore_config_910b.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        
        aicore_config_910b.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
        this->AICore().AddConfig("ascend910b", aicore_config_910b);
        this->MC2().HcclGroup("group");
    }
};

OP_ADD(MatmulAlltoAll);

} // namespace ops