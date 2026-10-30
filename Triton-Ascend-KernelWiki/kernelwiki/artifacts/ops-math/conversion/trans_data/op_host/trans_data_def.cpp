/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_data_def.cpp
 * \brief op store info of TransData
 */
#include "register/op_def_registry.h"

namespace ops
{
class TransData : public OpDef
{
public:
    const std::vector<ge::DataType> dType = {ge::DT_INT8,          ge::DT_UINT8,       ge::DT_INT16, ge::DT_UINT16,
                                             ge::DT_FLOAT16,       ge::DT_BF16,        ge::DT_INT32, ge::DT_UINT32,
                                             ge::DT_FLOAT,         ge::DT_FLOAT,       ge::DT_FLOAT, ge::DT_FLOAT8_E4M3FN,
                                             ge::DT_FLOAT4_E2M1,   ge::DT_HIFLOAT8,    ge::DT_UINT8, ge::DT_FLOAT4_E2M1,
                                             ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1, ge::DT_INT8,  
                                             ge::DT_INT8,          ge::DT_INT8,        ge::DT_UINT8,       
                                             ge::DT_UINT8,         ge::DT_UINT8,       ge::DT_FLOAT8_E4M3FN, 
                                             ge::DT_FLOAT16,       ge::DT_BF16,        ge::DT_INT32,         
                                             ge::DT_INT32,         ge::DT_INT32,       ge::DT_INT32,         
                                             ge::DT_INT32,         ge::DT_FLOAT,       ge::DT_FLOAT,         
                                             ge::DT_FLOAT,         ge::DT_FLOAT,       ge::DT_FLOAT,
                                             ge::DT_FLOAT4_E1M2};
    const std::vector<ge::Format> srcFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                               ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ, 
                                               ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ,
                                               ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ,
                                               ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,  
                                               ge::FORMAT_FRACTAL_NZ_C0_2,  ge::FORMAT_FRACTAL_NZ_C0_4,  ge::FORMAT_FRACTAL_NZ_C0_16,
                                               ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ_C0_2,
                                               ge::FORMAT_FRACTAL_NZ_C0_4,  ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_32,
                                               ge::FORMAT_ND};
    const std::vector<ge::Format> dstFormat = {
        ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,
        ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ_C0_16,
        ge::FORMAT_FRACTAL_NZ_C0_16, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ,
        ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ,       ge::FORMAT_FRACTAL_NZ,
        ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
        ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
        ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
        ge::FORMAT_FRACTAL_NZ};

    explicit TransData(const char* name) : OpDef(name)
    {
        this->Input("src").ParamType(REQUIRED).DataType(dType).Format(srcFormat).UnknownShapeFormat(srcFormat);
        this->Output("dst").ParamType(REQUIRED).DataType(dType).Format(dstFormat).UnknownShapeFormat(dstFormat);
        this->Attr("src_format").AttrType(REQUIRED).String();
        this->Attr("dst_format").AttrType(REQUIRED).String();
        this->Attr("src_subformat").AttrType(OPTIONAL).Int(0);
        this->Attr("dst_subformat").AttrType(OPTIONAL).Int(0);
        this->Attr("groups").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "trans_data_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(TransData);
}  // namespace ops
