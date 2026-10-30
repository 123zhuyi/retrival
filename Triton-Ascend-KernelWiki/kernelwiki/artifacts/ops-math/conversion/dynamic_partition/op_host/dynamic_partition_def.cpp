/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_partition_def.cpp
 * \brief op config of DynamicPartition
 */

#include "register/op_def_registry.h"

namespace ops
{
static const std::vector<ge::DataType> xDType = {
    ge::DT_INT8,   ge::DT_UINT8,     ge::DT_INT16,    ge::DT_UINT16,      ge::DT_INT32,        ge::DT_UINT32,
    ge::DT_INT64,  ge::DT_UINT64,    ge::DT_BOOL,     ge::DT_FLOAT16,     ge::DT_BF16,         ge::DT_FLOAT,
    ge::DT_DOUBLE, ge::DT_COMPLEX64, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::Format> xFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> partDType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

class DynamicPartition : public OpDef
{
public:
    explicit DynamicPartition(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("partitions").ParamType(REQUIRED).DataType(partDType).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Output("y")
            .OutputShapeDependOnCompute()
            .ParamType(DYNAMIC)
            .DataType(xDType)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Attr("num_partitions").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "dynamic_partition_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(DynamicPartition);
}  // namespace ops