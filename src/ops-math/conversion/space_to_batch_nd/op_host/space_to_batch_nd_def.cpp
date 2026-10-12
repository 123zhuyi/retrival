/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"
#include <array>
#include <vector>

namespace ops {
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,  ge::DT_UINT16,    ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64,  ge::DT_UINT64, ge::DT_BF16,      ge::DT_FLOAT16,
    ge::DT_FLOAT,  ge::DT_DOUBLE, ge::DT_BOOL,   ge::DT_COMPLEX32, ge::DT_COMPLEX64,
};
static constexpr std::array INDEX_DATA_TYPE_ALL{ge::DT_INT32, ge::DT_INT64};
static constexpr size_t C = VALUE_DATA_TYPE_ALL.size() * INDEX_DATA_TYPE_ALL.size() * INDEX_DATA_TYPE_ALL.size();

static auto CombineDataTypes()
{
    std::array<ge::DataType, C> xDTs{}, bsDTs{}, padDTs{};
    size_t idx = 0;
    for (size_t i3 = 0; i3 < INDEX_DATA_TYPE_ALL.size(); ++i3) {
        for (size_t i2 = 0; i2 < INDEX_DATA_TYPE_ALL.size(); ++i2) {
            for (size_t i1 = 0; i1 < VALUE_DATA_TYPE_ALL.size(); ++i1) {
                xDTs[idx] = VALUE_DATA_TYPE_ALL[i1];
                bsDTs[idx] = INDEX_DATA_TYPE_ALL[i2];
                padDTs[idx] = INDEX_DATA_TYPE_ALL[i3];
                ++idx;
            }
        }
    }
    return std::array{std::vector<ge::DataType>(xDTs.begin(), xDTs.end()),
                      std::vector<ge::DataType>(bsDTs.begin(), bsDTs.end()),
                      std::vector<ge::DataType>(padDTs.begin(), padDTs.end())};
}
static const auto DATA_TYPE_LISTS = CombineDataTypes();
static const auto& X_DT_LIST = std::get<0>(DATA_TYPE_LISTS);
static const auto& BS_DT_LIST = std::get<1>(DATA_TYPE_LISTS);
static const auto& PD_DT_LIST = std::get<2>(DATA_TYPE_LISTS);
static const auto DATA_FORMAT_LIST = std::vector<ge::Format>(C, ge::FORMAT_ND);

class SpaceToBatchND : public OpDef {
public:
    explicit SpaceToBatchND(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(X_DT_LIST)
            .Format(DATA_FORMAT_LIST)
            .UnknownShapeFormat(DATA_FORMAT_LIST);
        this->Input("block_shape")
            .ParamType(REQUIRED)
            .DataType(BS_DT_LIST)
            .Format(DATA_FORMAT_LIST)
            .UnknownShapeFormat(DATA_FORMAT_LIST)
            .ValueDepend(OPTIONAL);
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType(PD_DT_LIST)
            .Format(DATA_FORMAT_LIST)
            .UnknownShapeFormat(DATA_FORMAT_LIST)
            .ValueDepend(OPTIONAL);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(X_DT_LIST)
            .Format(DATA_FORMAT_LIST)
            .UnknownShapeFormat(DATA_FORMAT_LIST);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "space_to_batch_nd_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(SpaceToBatchND);
} // namespace ops
