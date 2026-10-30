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

namespace ops {
// x data type: BasicType (9 types)
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32,
    ge::DT_INT64, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_DOUBLE,
};
// paddings index type
static constexpr std::array INDEX_DATA_TYPE_ALL{ge::DT_INT32, ge::DT_INT64};
// total combinations: 9 x dtype × 2 paddings dtype = 18
static constexpr size_t DATA_TYPE_COMBINE_COUNT = VALUE_DATA_TYPE_ALL.size() * INDEX_DATA_TYPE_ALL.size();

static constexpr std::array<std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT>, 2> CombineDataTypes()
{
    std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT> xDTs{}, paddingsDTs{};

    std::size_t idx = 0;
    for (std::size_t i2 = 0; i2 < INDEX_DATA_TYPE_ALL.size(); ++i2) {
        for (std::size_t i1 = 0; i1 < VALUE_DATA_TYPE_ALL.size(); ++i1) {
            xDTs[idx] = VALUE_DATA_TYPE_ALL[i1];
            paddingsDTs[idx] = INDEX_DATA_TYPE_ALL[i2];
            ++idx;
        }
    }
    std::array res{xDTs, paddingsDTs};
    return res;
}
static constexpr auto DATA_TYPE_LIST = CombineDataTypes();
static constexpr auto& X_DATA_TYPE_LIST = std::get<0>(DATA_TYPE_LIST);
static constexpr auto& PADDINGS_DATA_TYPE_LIST = std::get<1>(DATA_TYPE_LIST);
static const auto NHWC_FORMAT_LIST = std::vector<ge::Format>(DATA_TYPE_COMBINE_COUNT, ge::FORMAT_NHWC);
static const auto ND_FORMAT_LIST = std::vector<ge::Format>(DATA_TYPE_COMBINE_COUNT, ge::FORMAT_ND);

class SpaceToBatch : public OpDef {
public:
    explicit SpaceToBatch(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(X_DATA_TYPE_LIST.begin(), X_DATA_TYPE_LIST.end()))
            .Format(NHWC_FORMAT_LIST);
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(PADDINGS_DATA_TYPE_LIST.begin(), PADDINGS_DATA_TYPE_LIST.end()))
            .Format(ND_FORMAT_LIST)
            .ValueDepend(OPTIONAL);
        this->Output("y").Follow("x");

        this->Attr("block_size").AttrType(REQUIRED).Int();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "space_to_batch_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(SpaceToBatch);
} // namespace ops
