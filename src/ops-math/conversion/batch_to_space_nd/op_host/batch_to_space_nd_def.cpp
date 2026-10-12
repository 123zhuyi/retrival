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
// x data type 所有取值
static constexpr std::array VALUE_DATA_TYPE_ALL{
    ge::DT_INT8,   ge::DT_UINT8,  ge::DT_INT16,  ge::DT_UINT16,    ge::DT_INT32,
    ge::DT_UINT32, ge::DT_INT64,  ge::DT_UINT64, ge::DT_BF16,      ge::DT_FLOAT16,
    ge::DT_FLOAT,  ge::DT_DOUBLE, ge::DT_BOOL,   ge::DT_COMPLEX32, ge::DT_COMPLEX64,
};
// 索引类型 data type 所有取值
static constexpr std::array INDEX_DATA_TYPE_ALL{ge::DT_INT32, ge::DT_INT64};
// 所有组合数量
static constexpr size_t DATA_TYPE_COMBINE_COUNT = VALUE_DATA_TYPE_ALL.size() * INDEX_DATA_TYPE_ALL.size() *
                                                  INDEX_DATA_TYPE_ALL.size();
// 计算各输入的 data type 组合
static constexpr std::array<std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT>, 3> CombineDataTypes()
{
    std::array<ge::DataType, DATA_TYPE_COMBINE_COUNT> xDTs{}, blockShapeDTs{}, cropsDTs{};

    std::size_t idx = 0;
    for (std::size_t i3 = 0; i3 < INDEX_DATA_TYPE_ALL.size(); ++i3) {
        for (std::size_t i2 = 0; i2 < INDEX_DATA_TYPE_ALL.size(); ++i2) {
            for (std::size_t i1 = 0; i1 < VALUE_DATA_TYPE_ALL.size(); ++i1) {
                xDTs[idx] = VALUE_DATA_TYPE_ALL[i1];
                blockShapeDTs[idx] = INDEX_DATA_TYPE_ALL[i2];
                cropsDTs[idx] = INDEX_DATA_TYPE_ALL[i3];
                ++idx;
            }
        }
    }
    std::array res{xDTs, blockShapeDTs, cropsDTs};
    return res;
}
static constexpr auto DATA_TYPE_LIST = CombineDataTypes();
static constexpr auto& X_DATA_TYPE_LIST = std::get<0>(DATA_TYPE_LIST);
static constexpr auto& BS_DATA_TYPE_LIST = std::get<1>(DATA_TYPE_LIST);
static constexpr auto& CROPS_DATA_TYPE_LIST = std::get<2>(DATA_TYPE_LIST);
static const auto DATA_FORMAT_LIST = std::vector<ge::Format>(DATA_TYPE_COMBINE_COUNT, ge::FORMAT_ND);

class BatchToSpaceND : public OpDef {
public:
    explicit BatchToSpaceND(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(X_DATA_TYPE_LIST.begin(), X_DATA_TYPE_LIST.end()))
            .Format(DATA_FORMAT_LIST);
        // block_shape 参数（1D 张量）
        this->Input("block_shape")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(BS_DATA_TYPE_LIST.begin(), BS_DATA_TYPE_LIST.end()))
            .Format(DATA_FORMAT_LIST)
            .ValueDepend(OPTIONAL);
        // crops 参数（2D 张量）
        this->Input("crops")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(CROPS_DATA_TYPE_LIST.begin(), CROPS_DATA_TYPE_LIST.end()))
            .Format(DATA_FORMAT_LIST)
            .ValueDepend(OPTIONAL);
        // 输出张量 y
        this->Output("y").Follow("x");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "batch_to_space_nd_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(BatchToSpaceND);
} // namespace ops
