/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "../../../op_kernel/arch22/acos_grad_v2_tiling_data.h"

namespace optiling {
struct AcosGradV2CompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};
} // namespace optiling

namespace NsAcosGradV2 {
bool operator==(const AcosGradV2TilingData& a, const AcosGradV2TilingData& b)
{
    return a.totalLength == b.totalLength && a.blockFormer == b.blockFormer && a.blockNum == b.blockNum &&
           a.ubFormer == b.ubFormer;
}
} // namespace NsAcosGradV2

namespace {
constexpr size_t WORKSPACE_SIZE = 0;
}

class AcosGradV2Tiling : public testing::Test {};

// FP32 8192 元素，20 核：tiling 应成功，totalLength=8192，blockNum≥1，blockFormer 按 512 对齐
TEST_F(AcosGradV2Tiling, fp32_8192)
{
    optiling::AcosGradV2CompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara para("AcosGradV2",
                                 {
                                     {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{8192}, {8192}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                 },
                                 {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(AcosGradV2TilingData));
    auto* td = reinterpret_cast<AcosGradV2TilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalLength, 8192ULL);
    EXPECT_GE(td->blockNum, 1U);
    EXPECT_GE(td->blockFormer, 512U); // ELEM_ALIGN
}

// FP16 多维：tiling 应成功
TEST_F(AcosGradV2Tiling, fp16_multi_dim)
{
    optiling::AcosGradV2CompileInfo compileInfo{20, 192 * 1024, false};
    gert::TilingContextPara para("AcosGradV2",
                                 {
                                     {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {
                                     {{{2, 3, 5}, {2, 3, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                 },
                                 {}, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(AcosGradV2TilingData));
    auto* td = reinterpret_cast<AcosGradV2TilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalLength, 30ULL); // 2*3*5
}
