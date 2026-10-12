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
 * \file test_exp_segsum_grad_tiling_arch35.cpp
 * \brief ExpSegsumGrad tiling UT for arch35 / Ascend950.
 */

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "platform/platform_info.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/exp_segsum_grad_tiling_arch35.h"
#include "../../../../op_kernel/arch35/exp_segsum_grad_tiling_data.h"

using namespace ge;
using namespace gert;
using optiling::ExpSegsumGradCompileInfoArch35;

#define EXP_SEGSUM_GRAD_ARCH35_STR_IMPL(x) #x
#define EXP_SEGSUM_GRAD_ARCH35_STR(x) EXP_SEGSUM_GRAD_ARCH35_STR_IMPL(x)

namespace {
constexpr uint64_t EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL = 1;
constexpr size_t EXP_SEGSUM_GRAD_A2_WORKSPACE_SIZE = 32 * 1024 * 1024;
constexpr size_t EXP_SEGSUM_GRAD_ARCH35_SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE = 10912;

bool IsAscend950Build() { return std::string(EXP_SEGSUM_GRAD_ARCH35_STR(BUILD_SOC_VERSION)) == "ascend950"; }

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (auto dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakeTilingPara(const std::vector<int64_t>& dims, ge::DataType dtype, void* compileInfo)
{
    auto shape = MakeStorageShape(dims);
    return gert::TilingContextPara("ExpSegsumGrad",
                                   {
                                       {shape, dtype, ge::FORMAT_ND},
                                       {shape, dtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {shape, dtype, ge::FORMAT_ND},
                                   },
                                   compileInfo);
}

void ExpectBatchBounds(const ExpSegsumGradTilingDataArch35& tiling, int64_t batches, uint32_t expectNeedCoreNum)
{
    ASSERT_EQ(tiling.needCoreNum, expectNeedCoreNum);
    if (batches == 0) {
        for (uint16_t i = 0; i < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; ++i) {
            EXPECT_EQ(tiling.batchStart[i], 0);
            EXPECT_EQ(tiling.batchEnd[i], 0);
        }
        return;
    }

    auto averageBatches = (batches + EXP_SEGSUM_GRAD_MAX_CORE_ARCH35 - 1) / EXP_SEGSUM_GRAD_MAX_CORE_ARCH35;
    for (uint32_t i = 0; i < expectNeedCoreNum; ++i) {
        EXPECT_EQ(tiling.batchStart[i], static_cast<int32_t>(i * averageBatches));
        EXPECT_EQ(tiling.batchEnd[i], static_cast<int32_t>(std::min<int64_t>((i + 1) * averageBatches, batches)));
    }
    for (uint32_t i = expectNeedCoreNum; i < EXP_SEGSUM_GRAD_MAX_CORE_ARCH35; ++i) {
        EXPECT_EQ(tiling.batchStart[i], 0);
        EXPECT_EQ(tiling.batchEnd[i], 0);
    }
}

const ExpSegsumGradTilingDataArch35* GetArch35TilingData(const TilingInfo& tilingInfo)
{
    EXPECT_GE(tilingInfo.tilingDataSize, sizeof(ExpSegsumGradTilingDataArch35));
    return reinterpret_cast<const ExpSegsumGradTilingDataArch35*>(tilingInfo.tilingData.get());
}
} // namespace

class ExpSegsumGradTilingArch35Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::OptionalInfos optiCompilation;
        optiCompilation.Init();
        optiCompilation.SetSocVersion("Ascend950");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(optiCompilation);
    }

    static void TearDownTestCase()
    {
        fe::OptionalInfos optiCompilation;
        optiCompilation.Init();
        optiCompilation.SetSocVersion("soc_version");
        fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(optiCompilation);
    }

    void SetUp() override
    {
        if (!IsAscend950Build()) {
            GTEST_SKIP() << "ExpSegsumGrad arch35 tiling UT only runs in ascend950 build.";
        }
    }
};

TEST_F(ExpSegsumGradTilingArch35Test, small_fp32_uses_arch35_tiling_data)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({1, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 1);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1);
    EXPECT_EQ(tilingInfo.workspaceSizes[0],
              EXP_SEGSUM_GRAD_A2_WORKSPACE_SIZE + EXP_SEGSUM_GRAD_ARCH35_SYS_WORKSPACE_SIZE);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 1);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE);
    ExpectBatchBounds(*tiling, 1, 1);
}

TEST_F(ExpSegsumGradTilingArch35Test, multi_batch_splits_across_ascend950_cores)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({65, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 33);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 65);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, EXP_SEGSUM_GRAD_ARCH35_FP32_SLIDE_SIZE);
    ExpectBatchBounds(*tiling, 65, 33);
}

TEST_F(ExpSegsumGradTilingArch35Test, empty_batch_keeps_valid_block_dim)
{
    ExpSegsumGradCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({0, 1, 12, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, EXP_SEGSUM_GRAD_ARCH35_TILING_KEY_SMALL);
    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 0);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->slideSize, 0);
    ExpectBatchBounds(*tiling, 0, 1);
}
