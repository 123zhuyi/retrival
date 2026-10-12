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
 * \file test_segsum_tiling_arch35.cpp
 * \brief Segsum tiling UT for arch35 / Ascend950.
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
#include "../../../../op_kernel/arch35/segsum_tiling_data.h"

using namespace ge;
using namespace gert;

#define SEGSUM_ARCH35_STR_IMPL(x) #x
#define SEGSUM_ARCH35_STR(x) SEGSUM_ARCH35_STR_IMPL(x)

namespace {
constexpr uint64_t SEGSUM_ARCH35_TILING_KEY_STRIPE = 0;
constexpr uint64_t SEGSUM_ARCH35_TILING_KEY_ROW_BLOCK = 1;

struct SegsumCompileInfoArch35 {};

bool IsAscend950Build() { return std::string(SEGSUM_ARCH35_STR(BUILD_SOC_VERSION)) == "ascend950"; }

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
    auto inShape = MakeStorageShape(dims);
    std::vector<int64_t> outDims = dims;
    outDims.push_back(dims.back());
    auto outShape = MakeStorageShape(outDims);
    return gert::TilingContextPara("Segsum",
                                   {
                                       {inShape, dtype, ge::FORMAT_ND},
                                   },
                                   {
                                       {outShape, dtype, ge::FORMAT_ND},
                                   },
                                   compileInfo);
}

const SegsumTilingDataArch35* GetArch35TilingData(const TilingInfo& tilingInfo)
{
    EXPECT_GE(tilingInfo.tilingDataSize, sizeof(SegsumTilingDataArch35));
    return reinterpret_cast<const SegsumTilingDataArch35*>(tilingInfo.tilingData.get());
}
} // namespace

class SegsumTilingArch35Test : public testing::Test {
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
            GTEST_SKIP() << "Segsum arch35 tiling UT only runs in ascend950 build.";
        }
    }
};

TEST_F(SegsumTilingArch35Test, small_fp32_uses_row_block_template)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({1, 1, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, SEGSUM_ARCH35_TILING_KEY_ROW_BLOCK);
    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 1);
    EXPECT_EQ(tiling->tailDimLength, 12);
    EXPECT_EQ(tiling->rowLen, 16); // 12 rounded up to 32B for float
    EXPECT_EQ(tiling->rowNum, 12); // one batch is 12 rows and they all fit
    EXPECT_EQ(tiling->stripeLen, 0);
    EXPECT_EQ(tiling->averageBatches, 1);
}

TEST_F(SegsumTilingArch35Test, small_fp16_row_len_is_16_aligned)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({2, 3, 20}, ge::DT_FLOAT16, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, SEGSUM_ARCH35_TILING_KEY_ROW_BLOCK);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 6);
    EXPECT_EQ(tiling->tailDimLength, 20);
    EXPECT_EQ(tiling->rowLen, 32); // 20 rounded up to 32B for half
    EXPECT_EQ(tiling->rowNum, 20);
}

TEST_F(SegsumTilingArch35Test, batches_split_across_ascend950_cores)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({65, 1, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 65);
    EXPECT_GT(tiling->averageBatches, 0);
    EXPECT_EQ(tilingInfo.blockNum, tiling->needCoreNum);
    // every batch is owned by exactly one core
    EXPECT_GE(tiling->needCoreNum * tiling->averageBatches, tiling->batches);
    EXPECT_LT((tiling->needCoreNum - 1) * tiling->averageBatches, tiling->batches);
}

TEST_F(SegsumTilingArch35Test, huge_tail_dim_falls_back_to_column_stripes)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({1, 1, 40000}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, SEGSUM_ARCH35_TILING_KEY_STRIPE);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->tailDimLength, 40000);
    EXPECT_EQ(tiling->rowNum, 0);
    EXPECT_GT(tiling->stripeLen, 0);
    EXPECT_LE(tiling->stripeLen, tiling->rowLen);
    EXPECT_EQ(tiling->stripeLen % 8, 0); // 32B aligned for float
}

TEST_F(SegsumTilingArch35Test, empty_tensor_keeps_valid_block_dim)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({0, 1, 12}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.tilingKey, SEGSUM_ARCH35_TILING_KEY_ROW_BLOCK);
    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->batches, 0);
    EXPECT_EQ(tiling->rowNum, 0);
    EXPECT_EQ(tiling->averageBatches, 0);
}

TEST_F(SegsumTilingArch35Test, zero_tail_dim_keeps_valid_block_dim)
{
    SegsumCompileInfoArch35 compileInfo = {};
    auto tilingContextPara = MakeTilingPara({2, 3, 0}, ge::DT_FLOAT, &compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));

    EXPECT_EQ(tilingInfo.blockNum, 1);

    auto tiling = GetArch35TilingData(tilingInfo);
    ASSERT_NE(tiling, nullptr);
    EXPECT_EQ(tiling->tailDimLength, 0);
    EXPECT_EQ(tiling->rowNum, 0);
}
