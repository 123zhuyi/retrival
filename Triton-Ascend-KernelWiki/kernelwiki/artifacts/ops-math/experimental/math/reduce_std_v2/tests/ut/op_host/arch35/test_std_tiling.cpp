/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "math/reduce_var/op_host/arch35/reduce_var_tiling.h"

using namespace std;
using namespace ge;

class TestStdTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestStdTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestStdTiling TearDown" << std::endl;
    }
};

static void ExecuteStdTilingSmokeCase(const gert::StorageShape& inputStorageShape, ge::DataType dtype,
    const gert::StorageShape& outputStorageShape, const std::vector<int64_t>& dim, int64_t correction, bool keepdim)
{
    optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc = {
        gert::TilingContextPara::TensorDescription(inputStorageShape, dtype, ge::FORMAT_ND),
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc = {
        gert::TilingContextPara::TensorDescription(outputStorageShape, dtype, ge::FORMAT_ND),
        gert::TilingContextPara::TensorDescription(outputStorageShape, dtype, ge::FORMAT_ND),
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(dim)),
        gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(correction)),
        gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)),
        gert::TilingContextPara::OpAttr("is_mean_out", Ops::Math::AnyValue::CreateFrom<bool>(true)),
    };
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2", inputTensorDesc, outputTensorDesc, attrs, &compileInfo);

    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_GT(tilingInfo.tilingKey, 0);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 16777216);
    EXPECT_GT(tilingInfo.tilingDataSize, 0U);
}

TEST_F(TestStdTiling, ReduceStdV2_test_tiling_001)
{
    optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
    gert::TilingContextPara tilingContextPara(
        "ReduceStdV2",
        {
            {{{64, 10240}, {64, 10240}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{64, 1}, {64, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
         gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(8)),
         gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("is_mean_out", Ops::Math::AnyValue::CreateFrom<bool>(true))},
        &compileInfo);
    uint64_t expectTilingKey = 5143;
    string expectTilingData =
        "1 64 1 1 1 1 1 64 27648 512 64 952945869 64 10240 0 0 0 0 0 0 0 10240 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 8 0 1 "
        "0 4092916413600104448 952945869 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(TestStdTiling, ReduceStdV2_test_tiling_float32_dim1)
{
    ExecuteStdTilingSmokeCase({{32, 128}, {32, 128}}, ge::DT_FLOAT, {{32, 1}, {32, 1}}, {1}, 1, true);
}

TEST_F(TestStdTiling, ReduceStdV2_test_tiling_keepdim_false)
{
    ExecuteStdTilingSmokeCase({{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT16, {{8, 32}, {8, 32}}, {1}, 1, false);
}

TEST_F(TestStdTiling, ReduceStdV2_test_tiling_multi_dim_correction_not_one)
{
    ExecuteStdTilingSmokeCase({{4, 8, 16}, {4, 8, 16}}, ge::DT_FLOAT, {{4}, {4}}, {1, 2}, 2, false);
}
