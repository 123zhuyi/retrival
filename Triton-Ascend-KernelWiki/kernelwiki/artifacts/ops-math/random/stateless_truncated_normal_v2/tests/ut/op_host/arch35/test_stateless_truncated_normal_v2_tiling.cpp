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
 * \file test_stateless_truncated_normal_v2_tiling.cpp
 * \brief StatelessTruncatedNormalV2 tiling UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_truncated_normal_v2_tiling_arch35.h"

class StatelessTruncatedNormalV2TilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessTruncatedNormalV2TilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessTruncatedNormalV2TilingTest TearDown" << std::endl;
  }
};

// Test 1: int32 shape, fp32 output, 2D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int32_fp32_2d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {32, 64};
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2,}, {2,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{32, 64}, {32, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 2: int32 shape, fp16 output, 1D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int32_fp16_1d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {512};
    uint64_t keyValue = 100;
    uint64_t counterValue[2] = {1, 2};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{512,}, {512,}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 3: int32 shape, bf16 output, 4D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int32_bf16_4d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {4, 8, 16, 32};
    uint64_t keyValue = 999;
    uint64_t counterValue[2] = {100, 200};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{4,}, {4,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{4, 8, 16, 32}, {4, 8, 16, 32}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(27)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 4: int64 shape, fp32 output, 3D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int64_fp32_3d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int64_t> shapeValue = {8, 16, 32};
    uint64_t keyValue = 55;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{3,}, {3,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{8, 16, 32}, {8, 16, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 5: int64 shape, fp16 output, 2D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int64_fp16_2d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int64_t> shapeValue = {100, 200};
    uint64_t keyValue = 12345;
    uint64_t counterValue[2] = {67890, 11111};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{100, 200}, {100, 200}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 6: int64 shape, bf16 output, 5D
TEST_F(StatelessTruncatedNormalV2TilingTest, test_int64_bf16_5d)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int64_t> shapeValue = {3, 4, 5, 6, 7};
    uint64_t keyValue = 777;
    uint64_t counterValue[2] = {0, 1};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{5,}, {5,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{3, 4, 5, 6, 7}, {3, 4, 5, 6, 7}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(27)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 7: 0-dim scalar output (empty shape tensor → outputSize=1)
TEST_F(StatelessTruncatedNormalV2TilingTest, test_0dim_scalar)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    int32_t shapeDummy = 0;  // valid pointer required by faker even for 0-element tensor
    uint64_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{0,}, {0,}}, ge::DT_INT32, ge::FORMAT_ND, true, &shapeDummy},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 8: Large output (trigger multi-core split)
TEST_F(StatelessTruncatedNormalV2TilingTest, test_large_output)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {1024, 1024};
    uint64_t keyValue = 88;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{2,}, {2,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 9: Error case - key dtype wrong (int32 instead of uint64)
TEST_F(StatelessTruncatedNormalV2TilingTest, test_key_dtype_error)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {1024};
    int32_t keyValue = 42;
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{1024,}, {1024,}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test 10: Error case - counter shape wrong ([1] instead of [2])
TEST_F(StatelessTruncatedNormalV2TilingTest, test_counter_shape_error)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {1024};
    uint64_t keyValue = 42;
    uint64_t counterValue = 0;
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &keyValue},
            {{{1,}, {1,}}, ge::DT_UINT64, ge::FORMAT_ND, true, &counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{1024,}, {1024,}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Test 11: Error case - key shape wrong ([2] instead of [1])
TEST_F(StatelessTruncatedNormalV2TilingTest, test_key_shape_error)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    std::vector<int32_t> shapeValue = {1024};
    uint64_t keyValue[2] = {42, 43};
    uint64_t counterValue[2] = {0, 0};
    int32_t algValue = 1;
    gert::TilingContextPara tilingContextPara(
        "StatelessTruncatedNormalV2",
        {
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, shapeValue.data()},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, keyValue},
            {{{2,}, {2,}}, ge::DT_UINT64, ge::FORMAT_ND, true, counterValue},
            {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND, true, &algValue},
        },
        {
            {{{1024,}, {1024,}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
