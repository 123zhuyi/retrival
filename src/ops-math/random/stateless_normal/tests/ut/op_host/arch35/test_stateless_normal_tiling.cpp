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
 * \file test_stateless_normal_tiling.cpp
 * \brief StatelessNormal V4 tiling UT
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/stateless_normal_tiling_arch35.h"

class StatelessNormalTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "StatelessNormalTilingTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "StatelessNormalTilingTest TearDown" << std::endl;
  }
};

// Test 1: Both tensor mean/stdev (L2 已广播), output FLOAT32
TEST_F(StatelessNormalTilingTest, test_both_scalar_float32)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {32, 512};
    int64_t seedValue = 12345;
    int64_t offsetValue = 0;
    // L2 层已将 scalar mean/stdev 广播到 output shape，tiling 侧看到的是 full tensor
    constexpr int64_t outputSize = 32 * 512;
    std::vector<float> meanData(outputSize, 0.0f);
    std::vector<float> stdevData(outputSize, 1.0f);
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, meanData.data()},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{32, 512}, {32, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    // tilingKey = 100 (基类默认, 统一 BothTensor 路径)
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 2: Both tensor mean/stdev, output BF16
TEST_F(StatelessNormalTilingTest, test_both_tensor_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {1024};
    int64_t seedValue = 42;
    int64_t offsetValue = 100;
    std::vector<uint16_t> meanData(1024, 0x3F80);   // bf16 ~1.0
    std::vector<uint16_t> stdevData(1024, 0x4000);   // bf16 ~2.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{1024,}, {1024,}}, ge::DT_BF16, ge::FORMAT_ND, true, meanData.data()},
            {{{1024,}, {1024,}}, ge::DT_BF16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{1024,}, {1024,}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    // tilingKey = 100 (基类默认, 统一 BothTensor 路径)
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 3: Both tensor mean/stdev (L2 已广播), output FP16
TEST_F(StatelessNormalTilingTest, test_mean_scalar_stdev_tensor_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {256, 256};
    int64_t seedValue = 99;
    int64_t offsetValue = 40;
    constexpr int64_t outputSize = 256 * 256;
    // L2 已将 mean scalar 广播到 output shape
    std::vector<uint16_t> meanData(outputSize, 0x3800);    // fp16 ~0.5
    std::vector<uint16_t> stdevData(outputSize, 0x3E00);   // fp16 ~1.5
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, meanData.data()},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 4: Both tensor mean/stdev (L2 已广播), output FLOAT32
TEST_F(StatelessNormalTilingTest, test_mean_tensor_stdev_scalar_float32)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {4096};
    int64_t seedValue = 7;
    int64_t offsetValue = 0;
    std::vector<float> meanData(4096, -1.0f);
    std::vector<float> stdevData(4096, 0.5f);
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{4096,}, {4096,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, meanData.data()},
            {{{4096,}, {4096,}}, ge::DT_FLOAT, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{4096,}, {4096,}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 5: Both tensor mean/stdev (L2 已广播), output BF16
TEST_F(StatelessNormalTilingTest, test_both_scalar_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {2048};
    int64_t seedValue = 55;
    int64_t offsetValue = 0;
    std::vector<uint16_t> meanData(2048, 0x0000);   // bf16 0.0
    std::vector<uint16_t> stdevData(2048, 0x3F80);   // bf16 1.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{2048,}, {2048,}}, ge::DT_BF16, ge::FORMAT_ND, true, meanData.data()},
            {{{2048,}, {2048,}}, ge::DT_BF16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{2048,}, {2048,}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 6: Both tensor mean/stdev (L2 已广播), output FP16
TEST_F(StatelessNormalTilingTest, test_both_scalar_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {512};
    int64_t seedValue = 66;
    int64_t offsetValue = 0;
    std::vector<uint16_t> meanData(512, 0x0000);    // fp16 0.0
    std::vector<uint16_t> stdevData(512, 0x3C00);   // fp16 1.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{512,}, {512,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, meanData.data()},
            {{{512,}, {512,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{512,}, {512,}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 7: Both tensor mean/stdev (L2 已广播), output FP16
TEST_F(StatelessNormalTilingTest, test_mean_tensor_stdev_scalar_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {2048};
    int64_t seedValue = 77;
    int64_t offsetValue = 0;
    std::vector<uint16_t> meanData(2048, 0x4200);    // fp16 ~3.0
    std::vector<uint16_t> stdevData(2048, 0x3C00);   // fp16 ~1.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{2048,}, {2048,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, meanData.data()},
            {{{2048,}, {2048,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{2048,}, {2048,}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 8: Both tensor mean/stdev (L2 已广播), output BF16
TEST_F(StatelessNormalTilingTest, test_mean_tensor_stdev_scalar_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {4096};
    int64_t seedValue = 88;
    int64_t offsetValue = 0;
    std::vector<uint16_t> meanData(4096, 0xBF80);   // bf16 ~-1.0
    std::vector<uint16_t> stdevData(4096, 0x3F00);   // bf16 ~0.5
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{4096,}, {4096,}}, ge::DT_BF16, ge::FORMAT_ND, true, meanData.data()},
            {{{4096,}, {4096,}}, ge::DT_BF16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{4096,}, {4096,}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 9: Both tensor mean/stdev (L2 已广播), output BF16
TEST_F(StatelessNormalTilingTest, test_mean_scalar_stdev_tensor_bf16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {512};
    int64_t seedValue = 111;
    int64_t offsetValue = 0;
    std::vector<uint16_t> meanData(512, 0x4040);    // bf16 ~3.0
    std::vector<uint16_t> stdevData(512, 0x3F80);   // bf16 ~1.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{512,}, {512,}}, ge::DT_BF16, ge::FORMAT_ND, true, meanData.data()},
            {{{512,}, {512,}}, ge::DT_BF16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{512,}, {512,}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 10: Both tensor mean/stdev, output FP16, 多维 shape
TEST_F(StatelessNormalTilingTest, test_both_tensor_fp16)
{
    optiling::RandomOperatorCompileInfo compileInfo = {40, 196608};
    vector<int64_t> shapeValue = {128, 64};
    int64_t seedValue = 222;
    int64_t offsetValue = 0;
    constexpr int64_t outputSize = 128 * 64;
    std::vector<uint16_t> meanData(outputSize, 0x4000);   // fp16 ~2.0
    std::vector<uint16_t> stdevData(outputSize, 0x3C00);  // fp16 ~1.0
    gert::TilingContextPara tilingContextPara(
        "StatelessNormal",
        {
            {{{2,}, {2,}}, ge::DT_INT64, ge::FORMAT_ND, true, shapeValue.data()},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &seedValue},
            {{{1,}, {1,}}, ge::DT_INT64, ge::FORMAT_ND, true, &offsetValue},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, meanData.data()},
            {{{outputSize,}, {outputSize,}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, stdevData.data()},
        },
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    uint64_t expectTilingKey = 100;
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}
