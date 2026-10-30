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
#include "../../../../op_kernel/arch35/adds_tiling_data.h"
#include "../../../../op_host/arch35/adds_tiling_arch35.h"

using namespace std;
using namespace ge;
using optiling::AddsCompileInfo;

class AddsTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddsTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddsTiling TearDown" << std::endl;
    }
};

// Test 1: FP32 basic shape
TEST_F(AddsTilingTest, adds_fp32_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(1.5f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 2: FP16 basic shape
TEST_F(AddsTilingTest, adds_fp16_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // input
        },
        {
            {{{1024, 1024}, {1024, 1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(2.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 3;  // schMode=0, dtype=1 (FP16)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 3: BF16 basic shape
TEST_F(AddsTilingTest, adds_bf16_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},  // input
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_BF16, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(0.5f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 5;  // schMode=0, dtype=2 (BF16)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 4: INT32 basic shape
TEST_F(AddsTilingTest, adds_int32_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{256, 256}, {256, 256}}, ge::DT_INT32, ge::FORMAT_ND},  // input
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_INT32, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(10.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 11;  // schMode=0, dtype=5 (INT32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 5: INT16 basic shape
TEST_F(AddsTilingTest, adds_int16_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT16, ge::FORMAT_ND},  // input
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT16, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(5.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 9;  // schMode=0, dtype=4 (INT16)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 6: INT64 basic shape
TEST_F(AddsTilingTest, adds_int64_basic)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(15.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 7: 1D shape - FP32
TEST_F(AddsTilingTest, adds_fp32_1d)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{1024}, {1024}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 8: INT64 1D shape
TEST_F(AddsTilingTest, adds_int64_1d)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{2048}, {2048}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(100.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 9: INT64 4D shape
TEST_F(AddsTilingTest, adds_int64_4d)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(50.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 10: INT64 negative scalar
TEST_F(AddsTilingTest, adds_int64_negative_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{256, 256}, {256, 256}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(-100.0f)},  // negative scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 11: INT64 zero scalar
TEST_F(AddsTilingTest, adds_int64_zero_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},  // zero scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 12: INT64 large scalar
TEST_F(AddsTilingTest, adds_int64_large_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{64, 64}, {64, 64}}, ge::DT_INT64, ge::FORMAT_ND},  // input
        },
        {
            {{{64, 64}, {64, 64}}, ge::DT_INT64, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(10000.0f)},  // large scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 13;  // schMode=0, dtype=6 (INT64)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 13: 4D shape - FP32
TEST_F(AddsTilingTest, adds_fp32_4d)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{2, 4, 8, 16}, {2, 4, 8, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(3.0f)},  // scalar value
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 8: Negative scalar value - FP32
TEST_F(AddsTilingTest, adds_fp32_negative_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(-2.5f)},  // negative scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 9: Zero scalar value - FP32
TEST_F(AddsTilingTest, adds_fp32_zero_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(0.0f)},  // zero scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 10: Large scalar value - FP32
TEST_F(AddsTilingTest, adds_fp32_large_scalar)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 40;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{128, 128}, {128, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(1000.0f)},  // large scalar
        },
        &compileInfo,
        40,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 11: Different core count
TEST_F(AddsTilingTest, adds_fp32_different_cores)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 16;
    compileInfo.ubSize = 196608;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{512, 512}, {512, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},  // scalar
        },
        &compileInfo,
        16,
        196608,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}

// Test 12: Small UB size
TEST_F(AddsTilingTest, adds_fp32_small_ub)
{
    AddsCompileInfo compileInfo;
    compileInfo.coreNum = 8;
    compileInfo.ubSize = 32768;
    
    gert::TilingContextPara tilingContextPara(
        "Adds",
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},  // input
        },
        {
            {{{256, 256}, {256, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},  // output
        },
        {
            {"value", Ops::Math::AnyValue::CreateFrom<float>(1.0f)},  // scalar
        },
        &compileInfo,
        8,
        32768,
        4096);
    
    uint64_t expectTilingKey = 7;  // schMode=0, dtype=3 (FP32)
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectWorkspaces);
}