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
#include "math/histogram_fixed_width/op_host/arch35/histogram_fixed_width_tiling_arch35.h"
#include "math/histogram_fixed_width/op_kernel/arch35/histogram_fixed_width_tilingdata.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class HistogramFixedWidthTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HistogramFixedWidthTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "HistogramFixedWidthTilingTest TearDown" << std::endl; }
};

namespace {
optiling::HistogramFixedWidthCompileInfo compileInfo;
}

// Success case: fp32, full load mode
TEST_F(HistogramFixedWidthTilingTest, tiling_success_fp32_full_load)
{
    std::vector<float> rangeData = {0.0f, 10.0f};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    auto* tilingData = reinterpret_cast<HistogramFixedWidthSimtTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->bins, 5);
}

// Success case: int32, not full load mode
TEST_F(HistogramFixedWidthTilingTest, tiling_success_int32)
{
    std::vector<int32_t> rangeData = {0, 100};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{1000}, {1000}}, ge::DT_INT32, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
    auto* tilingData2 = reinterpret_cast<HistogramFixedWidthSimtTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData2->bins, 5);
}

// Success case: int64
TEST_F(HistogramFixedWidthTilingTest, tiling_success_int64)
{
    std::vector<int64_t> rangeData = {0, 100};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
}

// Success case: fp16
TEST_F(HistogramFixedWidthTilingTest, tiling_success_fp16)
{
    std::vector<uint16_t> rangeData = {0x0000, 0x4900}; // 0.0 and 10.0 in fp16
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_TRUE(tilingRet);
}

// Fail case: min >= max
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_min_eq_max)
{
    std::vector<float> rangeData = {5.0f, 5.0f};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Fail case: min > max
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_min_gt_max)
{
    std::vector<float> rangeData = {10.0f, 0.0f};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Fail case: range shape size != 2
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_range_shape_not_2)
{
    std::vector<float> rangeData = {0.0f, 10.0f, 20.0f};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Fail case: range dtype != x dtype
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_range_dtype_mismatch)
{
    std::vector<int32_t> rangeData = {0, 10};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Fail case: unsupported x dtype
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_unsupported_dtype)
{
    std::vector<int8_t> rangeData = {0, 10};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_INT8, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT8, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{5}, {5}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Fail case: output shape != bins
TEST_F(HistogramFixedWidthTilingTest, tiling_fail_output_shape_mismatch)
{
    std::vector<float> rangeData = {0.0f, 10.0f};
    int32_t nbinsData = 5;
    gert::TilingContextPara tilingContextPara("HistogramFixedWidth",
                                              {{{{100}, {100}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND, true, rangeData.data()},
                                               {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &nbinsData}},
                                              {{{{10}, {10}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{"dtype", Ops::Math::AnyValue::CreateFrom<int64_t>(3)}}, &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
