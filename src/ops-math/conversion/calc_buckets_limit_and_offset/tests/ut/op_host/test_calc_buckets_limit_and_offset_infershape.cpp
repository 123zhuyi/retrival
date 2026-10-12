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
 * \file test_calc_buckets_limit_and_offset_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;
class CalcBucketsLimitAndOffsetInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CalcBucketsLimitAndOffsetInferShapeTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CalcBucketsLimitAndOffsetInferShapeTest TearDown" << std::endl; }
};

// Migrated from canndev: bucket_list{100}, ivf_counts{200}, ivf_offset{200}, all DT_INT32.
// Expect: both outputs inherit bucket_list shape {100}; buckets_limit dtype = DT_INT32,
// buckets_offset dtype = ivf_offset dtype (DT_INT32).
TEST_F(CalcBucketsLimitAndOffsetInferShapeTest, calc_buckets_limit_and_offset_infershape_success)
{
    gert::InfershapeContextPara::TensorDescription bucketList({{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription ivfCounts({{200}, {200}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription ivfOffset({{200}, {200}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription bucketsLimit({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription bucketsOffset({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("CalcBucketsLimitAndOffset", {bucketList, ivfCounts, ivfOffset},
                                                      {bucketsLimit, bucketsOffset});
    std::vector<std::vector<int64_t>> expectOutputShape = {{100}, {100}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ivf_offset is DT_INT64: verify buckets_offset dtype follows ivf_offset (DT_INT64),
// while buckets_limit stays DT_INT32 from bucket_list.
TEST_F(CalcBucketsLimitAndOffsetInferShapeTest, calc_buckets_limit_and_offset_infershape_int64_offset)
{
    gert::InfershapeContextPara::TensorDescription bucketList({{50}, {50}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription ivfCounts({{100}, {100}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription ivfOffset({{100}, {100}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription bucketsLimit({{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription bucketsOffset({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND);
    gert::InfershapeContextPara infershapeContextPara("CalcBucketsLimitAndOffset", {bucketList, ivfCounts, ivfOffset},
                                                      {bucketsLimit, bucketsOffset});
    std::vector<std::vector<int64_t>> expectOutputShape = {{50}, {50}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
