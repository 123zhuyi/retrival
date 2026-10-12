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
 * \file test_sort_with_index_branch_infershape.cpp
 * \brief Infershape branch coverage for SortWithIndex (ascend910b).
 *
 * dtype scope: the 910B first release declares ONLY the 4 int32-index combinations
 * value{fp16,fp32,bf16,int32} x index{int32}. int64-index is not a declared combination. (InferShape
 * itself is dtype-agnostic -- it only copies shapes -- so restricting to int32-index here keeps the
 * UT aligned with the declared dtype scope rather than implying int64 is supported.)
 *
 * The main-line infershape UT covers the (fp16, int32) same-shape main path, 1-D, axis-len-1, and
 * the static shape_mismatch -> GRAPH_FAILED. This file extends to the remaining 3 declared value
 * dtypes (fp32 / bf16 / int32, each with int32 index) so the y.shape = x.shape /
 * sorted_index.shape = index.shape pass-through is exercised across all declared value dtypes, plus
 * higher rank, the empty tensor, and a non-fp16 shape mismatch.
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class SortWithIndexBranchInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SortWithIndexBranchInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SortWithIndexBranchInfershape TearDown" << std::endl;
    }
};

// fp32 value + int32 index, 3-D, same shape -> y.shape = x.shape, sorted_index.shape = index.shape.
TEST_F(SortWithIndexBranchInfershape, infershape_fp32_int32_same_shape_3d)
{
    gert::InfershapeContextPara para("SortWithIndex",
        {
            {{{2, 3, 16}, {2, 3, 16}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 16}, {2, 3, 16}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expect = {{2, 3, 16}, {2, 3, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// bf16 value + int32 index, same shape -> pass-through.
TEST_F(SortWithIndexBranchInfershape, infershape_bf16_int32_same_shape_2d)
{
    gert::InfershapeContextPara para("SortWithIndex",
        {
            {{{5, 32}, {5, 32}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{5, 32}, {5, 32}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(true)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expect = {{5, 32}, {5, 32}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// int32 value + int32 index, empty tensor {0} -> shapes still pass through (y={0}, sorted_index={0}).
TEST_F(SortWithIndexBranchInfershape, infershape_int32_int32_empty_tensor)
{
    gert::InfershapeContextPara para("SortWithIndex",
        {
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    std::vector<std::vector<int64_t>> expect = {{0}, {0}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expect);
}

// Non-fp16 dtype combo with x.shape != index.shape -> shape_mismatch -> GRAPH_FAILED.
TEST_F(SortWithIndexBranchInfershape, infershape_fp32_int32_shape_mismatch_fail)
{
    gert::InfershapeContextPara para("SortWithIndex",
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 7}, {4, 7}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)},
            {"descending", Ops::Math::AnyValue::CreateFrom<bool>(false)},
            {"stable", Ops::Math::AnyValue::CreateFrom<bool>(false)},
        });
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
