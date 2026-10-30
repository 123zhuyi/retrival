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
 * \file test_split_v_tiling.cpp
 * \brief SplitV tiling UT.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "../../../op_kernel/split_v_tiling_data.h"
#include "../../../op_kernel/split_v_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class SplitVTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SplitVTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SplitVTiling TearDown" << std::endl; }
};

struct SplitVTestCompileInfo {};

TEST_F(SplitVTiling, split_v_tiling_pure_copy_single_split)
{
    SplitVTestCompileInfo compileInfo;
    int64_t sizeSplits[] = {1820};
    int64_t splitDim[] = {0};
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplits},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDim},
        },
        {
            {{{1820, 232}, {1820, 232}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        },
        &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TILING_KEY_SPLIT_V_PURE_COPY, false,
                          EMPTY_EXPECT_TILING_DATA, {sizeof(SplitVTilingDataPureCopy)});
}

TEST_F(SplitVTiling, split_v_tiling_one_row_pure_copy)
{
    SplitVTestCompileInfo compileInfo;
    int64_t sizeSplits[] = {2, 3, 5};
    int64_t splitDim[] = {0};
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{10}, {10}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplits},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDim},
        },
        {
            {{{2}, {2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
        },
        &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TILING_KEY_SPLIT_V_ONE_ROW_PURE_COPY, false,
                          EMPTY_EXPECT_TILING_DATA, {0});
}

TEST_F(SplitVTiling, split_v_tiling_invalid_split_dim)
{
    SplitVTestCompileInfo compileInfo;
    int64_t sizeSplits[] = {4, 4};
    int64_t splitDim[] = {2};
    gert::TilingContextPara tilingContextPara(
        "SplitV",
        {
            {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, sizeSplits},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, splitDim},
        },
        {
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("num_split", Ops::Math::AnyValue::CreateFrom<int64_t>(2)),
        },
        &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_FAILED, false, 0, false, EMPTY_EXPECT_TILING_DATA, {});
}
