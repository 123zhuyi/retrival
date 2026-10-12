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
#include "../../../../op_host/arch35/strided_slice_v2_tiling_arch35.h"

using namespace ge;

class StridedSliceV2TilingTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "StridedSliceV2TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "StridedSliceV2TilingTest TearDown" << std::endl;
    }
};


TEST_F(StridedSliceV2TilingTest, StridedSlice_test_tiling_001)
{
    optiling::StridedSliceCompileInfo compileInfo = {2, 253952, 32, 10, true};
    vector<int64_t> beginValue = {0};
    vector<int64_t> endValue = {1};
    vector<int64_t> stridesValue = {1};
    gert::TilingContextPara tilingContextPara(
        "StridedSliceV2",
        {
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, beginValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, endValue.data()},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, stridesValue.data()},
        },
        {
            {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            gert::TilingContextPara::OpAttr("begin_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("end_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("ellipsis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("new_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
            gert::TilingContextPara::OpAttr("shrink_axis_mask", Ops::Math::AnyValue::CreateFrom<int64_t>(1)),
        },
        &compileInfo);
    uint64_t expectTilingKey = 103;
    string expectTilingData = "1 0 1 1 4295221248 0 65537 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1"
        " 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 ";

    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}