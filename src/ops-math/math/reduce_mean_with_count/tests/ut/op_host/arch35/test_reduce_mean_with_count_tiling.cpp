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
#include "atvoss/reduce/reduce_tiling.h"

using namespace std;
using namespace ge;

class ReduceMeanWithCountTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceMeanWithCountTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceMeanWithCountTiling TearDown" << std::endl;
    }
};

// Test: 4D input, reduce along axes [1, 2], keep_dims=true, float32
TEST_F(ReduceMeanWithCountTiling, reduce_mean_with_count_tiling_4d_fp32)
{
    Ops::Base::ReduceOpCompileInfo compileInfo;
    gert::StorageShape xShape = {{3, 5, 16, 16}, {3, 5, 16, 16}};
    gert::StorageShape countShape = {{3, 5, 16, 16}, {3, 5, 16, 16}};
    gert::StorageShape countSumShape = {{3, 5, 16, 16}, {3, 5, 16, 16}};
    gert::StorageShape yShape = {{3, 1, 1, 16}, {3, 1, 1, 16}};
    gert::TilingContextPara::TensorDescription xDesc(xShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription countDesc(countShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription countSumDesc(countSumShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription yDesc(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara(
        "ReduceMeanWithCount",
        {xDesc, countDesc, countSumDesc},
        {yDesc},
        {
            gert::TilingContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1, 2})),
            gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 5161;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test: 3D input, reduce along axes [1], keep_dims=false, float16
TEST_F(ReduceMeanWithCountTiling, reduce_mean_with_count_tiling_3d_fp16)
{
    Ops::Base::ReduceOpCompileInfo compileInfo;
    gert::StorageShape xShape = {{2, 100, 4}, {2, 100, 4}};
    gert::StorageShape countShape = {{2, 100, 4}, {2, 100, 4}};
    gert::StorageShape countSumShape = {{2, 100, 4}, {2, 100, 4}};
    gert::StorageShape yShape = {{2, 4}, {2, 4}};
    gert::TilingContextPara::TensorDescription xDesc(xShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription countDesc(countShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription countSumDesc(countSumShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription yDesc(yShape, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::TilingContextPara tilingContextPara(
        "ReduceMeanWithCount",
        {xDesc, countDesc, countSumDesc},
        {yDesc},
        {
            gert::TilingContextPara::OpAttr("axes", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({1})),
            gert::TilingContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))
        },
        &compileInfo);
    uint64_t expectedTilingKey = 5161;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}
