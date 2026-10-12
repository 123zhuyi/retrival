/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

class ReduceSumDavidTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceSumDavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceSumDavidTiling TearDown" << std::endl;
    }
};

// TEST_F(ReduceSumDavidTiling, reduce_sum_david_tiling1)
// {
//     Ops::Base::ReduceOpCompileInfo compileInfo;
//     gert::StorageShape inputShape = {{2048, 2, 48, 2, 2, 2}, {2048, 2, 48, 2, 2, 2}};
//     gert::StorageShape axesShape = {{3}, {3}};
//     std::vector<int32_t> axesValue = {1, 3, 5};
//     gert::StorageShape yShape = {{2048, 48, 2}, {2048, 48, 2}};
//     gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
//     gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
//     gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
//     gert::TilingContextPara tilingContextPara(
//         "ReduceSum",
//         {input, axes},
//         {y},
//         {
//             gert::TilingContextPara::OpAttr("keep_dim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
//             gert::TilingContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))
//         },
//         &compileInfo);
//     uint64_t expectedTilingKey = 5191;
//     std::vector<size_t> expectedWorkspaces = { 16777216 };
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
// }

// TEST_F(ReduceSumDavidTiling, reduce_sum_david_tiling2)
// {
//     Ops::Base::ReduceOpCompileInfo compileInfo;
//     gert::StorageShape inputShape = {{2048, 2, 48, 2, 2}, {2048, 2, 48, 2, 2}};
//     gert::StorageShape axesShape = {{2}, {2}};
//     std::vector<int64_t> axesValue = {1, 3};
//     gert::StorageShape yShape = {{2048, 48, 2}, {2048, 48, 2}};
//     gert::TilingContextPara::TensorDescription input(inputShape, ge::DT_FLOAT, ge::FORMAT_ND);
//     gert::TilingContextPara::TensorDescription axes(axesShape, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data());
//     gert::TilingContextPara::TensorDescription y(yShape, ge::DT_FLOAT, ge::FORMAT_ND);
//     gert::TilingContextPara tilingContextPara(
//         "ReduceSum",
//         {input, axes},
//         {y},
//         {
//             gert::TilingContextPara::OpAttr("keep_dim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
//             gert::TilingContextPara::OpAttr("noop_with_empty_axes", Ops::Math::AnyValue::CreateFrom<bool>(false))
//         },
//         &compileInfo);
//     uint64_t expectedTilingKey = 5908;
//     std::vector<size_t> expectedWorkspaces = { 16826368 };
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
// }
