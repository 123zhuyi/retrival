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
 * \file test_tile_with_axis_infershape.cpp
 * \brief TileWithAxis Infershape UT
 *
 * 验证 shape 推导正确性:
 *   - 输出 y.shape = x.shape with y.shape[axis] = x.shape[axis] * tiles
 *   - rank=0 (标量) → output shape = [tiles]
 *   - 负轴归一化
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

namespace TileWithAxisUT {
using namespace std;
using namespace ge;
using namespace gert;

static const string OP_NAME = "TileWithAxis";

// ============================================================================
// Helper: create InfershapeContextPara with axis/tiles attributes
// ============================================================================
static InfershapeContextPara MakeInfershapePara(
    const string& caseName,
    initializer_list<int64_t> xShape,
    DataType dtype,
    int64_t axis,
    int64_t tiles)
{
    StorageShape inputShape = {xShape, xShape};
    // Output placeholder: unknown shape
    StorageShape outputPlaceholder = {{}, {}};

    vector<InfershapeContextPara::TensorDescription> inputs = {
        {inputShape, dtype, FORMAT_ND}
    };
    vector<InfershapeContextPara::TensorDescription> outputs = {
        {outputPlaceholder, dtype, FORMAT_ND}
    };
    vector<InfershapeContextPara::OpAttr> attrs = {
        {"axis",  Ops::Math::AnyValue::CreateFrom<int64_t>(axis)},
        {"tiles", Ops::Math::AnyValue::CreateFrom<int64_t>(tiles)}
    };

    return InfershapeContextPara(OP_NAME, inputs, outputs, attrs);
}

// ============================================================================
// P0: Normal 3D input, axis=1, tiles=3
// Input shape=[2, 3, 4] → Output shape=[2, 9, 4]
// ============================================================================
TEST(TileWithAxisInfershape, normal_3d_axis1_tiles3)
{
    auto para = MakeInfershapePara("normal_3d", {2, 3, 4}, DT_FLOAT16, 1, 3);
    vector<vector<int64_t>> expected = {{2, 9, 4}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: axis=0, tiles=3
// Input shape=[2, 3, 4] → Output shape=[6, 3, 4]
// ============================================================================
TEST(TileWithAxisInfershape, axis0_tiles3)
{
    auto para = MakeInfershapePara("axis0", {2, 3, 4}, DT_FLOAT16, 0, 3);
    vector<vector<int64_t>> expected = {{6, 3, 4}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: Negative axis: axis=-1 on 3D → axis=2
// Input shape=[2, 3, 4] → Output shape=[2, 3, 20] (4*5=20)
// ============================================================================
TEST(TileWithAxisInfershape, negative_axis_minus1)
{
    auto para = MakeInfershapePara("neg_axis", {2, 3, 4}, DT_FLOAT, -1, 5);
    vector<vector<int64_t>> expected = {{2, 3, 20}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: Negative axis=-2 on 3D → axis=1
// Input shape=[2, 3, 4] → Output shape=[2, 6, 4] (3*2=6)
// ============================================================================
TEST(TileWithAxisInfershape, negative_axis_minus2)
{
    auto para = MakeInfershapePara("neg_axis2", {2, 3, 4}, DT_FLOAT16, -2, 2);
    vector<vector<int64_t>> expected = {{2, 6, 4}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: Scalar input (rank=0)
// Input shape=[] → Output shape=[tiles]=[5]
// ============================================================================
TEST(TileWithAxisInfershape, scalar_input)
{
    auto para = MakeInfershapePara("scalar", {}, DT_FLOAT16, 0, 5);
    vector<vector<int64_t>> expected = {{5}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: tiles=1 (identity)
// Input shape=[2, 3, 4] → Output shape=[2, 3, 4]
// ============================================================================
TEST(TileWithAxisInfershape, tiles_equals_one)
{
    auto para = MakeInfershapePara("tiles1", {2, 3, 4}, DT_INT32, 1, 1);
    vector<vector<int64_t>> expected = {{2, 3, 4}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: tiles=3 with last axis (axis=-1, i.e. axis=2 for 3D)
// Input shape=[2, 3, 256] → Output shape=[2, 3, 768]
// ============================================================================
TEST(TileWithAxisInfershape, last_axis_large)
{
    auto para = MakeInfershapePara("last_axis", {2, 3, 256}, DT_FLOAT16, 2, 3);
    vector<vector<int64_t>> expected = {{2, 3, 768}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: 4D input, axis=2, tiles=3
// Input shape=[2, 4, 8, 16] → Output shape=[2, 4, 24, 16]
// ============================================================================
TEST(TileWithAxisInfershape, four_dim_axis2)
{
    auto para = MakeInfershapePara("4d", {2, 4, 8, 16}, DT_FLOAT, 2, 3);
    vector<vector<int64_t>> expected = {{2, 4, 24, 16}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

// ============================================================================
// P0: 4D input, axis=1, tiles=2
// Input shape=[5, 10, 20, 30] → Output shape=[5, 20, 20, 30]
// ============================================================================
TEST(TileWithAxisInfershape, four_dim_axis1)
{
    auto para = MakeInfershapePara("4d_axis1", {5, 10, 20, 30}, DT_FLOAT16, 1, 2);
    vector<vector<int64_t>> expected = {{5, 20, 20, 30}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expected);
}

} // namespace TileWithAxisUT
