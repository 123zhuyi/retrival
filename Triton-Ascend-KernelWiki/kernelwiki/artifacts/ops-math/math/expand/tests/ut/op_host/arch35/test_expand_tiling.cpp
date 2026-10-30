
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
 * \file test_broadcast_to_tiling.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/expand_tiling_arch35.h"

using namespace std;
using namespace ge;

class ExpandTilingTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ExpandTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ExpandTiling TearDown" << std::endl;
  }
};

TEST_F(ExpandTilingTest, Expand_tiling_test_success_1) {
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};

    gert::TilingContextPara tilingContextPara(
        "Expand",
        {{ inshape, ge::DT_UINT8, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{ outshape, ge::DT_UINT8, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

TEST_F(ExpandTilingTest, Expand_tiling_test_fail_2) {
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1, 313, 198}, {1, 1, 313, 198}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};

    gert::TilingContextPara tilingContextPara(
        "Expand",
        {{ inshape, ge::DT_UINT8, ge::FORMAT_ND }, { shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{ outshape, ge::DT_UINT8, ge::FORMAT_ND }},
        &compileInfo);
    uint64_t expectedTilingKey = 11002;
    std::vector<size_t> expectedWorkspaces = { 16777216 };
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with INT64 dtype for shape tensor to cover the INT64 branch in ReadOutShapeFromTensor
TEST_F(ExpandTilingTest, Expand_tiling_int64_shape_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1}, {1, 1}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int64_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT64, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with invalid dtype (DT_FLOAT) for shape tensor to cover dtype check failure in
// ReadOutShapeFromTensor
TEST_F(ExpandTilingTest, Expand_tiling_invalid_dtype_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_FLOAT, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with empty tensor (dim=0) to cover IsEmptyTensor failure path (xShape with 0 dim)
TEST_F(ExpandTilingTest, Expand_tiling_empty_tensor_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 0}, {1, 0}};
    gert::StorageShape outshape = {{1, 0}, {1, 0}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 0};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with empty tensor in y but not x to cover IsEmptyTensor yShape 0 dim check (line 197)
TEST_F(ExpandTilingTest, Expand_tiling_empty_tensor_y_only_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 3}, {1, 3}};
    gert::StorageShape outshape = {{1, 0}, {1, 0}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 0};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand where input has more dims than output shape to cover GetShapeInfo failure (inDimNum >
// outDimNum)
TEST_F(ExpandTilingTest, Expand_tiling_input_more_dims_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1, 1, 1}, {1, 1, 1, 1}};
    gert::StorageShape outshape = {{1, 1}, {1, 1}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 1};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand where output dims exceed max (BRCTO_MAX_DIM_NUM=8) to cover GetShapeInfo failure
TEST_F(ExpandTilingTest, Expand_tiling_output_exceed_max_dims_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape shape1 = {{9}, {9}};
    int32_t shapes[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with -1 in shape tensor to cover ApplyBroadcastRules outDim == -1 branch
TEST_F(ExpandTilingTest, Expand_tiling_negative_one_shape_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{3, 7}, {3, 7}};
    gert::StorageShape outshape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {-1, 7};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand where target dim is 1 but x dim is not 1 to cover ApplyBroadcastRules outDim==1 && xDim!=1
TEST_F(ExpandTilingTest, Expand_tiling_out_dim_1_x_not_1_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{3, 7}, {3, 7}};
    gert::StorageShape outshape = {{3, 7}, {3, 7}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 1};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11001;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with mixed B/A axes to cover MergeAxis else branch (different abInfo flags)
TEST_F(ExpandTilingTest, Expand_tiling_mixed_axes_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 3, 5}, {1, 3, 5}};
    gert::StorageShape outshape = {{7, 3, 5}, {7, 3, 5}};
    gert::StorageShape shape1 = {{3}, {3}};
    int32_t shapes[3] = {7, 3, 5};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11005;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with 2D broadcast leading to 1D after DeleteOneSizeAxis to cover MergeAxis dimNum==1 path
TEST_F(ExpandTilingTest, Expand_tiling_1d_after_delete_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1}, {1, 1}};
    gert::StorageShape outshape = {{1, 100}, {1, 100}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {1, 100};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 11003;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with negative coreNum to cover GetHardwareInfo failure path (coreNum <= 0)
TEST_F(ExpandTilingTest, Expand_tiling_negative_corenum_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = -2;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 1, 1}, {1, 1, 1}};
    gert::StorageShape outshape = {{1, 1, 313, 199}, {1, 1, 313, 199}};
    gert::StorageShape shape1 = {{4}, {4}};
    int32_t shapes[4] = {1, 1, 313, 199};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_UINT8, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_UINT8, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with zero ubSize to cover GetHardwareInfo failure for ubSize <= 0
TEST_F(ExpandTilingTest, Expand_tiling_zero_ubsize_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 0;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with zero clSize to cover GetHardwareInfo failure for clSize <= 0
TEST_F(ExpandTilingTest, Expand_tiling_zero_clsize_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 0;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with zero blockSize to cover GetHardwareInfo failure for blockSize <= 0
TEST_F(ExpandTilingTest, Expand_tiling_zero_blocksize_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 256;
    compileInfo.blockSize = 0;

    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}

// Test scenario: expand with zero vRegSize to cover GetHardwareInfo failure for vRegSize <= 0
TEST_F(ExpandTilingTest, Expand_tiling_zero_vregsize_001)
{
    optiling::ExpandCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    compileInfo.ubSize = 245760;
    compileInfo.clSize = 128;
    compileInfo.vRegSize = 0;
    compileInfo.blockSize = 32;

    gert::StorageShape inshape = {{1, 64}, {1, 64}};
    gert::StorageShape outshape = {{3, 64}, {3, 64}};
    gert::StorageShape shape1 = {{2}, {2}};
    int32_t shapes[2] = {3, 64};

    gert::TilingContextPara tilingContextPara(
        "Expand", {{inshape, ge::DT_FLOAT16, ge::FORMAT_ND}, {shape1, ge::DT_INT32, ge::FORMAT_ND, true, &shapes}},
        {{outshape, ge::DT_FLOAT16, ge::FORMAT_ND}}, &compileInfo);
    uint64_t expectedTilingKey = 0;
    std::vector<size_t> expectedWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, expectedTilingKey, expectedWorkspaces);
}
