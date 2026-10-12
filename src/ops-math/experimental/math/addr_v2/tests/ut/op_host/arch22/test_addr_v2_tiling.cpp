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
 * \file test_addr_v2_tiling.cpp
 * \brief UT test cases for AddrV2 tiling error paths (arch22 / Ascend910B)
 *
 * Design ref: DESIGN.md §4.3 - Tiling error path coverage
 *   - dtype mismatch, shape validation, broadcast check, empty tensor, unsupported dtype
 *   - CompileInfo: {coreNumAiv=24, ubSize=196608} (Ascend910B3, UB=192KB)
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch22/addr_v2_struct.h"
#include "../../../../op_host/arch22/addr_v2_tiling.h"

using namespace std;

class AddrV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddrV2Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "AddrV2Tiling TearDown" << std::endl; }
};

// 1. dtype不一致（x1=FP32, x2=FP16），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_dtype_mismatch)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 2. x2不是1D（2D输入），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_x2_not_1d)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 3. x3不是1D（2D输入），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_x3_not_1d)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 4. x1为1D但不可broadcast（x1=[5], x2=[4], x3=[3]），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_x1_1d_broadcast)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 5. x1为2D但不可broadcast（x1=[5, 7], x2=[4], x3=[3]），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_x1_2d_broadcast)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{5, 7}, {5, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 6. x1维度大于2（3D输入），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_x1_dim_3)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{2, 4, 3}, {2, 4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 7. y shape不匹配（y=[4,5] 而应该是 [4,3]），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_y_shape_mismatch)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 8. 空tensor（x2 shape为0），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_empty_tensor)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{0}, {0}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 9. 不支持的dtype（INT32），期望报错
TEST_F(AddrV2Tiling, ascend910B_test_tiling_failed_unsupported_dtype)
{
    optiling::AddrV2CompileInfo compileInfo = {24, 196608};
    gert::TilingContextPara tilingContextPara("AddrV2",
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
