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
 * \file test_tensor_move_tiling.cpp
 * \brief TensorMove tiling UT.
 */

#include <gtest/gtest.h>

#include <iostream>

#include "../../../op_kernel/tensor_move_tiling_data.h"
#include "../../../op_kernel/tensor_move_tiling_key.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

using namespace std;

class TensorMoveTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TensorMoveTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TensorMoveTiling TearDown" << std::endl; }
};

struct TensorMoveTestCompileInfo {};

TEST_F(TensorMoveTiling, tiling_float16_small)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{11}, {11}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{11}, {11}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_1, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}

TEST_F(TensorMoveTiling, tiling_float32_2d)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{1024, 32}, {1024, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 32}, {1024, 32}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_2, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}

TEST_F(TensorMoveTiling, tiling_int8_small)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{12}, {12}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{12}, {12}}, ge::DT_INT8, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_0, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}

TEST_F(TensorMoveTiling, tiling_int64_small)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{20}, {20}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{20}, {20}}, ge::DT_INT64, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_3, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}

TEST_F(TensorMoveTiling, tiling_large_float32_multicore)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{4096, 4096}, {4096, 4096}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_2, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}

TEST_F(TensorMoveTiling, tiling_float16_2d_multicore)
{
    TensorMoveTestCompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara("TensorMove",
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo);
    ExecuteTestCaseForEle(tilingContextPara, ge::GRAPH_SUCCESS, true, TENSOR_MOVE_TPL_SCH_MODE_1, false,
                          EMPTY_EXPECT_TILING_DATA, {32});
}
