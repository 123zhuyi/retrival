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
 * \file test_addr_tiling.cpp
 * \brief UT test cases for Addr tiling error paths
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "math/addr/op_kernel/arch35/addr_struct.h"
#include "math/addr/op_host/arch35/addr_tiling.h"

using namespace std;

class AddrTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddrTiling SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "AddrTiling TearDown" << std::endl;
    }
};

// 1. dtype不一致（x1=FP32, x2=FP16），期望报错 (EZ0021)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_dtype_mismatch)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 2. x2不是1D（2D输入），期望报错 (EZ0011)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_x2_not_1d)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 3. x3不是1D（2D输入），期望报错 (EZ0011)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_x3_not_1d)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 4. x1为1D但不可broadcast（x1=[5], x2=[4], x3=[3]），期望报错 (EZ0010)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_x1_1d_broadcast)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 5. x1为2D但不可broadcast（x1=[5, 7], x2=[4], x3=[3]），期望报错 (EZ0010)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_x1_2d_broadcast)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 6. x1维度大于2（3D输入），期望报错 (EZ0012)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_x1_dim_3)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 7. y shape不匹配（y=[4,5] 而应该是 [4,3]），期望报错 (EZ0008)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_y_shape_mismatch)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 8. 空tensor（x2 shape为0），期望报错 (EZ0016)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_empty_tensor)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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

// 9. 不支持的dtype（INT32），期望报错 (EZ0020)
TEST_F(AddrTiling, ascend910D1_test_tiling_failed_unsupported_dtype)
{
    optiling::AddrCompileInfo compileInfo = {64, 262144};
    gert::TilingContextPara tilingContextPara(
        "Addr",
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
