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
 * \file test_tensor_redirect_infershape.cpp
 * \brief TensorRedirect op_host InferShape UT
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class TensorRedirectInfershapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TensorRedirectInfershapeTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TensorRedirectInfershapeTest TearDown" << std::endl; }
};

// 1D：output_x.shape == x.shape
TEST_F(TensorRedirectInfershapeTest, infershape_1d_fp16_same_shape)
{
    gert::StorageShape shape = {{128}, {128}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{128}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 非 64 长度 1D 输入，输出须同 shape
TEST_F(TensorRedirectInfershapeTest, infershape_1d_not_degrade_to_fixed_64)
{
    gert::StorageShape shape = {{1023}, {1023}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1023}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 2D
TEST_F(TensorRedirectInfershapeTest, infershape_2d_fp32)
{
    gert::StorageShape shape = {{32, 64}, {32, 64}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 64}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 3D
TEST_F(TensorRedirectInfershapeTest, infershape_3d_int32)
{
    gert::StorageShape shape = {{4, 8, 16}, {4, 8, 16}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{shape, ge::DT_INT32, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 8, 16}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 8D：rank 上界（spec inputs[0].rank_range = [1,8]）
TEST_F(TensorRedirectInfershapeTest, infershape_8d_rank_upper_bound)
{
    gert::StorageShape shape = {{2, 2, 2, 2, 2, 2, 2, 2}, {2, 2, 2, 2, 2, 2, 2, 2}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_INT8, ge::FORMAT_ND}},
                                     {{shape, ge::DT_INT8, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2, 2, 2, 2, 2, 2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 空 Tensor [0,3]：保留 0 维度
TEST_F(TensorRedirectInfershapeTest, infershape_empty_tensor_0x3)
{
    gert::StorageShape shape = {{0, 3}, {0, 3}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 空 Tensor [2,0,3]：0 维度在中间
TEST_F(TensorRedirectInfershapeTest, infershape_empty_tensor_2x0x3)
{
    gert::StorageShape shape = {{2, 0, 3}, {2, 0, 3}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 0, 3}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 单元素 [1]
TEST_F(TensorRedirectInfershapeTest, infershape_single_element)
{
    gert::StorageShape shape = {{1}, {1}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_BF16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_BF16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 大 shape：InferShape 与 numel 大小无关
TEST_F(TensorRedirectInfershapeTest, infershape_large_shape)
{
    gert::StorageShape shape = {{4096, 4096}, {4096, 4096}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{shape, ge::DT_INT64, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{4096, 4096}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// 动态shape（含-1）
TEST_F(TensorRedirectInfershapeTest, infershape_dynamic_dim_minus1)
{
    gert::StorageShape shape = {{-1, 32}, {-1, 32}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 32}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

// UNKNOWN_RANK（-2）
TEST_F(TensorRedirectInfershapeTest, infershape_unknown_rank_minus2)
{
    gert::StorageShape shape = {{-2}, {-2}};
    gert::InfershapeContextPara para("TensorRedirect", {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{shape, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
