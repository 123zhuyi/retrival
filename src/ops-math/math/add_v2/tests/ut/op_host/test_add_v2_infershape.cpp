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
 * \file test_add_v2_infershape.cpp
 * \brief add_v2 infershape UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class AddV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddV2Infershape TearDown" << std::endl; }
};

TEST_F(AddV2Infershape, add_v2_infer_shape_eq)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_broadcast)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_dynamic)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_dynamic_dim_3d)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, -1, -1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_bf16)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                          {{{4, 4}, {4, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_int8)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_int16)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT16, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_int32)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_int64)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_uint8)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                          {{{8, 8}, {8, 8}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {8, 8},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddV2Infershape, add_v2_infer_shape_complex64)
{
    gert::InfershapeContextPara infershapeContextPara("AddV2",
                                                      {
                                                          {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                          {{{4, 4}, {4, 4}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_COMPLEX64, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
