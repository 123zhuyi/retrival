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
 * \file test_axpy_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class AxpyV2Infershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AxpyV2Infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AxpyV2Infershape TearDown" << std::endl;
    }
};

TEST_F(AxpyV2Infershape, axpy_v2_infershape_diff_test) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AxpyV2Infershape, axpy_v2_verify_int8) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_INT8, ge::FORMAT_ND},
                                                        {{{1}, {1}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AxpyV2Infershape, axpy_v2_infershape_same_test) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{1, 3, 4}, {1, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AxpyV2Infershape, axpy_v2_infershape_dynamic_test) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AxpyV2Infershape, axpy_v2_alpha_int64_test) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AxpyV2Infershape, axpy_v2_input_int64_test) {
    gert::InfershapeContextPara infershapeContextPara("AxpyV2",
                                                      {
                                                        {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT64, ge::FORMAT_ND},
                                                        {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}