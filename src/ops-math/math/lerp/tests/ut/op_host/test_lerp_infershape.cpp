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
 * \file test_lerp_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class LerpInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "LerpInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "LerpInfershape TearDown" << std::endl;
    }
};

TEST_F(LerpInfershape, InfershapeLerp_001) {
    gert::InfershapeContextPara infershapeContextPara("Lerp",
                                                      {
                                                        {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{5, 4}, {5, 4}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{3, 2, 1}, {3, 2, 1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(LerpInfershape, InfershapeLerp_002) {
    gert::InfershapeContextPara infershapeContextPara("Lerp",
                                                      {
                                                        {{{2, 2, 2}, {2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{2, 1, 2}, {2, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{2, 1, 2}, {2, 1, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LerpInfershape, InfershapeLerp_003) {
    gert::InfershapeContextPara infershapeContextPara("Lerp",
                                                      {
                                                        {{{2, 1}, {2, 1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{2, 1, 1}, {2, 1, 1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{2, 2, 2, 2}, {2, 2, 2, 2}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 2, 2, 2},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LerpInfershape, InfershapeLerp_unknown_shape_case) {
    gert::InfershapeContextPara infershapeContextPara("Lerp",
                                                      {
                                                        {{{1, 1, -1}, {1, 1, -1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{1, 1, -1}, {1, 1, -1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{1, 1, -1}, {1, 1, -1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 1, -1},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(LerpInfershape, InfershapeLerp_empty_tensor) {
    gert::InfershapeContextPara infershapeContextPara("Lerp",
                                                      {
                                                        {{{9, 8, 0, 9}, {9, 8, 0, 9}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{9, 8, 0, 9}, {9, 8, 0, 9}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                        {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_NCHW},
                                                      },
                                                      {
                                                        {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      }
                                                     );
    std::vector<std::vector<int64_t>> expectOutputShape = {{9, 8, 0, 9},};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}