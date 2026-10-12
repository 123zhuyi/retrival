/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

class CoshInferShapeTest : public testing::Test {};

TEST_F(CoshInferShapeTest, DynamicDimension)
{
    gert::InfershapeContextPara contextPara("Cosh",
                                            {
                                                {{{-1, 4}, {-1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{-1, 4}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}

TEST_F(CoshInferShapeTest, UnknownRank)
{
    gert::InfershapeContextPara contextPara("Cosh",
                                            {
                                                {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            },
                                            {
                                                {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                            });
    std::vector<std::vector<int64_t>> expectedOutputShapes = {{-2}};
    ExecuteTestCase(contextPara, ge::GRAPH_SUCCESS, expectedOutputShapes);
}
