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
#include <iostream>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

// AcosGradV2: 输出 z 的 shape = 输入 y 的 shape
class AcosGradV2Infershape : public testing::Test {};

TEST_F(AcosGradV2Infershape, fp32_same_shape)
{
    gert::InfershapeContextPara para("AcosGradV2",
                                     {
                                         {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{4, 3, 4}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AcosGradV2Infershape, fp16_one_dim)
{
    gert::InfershapeContextPara para("AcosGradV2",
                                     {
                                         {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{1024}, {1024}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1024}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AcosGradV2Infershape, bf16_two_dim)
{
    gert::InfershapeContextPara para("AcosGradV2",
                                     {
                                         {{{32, 256}, {32, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 256}, {32, 256}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 256}};
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectOutputShape);
}
