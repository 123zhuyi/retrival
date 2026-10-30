// ----------------------------------------------------------------------------
// Copyright (c) Huawei Device Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

using namespace ge;

class MaskedScaleInfershapeTest : public testing::Test {};

TEST_F(MaskedScaleInfershapeTest, fp32_shape_equal_success)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{2, 3}, {2, 3}}, DT_FLOAT, FORMAT_ND},
                                         {{{2, 3}, {2, 3}}, DT_UINT8, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_FLOAT, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{2, 3}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectShape);
}

TEST_F(MaskedScaleInfershapeTest, bf16_shape_equal_success)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{4, 1, 8}, {4, 1, 8}}, DT_BF16, FORMAT_ND},
                                         {{{4, 1, 8}, {4, 1, 8}}, DT_FLOAT16, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_BF16, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{4, 1, 8}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectShape);
}

TEST_F(MaskedScaleInfershapeTest, fp16_int8_shape_equal_success)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{32}, {32}}, DT_FLOAT16, FORMAT_ND},
                                         {{{32}, {32}}, DT_INT8, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_FLOAT16, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{32}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectShape);
}

TEST_F(MaskedScaleInfershapeTest, empty_tensor_success)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{0}, {0}}, DT_FLOAT, FORMAT_ND},
                                         {{{0}, {0}}, DT_FLOAT16, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_FLOAT, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{0}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectShape);
}

TEST_F(MaskedScaleInfershapeTest, high_dim_shape_success)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{2, 3, 4, 5}, {2, 3, 4, 5}}, DT_FLOAT16, FORMAT_ND},
                                         {{{2, 3, 4, 5}, {2, 3, 4, 5}}, DT_FLOAT, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_FLOAT16, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{2, 3, 4, 5}};
    ExecuteTestCase(para, GRAPH_SUCCESS, expectShape);
}

TEST_F(MaskedScaleInfershapeTest, shape_size_mismatch_failed)
{
    gert::InfershapeContextPara para("MaskedScale",
                                     {
                                         {{{2, 3}, {2, 3}}, DT_FLOAT16, FORMAT_ND},
                                         {{{2, 4}, {2, 4}}, DT_INT8, FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, DT_FLOAT16, FORMAT_ND},
                                     });
    std::vector<std::vector<int64_t>> expectShape = {{}};
    ExecuteTestCase(para, GRAPH_FAILED, expectShape);
}
