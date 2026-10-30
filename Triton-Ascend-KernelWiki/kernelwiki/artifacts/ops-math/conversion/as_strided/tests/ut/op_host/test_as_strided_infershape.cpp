/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class AsStridedInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "as_strided_test_infershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "as_strided_test_infershape SetUp" << std::endl;
    }
};

TEST_F(AsStridedInfershape, as_strided_infershape_test1)
{
    std::vector<int64_t> inputSizeValues = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {{{{101, 78, 67, 2, 78, 2}, {101, 78, 67, 2, 78, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsStridedInfershape, as_strided_infershape_test2)
{
    std::vector<int64_t> inputSizeValues = {3, 1, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {{{{101, 78, 67, 2, 78, 2}, {101, 78, 67, 2, 78, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{3}, {3}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AsStridedInfershape, as_strided_infershape_test3)
{
    std::vector<int64_t> inputSizeValues = {3, 5, 5};
    gert::InfershapeContextPara infershapeContextPara(
        "AsStrided",
        {{{{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},
         {{{9}, {9}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()}},
        {
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}