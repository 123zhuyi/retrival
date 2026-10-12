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

using namespace ge;

class ClipByValueV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ClipByValueV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ClipByValueV2Test TearDown" << std::endl;
    }
};

TEST_F(ClipByValueV2Test, clip_by_value_infershape_scalar_test_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_infershape_test_sucsess)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_dynamic_infershape_sucsess)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, -1, 5}, {3, -1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, -1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_empty_tensor_test_sucsess)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 0, 5}, {3, 0, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 0, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_verify_test_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_verify_scalar_test_success)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueV2Test, clip_by_value_verify_dtype_test_failed)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValueV2",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 1, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
