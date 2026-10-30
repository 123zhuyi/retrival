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
 * \file test_clip_by_value_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace ge;

class ClipByValueTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ClipByValueTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ClipByValueTest TearDown" << std::endl;
    }
};

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_4)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_5)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_6)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_7)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 6}, {1, 6}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_9)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, -1}, {1, -1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 1, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_10)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{0, -1}, {0, -1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_11)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{0, -1}, {0, -1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_12)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 5}, {3, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 0}, {1, 0}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_13)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{3, 1, 0}, {3, 1, 0}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{1, 5}, {1, 5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 0, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(ClipByValueTest, clip_by_value_infershape_diff_test_14)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
            {{{7, 9, 0, 9, 8, 7, 8, 9}, {7, 9, 0, 9, 8, 7, 8, 9}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_is_scalar)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_has_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_clip_value_dim_num_equal)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1}, {-1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_dim_num_great_than_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{16, 16}, {16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{16, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_dim_num_great_than_1_and_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{-1, 16}, {-1, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ClipByValueTest, infer_axis_type_with_input_all_unknown_dim)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ClipByValue",
        {
            {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
