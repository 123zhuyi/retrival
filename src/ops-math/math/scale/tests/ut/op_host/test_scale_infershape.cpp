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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ScaleInferShapeTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ScaleInferShapeTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ScaleInferShapeTest TearDown" << std::endl;
  }
};

TEST_F(ScaleInferShapeTest, infershape_float_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_float16_3d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_bf16_4d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 5}, {4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{128}, {128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {128},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScaleInferShapeTest, infershape_scalar_scale)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Scale",
        {
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
