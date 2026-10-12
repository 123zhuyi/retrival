/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CumulativeLogsumexpInferShape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CumulativeLogsumexpInferShape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CumulativeLogsumexpInferShape TearDown" << std::endl; }
};

TEST_F(CumulativeLogsumexpInferShape, fp32_rank2_axis_int32)
{
    int32_t axisValue = 1;
    gert::InfershapeContextPara infershapeContextPara(
        "CumulativeLogsumexp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(false)),
            gert::InfershapeContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(false)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CumulativeLogsumexpInferShape, fp16_rank3_negative_axis_reverse)
{
    int64_t axisValue = -1;
    gert::InfershapeContextPara infershapeContextPara(
        "CumulativeLogsumexp",
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &axisValue},
        },
        {
            {{{3, 5, 7}, {3, 5, 7}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("exclusive", Ops::Math::AnyValue::CreateFrom<bool>(true)),
            gert::InfershapeContextPara::OpAttr("reverse", Ops::Math::AnyValue::CreateFrom<bool>(true)),
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 5, 7},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CumulativeLogsumexpInferShape, dynamic_rank2_shape_passthrough)
{
    int16_t axisValue = 0;
    gert::InfershapeContextPara infershapeContextPara("CumulativeLogsumexp",
                                                      {
                                                          {{{-1, 64}, {-1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{1}, {1}}, ge::DT_INT16, ge::FORMAT_ND, true, &axisValue},
                                                      },
                                                      {
                                                          {{{-1, 64}, {-1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 64},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
