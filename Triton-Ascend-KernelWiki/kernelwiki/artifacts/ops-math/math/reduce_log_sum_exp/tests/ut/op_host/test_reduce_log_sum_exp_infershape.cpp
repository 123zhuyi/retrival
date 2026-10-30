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

class ReduceLogSumExpInferShape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceLogSumExpInferShape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceLogSumExpInferShape TearDown" << std::endl;
    }
};

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_0)
{
    std::vector<int64_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1, 4}, {1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_1)
{
    std::vector<int64_t> axesValue = {0, 1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1, 1, 4}, {1, 1, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_2)
{
    std::vector<int64_t> axesValue = {-1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{2, 3}, {2, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_3)
{
    std::vector<int32_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{2, 4}, {2, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1, 4}, {1, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_4)
{
    std::vector<int64_t> axesValue = {};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0}, {0}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{2, 3, 4, 5}, {2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4, 5},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_5)
{
    std::vector<int64_t> axesValue = {1};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{-1, 2}, {-1, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{-1, 1}, {-1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-1, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_6)
{
    std::vector<int64_t> axesValue = {2, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1, 2, 1, 4, 1}, {1, 2, 1, 4, 1}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 1, 4, 1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_7)
{
    std::vector<int64_t> axesValue = {2, 4};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1, 2, 4}, {1, 2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(false))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 2, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ReduceLogSumExpInferShape, reduce_log_sum_exp_infershape_test_8)
{
    std::vector<int64_t> axesValue = {0};
    gert::InfershapeContextPara infershapeContextPara(
        "ReduceLogSumExp",
        {
            {{{-1}, {-1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, axesValue.data()},
        },
        {
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {gert::InfershapeContextPara::OpAttr("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(true))});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
