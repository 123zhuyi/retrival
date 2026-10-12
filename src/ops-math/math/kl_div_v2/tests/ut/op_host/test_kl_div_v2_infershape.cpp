/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kl_div_v2_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class KlDivV2Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "KlDivV2Infershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "KlDivV2Infershape TearDown" << std::endl; }
};

// KLDivV2 has 2 inputs (input, target) and 1 output.
// Attr "reduction": "none" -> output shape equals input shape.
//                   "batchmean"/"sum" -> output is a scalar (0-dim).
TEST_F(KlDivV2Infershape, kl_div_v2_infershape_float_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KlDivV2Infershape, kl_div_v2_infershape_float16_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4, 3, 4}, {4, 3, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KlDivV2Infershape, kl_div_v2_infershape_int32_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KlDivV2Infershape, kl_div_v2_infershape_bf16_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KlDivV2Infershape, kl_div_v2_infershape_1d_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{10}, {10}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {10},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(KlDivV2Infershape, kl_div_v2_infershape_empty_none_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{0, 3, 4}, {0, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {0, 3, 4},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// reduction == "batchmean" -> output is a scalar (0-dim)
TEST_F(KlDivV2Infershape, kl_div_v2_infershape_float_batchmean_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("batchmean")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// reduction == "sum" -> output is a scalar (0-dim)
TEST_F(KlDivV2Infershape, kl_div_v2_infershape_float_sum_test)
{
    gert::InfershapeContextPara infershapeContextPara(
        "KLDivV2",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"reduction", Ops::Math::AnyValue::CreateFrom<std::string>("sum")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
