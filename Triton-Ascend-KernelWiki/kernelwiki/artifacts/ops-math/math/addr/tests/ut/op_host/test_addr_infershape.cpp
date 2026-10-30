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

using namespace ge;

class AddrInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddrInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddrInfershape TearDown" << std::endl;
    }
};

TEST_F(AddrInfershape, addr_infershape_success_x1_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_x1_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_x1_2d_broadcast_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{1, 3}, {1, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_x1_2d_broadcast_2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 1}, {4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_x1_2d_broadcast_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{1, 1}, {1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_bf16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_BF16, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_int32)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_int8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT8, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_INT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_uint8)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_UINT8, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_UINT8, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_success_bool)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_BOOL, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_BOOL, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 3},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_unknown_rank_x2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_unknown_rank_x3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {-2},
    };
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AddrInfershape, addr_infershape_failed_x1_dim_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(AddrInfershape, addr_infershape_failed_x1_dim_3)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3, 2}, {4, 3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(AddrInfershape, addr_infershape_failed_x2_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2}, {4, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

TEST_F(AddrInfershape, addr_infershape_failed_x3_not_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "Addr",
        {
            {{{4, 3}, {4, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        });
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
