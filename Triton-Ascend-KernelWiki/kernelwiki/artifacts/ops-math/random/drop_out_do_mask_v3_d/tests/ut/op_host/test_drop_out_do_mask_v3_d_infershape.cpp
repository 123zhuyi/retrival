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
 * \file test_drop_out_do_mask_v3_d_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class DropOutDoMaskV3DInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "DropOutDoMaskV3DInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DropOutDoMaskV3DInfershape TearDown" << std::endl;
    }
};

TEST_F(DropOutDoMaskV3DInfershape, DropOutDoMaskV3D_test_infershape_float) {
    gert::InfershapeContextPara infershapeContextPara("DropOutDoMaskV3D",
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{96}, {96}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{2, 3, 4}, {2, 3, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
        }
        );
    std::vector<std::vector<int64_t>> expectOutputShape = {{2, 3, 4}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DropOutDoMaskV3DInfershape, DropOutDoMaskV3D_test_infershape_float16) {
    gert::InfershapeContextPara infershapeContextPara("DropOutDoMaskV3D",
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1024}, {1024}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{8, 16}, {8, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        }
        );
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 16}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}