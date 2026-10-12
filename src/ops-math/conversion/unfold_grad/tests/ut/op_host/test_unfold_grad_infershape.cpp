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
 * \file test_unfold_grad_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class UnfoldGradInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "UnfoldGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "UnfoldGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(UnfoldGradInferShapeTest, unfold_grad_test_success)
{   
    std::vector<int64_t> inputSizeValues = {1, 3, 3, 658, 658};
    gert::InfershapeContextPara infershapeContextPara("UnfoldGrad",
                                                      {{{{1, 3, 3, 658, 320, 20}, {1, 3, 3, 658, 320, 20}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                       {{{5}, {5}}, ge::DT_INT64, ge::FORMAT_ND, true, inputSizeValues.data()},},
                                                      {{{{1, 3, 3, 658, 658}, {1, 3, 3, 658, 658}}, ge::DT_FLOAT16, ge::FORMAT_ND},},
                                                      {gert::InfershapeContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<int64_t>(3)),
                                                       gert::InfershapeContextPara::OpAttr("size", Ops::Math::AnyValue::CreateFrom<int64_t>(20)),
                                                       gert::InfershapeContextPara::OpAttr("step", Ops::Math::AnyValue::CreateFrom<int64_t>(2))});
    
    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 3, 3, 658, 658}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}
