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

#include <cstdint>
#include <vector>

class fill : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "fill Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "fill Proto Test TearDown" << std::endl;
    }
};

TEST_F(fill, fill_infershape_diff_test)
{
    gert::StorageShape valueShape = {{6, 2}, {6, 2}};
    gert::StorageShape dimsShape = {{3}, {3}};
    gert::StorageShape yShape = {{6, 7, 2}, {6, 7, 2}};

    std::vector<int32_t> dims_values = {6, 7, 2};
    gert::InfershapeContextPara::TensorDescription value(valueShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription dims(
        dimsShape, ge::DT_INT32, ge::FORMAT_ND, true, dims_values.data());
    gert::InfershapeContextPara::TensorDescription y(yShape, ge::DT_INT32, ge::FORMAT_ND);

    gert::InfershapeContextPara infershapeContextPara("Fill", {dims, value}, {y});
    std::vector<std::vector<int64_t>> expectOutputShape = {{6, 7, 2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}