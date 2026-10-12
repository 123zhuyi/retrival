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

#include "opdev/make_op_executor.h"
#include "aclnn_kernels/pad.h"

const int64_t DATA_SIZE = 1024 * 1024;

class PadTest : public ::testing::Test {
public:
    PadTest() : l0Executor(nullptr) {}

    aclTensor* CreateContiguousAclTensor(std::vector<int64_t> viewShape, aclDataType dtype)
    {
        std::vector<int64_t> stride(viewShape.size(), 1);
        for (int i = viewShape.size() - 2; i >= 0; i--) {
            stride[i] = stride[i + 1] * viewShape[i];
        }
        return aclCreateTensor(viewShape.data(), viewShape.size(), dtype, stride.data(), 0, ACL_FORMAT_ND,
                               viewShape.data(), viewShape.size(), data);
    }

    void Clear() { l0Executor->kernelLaunchObjList_.clear(); }

    void SetUp() override
    {
        auto l2Executor = &l0Executor;
        auto uniqueExecutor = CREATE_EXECUTOR();
        uniqueExecutor.ReleaseTo(l2Executor);
    }

    void TearDown() override { delete l0Executor; }

public:
    aclOpExecutor* l0Executor;
    int64_t data[DATA_SIZE] = {0};
};

TEST_F(PadTest, Pad)
{
    auto self = CreateContiguousAclTensor({4, 5, 6, 7}, ACL_FLOAT16);
    op::ShapeVector paddingsVec({0, 0, 0, 0, 0, 0, 0, 9});
    auto paddings = l0Executor->AllocIntArray(paddingsVec.data(), paddingsVec.size());
    auto reshapedTensor = l0op::Pad(self, l0Executor->ConvertToTensor(paddings, op::DataType::DT_INT64), l0Executor);
    ASSERT_NE(reshapedTensor, nullptr);

    op::ShapeVector expectShapeReshapedTensor({4, 5, 6, 16});
    EXPECT_EQ(op::ToShapeVector(reshapedTensor->GetViewShape()), expectShapeReshapedTensor);
}

TEST_F(PadTest, ascend950_Pad)
{
    auto self = CreateContiguousAclTensor({4, 5, 6, 7}, ACL_FLOAT16);
    op::ShapeVector paddingsVec({0, 0, 0, 0, 0, 0, 0, 9});
    auto paddings = l0Executor->AllocIntArray(paddingsVec.data(), paddingsVec.size());
    auto reshapedTensor = l0op::Pad(self, l0Executor->ConvertToTensor(paddings, op::DataType::DT_INT64), l0Executor);
    ASSERT_NE(reshapedTensor, nullptr);

    op::ShapeVector expectShapeReshapedTensor({4, 5, 6, 16});
    EXPECT_EQ(op::ToShapeVector(reshapedTensor->GetViewShape()), expectShapeReshapedTensor);
}
