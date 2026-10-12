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
#include "opdev/make_op_executor.h"
#include "math/axpy_v2/op_api/axpy_v2.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 64;

class AxpyV2Test : public ::testing::Test {
public:
    AxpyV2Test() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    int64_t data[DATA_SIZE] = {1};
};

TEST_F(AxpyV2Test, AxpyV2_SUCC_FLOAT)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    auto other = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    auto alpha = CreateAclTensor({1}, ACL_FLOAT);
    auto out = l0op::AxpyV2(self, other, alpha, exe);
    ASSERT_NE(out, nullptr);
}

TEST_F(AxpyV2Test, AxpyV2_SUCC_FLOAT16)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT16);
    auto other = CreateAclTensor({3, 4, 5}, ACL_FLOAT16);
    auto alpha = CreateAclTensor({1}, ACL_FLOAT16);
    auto out = l0op::AxpyV2(self, other, alpha, exe);
    ASSERT_NE(out, nullptr);
}

TEST_F(AxpyV2Test, AxpyV2_SUCC_BROADCAST)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    auto other = CreateAclTensor({1, 4, 5}, ACL_FLOAT);
    auto alpha = CreateAclTensor({1}, ACL_FLOAT);
    auto out = l0op::AxpyV2(self, other, alpha, exe);
    ASSERT_NE(out, nullptr);
}

TEST_F(AxpyV2Test, AxpyV2_SUCC_1D)
{
    auto self = CreateAclTensor({10}, ACL_FLOAT);
    auto other = CreateAclTensor({10}, ACL_FLOAT);
    auto alpha = CreateAclTensor({1}, ACL_FLOAT);
    auto out = l0op::AxpyV2(self, other, alpha, exe);
    ASSERT_NE(out, nullptr);
}
