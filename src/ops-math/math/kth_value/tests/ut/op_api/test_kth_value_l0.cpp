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
#include "math/kth_value/op_api/kth_value.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 64;

class KthValueTest : public ::testing::Test {
public:
    KthValueTest() : exe(nullptr) {}

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

TEST_F(KthValueTest, KthValue_SUCC_FLOAT)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT);
    int64_t k = 2;
    int64_t dim = 1;
    auto [values, indices] = l0op::KthValue(self, k, dim, exe);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(KthValueTest, KthValue_SUCC_FLOAT16)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_FLOAT16);
    int64_t k = 2;
    int64_t dim = 1;
    auto [values, indices] = l0op::KthValue(self, k, dim, exe);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(KthValueTest, KthValue_SUCC_INT32)
{
    auto self = CreateAclTensor({3, 4, 5}, ACL_INT32);
    int64_t k = 1;
    int64_t dim = 2;
    auto [values, indices] = l0op::KthValue(self, k, dim, exe);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(KthValueTest, KthValue_SUCC_1D)
{
    auto self = CreateAclTensor({10}, ACL_FLOAT);
    int64_t k = 3;
    int64_t dim = 0;
    auto [values, indices] = l0op::KthValue(self, k, dim, exe);
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}
