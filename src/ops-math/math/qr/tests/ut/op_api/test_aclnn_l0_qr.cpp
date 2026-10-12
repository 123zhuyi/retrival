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
#include "math/qr/op_api/qr.h"

const int64_t DATA_SIZE = 1024 * 1024;
class QrTest : public ::testing::Test {
public:
    QrTest() : exe(nullptr) {}
    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dataType)
    {
        return aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                               shape.size(), data);
    }

    void Clear() {}

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    int64_t data[DATA_SIZE] = {0};
};

TEST_F(QrTest, SOME_TRUE_SUCC)
{
    auto self = CreateAclTensor({4, 3}, ACL_FLOAT);
    auto [Q, R] = l0op::Qr(self, true, exe);
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(R, nullptr);
}

TEST_F(QrTest, SOME_FALSE_SUCC)
{
    auto self = CreateAclTensor({4, 3}, ACL_FLOAT);
    auto [Q, R] = l0op::Qr(self, false, exe);
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(R, nullptr);
}

TEST_F(QrTest, DATA_TYPE_DOUBLE_SUCC)
{
    auto self = CreateAclTensor({5, 4}, ACL_DOUBLE);
    auto [Q, R] = l0op::Qr(self, true, exe);
    ASSERT_NE(Q, nullptr);
    ASSERT_NE(R, nullptr);
}
