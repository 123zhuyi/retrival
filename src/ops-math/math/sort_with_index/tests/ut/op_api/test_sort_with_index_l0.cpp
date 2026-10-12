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
#include "math/sort_with_index/op_api/sort_with_index.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 64;

class SortWithIndexTest : public ::testing::Test {
public:
    SortWithIndexTest() : exe(nullptr) {}

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

TEST_F(SortWithIndexTest, SortWithIndex_SUCC_FLOAT)
{
    auto self = CreateAclTensor({2, 8}, ACL_FLOAT);
    auto index = CreateAclTensor({2, 8}, ACL_INT64);
    int64_t axis = 1;
    bool descending = false;
    bool stable = false;
    auto result = l0op::SortWithIndex(self, index, axis, descending, stable, exe);
    auto [values, indices] = result;
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(SortWithIndexTest, SortWithIndex_SUCC_FLOAT16)
{
    auto self = CreateAclTensor({2, 8}, ACL_FLOAT16);
    auto index = CreateAclTensor({2, 8}, ACL_INT64);
    int64_t axis = 1;
    bool descending = false;
    bool stable = false;
    auto result = l0op::SortWithIndex(self, index, axis, descending, stable, exe);
    auto [values, indices] = result;
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(SortWithIndexTest, SortWithIndex_SUCC_DESCENDING)
{
    auto self = CreateAclTensor({2, 8}, ACL_FLOAT);
    auto index = CreateAclTensor({2, 8}, ACL_INT64);
    int64_t axis = 1;
    bool descending = true;
    bool stable = false;
    auto result = l0op::SortWithIndex(self, index, axis, descending, stable, exe);
    auto [values, indices] = result;
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}

TEST_F(SortWithIndexTest, SortWithIndex_SUCC_1D)
{
    auto self = CreateAclTensor({16}, ACL_FLOAT);
    auto index = CreateAclTensor({16}, ACL_INT64);
    int64_t axis = 0;
    bool descending = false;
    bool stable = false;
    auto result = l0op::SortWithIndex(self, index, axis, descending, stable, exe);
    auto [values, indices] = result;
    ASSERT_NE(values, nullptr);
    ASSERT_NE(indices, nullptr);
}
