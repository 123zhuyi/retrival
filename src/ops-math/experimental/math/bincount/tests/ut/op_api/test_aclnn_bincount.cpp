/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_bincount.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class test_aclnn_bincount : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "test_aclnn_bincount SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "test_aclnn_bincount TearDown" << std::endl; }
};

// 计数：self int32（无 weights），out int32
TEST_F(test_aclnn_bincount, count_int32)
{
    auto self = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{2, 0, 1, 1, 3, 2});
    int64_t minlength = 0;
    auto out = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 负数：按需求改为「检查 + 报错」。负数检查是数据相关的，发生在 kernel 运行期
// （kernel 求 min，min<0 触发 ascendc_assert 报错中止）；L2 host 侧拿不到 device 数据值，
// 故 GetWorkspaceSize 仍返回 SUCCESS。负数真正被拒绝的行为由 example / AscendOpTest 真机覆盖，
// 此处仅确认 L2 不会因含负数而提前误报错。
TEST_F(test_aclnn_bincount, negative_checked_at_runtime_l2_passes)
{
    auto self = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{-2, 0, 1, 1, 3, -2});
    int64_t minlength = 0;
    auto out = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// minlength 生效：高位补 0
TEST_F(test_aclnn_bincount, count_minlength)
{
    auto self = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{1, 2, 3});
    int64_t minlength = 6;
    auto out = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 计数：self int8 -> out int32
TEST_F(test_aclnn_bincount, count_int8)
{
    auto self = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 5);
    int64_t minlength = 0;
    auto out = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 计数：self uint8 -> out int32
TEST_F(test_aclnn_bincount, count_uint8)
{
    auto self = TensorDesc({8}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 5);
    int64_t minlength = 0;
    auto out = TensorDesc({6}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 计数：self int64 -> out int64
TEST_F(test_aclnn_bincount, count_int64_out64)
{
    auto self = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 5);
    int64_t minlength = 0;
    auto out = TensorDesc({6}, ACL_INT64, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 加权：weights float -> out float
TEST_F(test_aclnn_bincount, weighted_float)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    auto weights = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t minlength = 0;
    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, weights, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 加权：weights int32（L2 Cast 成 float）-> out float
TEST_F(test_aclnn_bincount, weighted_int32)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    auto weights = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 4);
    int64_t minlength = 0;
    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, weights, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 加权：weights bool（L2 Cast 成 float）-> out float
TEST_F(test_aclnn_bincount, weighted_bool)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    auto weights = TensorDesc({4}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    int64_t minlength = 0;
    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, weights, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 加权：out double（L2 Cast float->double）
TEST_F(test_aclnn_bincount, weighted_out_double)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    auto weights = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t minlength = 0;
    auto out = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, weights, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常：self 为空指针
TEST_F(test_aclnn_bincount, self_nullptr)
{
    int64_t minlength = 0;
    auto out = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(nullptr, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常：minlength 为负（与 torch.bincount 一致：minlength should be >= 0）
TEST_F(test_aclnn_bincount, minlength_negative)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    int64_t minlength = -1;
    auto out = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 边界：空输入（self 长度 0）+ minlength>0 -> out 应被清零返回（torch 语义），接口成功
TEST_F(test_aclnn_bincount, empty_self_returns_zeros)
{
    auto self = TensorDesc({0}, ACL_INT32, ACL_FORMAT_ND);
    int64_t minlength = 5;
    auto out = TensorDesc({5}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 异常：out 非 1 维 -> ACLNN_ERR_PARAM_INVALID（out 维度校验）
TEST_F(test_aclnn_bincount, out_wrong_dimension)
{
    auto self = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{0, 1, 1, 2});
    int64_t minlength = 0;
    auto out = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBincount, INPUT(self, nullptr, minlength), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
