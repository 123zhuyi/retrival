/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_eq_scalar.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_eq_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "eq_scalar_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "eq_scalar_test TearDown" << std::endl;
    }
};

// 正常场景
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal01)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eq_scalar_test, test_eq_scalar_normal01_with_double)
{
    auto tensor_self =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{3, 4, 9, 6, 7, 11});

    auto scalar_desc = ScalarDesc(static_cast<double>(1.0));

    auto out_tensor_desc =
        TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast,self:float16 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal06)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_eq_scalar_test, test_eq_scalar_normal06_with_double)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(static_cast<double>(10.0));

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast,self:bfloat16 & other:int
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal08)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    float value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        // 当前不支持bf16类型的scalar
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// 正常场景 做dtype cast, self:int32 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal02)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast, self:int64 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal03)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(1.0f);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast, self:int8 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal11)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast, self:uint8 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal05)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int8_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast,self:bool & other:int
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal07)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<int32_t>{false, false, false, false, false, false});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 做dtype cast, self:uint32 & other:float
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal12)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    auto scalar_desc = ScalarDesc(uint32_t(1));

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 正常场景 做dtype cast, self:uint64 & other:uint64
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal04)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<uint64_t>{1, 2, 3, 1, 2, 3});

    uint64_t value = 3;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 正常场景 空tensor
TEST_F(l2_eq_scalar_test, test_eq_scalar_normal13)
{
    auto tensor_self = TensorDesc({2, 3, 0}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 0}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常场景 空指针
TEST_F(l2_eq_scalar_test, test_eq_scalar_abnormal01)
{
    auto tensor_self = nullptr;

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景 空指针
TEST_F(l2_eq_scalar_test, test_eq_scalar_abnormal02)
{
    auto tensor_self =
        TensorDesc({2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = nullptr;

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景
TEST_F(l2_eq_scalar_test, test_eq_scalar_abnormal03)
{
    auto tensor_self = TensorDesc({2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc = TensorDesc({2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 不校验 私有format
TEST_F(l2_eq_scalar_test, test_eq_scalar_abnormal05)
{
    auto tensor_self = TensorDesc({2, 3, 1}, ACL_BF16, ACL_FORMAT_NC1HWC0)
                           .ValueRange(-10, 10)
                           .Value(vector<int32_t>{1, 2, 3, 1, 2, 3});

    float value = 10;
    auto scalar_desc = ScalarDesc(value);

    auto out_tensor_desc =
        TensorDesc({2, 3, 1}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{false, false, false, false, false, false});

    auto ut = OP_API_UT(aclnnEqScalar, INPUT(tensor_self, scalar_desc), OUTPUT(out_tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    if (GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}