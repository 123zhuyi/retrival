/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_aclnn_slogdet.cpp
 * \brief Slogdet aclnn (op_api) 参数校验 UT（迭代一 P1）。
 *
 * 实验树 slogdet 原生实现仅支持 fp32（CP1 锁定），故 dtype 校验与 L0 真值源
 * (math/slogdet 支持 double/complex) 不同：fp16/double/complex 一律报 ACLNN_ERR_PARAM_INVALID。
 *
 * 覆盖点：
 *   - 空指针（self / signOut / logOut）⇒ ACLNN_ERR_PARAM_NULLPTR；
 *   - 非 fp32 dtype（self/signOut/logOut）⇒ ACLNN_ERR_PARAM_INVALID；
 *   - 非方阵（末两维不等）⇒ ACLNN_ERR_PARAM_INVALID；
 *   - rank<2 ⇒ ACLNN_ERR_PARAM_INVALID；
 *   - 输出 shape 不等于 self.shape[:-2] ⇒ ACLNN_ERR_PARAM_INVALID；
 *   - 正常 fp32 路径 ⇒ ACLNN_SUCCESS（含标量输出、batch、空 Tensor）。
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_slogdet_native.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class test_aclnn_slogdet : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "test_aclnn_slogdet SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "test_aclnn_slogdet TearDown" << endl;
    }
};

// self == nullptr ⇒ NULLPTR
TEST_F(test_aclnn_slogdet, case_nullptr_self)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT((aclTensor*)nullptr), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// signOut == nullptr ⇒ NULLPTR
TEST_F(test_aclnn_slogdet, case_nullptr_sign_out)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT((aclTensor*)nullptr, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// logOut == nullptr ⇒ NULLPTR
TEST_F(test_aclnn_slogdet, case_nullptr_log_out)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, (aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 非 fp32：self fp16 ⇒ INVALID（实验树仅 fp32）
TEST_F(test_aclnn_slogdet, case_self_fp16_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非 fp32：self double ⇒ INVALID（实验树仅 fp32，不支持 double）
TEST_F(test_aclnn_slogdet, case_self_double_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非 fp32：signOut fp16 ⇒ INVALID
TEST_F(test_aclnn_slogdet, case_sign_out_fp16_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非方阵（末两维不等）⇒ INVALID
TEST_F(test_aclnn_slogdet, case_not_square_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// rank<2（self=[4]）⇒ INVALID（OP_CHECK_MIN_DIM）
TEST_F(test_aclnn_slogdet, case_rank_lt2_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// signOut shape != self.shape[:-2] ⇒ INVALID
TEST_F(test_aclnn_slogdet, case_sign_out_shape_wrong_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logOut shape != self.shape[:-2] ⇒ INVALID
TEST_F(test_aclnn_slogdet, case_log_out_shape_wrong_invalid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 正常 fp32 路径：batch ⇒ SUCCESS
TEST_F(test_aclnn_slogdet, case_float_batch_ok)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({3, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto signOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 正常 fp32 路径：标量输出（self=[5,5] ⇒ 输出 []）⇒ SUCCESS
TEST_F(test_aclnn_slogdet, case_float_scalar_output_ok)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto signOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空 Tensor：self=[0,5,5] ⇒ workspaceSize=0，返回 SUCCESS
TEST_F(test_aclnn_slogdet, case_empty_tensor_ok)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({0, 5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto signOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto logOut = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto ut = OP_API_UT(aclnnSlogdet, INPUT(self), OUTPUT(signOut, logOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}
