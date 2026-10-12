/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../mul/op_api/aclnn_mul.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_muls_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "muls_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "muls_test TearDown" << endl;
    }
};

TEST_F(l2_muls_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1, 16, 1, 1}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMuls, INPUT((aclTensor*)nullptr, (aclScalar*)nullptr), OUTPUT(tensor_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_muls_test, case_001)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_muls_test, case_002)
{
    auto tensor_desc = TensorDesc({10, 5, 1}, ACL_INT64, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5, 1}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_muls_test, case_005)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_muls_test, case_006)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_muls_test, case_007)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_muls_test, case_008)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(static_cast<int64_t>(5));
    auto out_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnMuls, INPUT(tensor_desc, scalar_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}