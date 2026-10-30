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

#include "../../../op_api/aclnn_addcdiv.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_addcdiv_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "addcdiv_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "addcdiv_test TearDown" << endl;
    }
};

// 普通场景（含精度）
TEST_F(l2_addcdiv_test, case_1)
{
    auto input = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto tensor1 = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto tensor2 = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto out = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self空指针
TEST_F(l2_addcdiv_test, case_2)
{
    auto input = nullptr;

    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// tensor1空指针
TEST_F(l2_addcdiv_test, case_3)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = nullptr;

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// tensor2空指针
TEST_F(l2_addcdiv_test, case_4)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto tensor2 = nullptr;

    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// value空指针
TEST_F(l2_addcdiv_test, case_5)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});
    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});
    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});
    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = nullptr;

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out空指针
TEST_F(l2_addcdiv_test, case_6)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});
    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});
    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});
    auto out = nullptr;

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// tensor1、tensor2 dtype不支持推导
TEST_F(l2_addcdiv_test, case_7)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self 与tensor1、tensor2 dtype不支持推导
TEST_F(l2_addcdiv_test, case_8)
{
    auto input = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out与self、tensor1、tensor2 dtype不支持推导
TEST_F(l2_addcdiv_test, case_9)
{
    auto input = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// rank=9场景
TEST_F(l2_addcdiv_test, case_10)
{
    auto input = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto tensor1 = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto tensor2 = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto out = TensorDesc({3, 3, 3, 3, 3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// tensor1、tensor2 不满足broardcast
TEST_F(l2_addcdiv_test, case_11)
{
    auto input = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor1 = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto out = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self与 tensor1、tensor2 不满足broardcast
TEST_F(l2_addcdiv_test, case_12)
{
    auto input = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor1 = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto out = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out 与 self、tensor1、tensor2 shape不匹配
TEST_F(l2_addcdiv_test, case_13)
{
    auto input = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor1 = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto out = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor场景
TEST_F(l2_addcdiv_test, case_14)
{
    auto input = TensorDesc({4, 1, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor1 = TensorDesc({4, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto tensor2 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto out = TensorDesc({4, 0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float16场景
TEST_F(l2_addcdiv_test, case_16)
{
    auto input = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16场景
TEST_F(l2_addcdiv_test, case_17)
{
    auto input = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// int32场景
TEST_F(l2_addcdiv_test, case_18)
{
    auto input = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int32场景
TEST_F(l2_addcdiv_test, case_20)
{
    auto input = TensorDesc({3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_INT16, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_INT16, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// int8场景
TEST_F(l2_addcdiv_test, case_21)
{
    auto input = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// uint8场景
TEST_F(l2_addcdiv_test, case_22)
{
    auto input = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// bool场景
TEST_F(l2_addcdiv_test, case_23)
{
    auto input = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_BOOL, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// complex64场景
TEST_F(l2_addcdiv_test, case_24)
{
    auto input = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// complex128场景
TEST_F(l2_addcdiv_test, case_25)
{
    auto input = TensorDesc({3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// abnormal场景
TEST_F(l2_addcdiv_test, case_26)
{
    auto input = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 2, 3});

    auto tensor1 = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{4, 4, 4});

    auto tensor2 = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND).ValueRange(0, 10).Value(vector<float>{1, 1, 1});

    auto out = TensorDesc({3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// nchw场景
TEST_F(l2_addcdiv_test, case_27)
{
    auto input = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto tensor1 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto tensor2 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto out = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// nchw场景
TEST_F(l2_addcdiv_test, case_28)
{
    auto input = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto tensor1 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto tensor2 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto out = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// nchw场景
TEST_F(l2_addcdiv_test, case_29)
{
    auto input = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto tensor1 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto tensor2 = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto out = TensorDesc({3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ndhwc场景
TEST_F(l2_addcdiv_test, case_30)
{
    auto input = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto tensor1 = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto tensor2 = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto out = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// ncdhw场景
TEST_F(l2_addcdiv_test, case_31)
{
    auto input = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto tensor1 = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto tensor2 = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto out = TensorDesc({3, 3, 3, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_addcdiv_test, l2_addcdiv_test_case_32_Test)
{
    auto input = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 2}, 0, {4, 2});

    auto tensor1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 2}, 0, {4, 2});

    auto tensor2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW, {1, 2}, 0, {4, 2});

    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto scalar = ScalarDesc(1.0f);

    auto ut = OP_API_UT(aclnnAddcdiv, INPUT(input, tensor1, tensor2, scalar), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}