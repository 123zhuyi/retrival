/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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

#include "../../../../op_api/aclnn_asin.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_asin_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Asin Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Asin Test TearDown" << std::endl; }
};

TEST_F(l2_asin_test, case_1)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_2)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, ascend910B2_case_3)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_4)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_5)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_6)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_7)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_8)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_9)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_10)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_11)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_12)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_13)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_14)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_15)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_16)
{
    auto tensor_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_17)
{
    auto tensor_desc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_18)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_19)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_20)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_21)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_22)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_23)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(nullptr), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_asin_test, case_24)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_25)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_26)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_27)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_28)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_29)
{
    auto tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_30)
{
    auto tensor_desc = TensorDesc({5, 6, 0, 8}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({5, 6, 0, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_asin_test, case_31)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_32)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnAsin, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_asin_test, case_33)
{
    auto tensor_desc = TensorDesc({5, 4}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto ut = OP_API_UT(aclnnInplaceAsin, INPUT(tensor_desc), OUTPUT());

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}