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

#include "math/q_r/op_host/op_api/aclnn_qr.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_qr_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "Qr Test Setup" << endl; }
    static void TearDownTestCase() { cout << "Qr Test TearDown" << endl; }
};

TEST_F(l2_qr_test, case_normal)
{
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    bool some = false;
    auto q_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // // SAMPLE: precision simulate
}

// 空tensor 场景二：最后一个维度为0 some为true
TEST_F(l2_qr_test, case_empty_n_true)
{
    auto self_desc = TensorDesc({1, 1, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    bool some = true;
    auto q_desc = TensorDesc({1, 1, 4, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 1, 0, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // // SAMPLE: precision simulate
}

// 空tensor 场景三: 倒数第二个维度为0， some为true and false
TEST_F(l2_qr_test, case_empty_m)
{
    auto self_desc = TensorDesc({1, 1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool some = false;
    auto q_desc = TensorDesc({1, 1, 0, 0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 1, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    for (int i = 0; i < 2; i++) {
        some = !some;
        auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// 空tensor 场景四: 其他维度为0， some为true and false
TEST_F(l2_qr_test, case_empty_other)
{
    auto self_desc = TensorDesc({1, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    bool some = false;
    auto q_desc = TensorDesc({1, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 0, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    for (int i = 0; i < 2; i++) {
        some = !some;
        auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// check nullptr
TEST_F(l2_qr_test, case_nullptr)
{
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    bool some = false;
    auto q_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut1 = OP_API_UT(aclnnQr, INPUT(nullptr, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize1 = 0;
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspaceSize1);
    EXPECT_EQ(aclRet1, ACLNN_ERR_INNER_NULLPTR);

    auto ut2 = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(nullptr, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspaceSize2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_INNER_NULLPTR);

    auto ut3 = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize3 = 0;
    aclnnStatus aclRet3 = ut3.TestGetWorkspaceSize(&workspaceSize3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_INNER_NULLPTR);
}

// check dtype valid
TEST_F(l2_qr_test, case_dtype_valid)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128,
        ACL_INT32};
    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        bool some = false;
        auto q_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);
        auto r_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

        auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (ValidList[i] != ACL_INT32) {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        } else {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

// check different dtype between input and output
TEST_F(l2_qr_test, case_dtype_diff)
{
    vector<aclDataType> ValidList = {
        ACL_FLOAT,
        ACL_FLOAT16,
        ACL_DOUBLE,
        ACL_COMPLEX64,
        ACL_COMPLEX128,
        ACL_INT32};
    int length = ValidList.size();
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    bool some = false;
    for (int i = 0; i < length; i++) {
        auto q_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND).Precision(0.0001, 0.0001);
        auto r_desc = TensorDesc({1, 1, 4, 4}, ValidList[i], ACL_FORMAT_ND).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        if (ValidList[i] != ACL_INT32) {
            EXPECT_EQ(aclRet, ACLNN_SUCCESS);
        } else {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        }
    }
}

// check format
TEST_F(l2_qr_test, case_format_valid)
{
    vector<aclFormat> ValidList = {
        ACL_FORMAT_NCHW,
        ACL_FORMAT_NHWC,
        ACL_FORMAT_ND,
        ACL_FORMAT_NC1HWC0,
        ACL_FORMAT_FRACTAL_Z,
        ACL_FORMAT_NC1HWC0_C04,
        ACL_FORMAT_HWCN,
        ACL_FORMAT_NDHWC,
        ACL_FORMAT_FRACTAL_NZ,
        ACL_FORMAT_NCDHW,
        ACL_FORMAT_NDC1HWC0,
        ACL_FRACTAL_Z_3D};
    int length = ValidList.size();
    bool some = false;
    for (int i = 0; i < length; i++) {
        auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i])
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto q_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]).Precision(0.0001, 0.0001);
        auto r_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ValidList[i]).Precision(0.0001, 0.0001);
        auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    }
}

// checkshape dim < 2
TEST_F(l2_qr_test, case_dim_smaller_2)
{
    auto self_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4});
    bool some = false;
    auto q_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// checkshape dim > 8
TEST_F(l2_qr_test, case_dim_greater_8)
{
    auto self_desc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4});
    bool some = false;
    auto q_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// checkshape some = false
TEST_F(l2_qr_test, case_shape_some_false)
{
    auto self_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    bool some = false;
    auto q_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    auto q_desc_2 = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc_2 = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut2 = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc_2, r_desc_2));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspaceSize2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_INVALID);
}

// checkshape some = true
TEST_F(l2_qr_test, case_shape_some_true)
{
    auto self_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    bool some = true;
    auto q_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto q_desc_2 = TensorDesc({1, 1, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc_2 = TensorDesc({1, 1, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut2 = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc_2, r_desc_2));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspaceSize2);
    EXPECT_EQ(aclRet2, ACLNN_SUCCESS);
}

// check not contiguous usr transfer
TEST_F(l2_qr_test, case_not_contiguous)
{
    auto self_desc = TensorDesc({5, 4}, ACL_COMPLEX64, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1000, 1000);
    bool some = false;
    auto q_desc = TensorDesc({5, 5}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto r_desc = TensorDesc({5, 4}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnQr, INPUT(self_desc, some), OUTPUT(q_desc, r_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}