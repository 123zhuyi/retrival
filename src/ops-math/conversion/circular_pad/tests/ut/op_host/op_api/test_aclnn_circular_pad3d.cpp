/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "conversion/circular_pad/op_host/op_api/aclnn_circular_pad3d.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class circular_pad3d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "circular_pad3d_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "circular_pad3d_test TearDown" << std::endl;
    }
};

TEST_F(circular_pad3d_test, case_1)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// empty tensor, first dim is 0
TEST_F(circular_pad3d_test, case_2)
{
    auto self_tensor_desc = TensorDesc({0, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({0, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull self padding
TEST_F(circular_pad3d_test, case_3)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(nullptr, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, nullptr), OUTPUT(out_desc));
    workspace_size = 0;
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(circular_pad3d_test, case_4)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(nullptr));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckShape padding dim
TEST_F(circular_pad3d_test, case_5)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape self dim
TEST_F(circular_pad3d_test, case_6)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape diffrent dim num of self and out
TEST_F(circular_pad3d_test, case_7)
{
    auto self_tensor_desc = TensorDesc({1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckFormat diffrent format
TEST_F(circular_pad3d_test, case_8)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_UNDEFINED);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype support
TEST_F(circular_pad3d_test, case_9)
{
    vector<aclDataType> ValidList = {ACL_FLOAT16, ACL_FLOAT, ACL_BF16, ACL_INT32, ACL_INT8};

    int length = ValidList.size();
    for (int i = 0; i < length; i++) {
        auto self_tensor_desc = TensorDesc({1, 2, 2, 2}, ValidList[i], ACL_FORMAT_ND);
        auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
        auto out_desc = TensorDesc({1, 4, 4, 4}, ValidList[i], ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// CheckDtype not support
TEST_F(circular_pad3d_test, case_10)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtype diffrent dtype of self and out
TEST_F(circular_pad3d_test, case_11)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 4, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape out dim value
TEST_F(circular_pad3d_test, case_12)
{
    auto self_tensor_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// empty tensor, second dim is 0
TEST_F(circular_pad3d_test, case_13)
{
    auto self_tensor_desc = TensorDesc({1, 0, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({1, 0, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(circular_pad3d_test, case_14)
{
    auto self_tensor_desc = TensorDesc({0, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({0, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Outputshape of non-filled axis is not equal to inputshape
TEST_F(circular_pad3d_test, case_15)
{
    auto self_tensor_desc = TensorDesc({2, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({3, 1, 4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Outputshape of filled axis is not equal to inputshape
TEST_F(circular_pad3d_test, case_16)
{
    auto self_tensor_desc = TensorDesc({2, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({2, 1, 4, 5, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Outputshape of filled axis is not equal to inputshape
TEST_F(circular_pad3d_test, case_17)
{
    auto self_tensor_desc = TensorDesc({2, 1, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto padding_desc = IntArrayDesc(vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto out_desc = TensorDesc({2, 1, 4, 5, 4}, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
    auto ut = OP_API_UT(aclnnCircularPad3d, INPUT(self_tensor_desc, padding_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}