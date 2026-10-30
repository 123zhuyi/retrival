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

#include "../../../op_api/aclnn_cat.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_cat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_cat_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_cat_test TearDown" << endl;
    }

    void TearDown() override
    {
        op::SetPlatformNpuArch(NpuArch::DAV_2201);
    }
};

TEST_F(l2_cat_test, cat_dtype_all_support)
{
    vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16, ACL_INT8, ACL_INT32,     ACL_UINT8, ACL_INT16,
                                   ACL_INT64, ACL_DOUBLE,  ACL_BOOL, ACL_COMPLEX64, ACL_BOOL,  ACL_COMPLEX128};
    for (auto dtype : dtype_list) {
        cout << "+++++++++++++++++++++++ start to test dtype " << String(dtype) << endl;
        auto tensor_1_desc = TensorDesc({3, 5}, dtype, ACL_FORMAT_ND)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        auto tensor_2_desc =
            TensorDesc({2, 5}, dtype, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
        auto out_tensor_desc = TensorDesc({5, 5}, dtype, ACL_FORMAT_ND);
        auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

        int64_t dim = 0;
        auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        if (dtype == ACL_COMPLEX128) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
        }
    }
}

TEST_F(l2_cat_test, cat_different_dtype_format)
{
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);

    EXPECT_EQ(aclRet, ACL_SUCCESS);

    out_tensor_desc = TensorDesc({5, 5}, ACL_INT32, ACL_FORMAT_ND);
    auto ut1 = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut2 = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    tensor_2_desc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20});
    out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut3 = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cat_test, cat_dtype_all_format)
{
    vector<aclFormat> format_list{ACL_FORMAT_NC1HWC0, ACL_FORMAT_NCHW,  ACL_FORMAT_NHWC, ACL_FORMAT_ND,
                                  ACL_FORMAT_HWCN,    ACL_FORMAT_NDHWC, ACL_FORMAT_NCDHW};
    for (auto format : format_list) {
        cout << "+++++++++++++++++++++++ start to test format " << format << endl;
        auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, format)
                                 .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
        auto tensor_2_desc =
            TensorDesc({2, 5}, ACL_FLOAT, format).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
        auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, format);
        auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

        int64_t dim = 0;
        auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        if (format == ACL_FORMAT_NC1HWC0) {
            EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
        } else {
            EXPECT_EQ(aclRet, ACL_SUCCESS);
        }
    }
}

TEST_F(l2_cat_test, cat_nullptr)
{
    int64_t dim = 0;
    auto tensor_1_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                             .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto tensor_2_desc =
        TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
    auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    auto ut_1 = OP_API_UT(aclnnCat, INPUT(nullptr, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(nullptr));
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cat_test, cat_dim_all)
{
    auto tensor_1_desc = TensorDesc({3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc =
        TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto out_tensor_desc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 1;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = -1;
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = -2;
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    dim = 2;
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cat_test, cat_many_tensors)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_list_desc = TensorListDesc(34, tensor_1_desc);
    auto out_tensor_desc = TensorDesc({68, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    dim = 1;
    out_tensor_desc = TensorDesc({2, 198}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc_2 = TensorListDesc(66, tensor_1_desc);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc_2, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, cat_empty_tensors)
{
    auto tensor_1_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_3_desc = TensorDesc({3, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    // 2 tensor left except empty tensor
    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc, tensor_3_desc});
    auto out_tensor_desc = TensorDesc({5, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 0 tensor left except empty tensor
    tensor_list_desc = TensorListDesc(3, tensor_2_desc);
    out_tensor_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 1 tensor left except empty tensor
    tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 1 tensor left except empty tensor, empty is first
    tensor_list_desc = TensorListDesc({tensor_2_desc, tensor_1_desc});
    out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, cat_only_one_tensor)
{
    auto tensor_1_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    // only one tensor
    int64_t dim = 0;
    auto tensor_list_desc = TensorListDesc({tensor_1_desc});
    auto out_tensor_desc = TensorDesc({2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // only empty tensor
    tensor_list_desc = TensorListDesc(3, tensor_2_desc);
    out_tensor_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // only one empty tensor
    tensor_list_desc = TensorListDesc(1, tensor_2_desc);
    out_tensor_desc = TensorDesc({0, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, cat_non_contiguous)
{
    auto tensor_1_desc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {3, 5}).ValueRange(-1, 1);
    auto tensor_2_desc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {2, 5}).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 1;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, cat_9dim)
{
    auto tensor_1_desc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({4, 2, 3, 2, 2, 3, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, cat_one_dim1_empty_tensor)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_cat_test, ascend310P_bfloat16)
{
    op::SetPlatformNpuArch(NpuArch::DAV_2002);
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({1, 3}, ACL_BF16, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cat_test, dtype_promote_to_complex128)
{
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto tensor_2_desc = TensorDesc({1, 3}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0, 2);
    auto out_tensor_desc = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});

    int64_t dim = 0;
    auto ut = OP_API_UT(aclnnCat, INPUT(tensor_list_desc, dim), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}