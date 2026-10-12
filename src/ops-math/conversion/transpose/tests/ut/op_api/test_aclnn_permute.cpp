// /**
//  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
//  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
//  * CANN Open Software License Agreement Version 2.0 (the "License").
//  * Please refer to the License for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <vector>
// #include <array>
// #include "gtest/gtest.h"
// #include "platform/platform_info.h"
// #include "../../../op_api/aclnn_permute.h"
// #include "op_api_ut_common/op_api_ut.h"
// #include "op_api_ut_common/scalar_desc.h"
// #include "op_api_ut_common/tensor_desc.h"
// #include "opdev/platform.h"

// using namespace std;
// using namespace op;

// class l2_transpose_permute_test : public testing::Test {
// protected:
//     static void SetUpTestCase()
//     {
//         cout << "transpose_permute_test SetUp" << endl;
//     }

//     static void TearDownTestCase()
//     {
//         cout << "transpose_permute_test TearDown" << endl;
//     }
// };

// TEST_F(l2_transpose_permute_test, case_float)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_float16)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_int8)
// {
//     auto self = TensorDesc({4, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_INT8, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_int16)
// {
//     auto self = TensorDesc({4, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_int32)
// {
//     auto self = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_int64)
// {
//     auto self = TensorDesc({4, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_INT64, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_uint8)
// {
//     auto self = TensorDesc({4, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(0, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_UINT8, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_uint16)
// {
//     auto self = TensorDesc({4, 2}, ACL_UINT16, ACL_FORMAT_ND).ValueRange(0, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_UINT16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_uint32)
// {
//     auto self = TensorDesc({4, 2}, ACL_UINT32, ACL_FORMAT_ND).ValueRange(0, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_UINT32, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_uint64)
// {
//     auto self = TensorDesc({4, 2}, ACL_UINT64, ACL_FORMAT_ND).ValueRange(0, 10);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_UINT64, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_bool)
// {
//     auto self = TensorDesc({4, 2}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_double)
// {
//     auto self = TensorDesc({4, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_complex64)
// {
//     auto self = TensorDesc({4, 2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_complex128)
// {
//     auto self = TensorDesc({4, 2}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_1d)
// {
//     auto self = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0});
//     auto out = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_3d)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 2, 1});
//     auto out = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_4d)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 2, 1, 3});
//     auto out = TensorDesc({2, 4, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_5d)
// {
//     auto self = TensorDesc({2, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2, 4, 3});
//     auto out = TensorDesc({2, 2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_6d)
// {
//     auto self = TensorDesc({1, 2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2, 3, 5, 4});
//     auto out = TensorDesc({1, 2, 3, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_7d)
// {
//     auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2, 3, 4, 6, 5});
//     auto out = TensorDesc({1, 2, 3, 4, 5, 7, 6}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_8d)
// {
//     auto self = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2, 3, 4, 5, 7, 6});
//     auto out = TensorDesc({1, 2, 3, 4, 5, 6, 8, 7}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_negative_dims)
// {
//     auto self = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({-1, -2, -3});
//     auto out = TensorDesc({3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_mixed_negative_dims)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 2, 1});
//     auto out = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_identity_perm)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1});
//     auto out = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_empty_tensor)
// {
//     auto self = TensorDesc({4, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto dims = IntArrayDesc({0, 2, 1});
//     auto out = TensorDesc({4, 2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_empty_tensor_single_dim)
// {
//     auto self = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto dims = IntArrayDesc({0});
//     auto out = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_single_element)
// {
//     auto self = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_large_tensor)
// {
//     auto self = TensorDesc({100, 200}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({200, 100}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_small_shape)
// {
//     auto self = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_self_nullptr)
// {
//     auto self = nullptr;
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// TEST_F(l2_transpose_permute_test, case_dims_nullptr)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, nullptr), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// TEST_F(l2_transpose_permute_test, case_out_nullptr)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(nullptr));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dims_size)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dims_out_of_range)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 3});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dims_out_of_range_negative)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, -3});
//     auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dims_duplicate)
// {
//     auto self = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 0, 1});
//     auto out = TensorDesc({4, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_shape_mismatch)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dim_mismatch)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0, 2});
//     auto out = TensorDesc({2, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dtype_mismatch)
// {
//     auto self = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_invalid_dim_gt_8)
// {
//     auto self = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2, 3, 4, 5, 6, 7, 8});
//     auto out = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_021)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 2, 1});
//     auto out = TensorDesc({2, 4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_102)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0, 2});
//     auto out = TensorDesc({3, 2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_201)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({2, 0, 1});
//     auto out = TensorDesc({4, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_210)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({2, 1, 0});
//     auto out = TensorDesc({4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_120)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 2, 0});
//     auto out = TensorDesc({3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_012)
// {
//     auto self = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 2});
//     auto out = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_3012)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({3, 0, 1, 2});
//     auto out = TensorDesc({5, 2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_0213)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 2, 1, 3});
//     auto out = TensorDesc({2, 4, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_0132)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 1, 3, 2});
//     auto out = TensorDesc({2, 3, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_0312)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({0, 3, 1, 2});
//     auto out = TensorDesc({2, 5, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_1032)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0, 3, 2});
//     auto out = TensorDesc({3, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_2103)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({2, 1, 0, 3});
//     auto out = TensorDesc({4, 3, 2, 5}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_transpose_3210)
// {
//     auto self = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({3, 2, 1, 0});
//     auto out = TensorDesc({5, 4, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_bfloat16)
// {
//     auto self = TensorDesc({4, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     auto curSoc = GetCurrentPlatformInfo().GetSocVersion();
//     if (curSoc == SocVersion::ASCEND910B || curSoc == SocVersion::ASCEND910_93 || curSoc == SocVersion::ASCEND910E) {
//         EXPECT_EQ(aclRet, ACL_SUCCESS);
//     } else {
//         EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
//     }
// }

// TEST_F(l2_transpose_permute_test, case_2d_with_1_in_shape)
// {
//     auto self = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({4, 1}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// TEST_F(l2_transpose_permute_test, case_large_dim_values)
// {
//     auto self = TensorDesc({1000, 1000}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto dims = IntArrayDesc({1, 0});
//     auto out = TensorDesc({1000, 1000}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnPermute, INPUT(self, dims), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }
