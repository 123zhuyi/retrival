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

// #include "../../../op_api/aclnn_channel_shuffle.h"
// #include "op_api_ut_common/op_api_ut.h"
// #include "op_api_ut_common/scalar_desc.h"
// #include "op_api_ut_common/tensor_desc.h"

// using namespace std;
// using namespace op;

// class l2_channel_shuffle_test : public testing::Test {
// protected:
//     static void SetUpTestCase()
//     {
//         cout << "channel_shuffle_test SetUp" << endl;
//     }
//     static void TearDownTestCase()
//     {
//         cout << "channel_shuffle_test TearDown" << endl;
//     }
// };

// // bf16场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_bfloat16)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_BF16, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // float16场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_float16)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // float场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_float)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // double场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_double)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // uint8场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_uint8)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_UINT8, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_UINT8, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // int8场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_int8)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_INT8, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // int16场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_int16)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_INT16, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // int32场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_int32)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // long场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_long)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // bool场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_bool)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_BOOL, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // complex64场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_complex64)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_COMPLEX64, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // complex128场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_complex128)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_COMPLEX128, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }

// // self空指针
// TEST_F(l2_channel_shuffle_test, case_self_nullptr)
// {
//     auto self = nullptr;
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// // out空指针
// TEST_F(l2_channel_shuffle_test, case_out_nullptr)
// {
//     auto out = nullptr;
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
// }

// // groups小于等于0
// TEST_F(l2_channel_shuffle_test, case_groups_less_equal_0)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = -1;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // groups不能被channels整除
// TEST_F(l2_channel_shuffle_test, case_groups_can_not_be_divided)
// {
//     auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 3;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // 超过最大维度限制
// TEST_F(l2_channel_shuffle_test, case_more_than_max_dim)
// {
//     auto self = TensorDesc({1, 4, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // 小于最小维度限制
// TEST_F(l2_channel_shuffle_test, case_less_than_min_dim)
// {
//     auto self = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto out = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

// // 空tensor场景（含精度）
// TEST_F(l2_channel_shuffle_test, case_empty_tensor)
// {
//     auto self = TensorDesc({1, 4, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
//     auto out = TensorDesc({1, 4, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t groups = 2;

//     auto ut = OP_API_UT(aclnnChannelShuffle, INPUT(self, groups), OUTPUT(out));
//     uint64_t workspaceSize = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);
// }
