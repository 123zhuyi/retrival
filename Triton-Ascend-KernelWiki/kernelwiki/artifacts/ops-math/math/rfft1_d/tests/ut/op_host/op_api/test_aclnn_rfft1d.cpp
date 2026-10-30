// /**
//  * Copyright(c) Huawei Technologies Co., Ltd.2025. All rights reserved.
//  * This File is a part of the CANN Open Software.
//  * Licensed under CANN Open Software License Agreement Version 2.0 (the "License");
//  * Please refer to the Licence for details. You may not use this file except in compliance with the License.
//  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
//  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//  * See LICENSE in the root of the software repository for the full text of the License.
//  */

// #include <array>
// #include <vector>
// #include "gtest/gtest.h"

// #include "acl_rfft1d.h"

// #include "op_api_ut_common/inner/types.h"
// #include "op_api_ut_common/op_api_ut.h"
// #include "op_api_ut_common/scalar_desc.h"
// #include "op_api_ut_common/tensor_desc.h"
// #include <iostream>

// using namespace op;
// using namespace std;

// class l2_rfft1d_test : public testing::Test {
// protected:
//     static void SetUpTestCase()
//     {
//         cout << "rfft1d_test SetUp" << endl;
//     }

//     static void TearDownTestCase()
//     {
//         cout << "rfft1d_test TearDown" << endl;
//     }
// };

// ///////////////////////// Precision tests /////////////////////////

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_1)
// {
//     auto self_tensor_desc = TensorDesc({48000}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 48000L;
//     int64_t dim = -1;
//     int64_t norm = 1;
//     auto out_tensor_desc = TensorDesc({n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_2)
// {
//     auto self_tensor_desc = TensorDesc({1500}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 1500L;
//     int64_t dim = -1;
//     int64_t norm = 1;
//     auto out_tensor_desc = TensorDesc({n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_3)
// {
//     auto self_tensor_desc = TensorDesc({120, 700}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 700L;
//     int64_t dim = -1;
//     int64_t norm = 2;
//     auto out_tensor_desc = TensorDesc({120, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_4)
// {
//     auto self_tensor_desc = TensorDesc({20, 32, 2048}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 2048L;
//     int64_t dim = -1;
//     int64_t norm = 3;
//     auto out_tensor_desc = TensorDesc({20, 32, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_5)
// {
//     auto self_tensor_desc = TensorDesc({16, 20, 30, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 1024L;
//     int64_t dim = -1;
//     int64_t norm = 1;
//     auto out_tensor_desc = TensorDesc({16, 20, 30, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_6)
// {
//     auto self_tensor_desc = TensorDesc({5, 8, 11, 15, 777}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 777L;
//     int64_t dim = -1;
//     int64_t norm = 2;
//     auto out_tensor_desc = TensorDesc({5, 8, 11, 15, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_7)
// {
//     auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 999}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 999L;
//     int64_t dim = -1;
//     int64_t norm = 3;
//     auto out_tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACLNN_ERR_INNER_NULLPTR);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_8)
// {
//     auto self_tensor_desc = TensorDesc({4111}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 4111L;
//     int64_t dim = -1;
//     int64_t norm = 1;
//     auto out_tensor_desc = TensorDesc({n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }

// TEST_F(l2_rfft1d_test, ascend910B1_case_aicore_precision_9)
// {
//     auto self_tensor_desc = TensorDesc({16, 20, 30, 1024}, ACL_FLOAT, ACL_FORMAT_ND);
//     int64_t n = 1024L;
//     int64_t dim = -1;
//     int64_t norm = 2;
//     auto out_tensor_desc = TensorDesc({16, 20, 30, n / 2 + 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
//     auto ut = OP_API_UT(aclRfft1D, INPUT(self_tensor_desc, n, dim, norm, out_tensor_desc), OUTPUT());

//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
//     EXPECT_EQ(aclRet, ACL_SUCCESS);

//     // ut.TestPrecision();
// }