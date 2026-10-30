/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
 
 #include "../../../op_api/aclnn_chunk_cat.h"
 
 #include "op_api_ut_common/inner/types.h"
 #include "op_api_ut_common/op_api_ut.h"
 #include "op_api_ut_common/scalar_desc.h"
 #include "op_api_ut_common/tensor_desc.h"
 #include "opdev/platform.h"
 
 using namespace std;
 
 class l2_chunk_cat_test : public testing::Test {
 protected:
     static void SetUpTestCase()
     {
         cout << "l2_chunk_cat_test SetUp" << endl;
     }
 
     static void TearDownTestCase()
     {
         cout << "l2_chunk_cat_test TearDown" << endl;
     }
 
     void TearDown() override
     {
         op::SetPlatformNpuArch(NpuArch::DAV_2201);
     }
 };
 
 TEST_F(l2_chunk_cat_test, cat_dtype_all_support)
 {
     vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16};
     for (auto dtype : dtype_list) {
         cout << "+++++++++++++++++++++++ start to test dtype " << String(dtype) << endl;
         auto tensor_1_desc = TensorDesc({5, 3}, dtype, ACL_FORMAT_ND)
                                  .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
         auto tensor_2_desc =
             TensorDesc({5, 2}, dtype, ACL_FORMAT_ND).Value(vector<float>{16, 17, 18, 19, 20, 21, 22, 23, 24, 0});
         auto out_tensor_desc = TensorDesc({5, 5}, dtype, ACL_FORMAT_ND);
         auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
 
         int64_t dim = 0;
         int64_t num_chunks = 5;
         auto ut = OP_API_UT(aclnnChunkCat, INPUT(tensor_list_desc, dim, num_chunks), OUTPUT(out_tensor_desc));
         uint64_t workspace_size = 0;
         aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
         EXPECT_EQ(aclRet, ACL_SUCCESS);
     }
 }
 
 
 TEST_F(l2_chunk_cat_test, cat_one_tensor_support)
 {
     vector<aclDataType> dtype_list{ACL_FLOAT, ACL_FLOAT16};
     for (auto dtype : dtype_list) {
         cout << "+++++++++++++++++++++++ start to test dtype " << String(dtype) << endl;
         auto tensor_1_desc = TensorDesc({5, 3}, dtype, ACL_FORMAT_ND)
                                  .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
 
         auto out_tensor_desc = TensorDesc({5, 3}, dtype, ACL_FORMAT_ND);
         auto tensor_list_desc = TensorListDesc({tensor_1_desc});
 
         int64_t dim = 0;
         int64_t num_chunks = 5;
         auto ut = OP_API_UT(aclnnChunkCat, INPUT(tensor_list_desc, dim, num_chunks), OUTPUT(out_tensor_desc));
         uint64_t workspace_size = 0;
         aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
         EXPECT_EQ(aclRet, ACL_SUCCESS);
     }
 }