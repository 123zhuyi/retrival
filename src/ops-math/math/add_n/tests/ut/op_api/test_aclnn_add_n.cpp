/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <gtest/gtest.h>
#include "../../../op_api/aclnn_add_n.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_addn_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_addn_test SetUp" << endl;
    }
    
    static void TearDownTestCase()
    {
        cout << "l2_addn_test TearDown" << endl;
    }
};

// 输入x为空指针
TEST_F(l2_addn_test, l2_addn_test_nullptr_input) {
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAddN, INPUT(nullptr), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 输出out为空指针
TEST_F(l2_addn_test, l2_addn_test_nullptr_output) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

/* AICORE */

// bfloat16
TEST_F(l2_addn_test, l2_addn_test_dypte_bf16) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// float16
TEST_F(l2_addn_test, l2_addn_test_dypte_fp16) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}


// float
TEST_F(l2_addn_test, l2_addn_test_dypte_fp) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// int32
TEST_F(l2_addn_test, l2_addn_test_dypte_int32) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// int64
TEST_F(l2_addn_test, l2_addn_test_dypte_int64) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 数据格式nd
TEST_F(l2_addn_test, l2_addn_test_format_nd) {
    auto tensor_1_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto tensor_2_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4, 5, 6});
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 空tensor
TEST_F(l2_addn_test, l2_addn_test_empty_tensor) {
    auto tensor_1_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 10维tensor
TEST_F(l2_addn_test, l2_addn_test_10dims) {
    auto tensor_1_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_2_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensor_list_desc = TensorListDesc({tensor_1_desc, tensor_2_desc});
    auto out_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnAddN, INPUT(tensor_list_desc), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}