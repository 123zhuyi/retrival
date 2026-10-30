/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <float.h>

#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "aclnn_sign_bits_pack.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class sign_bits_pack_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "sign_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "sign_test TeastDown" << endl; }
};


// CheckNotNull self
TEST_F(sign_bits_pack_test, ascend910B2_case_null_self)
{   
    int64_t size = 2; 
    auto tensor_desc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(nullptr, size), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(sign_bits_pack_test, ascend910B2_case_null_out)
{   
    int64_t size = 2; 
    auto tensor_desc = TensorDesc({1,2}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(tensor_desc, size), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeVaild self
TEST_F(sign_bits_pack_test, ascend910B2_case_vaild_self)
{   
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({14}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeVaild out
TEST_F(sign_bits_pack_test, ascend910B2_case_vaild_out)
{   
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue size
TEST_F(sign_bits_pack_test, ascend910B2_case_value_size)
{   
    int64_t size = 0; 
    auto self_tensor_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue size 
TEST_F(sign_bits_pack_test, ascend910B2_case_value_size2)
{   
    int64_t size = 3; 
    auto self_tensor_desc = TensorDesc({14}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue shape 
TEST_F(sign_bits_pack_test, ascend910B2_case_shape_self)
{   
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({16,2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue shape 
TEST_F(sign_bits_pack_test, ascend910B2_case_shape_out)
{   
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({3}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Check empty_tensor
TEST_F(sign_bits_pack_test, ascend910B2_case_float_empty_tensor)
{
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,0}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

//not 910B
TEST_F(sign_bits_pack_test, case_float_float16)
{
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// Checkfloat_float16
TEST_F(sign_bits_pack_test, ascend910B2_case_float_float16)
{
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Checkfloat_float
TEST_F(sign_bits_pack_test, ascend910B2_case_float_float)
{
    int64_t size = 2; 
    auto self_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2,1}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsPack, INPUT(self_tensor_desc, size), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}


