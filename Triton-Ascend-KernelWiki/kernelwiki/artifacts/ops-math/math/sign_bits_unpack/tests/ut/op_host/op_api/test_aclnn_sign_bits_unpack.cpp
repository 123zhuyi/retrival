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
#include "aclnn_sign_bits_unpack.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class sign_bits_unpack_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "sign_bits_unpack_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "sign_bits_unpack_test TeastDown" << endl; }
};

// CheckNotNull self
TEST_F(sign_bits_unpack_test, ascend910B2_case_null_self)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(nullptr, size, dType), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(sign_bits_unpack_test, ascend910B2_case_null_out)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(tensor_desc, size, dType), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid self
TEST_F(sign_bits_unpack_test, ascend910B2_case_valid_self)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid out
TEST_F(sign_bits_unpack_test, ascend910B2_case_valid_out)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid dtype
TEST_F(sign_bits_unpack_test, ascend910B2_case_valid_dtype)
{
    int64_t size = 2;
    aclDataType dType = ACL_UINT8;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValidIsSame dtype and out
TEST_F(sign_bits_unpack_test, ascend910B2_case_valid_out_dtype)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT16;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue size is zero
TEST_F(sign_bits_unpack_test, ascend910B2_case_value_size_1)
{
    int64_t size = 0;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue size cannot be divided
TEST_F(sign_bits_unpack_test, ascend910B2_case_value_size_2)
{
    int64_t size = 3;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckValue out's dim is diff size
TEST_F(sign_bits_unpack_test, ascend910B2_case_value_outdim_size)
{
    int64_t size = 4;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape self
TEST_F(sign_bits_unpack_test, ascend910B2_case_shape_self)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({16, 2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape out
TEST_F(sign_bits_unpack_test, ascend910B2_case_shape_out)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(sign_bits_unpack_test, case_float_float16_910)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT16;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//  CheckEmpty_Tensor
TEST_F(sign_bits_unpack_test, ascend910B2_case_empty_tensor)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT16;
    auto self_tensor_desc = TensorDesc({0}, ACL_UINT8, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Normal scenario
// CheckFLoat_Flaot16
TEST_F(sign_bits_unpack_test, ascend910B2_case_float_float16)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT16;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{159, 15});
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<int16_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckFLoat_Flaot
TEST_F(sign_bits_unpack_test, ascend910B2_case_float_float)
{
    int64_t size = 2;
    aclDataType dType = ACL_FLOAT;
    auto self_tensor_desc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{159, 15});
    auto out_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto ut = OP_API_UT(aclnnSignBitsUnpack, INPUT(self_tensor_desc, size, dType), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}