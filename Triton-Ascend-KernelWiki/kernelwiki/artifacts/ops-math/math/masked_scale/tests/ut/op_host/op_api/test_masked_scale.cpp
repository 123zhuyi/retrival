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

#include "../../../../op_api/aclnn_masked_scale.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace op;

class l2_masked_scale_test : public testing::Test {
    protected:
        static void SetUpTestCase() { std::cout << "masked_scale_test SetUp" << std::endl; }

        static void TearDownTestCase() { std::cout << "masked_scale_test TearDown" << std::endl; }
};

// test nullptr
TEST_F(l2_masked_scale_test, ascend950_case_nullptr) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;
    uint64_t workspace_size = 0;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(nullptr, mask, scale), OUTPUT(out));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnMaskedScale, INPUT(self, nullptr, scale), OUTPUT(out));
    aclnnStatus aclRet1 = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet1, ACLNN_ERR_PARAM_NULLPTR);
    
    auto ut2 = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(nullptr));
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);
}

// test xdtype:float16, maskdtype:int8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp16_int8_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:bfloat16, maskdtype:int8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_bf16_int8_01) {
    auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float32, maskdtype:int8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp32_int8_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float16, maskdtype:uint8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp16_uint8_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:bfloat16, maskdtype:uint8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_bf16_uint8_01) {
    auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float32, maskdtype:uint8
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp32_uint8_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float16, maskdtype:float16
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp16_fp16_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:bfloat16, maskdtype:float16
TEST_F(l2_masked_scale_test, ascend950_case_dtype_bf16_fp16_01) {
    auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float32, maskdtype:float16
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp32_fp16_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float16, maskdtype:float32
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp16_fp32_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:bfloat16, maskdtype:float32
TEST_F(l2_masked_scale_test, ascend950_case_dtype_bf16_fp32_01) {
    auto self = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test xdtype:float32, maskdtype:float32
TEST_F(l2_masked_scale_test, ascend950_case_dtype_fp32_fp32_01) {
    auto self = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非法dype：不支持的dtype
TEST_F(l2_masked_scale_test, ascend950_case_dtype_invalid_0) {
    auto self = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape：shape不一致
TEST_F(l2_masked_scale_test, ascend950_case_shape_invalid_1) {
    auto self = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
    float scale = 1.0;
    
    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非法shape：shape的dim超过8维
TEST_F(l2_masked_scale_test, ascend950_case_shape_invalid_2) {
    auto self = TensorDesc({2, 2, 2, 1, 1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mask = TensorDesc({2, 2, 2, 1, 1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2, 2, 1, 1, 1, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    float scale = 1.0;

    auto ut = OP_API_UT(aclnnMaskedScale, INPUT(self, mask, scale), OUTPUT(out));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
