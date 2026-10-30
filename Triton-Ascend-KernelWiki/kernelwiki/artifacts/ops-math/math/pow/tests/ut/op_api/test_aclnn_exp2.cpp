/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_exp2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class L2_Exp2_Test : public testing::Test {
    protected:
        static void SetUpTestCase() {
            cout << "exp2_test SetUp" << endl;
        }

        static void TearDownTestCase() {
            cout << "exp2_test TearDown" << endl;
        }
};

//测试用例编号：aclnnExp2_001
//测试项：exp2.out输入支持FLOAT
TEST_F(L2_Exp2_Test, aclnnExp2_001_DType_FLOAT_ND) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_002
//测试项：exp2.out输入支持FLOAT16
TEST_F(L2_Exp2_Test, aclnnExp2_002_DType_FLOAT16_ND) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_005
//测试项：exp2.out输入支持NAN
TEST_F(L2_Exp2_Test, aclnnExp2_005_DType_NAN_ND) {

    auto selfDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float> {NAN, 1, NAN, 2, NAN, 0, 3, -1});
    auto outDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_006
//测试项：exp2.out输入支持INF
TEST_F(L2_Exp2_Test, aclnnExp2_006_DType_INF_ND) {

    auto selfDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float> {INFINITY, 1, -INFINITY, 2, INFINITY, 0, 3, -1});
    auto outDesc = TensorDesc({8, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_011
//测试项：exp2.out支持输入和输出类型不一致
TEST_F(L2_Exp2_Test, aclnnExp2_011_DType_InputFloat_OutputFloat16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_012
//测试项：exp2.out支持输入和输出类型不一致
TEST_F(L2_Exp2_Test, aclnnExp2_012_DType_InputFloat_OutputBFloat16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_014
//测试项：exp2.out支持输入和输出类型不一致
TEST_F(L2_Exp2_Test, aclnnExp2_014_DType_InputFloat16_OutputFloat) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_015
//测试项：exp2.out支持输入和输出类型不一致
TEST_F(L2_Exp2_Test, aclnnExp2_015_DType_InputFloat16_OutputBFloat16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_019
//测试项：数据格式覆盖测试
//前面测试已覆盖
TEST_F(L2_Exp2_Test, aclnnExp2_019_DFormat_ND) {
}

//测试用例编号：aclnnExp2_020
//测试项：数据格式覆盖测试
TEST_F(L2_Exp2_Test, aclnnExp2_020_DFormat_NCHW) {

    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_021
//测试项：数据格式覆盖测试
TEST_F(L2_Exp2_Test, aclnnExp2_021_DFormat_NHWC) {

    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_022
//测试项：数据格式覆盖测试
TEST_F(L2_Exp2_Test, aclnnExp2_022_DFormat_HWCN) {

    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_023
//测试项：数据格式覆盖测试
TEST_F(L2_Exp2_Test, aclnnExp2_023_DFormat_NDHWC) {

    auto selfDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_024
//测试项：数据格式覆盖测试
TEST_F(L2_Exp2_Test, aclnnExp2_024_DFormat_NCDHW) {

    auto selfDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto outDesc = TensorDesc({2, 3, 4, 5, 6}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_025
//测试项：exp2.out支持输入为空tensor
TEST_F(L2_Exp2_Test, aclnnExp2_025_Input_NullTensor) {

    auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}


//测试用例编号：aclnnExp2_026
//测试项：exp2.out支持输入为非连续Tensor
TEST_F(L2_Exp2_Test, aclnnExp2_026_Input_NotContigousTensor) {

    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

//测试用例编号：aclnnExp2_027
//测试项：exp2.out支持输入为空指针场景
TEST_F(L2_Exp2_Test, aclnnExp2_027_Input_NullPointer) {

    auto selfDesc = nullptr;
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

//测试用例编号：aclnnExp2_028
//测试项：exp2.out输入数据类型为int8
TEST_F(L2_Exp2_Test, aclnnExp2_028_Input_DType_int8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

//测试用例编号：aclnnExp2_029
//测试项：exp2.out输入数据类型为uint8
TEST_F(L2_Exp2_Test, aclnnExp2_029_Input_DType_uint8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT8, ACL_FORMAT_ND)
                               .ValueRange(0, 8);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}


//测试用例编号：aclnnExp2_030
//测试项：exp2.out输入数据类型为int16
TEST_F(L2_Exp2_Test, aclnnExp2_030_Input_DType_int16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

//测试用例编号：aclnnExp2_031
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_031_Input_ErrorType_uint16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_032
//测试项：exp2.out输入数据类型为int32
TEST_F(L2_Exp2_Test, aclnnExp2_032_Input_DType_int32) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}


//测试用例编号：aclnnExp2_033
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_033_Input_ErrorType_uint32) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_034
//测试项：exp2.out输入数据类型为int64
TEST_F(L2_Exp2_Test, aclnnExp2_034_Input_ErrorType_int64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

//测试用例编号：aclnnExp2_035
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_035_Input_ErrorType_uint64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_036
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_036_Input_DType_bool) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}

//测试用例编号：aclnnExp2_037
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_037_Input_ErrorType_complex64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_038
//测试项：exp2.out异常数据类型测试
TEST_F(L2_Exp2_Test, aclnnExp2_038_Input_ErrorType_complex128) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_COMPLEX128, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_039
//测试项：exp2.out输出out为空指针
TEST_F(L2_Exp2_Test, aclnnExp2_039_Output_NullTensor) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = nullptr;

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

//测试用例编号：aclnnExp2_040
//测试项：exp2.out输出out为非连续Tensor
TEST_F(L2_Exp2_Test, aclnnExp2_040_Output_NotContigous) {

}

//测试用例编号：aclnnExp2_041
//测试项：exp2.out输入维度超过8维场景
TEST_F(L2_Exp2_Test, aclnnExp2_041_Dim_Morethan_8) {

    auto selfDesc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND);;

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_042
//测试项：exp2.out输入输出shape不一致
TEST_F(L2_Exp2_Test, aclnnExp2_042_Input_Output_DiffShape) {

    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);;

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_043
//测试项：exp2.out输入输出format不一致
TEST_F(L2_Exp2_Test, aclnnExp2_043_Input_Output_DiffFormat) {

    auto selfDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto outDesc = TensorDesc({2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

//测试用例编号：aclnnExp2_044
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_044_InputNormal_OutputINT8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_045
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_045_InputNormal_OutputUINT8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_046
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_046_InputNormal_OutputINT16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


//测试用例编号：aclnnExp2_047
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_047_InputNormal_OutputUINT16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


//测试用例编号：aclnnExp2_048
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_048_InputNormal_OutputINT32) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_049
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_049_InputNormal_OutputUINT32) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_050
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_050_InputNormal_OutputINT64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_051
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_051_InputNormal_OutputUINT64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_052
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_052_InputNormal_OutputBOOL) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_055
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_055_InputNormal_OutputINT8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


//测试用例编号：aclnnExp2_056
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_056_InputNormal_OutputUINT8) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_UINT8, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_UINT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


//测试用例编号：aclnnExp2_057
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_057_InputNormal_OutputINT16) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


//测试用例编号：aclnnExp2_058
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_058_InputNormal_OutputINT32) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_059
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_059_InputNormal_OutputINT64) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT64, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT64, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_060
//测试项：exp2.out输入类型为合法值，输出数据类型为非法值
TEST_F(L2_Exp2_Test, aclnnExp2_060_InputNormal_OutputBOOL) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3, 4}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnExp2, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

//测试用例编号：aclnnExp2_061
//测试项：exp2_输入支持FLOAT
TEST_F(L2_Exp2_Test, aclnnExp2_061_DType_FLOAT_ND_INPLACE) {

    auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceExp2, INPUT(selfDesc), OUTPUT());

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

}