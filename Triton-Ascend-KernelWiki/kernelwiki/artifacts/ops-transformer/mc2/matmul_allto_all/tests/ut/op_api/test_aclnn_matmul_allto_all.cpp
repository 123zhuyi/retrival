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
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_matmul_allto_all.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class TestAclnnMatmulAlltoAll : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "TestAclnnMatmulAlltoAll SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "TestAclnnMatmulAlltoAll TearDown" << endl;
    }
};

// ut用例结构体
struct MatmulAlltoAllAclnnTestParam {
    // 用例名
    string caseName;
    // 通信域卡数，ut测试默认为2
 	int worldSize;
    // 数据形状
    vector<int64_t> x1Shape; // x1数据shape，正常为（BS，H1）
    vector<int64_t> x2Shape; // x2数据shape，正常为（H1，H2）
    vector<int64_t> biasShape; // bias数据shape，正常为（H2）
    vector<int64_t> outputShape; // output数据shape，正常为（BS * world_size，H2 / world_size）
    // 数据类型
    aclDataType x1Dtype; // x1数据dtype，仅支持bfloat16和float16
    aclDataType x2Dtype; // x2数据dtype，仅支持bfloat16和float16
    aclDataType biasDtype; // bias数据dtype，仅支持float32
    aclDataType outputDtype; // 输出数据dtype，支持bfloat16、float16和float32
    // 数据格式
    aclFormat x1Format; // x1数据format，仅支持ND
    aclFormat x2Format; // x2数据format，仅支持ND
    aclFormat biasFormat; // bias数据format，仅支持ND
    aclFormat outputFormat; // output数据format，仅支持ND
    // 其它属性
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-1,-2]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    // ut用例期望返回状态
    aclnnStatus aclnnStatusUt; //期望状态
};

static MatmulAlltoAllAclnnTestParam g_casesParams[] = {
    // 正常用例 13条
    // caseid按照[算子名-x1x2output_dtype-biasDtype-format-transpose-id]构成，按bias分组
    // ========================bfloat16 系列（6条）========================
    // 1. Bias=BF16 (2)
    {"AclnnMatmulAlltoAll-bf16-biasbf16-nd-notrans-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-bf16_biasbf16_nd_trans_02", 2, {256, 128}, {256, 128}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // 2. Bias=FLOAT32 (2)
    {"AclnnMatmulAlltoAll-bf16-biasf32-nd-notrans-03", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-bf16-biasf32-nd-trans-04", 2, {256, 128}, {256, 128}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // 3. Bias=Null (2)
    {"AclnnMatmulAlltoAll-bf16-biasnull-nd-notrans-05", 2, {256, 128}, {128, 256}, {}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-bf16-biasnull-nd-trans-06", 2, {256, 128}, {256, 128}, {}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_DT_UNDEFINED, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // ========================float16 系列（6条）========================
    // 1. Bias=FP16 (2)
    {"AclnnMatmulAlltoAll-fp16-biasfp16-nd-notrans-07", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-fp16-biasfp16-nd-trans-08", 2, {256, 128}, {256, 128}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // 2. Bias=FLOAT32 (2)
    {"AclnnMatmulAlltoAll-fp16-biasf32-nd-notrans-09", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-fp16-biasf32-nd-trans-10", 2, {256, 128}, {256, 128}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // 3. Bias=Null (2)
    {"AclnnMatmulAlltoAll-fp16-biasnull-nd-notrans-11", 2, {256, 128}, {128, 256}, {}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},
    {"AclnnMatmulAlltoAll-fp16-biasnull-nd-trans-12", 2, {256, 128}, {256, 128}, {}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_DT_UNDEFINED, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_SUCCESS},
    // 空tensor场景 1条
    {"AclnnMatmulAlltoAll-x1_empty_tensor", 2, {0, 128}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_SUCCESS},

    // 异常用例 23条，caseid按照[error-算子名-异常原因-id]构成
    // 1. x1 dtype不合法(ACL_INT8)
    {"error-AclnnMatmulAlltoAll-x1dtype_invalid-01", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_INT8, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 2. x2 dtype不合法 (ACL_UINT8)
    {"error-AclnnMatmulAlltoAll-x2dtype_invalid-02", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_UINT8, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 3. bias dtype不合法 不等于xdtype或float32(ACL_BF16)
    {"error-AclnnMatmulAlltoAll-biasdtype_invalid-03", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_BF16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 4. output dtype不合法 (ACL_FLOAT)
    {"error-AclnnMatmulAlltoAll-outdtype_mismatch_04", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 5. 空tensor (3条)
    // 5.1 x1有维度为0
 	{"error-AclnnMatmulAlltoAll-x1empty-05", 2, {256, 0}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
	    ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
 	    {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 5.2 x2有维度为0，first dim
    {"error-AclnnMatmulAlltoAll-x2empty-06", 2, {256, 128}, {0, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 5.3 x2有维度为0，second dim
    {"error-AclnnMatmulAlltoAll-x2empty-07", 2, {256, 128}, {128, 0}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 6. format为私有格式(4条)
    {"error-AclnnMatmulAlltoAll-private_fmt1-08", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnMatmulAlltoAll-private_fmt2-09", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnMatmulAlltoAll-private_fmt3-10", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnMatmulAlltoAll-private_fmt4-11", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_FRACTAL_Z,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 7. AlltoAllAxes不合法
    {"error-AclnnMatmulAlltoAll-invalid_axes-12", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {3, 2, 1}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 8. group不合法 (2条)
    // 8.1 group为空
    {"error-AclnnMatmulAlltoAll-group_empty-13", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "", false, false, ACLNN_ERR_PARAM_INVALID},
    // 8.2 group长度超过128(group自带'\0'，所以超过127就算异常)
    {"error-AclnnMatmulAlltoAll-group_extralong-14", 2, {256, 128}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567",
        false, false, ACLNN_ERR_PARAM_INVALID},
    // 9. transposeX1=true
    {"error-AclnnMatmulAlltoAll-transx1-15", 2, {128, 256}, {128, 256}, {256}, {512, 128},
        ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", true, false, ACLNN_ERR_PARAM_INVALID},
    // 10. shape不合法 (8条)
    // 10.1 x1维度不合法
    {"error-AclnnMatmulAlltoAll-invalid_x1dim-16", 2, {256, 128, 32}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.2 x2维度不合法
    {"error-AclnnMatmulAlltoAll-invalid_x2dim-17", 2, {256, 128}, {128, 256, 32}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.3 output维度不合法
    {"error-AclnnMatmulAlltoAll-invalid_outputdim-18", 2, {256, 128}, {128, 256}, {256}, {512, 128, 32},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.4 bias维度不合法
    {"error-AclnnMatmulAlltoAll-invalid_biasdim-19", 2, {256, 128, 32}, {128, 256}, {256, 32}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.5 x1和x2的k轴不匹配(x2不转置)
    {"error-AclnnMatmulAlltoAll-mismatch_kdim-20", 2, {256, 64}, {128, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.6 x1和x2的k轴不匹配(x2转置)
    {"error-AclnnMatmulAlltoAll-mismatch_kdim-21", 2, {256, 128}, {256, 64}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, true, ACLNN_ERR_PARAM_INVALID},
    // 10.7 k轴超出范围
    {"error-AclnnMatmulAlltoAll-outrange_kdim-22", 2, {256, 65536}, {65536, 256}, {256}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID},
    // 10.8 bias和x2不匹配
    {"error-AclnnMatmulAlltoAll-mismatch_kdim-23", 2, {256, 128}, {128, 256}, {128}, {512, 128},
        ACL_BF16, ACL_BF16, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        {-1, -2}, "ut_test_matmul_allto_all", false, false, ACLNN_ERR_PARAM_INVALID}
};

static void TestOneParamCase(const MatmulAlltoAllAclnnTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    // 从结构体list中获取实际用例属性
    vector<int64_t> x1Shape = param.x1Shape;
    vector<int64_t> x2Shape = param.x2Shape;
    vector<int64_t> biasShape = param.biasShape;
    vector<int64_t> outputShape = param.outputShape;
    aclDataType x1Dtype = param.x1Dtype;
    aclDataType x2Dtype = param.x2Dtype;
    aclDataType biasDtype = param.biasDtype;
    aclDataType outputDtype = param.outputDtype;
    aclFormat x1Format = param.x1Format;
    aclFormat x2Format = param.x2Format;
    aclFormat biasFormat = param.biasFormat;
    aclFormat outputFormat = param.outputFormat;
    vector<int64_t> axesAcl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axesAcl.data(), axesAcl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnnStatusUt;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet;
    if (biasShape.empty()) {
        auto ut = OP_API_UT(aclnnMatmulAlltoAll,
                        INPUT(x1, x2, nullptr, alltoAllAxesOptional, group, transposeX1, transposeX2),
                        OUTPUT(output));
        aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    } else {
        TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
        auto ut = OP_API_UT(aclnnMatmulAlltoAll,
                        INPUT(x1, x2, bias, alltoAllAxesOptional, group, transposeX1, transposeX2),
                        OUTPUT(output));
        aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    }
    if (retStatus == ACLNN_SUCCESS) {
        EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);
    } else {
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.caseName << std::endl;
}

TEST_F(TestAclnnMatmulAlltoAll, CasesParamsTest)
{
    if (std::size(g_casesParams) != 0) {
        uint64_t numCases = sizeof(g_casesParams) / sizeof(g_casesParams[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            TestOneParamCase(g_casesParams[idx]);
        }
    }
}
