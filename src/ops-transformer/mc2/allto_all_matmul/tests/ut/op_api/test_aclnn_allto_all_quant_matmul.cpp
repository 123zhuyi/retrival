/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <float.h>
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_api/aclnn_allto_all_quant_matmul.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class test_aclnn_allto_all_quant_matmul : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(NpuArch::DAV_3510);
        cout << "test_aclnn_allto_all_quant_matmul SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "test_aclnn_allto_all_quant_matmul TearDown" << endl;
    }
};

// ut用例结构体
struct AlltoAllQuantMatmulAclnnTestParam {
    // 用例名
    string case_name;
    // 通信域卡数，ut测试默认为2
    int world_size;
    // 量化模式
    int64_t x1_quantmode; // x1量化模式
    int64_t x2_quantmode; // x2量化模式
    // 数据形状
    vector<int64_t> x1_shape; // x1数据shape，正常为（BS，H）
    vector<int64_t> x2_shape; // x2数据shape，正常为（H * world_size，N）
    vector<int64_t> bias_shape; // bias数据shape，正常为（N）
    vector<int64_t> x1_scale_optional_shape; // x1ScaleOptional数据shape，正常为（BS/rankSize），mx量化为（BS/rankSize，ceil(H*rankSize/64)，2）
    vector<int64_t> x2_scale_shape; // x2scales数据shape，正常为（N），mx量化为（ceil(H*rankSize/64)，N，2）
    vector<int64_t> output_shape; // output数据shape，正常为（BS / world_size，N）
    vector<int64_t> alltoalloutput_shape; // alltoalloutput数据shape，正常为（BS / ranksize，H * ranksize）
    // 数据类型
    aclDataType x1_dtype; // x1数据dtype，仅支持bfloat16和float16
    aclDataType x2_dtype; // x2数据dtype，仅支持float8_e5m2和float8_e4m3fn
    aclDataType bias_dtype; // bias数据dtype，仅支持float32
    aclDataType x1_scale_optional_dtype; // x1ScaleOptional数据dtype，仅支持bfloat16和float16，要求和x1Dtype一致
    aclDataType x2_scale_dtype; // x2scales数据dtype，仅支持float32
    aclDataType output_dtype; // output数据dtype，支持bfloat16、float16和float32
    aclDataType alltoalloutput_dtype; // alltoalloutput数据dtype，仅支持bfloat16和float16，要求和x1Dtype一致
    // 数据格式
    aclFormat x1_format; // x1数据format，仅支持ND
    aclFormat x2_format; // x2数据format，仅支持ND
    aclFormat bias_format; // bias数据format，仅支持ND
    aclFormat x1_scale_optional_format; // x1ScaleOptional数据format，仅支持ND
    aclFormat x2_scale_format; // x2Scale数据format，仅支持ND
    aclFormat output_format; // output数据format，仅支持ND
    aclFormat alltoalloutput_format; // alltoalloutputoutput数据format，仅支持ND
    // 其它属性
    int64_t x1_quantdtype; // x1量化数据类型，仅支持配置35（表示ACL_FLOAT8_E5M2）或36（表示ACL_FLOAT8_E4M3FN）
    int64_t group_size; // groupSize，仅在perGroup（MX）量化时产生意义，表示分组量化时分组的大小，其余场景默认传0即可
    vector<int64_t> alltoAllAxesOptional; // alltoall数据交换的方向，只能为空或者[-2,-1]
    char* group; // 通信域标识，字符串，长度要求（0，128）
    bool transposeX1; // x1是否转置，现不支持为true
    bool transposeX2; // x2是否转置，为true时x2shape为（H2，H1）
    // ut用例期望返回状态
    aclnnStatus aclnn_status; //期望状态
};

// KC动态量化UT用例表
static AlltoAllQuantMatmulAclnnTestParam KCDynQuant_cases_params[] = {
    // 正常用例 192条，caseid按照[算子名-x1-x2-bias-x1scale-x2scale-output-alltoallout-format-transpose-x1quantdtype-id]构成
    // 等待补充
    {"AclnnAlltoAllQuantMatmul-bf16-e4m3-f32-bf16-f32-bf16-bf16-nd-notrans-35-01",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {0}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_SUCCESS},
    {"AclnnAlltoAllQuantMatmul-bf16-e4m3-f32-bf16-f32-bf16-bf16-nd-trans-35-02",
        2, 7, 2, {256, 64}, {256, 128}, {256}, {0}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS},

    // 异常用例 很多很多条，caseid按照[error-算子名-异常原因-id]构成
    // 等待补充
    {"error-AclnnAlltoAllQuantMatmul-x1dtype_invalid-01",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {0}, {256}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID},
    {"error-AclnnAlltoAllQuantMatmul-x2dtype_invalid-02",
        2, 7, 2, {256, 64}, {128, 256}, {256}, {0}, {256}, {128, 256}, {128, 128},
        ACL_BF16, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_FLOAT, ACL_BF16, ACL_BF16,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 0, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID},
};

// MX量化UT用例表
static AlltoAllQuantMatmulAclnnTestParam MXQuant_cases_params[] = {
    // 正常用例
    {"AAQMM_MX-bf16-success-001",
        2, 6, 6, {256, 64}, {256, 128}, {256}, {128, 2, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E5M2, ACL_FLOAT8_E5M2, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_BF16, ACL_FLOAT8_E5M2,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        35, 4295032864, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 输入fp8_e5m2，输出bf16
    {"AAQMM_MX-fp16-success-002",
        2, 6, 6, {256, 64}, {256, 128}, {256}, {128, 2, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT16, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 4295032864, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 输入fp8_e4m3fn，输出fp16
    {"AAQMM_MX-fp32-success-003",
        2, 6, 6, {256, 64}, {256, 128}, {256}, {128, 2, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 4295032864, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_SUCCESS}, // 输入fp8_e4m3fn，输出fp32

    // 异常用例 很多很多条，caseid按照[error-算子名-异常原因-id]构成
    // 等待补充
    {"AAQMM_MX-x1QuantDtype_missmatch-error-001",
        2, 6, 6, {256, 64}, {256, 128}, {256}, {128, 2, 2, 16}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 4295032864, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // x1scale维度不对
    {"AAQMM_MX-groupSize_invalid-error-002",
        2, 6, 6, {256, 64}, {256, 128}, {256}, {128, 2, 2}, {256, 2, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 31241, {-2, -1}, "ut_test_allto_all_quant_matmul", false, true, ACLNN_ERR_PARAM_INVALID}, // groupSize不是MX量化要的值
    {"AAQMM_MX-x2nottrans-error-003",
        2, 6, 6, {256, 64}, {128, 256}, {256}, {128, 2, 2}, {2, 256, 2}, {128, 256}, {128, 128},
        ACL_FLOAT8_E4M3FN, ACL_FLOAT8_E4M3FN, ACL_FLOAT, ACL_FLOAT8_E8M0, ACL_FLOAT8_E8M0, ACL_FLOAT, ACL_FLOAT8_E4M3FN,
        ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND, ACL_FORMAT_ND,
        36, 4295032864, {-2, -1}, "ut_test_allto_all_quant_matmul", false, false, ACLNN_ERR_PARAM_INVALID} // x2非转置
};

static void TestOneParamCase(const AlltoAllQuantMatmulAclnnTestParam& param)
{
    std::cout << "run case " << param.case_name << std::endl;
    // 从结构体list中获取实际用例属性
    int64_t x1quantmode = param.x1_quantmode;
    int64_t x2quantmode = param.x2_quantmode;
    vector<int64_t> x1Shape = param.x1_shape;
    vector<int64_t> x2Shape = param.x2_shape;
    vector<int64_t> biasShape = param.bias_shape;
    vector<int64_t> x1scalesShape = param.x1_scale_optional_shape;
    vector<int64_t> x2scalesShape = param.x2_scale_shape;
    vector<int64_t> outputShape = param.output_shape;
    vector<int64_t> alltoalloutShape = param.alltoalloutput_shape;
    aclDataType x1Dtype = param.x1_dtype;
    aclDataType x2Dtype = param.x2_dtype;
    aclDataType biasDtype = param.bias_dtype;
    aclDataType x1scalesDtype = param.x1_scale_optional_dtype;
    aclDataType x2scalesDtype = param.x2_scale_dtype;
    aclDataType outputDtype = param.output_dtype;
    aclDataType alltoalloutDtype = param.alltoalloutput_dtype;
    aclFormat x1Format = param.x1_format;
    aclFormat x2Format = param.x2_format;
    aclFormat biasFormat = param.bias_format;
    aclFormat x1_scale_format = param.x1_scale_optional_format;
    aclFormat x2_scale_format = param.x2_scale_format;
    aclFormat outputFormat = param.output_format;
    aclFormat alltoalloutFormat = param.alltoalloutput_format;
    int64_t x1quantdtype = param.x1_quantdtype;
    int64_t groupSize = param.group_size;
    vector<int64_t> axes_acl = param.alltoAllAxesOptional;
    aclIntArray *alltoAllAxesOptional = aclCreateIntArray(axes_acl.data(), axes_acl.size());
    const char* group = param.group;
    bool transposeX1 = param.transposeX1;
    bool transposeX2 = param.transposeX2;
    aclnnStatus retStatus = param.aclnn_status;
    TensorDesc x1 = TensorDesc(x1Shape, x1Dtype, x1Format);
    TensorDesc x2 = TensorDesc(x2Shape, x2Dtype, x2Format);
    TensorDesc bias = TensorDesc(biasShape, biasDtype, biasFormat);
    TensorDesc x1scales = TensorDesc(x1scalesShape, x1scalesDtype, x1_scale_format);
    TensorDesc x2scales = TensorDesc(x2scalesShape, x2scalesDtype, x2_scale_format);
    TensorDesc output = TensorDesc(outputShape, outputDtype, outputFormat);
    TensorDesc alltoallout = TensorDesc(alltoalloutShape, alltoalloutDtype, alltoalloutFormat);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    if (x1quantmode == 7) {
        auto ut = OP_API_UT(aclnnAlltoAllQuantMatmul,
            INPUT(x1, x2, bias, nullptr, x2scales, nullptr, nullptr, nullptr, group, alltoAllAxesOptional,
                x1quantmode, x2quantmode, 0, -1, x1quantdtype, groupSize, transposeX1, transposeX2),
            OUTPUT(output, alltoallout));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    } else {
        auto ut = OP_API_UT(aclnnAlltoAllQuantMatmul,
            INPUT(x1, x2, bias, x1scales, x2scales, nullptr, nullptr, nullptr, group, alltoAllAxesOptional,
                x1quantmode, x2quantmode, 0, -1, x1quantdtype, groupSize, transposeX1, transposeX2),
            OUTPUT(output, alltoallout));
        aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspace_size, executor);
        EXPECT_EQ(aclRet, retStatus);
    }
    std::cout << "end case " <<  param.case_name << std::endl;
}

// 测试KC动态量化场景下的UT用例
TEST_F(test_aclnn_allto_all_quant_matmul, KCDynQuant_cases_params)
{
    if (std::size(KCDynQuant_cases_params) != 0) {
        uint64_t numCases = sizeof(KCDynQuant_cases_params) / sizeof(KCDynQuant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            // 这是注释TestOneParamCase(KCDynQuant_cases_params[idx]);
        }
    }
}
// 测试MX量化场景下的UT用例
TEST_F(test_aclnn_allto_all_quant_matmul, MXQuant_cases_params)
{
    if (std::size(MXQuant_cases_params) != 0) {
        uint64_t numCases = sizeof(MXQuant_cases_params) / sizeof(MXQuant_cases_params[0]);
        for (size_t idx = 0; idx < numCases; idx += 1) {
            // 这是注释TestOneParamCase(MXQuant_cases_params[idx]);
        }
    }
}