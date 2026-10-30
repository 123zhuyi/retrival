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
#include <cstdlib>
#include "acl/acl.h"
#include <gmock/gmock.h>
#include "gtest/gtest.h"
#include <vector>

#include "../../../op_host/op_api/aclnn_weight_quant_preprocess.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "ut_stub.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

struct WeightQuantPreprocessTestParam {
    string caseName;
    vector<int64_t> weightViewShape;
    vector<int64_t> weightStorageShape;
    vector<int64_t> weightStrides;
    vector<int64_t> scaleViewShape;
    vector<int64_t> scaleStorageShape;
    vector<int64_t> scaleStrides;
    aclDataType weightDtype;
    aclDataType scaleDtype;
    aclFormat weightFormat;
    aclFormat scaleFormat;
    aclDataType xDtype;
    aclDataType xScaleDtype;
    int64_t kGroupSize;
    bool isGMM;
    aclnnStatus expectRet;
    bool weightIsNull;
    bool weightScaleIsNull;
    bool outWeightIsNull;
    bool outWeightScaleIsNull;
    bool hasWeightOffset;
    bool hasBias;
    vector<int64_t> biasViewShape;
    vector<int64_t> biasStorageShape;
    aclDataType biasDtype;
    aclFormat biasFormat;
    aclFormat outWeightFormat;
    aclDataType outWeightDtype;
    aclDataType outWeightScaleDtype;
};

class l2_weight_quant_preprocess_test : public testing::TestWithParam<WeightQuantPreprocessTestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_weight_quant_preprocess_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_weight_quant_preprocess_test TearDown" << endl; }
};

class l2_weight_quant_preprocess_gmm_test : public testing::TestWithParam<WeightQuantPreprocessTestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_weight_quant_preprocess_gmm_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_weight_quant_preprocess_gmm_test TearDown" << endl; }
};

class l2_weight_quant_preprocess_mm_test : public testing::TestWithParam<WeightQuantPreprocessTestParam> {
protected:
    static void SetUpTestCase() { cout << "l2_weight_quant_preprocess_mm_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "l2_weight_quant_preprocess_mm_test TearDown" << endl; }
};

static void TestOneParamCase(const WeightQuantPreprocessTestParam& param)
{
    std::cout << "run case start: " << param.caseName << std::endl;
    void* weightDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* outWeightDeviceAddr = nullptr;
    void* outScaleDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;

    int64_t weightStorageSize = 1;
    for (auto d : param.weightStorageShape)
        weightStorageSize *= d;
    int64_t weightBytes = weightStorageSize / 2;

    int64_t scaleStorageSize = 1;
    for (auto d : param.scaleStorageShape)
        scaleStorageSize *= d;
    int64_t scaleBytes = scaleStorageSize;

    aclTensor* weight = nullptr;
    if (!param.weightIsNull) {
        weight = aclCreateTensor(param.weightViewShape.data(), param.weightViewShape.size(), param.weightDtype,
                                 param.weightStrides.data(), 0, param.weightFormat, param.weightStorageShape.data(),
                                 param.weightStorageShape.size(), weightDeviceAddr);
    }

    aclTensor* weightScale = nullptr;
    if (!param.weightScaleIsNull) {
        weightScale = aclCreateTensor(param.scaleViewShape.data(), param.scaleViewShape.size(), param.scaleDtype,
                                      param.scaleStrides.data(), 0, param.scaleFormat, param.scaleStorageShape.data(),
                                      param.scaleStorageShape.size(), scaleDeviceAddr);
    }

    int64_t C0 = 32;
    auto outWeightViewShape = param.weightViewShape;
    vector<int64_t> outWeightStorageShape;
    if (!param.weightIsNull && !param.weightViewShape.empty()) {
        if (param.isGMM) {
            int64_t G = param.weightViewShape[0];
            int64_t K = param.weightViewShape[1];
            int64_t N = param.weightViewShape[2];
            outWeightStorageShape = {G, CEIL_DIV(K, C0), CEIL_DIV(N, 16), 16, C0};
        } else {
            int64_t K = param.weightViewShape[0];
            int64_t N = param.weightViewShape[1];
            outWeightStorageShape = {CEIL_DIV(K, C0), CEIL_DIV(N, 16), 16, C0};
        }
    }
    int64_t outWeightStorageSize = 1;
    for (auto d : outWeightStorageShape)
        outWeightStorageSize *= d;
    int64_t outWeightBytes = outWeightStorageSize / 2;

    aclTensor* outWeight = nullptr;
    if (!param.outWeightIsNull && !outWeightViewShape.empty()) {
        aclFormat outFormat = param.outWeightFormat != ACL_FORMAT_UNDEFINED ? param.outWeightFormat :
                                                                              ACL_FORMAT_FRACTAL_NZ_C0_32;
        aclDataType outDtype = param.outWeightDtype != ACL_DT_UNDEFINED ? param.outWeightDtype : param.weightDtype;
        outWeight = aclCreateTensor(outWeightViewShape.data(), outWeightViewShape.size(), outDtype, nullptr, 0,
                                    outFormat, outWeightStorageShape.data(), outWeightStorageShape.size(),
                                    outWeightDeviceAddr);
    }

    auto outScaleViewShape = param.scaleViewShape;
    auto outScaleStorageShape = param.scaleStorageShape;
    int64_t outScaleStorageSize = scaleStorageSize;

    aclTensor* outWeightScale = nullptr;
    if (!param.outWeightScaleIsNull && !outScaleViewShape.empty()) {
        aclDataType outScaleDtype = param.outWeightScaleDtype != ACL_DT_UNDEFINED ? param.outWeightScaleDtype :
                                                                                    param.scaleDtype;
        outWeightScale = aclCreateTensor(outScaleViewShape.data(), outScaleViewShape.size(), outScaleDtype,
                                         param.scaleStrides.data(), 0, param.scaleFormat, outScaleStorageShape.data(),
                                         outScaleStorageShape.size(), outScaleDeviceAddr);
    }

    aclTensor* biasOptional = nullptr;
    aclTensor* outBiasOptional = nullptr;
    if (param.hasBias && !param.biasViewShape.empty()) {
        biasOptional = aclCreateTensor(param.biasViewShape.data(), param.biasViewShape.size(), param.biasDtype, nullptr,
                                       0, param.biasFormat, param.biasStorageShape.data(),
                                       param.biasStorageShape.size(), biasDeviceAddr);
        outBiasOptional = aclCreateTensor(param.biasViewShape.data(), param.biasViewShape.size(), param.biasDtype,
                                          nullptr, 0, param.biasFormat, param.biasStorageShape.data(),
                                          param.biasStorageShape.size(), biasDeviceAddr);
    }

    aclTensor* weightOffsetOptional = nullptr;
    aclTensor* outWeightOffsetOptional = nullptr;
    if (param.hasWeightOffset) {
        weightOffsetOptional = aclCreateTensor(
            param.scaleViewShape.data(), param.scaleViewShape.size(), param.scaleDtype, nullptr, 0, param.scaleFormat,
            param.scaleStorageShape.data(), param.scaleStorageShape.size(), scaleDeviceAddr);
        outWeightOffsetOptional = aclCreateTensor(
            param.scaleViewShape.data(), param.scaleViewShape.size(), param.scaleDtype, nullptr, 0, param.scaleFormat,
            param.scaleStorageShape.data(), param.scaleStorageShape.size(), scaleDeviceAddr);
    }

    uint64_t workspaceSize = 0U;
    aclOpExecutor* exe = nullptr;
    void* workspaceAddr = nullptr;
    aclrtStream stream = nullptr;

    auto ret = aclnnWeightQuantPreprocessGetWorkspaceSize(
        weight, weightScale, weightOffsetOptional, biasOptional, param.xDtype, param.xScaleDtype, param.kGroupSize,
        outWeight, outWeightScale, outWeightOffsetOptional, outBiasOptional, &workspaceSize, &exe);
    std::cout << "aclnnWeightQuantPreprocessGetWorkspaceSize returned: " << ret << std::endl;

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnWeightQuantPreprocessGetWorkspaceSize failed. ERROR: %d\n", ret);
        EXPECT_EQ(ret, param.expectRet);
        if (weight)
            aclDestroyTensor(weight);
        if (weightScale)
            aclDestroyTensor(weightScale);
        if (outWeight)
            aclDestroyTensor(outWeight);
        if (outWeightScale)
            aclDestroyTensor(outWeightScale);
        if (biasOptional)
            aclDestroyTensor(biasOptional);
        if (outBiasOptional)
            aclDestroyTensor(outBiasOptional);
        if (weightOffsetOptional)
            aclDestroyTensor(weightOffsetOptional);
        if (outWeightOffsetOptional)
            aclDestroyTensor(outWeightOffsetOptional);
        return;
    }

    ret = aclnnWeightQuantPreprocess(workspaceAddr, workspaceSize, exe, stream);
    std::cout << "aclnnWeightQuantPreprocess returned: " << ret << std::endl;

    if (ret != ACL_SUCCESS) {
        LOG_PRINT("aclnnWeightQuantPreprocess failed. ERROR: %d\n", ret);
        EXPECT_EQ(ret, param.expectRet);
    }

    std::cout << "run case end: " << param.caseName << std::endl;
    if (param.expectRet == ACLNN_SUCCESS) {
        EXPECT_NE(exe, nullptr);
    }

    if (weight)
        aclDestroyTensor(weight);
    if (weightScale)
        aclDestroyTensor(weightScale);
    if (outWeight)
        aclDestroyTensor(outWeight);
    if (outWeightScale)
        aclDestroyTensor(outWeightScale);
    if (biasOptional)
        aclDestroyTensor(biasOptional);
    if (outBiasOptional)
        aclDestroyTensor(outBiasOptional);
    if (weightOffsetOptional)
        aclDestroyTensor(weightOffsetOptional);
    if (outWeightOffsetOptional)
        aclDestroyTensor(outWeightOffsetOptional);
}

static void TestOneParamCaseInSubprocess(const WeightQuantPreprocessTestParam& param)
{
    ASSERT_EXIT(
        {
            op::SetPlatformNpuArch(NpuArch::DAV_3510);
            TestOneParamCase(param);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST_P(l2_weight_quant_preprocess_gmm_test, gmmNormalTest)
{
    WeightQuantPreprocessTestParam param = GetParam();
    TestOneParamCaseInSubprocess(param);
}

TEST_P(l2_weight_quant_preprocess_mm_test, mmNormalTest)
{
    WeightQuantPreprocessTestParam param = GetParam();
    TestOneParamCaseInSubprocess(param);
}

TEST_P(l2_weight_quant_preprocess_test, generalTest)
{
    op::SetPlatformNpuArch(NpuArch::DAV_3510);
    WeightQuantPreprocessTestParam param = GetParam();
    TestOneParamCase(param);
}

static WeightQuantPreprocessTestParam gmmNormalCases[] = {
    {"ascend950_test_GMM_MX_A8W4_normal",
     {2, 64, 128},
     {2, 128, 64},
     {8192, 1, 64},
     {2, 1, 128, 2},
     {2, 128, 1, 2},
     {256, 2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     true,
     ACLNN_SUCCESS,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},
};

static WeightQuantPreprocessTestParam mmNormalCases[] = {
    {"ascend950_test_MM_MX_A8W4_normal",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_SUCCESS,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_MM_MX_A8W4_bias_1d",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_SUCCESS,
     false,
     false,
     false,
     false,
     false,
     true,
     {128},
     {128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_MM_MX_A8W4_bias_2d",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_SUCCESS,
     false,
     false,
     false,
     false,
     false,
     true,
     {1, 128},
     {1, 128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},
};

static WeightQuantPreprocessTestParam casesParams[] = {

    {"ascend950_test_GMM_kGroupSize_not_32",
     {2, 64, 128},
     {2, 128, 64},
     {8192, 1, 64},
     {2, 1, 128, 2},
     {2, 128, 1, 2},
     {256, 2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     64,
     true,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_nullptr",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_NULLPTR,
     true,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_nullptr",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_NULLPTR,
     false,
     true,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_outWeight_nullptr",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_NULLPTR,
     false,
     false,
     true,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_outWeightScale_nullptr",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_NULLPTR,
     false,
     false,
     false,
     true,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_empty",
     {0, 128},
     {128, 0},
     {1, 0},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_empty",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 0, 2},
     {0, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_format_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_NCHW,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_format_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_NHWC,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_outWeight_format_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_dtype_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_INT8,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_dtype_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_xDtype_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT16,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_outWeight_dtype_mismatch",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_INT8,
     ACL_DT_UNDEFINED},

    {"ascend950_test_outWeightScale_dtype_mismatch",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_INT8},

    {"ascend950_test_weight_dim_1d",
     {64},
     {64},
     {1},
     {1, 2},
     {2, 1},
     {1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_dim_4d",
     {2, 2, 64, 128},
     {2, 2, 128, 64},
     {16384, 8192, 128, 1},
     {2, 2, 1, 128, 2},
     {2, 2, 128, 1, 2},
     {256, 256, 2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weight_not_transposed",
     {64, 128},
     {64, 128},
     {128, 1},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_kGroupSize_not_32",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     64,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_kGroupSize_zero",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     0,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_shape_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {2, 128, 2},
     {128, 2, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightScale_not_transposed",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {1, 128, 2},
     {256, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_weightOffset_not_nullptr",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     true,
     false,
     {},
     {},
     ACL_DT_UNDEFINED,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_bias_dtype_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {1, 128},
     {1, 128},
     ACL_INT8,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_bias_shape_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {2, 128},
     {2, 128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_bias_1d_shape_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {127},
     {127},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_bias_format_invalid",
     {64, 128},
     {128, 64},
     {1, 64},
     {1, 128, 2},
     {128, 1, 2},
     {2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     false,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {1, 128},
     {1, 128},
     ACL_FLOAT16,
     ACL_FORMAT_NCHW,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_GMM_bias_shape_invalid",
     {2, 64, 128},
     {2, 128, 64},
     {8192, 1, 64},
     {2, 1, 128, 2},
     {2, 128, 1, 2},
     {256, 2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     true,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {1, 128},
     {1, 128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

    {"ascend950_test_GMM_bias_1d_shape_invalid",
     {2, 64, 128},
     {2, 128, 64},
     {8192, 1, 64},
     {2, 1, 128, 2},
     {2, 128, 1, 2},
     {256, 2, 2, 1},
     ACL_FLOAT4_E2M1,
     ACL_FLOAT8_E8M0,
     ACL_FORMAT_ND,
     ACL_FORMAT_ND,
     ACL_FLOAT8_E4M3FN,
     ACL_FLOAT8_E8M0,
     32,
     true,
     ACLNN_ERR_PARAM_INVALID,
     false,
     false,
     false,
     false,
     false,
     true,
     {128},
     {128},
     ACL_FLOAT16,
     ACL_FORMAT_ND,
     ACL_FORMAT_UNDEFINED,
     ACL_DT_UNDEFINED,
     ACL_DT_UNDEFINED},

};

INSTANTIATE_TEST_SUITE_P(WeightQuantPreprocessGMM, l2_weight_quant_preprocess_gmm_test,
                         testing::ValuesIn(gmmNormalCases));
INSTANTIATE_TEST_SUITE_P(WeightQuantPreprocessMM, l2_weight_quant_preprocess_mm_test, testing::ValuesIn(mmNormalCases));
INSTANTIATE_TEST_SUITE_P(WeightQuantPreprocess, l2_weight_quant_preprocess_test, testing::ValuesIn(casesParams));
