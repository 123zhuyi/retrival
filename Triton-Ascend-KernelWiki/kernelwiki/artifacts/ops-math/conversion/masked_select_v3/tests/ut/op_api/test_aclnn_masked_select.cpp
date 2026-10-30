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

#include "conversion/masked_select_v3/op_api/aclnn_masked_select.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_masked_select_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "masked_select_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "masked_select_test TearDown" << endl;
    }
};

int64_t sizes(const vector<int64_t>& arr)
{
    int64_t result = 1;
    for (const auto& elem : arr) {
        if (elem) {
            result *= elem;
        }
    }
    return result;
}

vector<bool> genRandomBoolVector(const vector<int64_t>& arr)
{
    int64_t size = sizes(arr);
    vector<bool> result(size);
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        result[i] = rand() % 2 == 0;
    }
    return result;
}

vector<uint8_t> genRandomUint8Vector(const vector<int64_t>& arr)
{
    int64_t size = sizes(arr);
    vector<uint8_t> result(size);
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        result[i] = rand() % 2;
    }
    return result;
}

template <typename T>
int getTrueNumInVector(const vector<T>& arr)
{
    int count = 0;
    for (const auto& elem : arr) {
        if (elem) {
            ++count;
        }
    }
    return count;
}

// 基础用例
TEST_F(l2_masked_select_test, aclnnMaskedSelect_base_case_1)
{
    // left input
    const vector<int64_t>& selfShape = {8};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {8};
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {8};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat)
                              .Value(vector<bool>{true, true, true, true, false, false, false, false});
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(4);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_base_case_2)
{
    // left input
    const vector<int64_t>& selfShape = {8};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {8};
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {8};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat)
                              .Value(vector<bool>{false, false, false, false, false, false, false, false});
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(0);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 简单随机用例
TEST_F(l2_masked_select_test, aclnnMaskedSelect_8_int32_nd_and_8_bool_nd_simple_random_case_1)
{
    // left input
    const vector<int64_t>& selfShape = {8};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {8};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_8_8_float_nd_and_8_8_bool_nd_simple_random_case_2)
{
    // left input
    const vector<int64_t>& selfShape = {8, 8};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {8, 8};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(
    l2_masked_select_test, ascend950_aclnnMaskedSelect_8_8_float_nd_and_6_8_8_bool_nd_simple_random_case_broadcast)
{
    // left input
    const vector<int64_t>& selfShape = {6, 8, 8, 8, 8};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& maskShapeBroadcast = {8, 8, 8, 8};
    vector<bool> boolMask = genRandomBoolVector(maskShapeBroadcast);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShapeBroadcast, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/* 各元素基本类型覆盖用例
 * 维度：1-8维
 * self基本数据类型：FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、DOUBLE、COMPLEX64、COMPLEX128，BFLOAT16,BOOL
 * mask基本数据类型：UINT8、BOOL
 * 数据格式：ND、NCHW、NHWC、HWCN、NDHWC、NCDHW
 */

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_12_14_16_int64_nd_and_2_4_6_8_10_12_14_16_bool_nd)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12, 14, 16};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12, 14, 16};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_12_14_float16_nchw_and_2_4_6_8_10_12_14_bool_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12, 14};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12, 14};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_2_2_2_2_2_2_float16_nchw_and_2_2_2_2_2_2_2_bool_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_2_2_2_2_2_2_2_float16_nchw_and_2_2_2_2_2_2_2_2_bool_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2, 2};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2, 2};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, ascend910B2_aclnnMaskedSelect_2_2_2_2_2_2_2_bfloat16_nchw_and_2_2_2_2_2_2_2_bool_nchw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 2, 2, 2, 2, 2, 2};
    aclDataType selfDtype = ACL_BF16;
    aclFormat selfFormat = ACL_FORMAT_NCHW;
    // right input
    const vector<int64_t>& maskShape = {2, 2, 2, 2, 2, 2, 2};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_BF16;
    aclFormat outFormat = ACL_FORMAT_NCHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).ValueRange(-1, 1);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_12_uint8_nhwc_and_2_4_6_8_10_12_bool_nhwc)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12};
    aclDataType selfDtype = ACL_UINT8;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NHWC;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_UINT8;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_12_uint8_nhwc_and_2_4_6_8_10_12_uint8_nhwc)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8, 10, 12};
    aclDataType selfDtype = ACL_UINT8;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8, 10, 12};
    vector<uint8_t> uint8Mask = genRandomUint8Vector(maskShape);
    aclDataType maskDtype = ACL_UINT8;
    aclFormat maskFormat = ACL_FORMAT_NHWC;
    // output
    int64_t shapeSize = getTrueNumInVector(uint8Mask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_UINT8;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(uint8Mask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_10_int8_hwcn_and_2_4_6_8_10_bool_hwcn)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8, 10};
    aclDataType selfDtype = ACL_INT8;
    aclFormat selfFormat = ACL_FORMAT_HWCN;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8, 10};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_HWCN;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT8;
    aclFormat outFormat = ACL_FORMAT_HWCN;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_8_int16_ndhwc_and_2_4_6_8_bool_ndhwc)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6, 8};
    aclDataType selfDtype = ACL_INT16;
    aclFormat selfFormat = ACL_FORMAT_NDHWC;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6, 8};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NDHWC;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT16;
    aclFormat outFormat = ACL_FORMAT_NDHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_6_int32_ncdhw_and_2_4_6_bool_ncdhw)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4, 6};
    aclDataType selfDtype = ACL_INT32;
    aclFormat selfFormat = ACL_FORMAT_NCDHW;
    // right input
    const vector<int64_t>& maskShape = {2, 4, 6};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NCDHW;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT32;
    aclFormat outFormat = ACL_FORMAT_NCDHW;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_int64_nd_and_2_4_uint8_nd)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<uint8_t> uint8Mask = genRandomUint8Vector(maskShape);
    aclDataType maskDtype = ACL_UINT8;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(uint8Mask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(uint8Mask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_double_nd_and_2_uint8_nd)
{
    // left input
    const vector<int64_t>& selfShape = {2};
    aclDataType selfDtype = ACL_DOUBLE;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2};
    vector<uint8_t> uint8Mask = genRandomUint8Vector(maskShape);
    aclDataType maskDtype = ACL_UINT8;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(uint8Mask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_DOUBLE;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(uint8Mask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_bool_nd_and_2_4_bool_nd)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_BOOL;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_BOOL;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/* 各元素特殊类型覆盖用例
 * 空tensor
 * 溢出值
 * 边界值
 * 特殊值：nan,inf
 * 非连续
 */
TEST_F(l2_masked_select_test, aclnnMaskedSelect_0_2_float_nd_and_0_2_bool_nd_empty_tensor)
{
    // left input
    const vector<int64_t>& selfShape = {0, 2};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {0, 2};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    const vector<int64_t>& outShape = {0};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_1_2_float16_nd_and_2_2_bool_nd_boundary_value)
{
    // left input
    const vector<int64_t>& selfShape = {1, 2};
    aclDataType selfDtype = ACL_FLOAT16;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 2};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(maskShape)};
    aclDataType outDtype = ACL_FLOAT16;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<float>{65504.0, -65504.0});
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.001, 0.001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_5_4_float_nd_and_5_4_bool_nd_not_contiguous)
{
    // left input
    const vector<int64_t>& selfShape = {5, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    const vector<int64_t>& selfViewDim = {1, 5};
    int64_t selfOffset = 0;
    const vector<int64_t>& selfStorageDim = {4, 5};
    // right input
    const vector<int64_t>& maskShape = {5, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    const vector<int64_t>& maskViewDim = {1, 5};
    int64_t maskOffset = 0;
    const vector<int64_t>& maskStorageDim = {4, 5};
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc =
        TensorDesc(selfShape, selfDtype, selfFormat, selfViewDim, selfOffset, selfStorageDim).ValueRange(-2, 2);
    auto maskTensorDesc =
        TensorDesc(maskShape, maskDtype, maskFormat, maskViewDim, maskOffset, maskStorageDim).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).Precision(0.0001, 0.0001).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

/* 报错类型覆盖用例
 * 空指针
 * self 或者mask的维度超过8维
 * self和mask的形状不能广播
 * out的形状不是一维
 * out的shapeSize不满足条件
 * self数据类型不支持uint64
 * self和out的数据类型不同,当前支持
 * out的数据类型不满足条件
 */
TEST_F(l2_masked_select_test, aclnnMaskedSelect_input_output_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_l = OP_API_UT(aclnnMaskedSelect, INPUT((aclTensor*)nullptr, tensor_desc), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnMaskedSelect, INPUT(tensor_desc, (aclTensor*)nullptr), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_r.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnMaskedSelect, INPUT(tensor_desc, tensor_desc), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_o.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_self_shape_dim_num_out_of_8)
{
    // left input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {1, 2, 3, 4, 5};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_mask_shape_dim_num_out_of_8)
{
    // left input
    const vector<int64_t>& selfShape = {1, 2, 3, 4, 5};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_float32_nd_and_3_4_bool_nd_shape_not_broadcast)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {3, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_float32_nd_and_2_4_bool_nd_out_shape_with_2_dim)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {2, 4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_2_4_float32_nd_and_2_4_bool_nd_out_shape_size_is_small)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {4};
    aclDataType outDtype = ACL_FLOAT;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_error_input_dtype_with_uint64)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_UINT64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_UINT64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    SetPlatformNpuArch(NpuArch::DAV_2201);
    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_error_self_out_diff_dtype)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_FLOAT;
    aclFormat selfFormat = ACL_FORMAT_NHWC;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_NHWC;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_INT64;
    aclFormat outFormat = ACL_FORMAT_NHWC;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_masked_select_test, aclnnMaskedSelect_error_input_dtype_with_uint64_out)
{
    // left input
    const vector<int64_t>& selfShape = {2, 4};
    aclDataType selfDtype = ACL_INT64;
    aclFormat selfFormat = ACL_FORMAT_ND;
    // right input
    const vector<int64_t>& maskShape = {2, 4};
    vector<bool> boolMask = genRandomBoolVector(maskShape);
    aclDataType maskDtype = ACL_BOOL;
    aclFormat maskFormat = ACL_FORMAT_ND;
    // output
    int64_t shapeSize = getTrueNumInVector(boolMask);
    const vector<int64_t>& outShape = {sizes(selfShape)};
    aclDataType outDtype = ACL_UINT64;
    aclFormat outFormat = ACL_FORMAT_ND;

    auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
    auto maskTensorDesc = TensorDesc(maskShape, maskDtype, maskFormat).Value(boolMask);
    auto outTensorDesc = TensorDesc(outShape, outDtype, outFormat).ValidCount(shapeSize);

    auto ut = OP_API_UT(aclnnMaskedSelect, INPUT(selfTensorDesc, maskTensorDesc), OUTPUT(outTensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}