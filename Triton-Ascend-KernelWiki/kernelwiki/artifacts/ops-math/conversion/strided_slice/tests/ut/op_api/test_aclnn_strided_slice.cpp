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
#include "conversion/strided_slice/op_api/aclnn_strided_slice.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;
class l2_strided_slice_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "strided_slice_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "strided_slice_test TearDown" << endl;
    }
};

// 异常场景：芯片类型不支持
TEST_F(l2_strided_slice_test, case_error_soc)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_2201);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：空指针
TEST_F(l2_strided_slice_test, case_NULLPTR)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut1 = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            nullptr, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask),
        OUTPUT(output_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, nullptr, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask),
        OUTPUT(output_desc));
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, nullptr, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(
        aclnnStridedSlice,
        INPUT(input_desc, begin_desc, end_desc, nullptr, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask),
        OUTPUT(output_desc));
    aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut5 = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(nullptr));
    aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景：outshape不符合要求
TEST_F(l2_strided_slice_test, case_error_output_shape)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 1, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：输入输出类型不一致
TEST_F(l2_strided_slice_test, case_error_dtype)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：输入不在API支持的数据类型范围之内
TEST_F(l2_strided_slice_test, case_error_input_dtype)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：输出不在API支持的数据类型范围之内
TEST_F(l2_strided_slice_test, case_error_output_dtype)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_DOUBLE, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：输出维度大于8
TEST_F(l2_strided_slice_test, case_input_dim_gt_8)
{
    auto input_desc = TensorDesc({6, 6, 6, 6, 6, 6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：strides存在0
TEST_F(l2_strided_slice_test, case_strides_zero)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 0, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：begin/end/strides长度不相等
TEST_F(l2_strided_slice_test, case_para_size)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：ellipsisMask 不止有一个bit位为1
TEST_F(l2_strided_slice_test, case_ellipsisMask_gt_one)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 3;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景：shrinkAxisMask 中bit位为1的索引，对应的strides需要小于0
TEST_F(l2_strided_slice_test, case_shrinkAxisMask_strides_lt_zero)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, -1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 4;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_strided_slice_test, ascend950_case_INT8)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_UINT8)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_UINT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_INT16)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT16, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT16, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_UINT16)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_UINT16, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_INT32)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT32, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT32, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_UINT32)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_UINT32, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT32, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_INT64)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_INT64, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_INT64, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_UINT64)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_UINT64, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_UINT64, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_BF16)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_BF16, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_BF16, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_FLOAT)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_FLOAT16)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_BOOL)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_BOOL, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_BOOL, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_COMPLEX32)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_COMPLEX32, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_COMPLEX32, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_COMPLEX64)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_COMPLEX64, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_HIFLOAT8)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_HIFLOAT8, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_FLOAT8_E5M2)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_strided_slice_test, ascend950_case_FLOAT8_E4M3FN)
{
    auto input_desc = TensorDesc({6, 6, 6, 6}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    auto output_desc = TensorDesc({2, 2, 2, 3}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
    vector<int64_t> begin = {4, 1, 2, 0};
    vector<int64_t> end = {6, 3, 4, 3};
    vector<int64_t> strides = {1, 1, 1, 1};
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;
    auto begin_desc = IntArrayDesc(begin);
    auto end_desc = IntArrayDesc(end);
    auto strides_desc = IntArrayDesc(strides);

    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto ut = OP_API_UT(
        aclnnStridedSlice,
        INPUT(
            input_desc, begin_desc, end_desc, strides_desc, beginMask, endMask, ellipsisMask, newAxisMask,
            shrinkAxisMask),
        OUTPUT(output_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}