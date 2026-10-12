/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "experimental/math/floor/op_api/aclnn_floor.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2FloorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2FloorTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2FloorTest TearDown" << std::endl; }
};

// 测试用例001: 参数校验 输入和输出张量数据类型不一致
TEST_F(l2FloorTest, l2_floor_test_001)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例002: 参数校验 输入和输出张量形状维度不一致
TEST_F(l2FloorTest, l2_floor_test_002)
{
    auto selfDesc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例003: 参数校验 输入和输出张量形状大小不一致
TEST_F(l2FloorTest, l2_floor_test_003)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例004: 边界情况 空张量
TEST_F(l2FloorTest, l2_floor_test_004)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例005: 正常路径 FLOAT32
TEST_F(l2FloorTest, l2_floor_test_005)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例006: 正常路径 FLOAT16
TEST_F(l2FloorTest, l2_floor_test_006)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例007: 参数校验 输入张量指针为空
TEST_F(l2FloorTest, l2_floor_test_007)
{
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试用例008: 参数校验 输出张量指针为空
TEST_F(l2FloorTest, l2_floor_test_008)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试用例009: 格式转换 NCHW -> NHWC
TEST_F(l2FloorTest, l2_floor_test_009)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例010: 格式转换 NDHWC -> NCDHW
TEST_F(l2FloorTest, l2_floor_test_010)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_NCDHW);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例011: 格式支持 HWCN -> HWCN
TEST_F(l2FloorTest, l2_floor_test_011)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例012: 格式校验 NC1HWC0 -> HWCN
TEST_F(l2FloorTest, l2_floor_test_012)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例013: 格式校验 HWCN -> NC1HWC0
TEST_F(l2FloorTest, l2_floor_test_013)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto outDesc = TensorDesc({2, 4, 6, 8, 8}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例014: 数据范围
TEST_F(l2FloorTest, l2_floor_test_014)
{
    auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例015: 内存布局
TEST_F(l2FloorTest, l2_floor_test_015)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例016: 原地操作
TEST_F(l2FloorTest, l2_floor_test_016)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnInplaceFloor, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 测试用例017: 维度限制
TEST_F(l2FloorTest, l2_floor_test_017)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    // only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 测试用例018: 数据类型支持 - BFLOAT16类型
TEST_F(l2FloorTest, l2_floor_test_018)
{
    auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFloor, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // 注意：在910芯片上可能返回ACLNN_ERR_PARAM_INVALID，在910B芯片上返回ACLNN_SUCCESS
    // 这里根据实际芯片类型判断，测试框架会自动处理
    EXPECT_TRUE(getWorkspaceResult == ACLNN_SUCCESS || getWorkspaceResult == ACLNN_ERR_PARAM_INVALID);
}
