/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "../../../op_api/aclnn_log_space.h"
#include "acl/acl.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

namespace {
// OP_API_UT 要求实参顺序为 (输入…, 输出…, 其余标量)，而 aclnnLogSpace 的 result 排在 steps/base
// 之后，故用薄封装调换顺序，不改变被测逻辑。
aclnnStatus aclnnLogSpaceForUtGetWorkspaceSize(const aclScalar* start, const aclScalar* end, const aclTensor* result,
                                               int64_t steps, double base, uint64_t* workspaceSize,
                                               aclOpExecutor** executor)
{
    return aclnnLogSpaceGetWorkspaceSize(start, end, steps, base, result, workspaceSize, executor);
}

aclnnStatus aclnnLogSpaceForUt(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnLogSpace(workspace, workspaceSize, executor, stream);
}
} // namespace

class l2_log_space_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2_log_space_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2_log_space_test TearDown" << endl; }

    void SetUp() override
    {
        // 默认 Atlas A2（DAV_2201）：浮点 + 整型输出全支持。需要 Ascend950 的用例在用例内改。
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    // resultDims 与 steps 解耦传入，便于构造 shape 与 steps 不一致的异常用例。
    void TestRun(const vector<int64_t>& resultDims, aclDataType outDtype, int64_t steps, aclnnStatus expectStatus,
                 double start = 0.0, double end = 2.0, double base = 10.0)
    {
        auto startScalar = ScalarDesc(start, ACL_FLOAT);
        auto endScalar = ScalarDesc(end, ACL_FLOAT);
        auto result = TensorDesc(resultDims, outDtype, ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnLogSpaceForUt, INPUT(startScalar, endScalar), OUTPUT(result), steps, base);
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), expectStatus);
    }
};

// ---------------- 浮点输出 dtype（Ascend950 原有能力，A2/A3 同样支持） ----------------
TEST_F(l2_log_space_test, case_01_float32) { TestRun({5}, ACL_FLOAT, 5, ACLNN_SUCCESS); }

TEST_F(l2_log_space_test, case_02_float16) { TestRun({5}, ACL_FLOAT16, 5, ACLNN_SUCCESS); }

TEST_F(l2_log_space_test, case_03_bfloat16) { TestRun({5}, ACL_BF16, 5, ACLNN_SUCCESS); }

// ---------------- 整型输出 dtype（本次在 Atlas A2/A3 上新增） ----------------
TEST_F(l2_log_space_test, case_04_int8)
{
    // base^x 取值域落在 int8 内：2^0..2^6 = 1..64
    TestRun({7}, ACL_INT8, 7, ACLNN_SUCCESS, 0.0, 6.0, 2.0);
}

TEST_F(l2_log_space_test, case_05_int16) { TestRun({5}, ACL_INT16, 5, ACLNN_SUCCESS, 0.0, 4.0, 3.0); }

TEST_F(l2_log_space_test, case_06_int32) { TestRun({6}, ACL_INT32, 6, ACLNN_SUCCESS, 0.0, 5.0, 2.0); }

TEST_F(l2_log_space_test, case_07_uint8) { TestRun({7}, ACL_UINT8, 7, ACLNN_SUCCESS, 0.0, 6.0, 2.0); }

// ---------------- steps 边界 ----------------
TEST_F(l2_log_space_test, case_08_steps_zero_empty_tensor)
{
    // steps==0：L2 短路返回成功且不下发 Kernel，workspaceSize 必须为 0
    auto startScalar = ScalarDesc(0.0, ACL_FLOAT);
    auto endScalar = ScalarDesc(2.0, ACL_FLOAT);
    auto result = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSpaceForUt, INPUT(startScalar, endScalar), OUTPUT(result), static_cast<int64_t>(0),
                        10.0);
    uint64_t workspaceSize = 1;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

TEST_F(l2_log_space_test, case_09_steps_one)
{
    // steps==1：out[0] = base^start，走 SINGLE 模式
    TestRun({1}, ACL_FLOAT, 1, ACLNN_SUCCESS);
}

TEST_F(l2_log_space_test, case_10_steps_one_int32) { TestRun({1}, ACL_INT32, 1, ACLNN_SUCCESS, 0.0, 5.0, 2.0); }

// ---------------- 异常参数 ----------------
TEST_F(l2_log_space_test, case_11_negative_steps)
{
    // steps<0 先于 shape 校验被拦截，故 result shape 用合法的 {5}
    TestRun({5}, ACL_FLOAT, -1, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_space_test, case_12_base_zero) { TestRun({5}, ACL_FLOAT, 5, ACLNN_ERR_PARAM_INVALID, 0.0, 2.0, 0.0); }

TEST_F(l2_log_space_test, case_13_base_negative)
{
    TestRun({5}, ACL_FLOAT, 5, ACLNN_ERR_PARAM_INVALID, 0.0, 2.0, -2.0);
}

TEST_F(l2_log_space_test, case_14_unsupported_result_dtype) { TestRun({5}, ACL_INT64, 5, ACLNN_ERR_PARAM_INVALID); }

TEST_F(l2_log_space_test, case_15_shape_not_match_steps)
{
    // result shape[0]=5 与 steps=6 不一致
    TestRun({5}, ACL_FLOAT, 6, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_log_space_test, case_16_result_not_1d) { TestRun({2, 3}, ACL_FLOAT, 6, ACLNN_ERR_PARAM_INVALID); }

TEST_F(l2_log_space_test, case_17_null_start)
{
    auto endScalar = ScalarDesc(2.0, ACL_FLOAT);
    auto result = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSpaceForUt, INPUT(nullptr, endScalar), OUTPUT(result), static_cast<int64_t>(5), 10.0);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_log_space_test, case_18_null_end)
{
    auto startScalar = ScalarDesc(0.0, ACL_FLOAT);
    auto result = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLogSpaceForUt, INPUT(startScalar, nullptr), OUTPUT(result), static_cast<int64_t>(5), 10.0);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_log_space_test, case_19_null_result)
{
    auto startScalar = ScalarDesc(0.0, ACL_FLOAT);
    auto endScalar = ScalarDesc(2.0, ACL_FLOAT);

    auto ut = OP_API_UT(aclnnLogSpaceForUt, INPUT(startScalar, endScalar), OUTPUT(nullptr), static_cast<int64_t>(5),
                        10.0);
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------- 芯片架构分流（Ascend950 原有能力 vs Atlas A2/A3 新增能力） ----------------
TEST_F(l2_log_space_test, case_20_ascend950_float_supported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    TestRun({5}, ACL_FLOAT, 5, ACLNN_SUCCESS);
}

TEST_F(l2_log_space_test, case_21_ascend950_int32_not_supported)
{
    // Ascend950（DAV_3510）仅支持 FLOAT/FLOAT16/BFLOAT16。整型在 L2 就按芯片被拦下，
    // 返回参数非法而非内部错误（不可放行到 L0 再坍缩成 ACLNN_ERR_INNER_NULLPTR）。
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    TestRun({6}, ACL_INT32, 6, ACLNN_ERR_PARAM_INVALID, 0.0, 5.0, 2.0);
}

TEST_F(l2_log_space_test, case_23_ascend950_int8_not_supported)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    TestRun({7}, ACL_INT8, 7, ACLNN_ERR_PARAM_INVALID, 0.0, 6.0, 2.0);
}

TEST_F(l2_log_space_test, case_22_ascend910_93_int8_supported)
{
    // Atlas A3（ascend910_93）与 A2 同属 DAV_2201，整型同样支持
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910_93);
    TestRun({7}, ACL_INT8, 7, ACLNN_SUCCESS, 0.0, 6.0, 2.0);
}
