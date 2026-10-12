/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_dropout_v3_grad_l0.cpp
 * \brief l0op::DropoutV3Grad L0 UT —— scale 固定 DT_FLOAT tensor，验证算子构图返回非空。
 */

#include <gtest/gtest.h>
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "random/drop_out_v3_grad/op_api/dropout_v3_grad.h"

using namespace op;
using namespace std;

const int64_t DATA_SIZE = 256;

class DropOutV3GradL0Test : public ::testing::Test {
public:
    DropOutV3GradL0Test() : exe(nullptr) {}

    aclTensor* CreateAclTensor(std::vector<int64_t> shape, aclDataType dtype)
    {
        return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND, shape.data(), shape.size(),
                               data);
    }

    void SetUp() override
    {
        auto executor = &exe;
        auto unique_executor = CREATE_EXECUTOR();
        unique_executor.ReleaseTo(executor);
    }

    void TearDown() override { delete exe; }

public:
    aclOpExecutor* exe;
    float data[DATA_SIZE] = {1.0f};
};

// case 1: float32 grad_y + uint8 mask + float32 scalar scale
TEST_F(DropOutV3GradL0Test, dropout_v3_grad_l0_float32)
{
    auto gradY = CreateAclTensor({256}, ACL_FLOAT);
    auto mask = CreateAclTensor({32}, ACL_UINT8);
    auto scale = CreateAclTensor({1}, ACL_FLOAT); // scale 恒 float32
    auto gradX = l0op::DropoutV3Grad(gradY, mask, scale, exe);
    ASSERT_NE(gradX, nullptr);
}

// case 2: float16 grad_y
TEST_F(DropOutV3GradL0Test, dropout_v3_grad_l0_float16)
{
    auto gradY = CreateAclTensor({256}, ACL_FLOAT16);
    auto mask = CreateAclTensor({32}, ACL_UINT8);
    auto scale = CreateAclTensor({1}, ACL_FLOAT);
    auto gradX = l0op::DropoutV3Grad(gradY, mask, scale, exe);
    ASSERT_NE(gradX, nullptr);
}

// case 3: bfloat16 grad_y，950 平台
TEST_F(DropOutV3GradL0Test, dropout_v3_grad_l0_bf16_950)
{
    SetPlatformNpuArch(NpuArch::DAV_3510);
    auto gradY = CreateAclTensor({256}, ACL_BF16);
    auto mask = CreateAclTensor({32}, ACL_UINT8);
    auto scale = CreateAclTensor({1}, ACL_FLOAT);
    auto gradX = l0op::DropoutV3Grad(gradY, mask, scale, exe);
    ASSERT_NE(gradX, nullptr);
}
