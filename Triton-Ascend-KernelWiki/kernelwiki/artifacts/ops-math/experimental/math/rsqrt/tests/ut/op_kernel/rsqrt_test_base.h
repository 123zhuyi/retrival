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
 * \file rsqrt_test_base.h
 * \brief Shared test base class and helper macros for rsqrt kernel UT.
 */

#ifndef RSQRT_TEST_BASE_H
#define RSQRT_TEST_BASE_H

#include <string>
#include <cstdlib>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

// 共享数据路径
const std::string RSQRT_ROOT_PATH = "../../../../";
const std::string RSQRT_DATA_PATH = RSQRT_ROOT_PATH + "experimental/math/rsqrt/tests/ut/op_kernel/rsqrt_data";

#define RSQRT_SETUP_TEARDOWN(TestClassName)                          \
protected:                                                           \
    static void SetUpTestCase()                                      \
    {                                                                \
        std::cout << #TestClassName " SetUp\n" << std::endl;         \
        const std::string cmd = "cp -rf " + RSQRT_DATA_PATH + " ./"; \
        int ret = system(cmd.c_str());                               \
        EXPECT_EQ(ret, 0);                                           \
        ret = system("chmod -R 755 ./rsqrt_data/");                  \
        EXPECT_EQ(ret, 0);                                           \
    }                                                                \
    static void TearDownTestCase() { std::cout << #TestClassName " TearDown\n" << std::endl; }

// 预分配 workspace 大小
#define RSQRT_WORKSPACE_SIZE (1024 * 1024 * 16)

#endif // RSQRT_TEST_BASE_H
