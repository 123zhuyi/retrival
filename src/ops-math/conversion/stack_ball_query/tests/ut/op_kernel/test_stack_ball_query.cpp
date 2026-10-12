/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <string>
#include <cstdint>
#include <sstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/stack_ball_query_tiling.h"

using namespace std;
extern "C" __global__ __aicore__ void stack_ball_query(
    GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR xyz_batch_cnt, GM_ADDR center_xyz_batch_cnt, GM_ADDR idx,
    GM_ADDR workspace, GM_ADDR tiling);

class stack_ball_query_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "stack_ball_query_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "stack_ball_query_test TearDown\n" << endl;
    }
};