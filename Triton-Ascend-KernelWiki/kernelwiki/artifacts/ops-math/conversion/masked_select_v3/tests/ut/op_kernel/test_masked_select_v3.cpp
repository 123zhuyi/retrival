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
#include <iostream>
#include <string>
#include <cstdint>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "masked_select_v3_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void masked_select_v3(
    GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR shapeOut, GM_ADDR workspace, GM_ADDR tiling);

class MaskedSelectV3Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "MaskedSelectV3Test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "MaskedSelectV3Test TearDown\n" << endl;
    }
};