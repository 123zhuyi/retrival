/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
#include "../../../op_host/hans_decode_tiling.h"
#include "data_utils.h"

using namespace std;
// using namespace AscendC;

extern "C" __global__ __aicore__ void hans_decode(GM_ADDR mantissa, GM_ADDR fixed, GM_ADDR var, GM_ADDR pdf,
                                                  GM_ADDR recover, GM_ADDR workspace, GM_ADDR tiling);

class hans_decode_test : public testing::Test {
   protected:
    static void SetUpTestCase() {
        cout << "hans_decode_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "hans_decode_test TearDown\n" << endl;
    }
};