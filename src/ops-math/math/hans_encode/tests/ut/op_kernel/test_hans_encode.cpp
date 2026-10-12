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
#include "../../../op_host/hans_encode_tiling.h"
#include "data_utils.h"

#include <cstdint>
using namespace std;
// using namespace AscendC;

extern "C" __global__ __aicore__ void hans_encode(GM_ADDR input_gm, GM_ADDR pdf_gm, GM_ADDR pdf_ref,
                                                  GM_ADDR output_mantissa_gm, GM_ADDR fixed_gm, GM_ADDR var_gm,
                                                  GM_ADDR workspace, GM_ADDR tiling);

class hans_encode_test : public testing::Test {
   protected:
    static void SetUpTestCase() {
        cout << "hans_encode_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "hans_encode_test TearDown\n" << endl;
    }
};