/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/mul_v2_tiling_data.h" //改名字

using namespace std;
using namespace ge;

class MulV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "mul_v2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "mul_v2Tiling TearDown" << std::endl; }
};
struct MulV2CompileInfo {};
// TEST_F(MulV2Tiling, mul_v2_tiling_001)
// {
//     int32_t start = 0;
//     int32_t stop = 8;
//     int32_t num = 10;
//     MulV2CompileInfo compileInfo = {};
//     gert::TilingContextPara tilingContextPara(
//         "MulV2",
//         {
//             {{{16,16,16,16},{16,16,16,16}}, ge::DT_FLOAT, ge::FORMAT_ND},  //输入x1，形状、类型
//             {{{16,16,16,16},{16,16,16,16}}, ge::DT_FLOAT, ge::FORMAT_ND},  //输入x2
//         },
//         {
//             {{{16,16,16,16},{16,16,16,16}}, ge::DT_FLOAT, ge::FORMAT_ND},  //输出y
//         },
//         &compileInfo);
//     uint64_t expectTilingKey = 0;//0：float  1:float16
//     string expectTilingData = "64 72 1 1 10920 64 72 0 "; //根据形状来计算对应tilingdata的数据
//     std::vector<size_t> expectWorkspaces = {16 * 1024 * 1024};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }
