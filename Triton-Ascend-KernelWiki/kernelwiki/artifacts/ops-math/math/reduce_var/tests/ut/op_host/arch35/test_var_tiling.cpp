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
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_host/arch35/reduce_var_tiling.h"

using namespace std;
using namespace ge;

class ReduceVarTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceVarTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceVarTiling TearDown" << std::endl;
    }
};

// TEST_F(ReduceVarTiling, ReduceVar_test_tiling_001)
// {
//     optiling::ReduceVarCompileInfo compileInfo = {64, 253952};
//     gert::TilingContextPara tilingContextPara(
//         "ReduceVar",
//         {
//             {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {
//             {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
//             {{{13, 5, 9, 13, 7, 13}, {13, 5, 9, 13, 7, 13}}, ge::DT_FLOAT, ge::FORMAT_ND},
//         },
//         {gert::TilingContextPara::OpAttr("dim", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2, 3, 4, 5})),
//          gert::TilingContextPara::OpAttr("correction", Ops::Math::AnyValue::CreateFrom<int64_t>(0)),
//          gert::TilingContextPara::OpAttr("keepdim", Ops::Math::AnyValue::CreateFrom<bool>(true)),
//          gert::TilingContextPara::OpAttr("is_mean_out", Ops::Math::AnyValue::CreateFrom<bool>(true))},
//         &compileInfo);
//     uint64_t expectTilingKey = 6175;
//     string expectTilingData =
//         "1 3 2 9 169 832 19 5 45568 512 64 921857298 1 13 5 10647 0 0 0 0 0 692055 53235 10647 1 0 0 0 0 0 5 5 1 1 0 0 "
//         "0 0 0 0 0 1 16384 3959346946488926208 921857298 7488 7319 7319 7488 7319 7319 7488 7319 7319 7488 7319 7319 "
//         "7319 7488 7319 7319 7488 7319 5655 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
//         "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
//         "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
//     std::vector<size_t> expectWorkspaces = {16809984};
//     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
// }