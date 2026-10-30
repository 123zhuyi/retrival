/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../matmul_all_reduce_host_ut_param.h"
#include "mc2_tiling_case_executor.h"

namespace MatmulAllReduceUT {

class MatmulAllReduceArch35TilingTest : public testing::TestWithParam<MatmulAllReduceTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceArch35TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulAllReduceArch35TilingTest, param)
{
    auto param = GetParam();
    struct MatmulAllReduceCompileInfo {} compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulAllReduce",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x3,
            param.antiquant_scale,
            param.antiquant_offset,
            param.dequant_scale,
            param.pertoken_scale,
            param.comm_quant_scale_1,
            param.comm_quant_scale_2
        },
        {
            param.y
        },
        {
            {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
            {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
            {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_group_size)},
            {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
            {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
            {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_quant_mode)}
        },
        param.inputInstance, param.outputInstance,
        &compileInfo,
        param.soc, param.coreNum, param.ubsize
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", param.ranksize}
    };
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
        param.expectTilingDataHash, {}, 0, true);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    MatmulAllReduceArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceTilingUtParam>
);

} // namespace MatmulAllReduceUT
