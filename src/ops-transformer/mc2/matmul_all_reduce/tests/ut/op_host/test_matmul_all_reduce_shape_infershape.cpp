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
#include "matmul_all_reduce_host_ut_param.h"
#include "mc2_infer_shape_case_executor.h"

namespace MatmulAllReduceUT {

class InferShapeTest : public testing::TestWithParam<MatmulAllReduceInferShapeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduce InferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduce InferShapeTest TearDown" << std::endl;
    }
};

TEST_P(InferShapeTest, param)
{
    auto param = GetParam();
    std::vector<gert::InfershapeContextPara::TensorDescription> inputTensorDesc;
    if (param.inputInstance[0] == 1) inputTensorDesc.emplace_back(param.x1);
    if (param.inputInstance[1] == 1) inputTensorDesc.emplace_back(param.x2);
    if (param.inputInstance[2] == 1) inputTensorDesc.emplace_back(param.bias);
    if (param.inputInstance[3] == 1) inputTensorDesc.emplace_back(param.x3);
    if (param.inputInstance[4] == 1) inputTensorDesc.emplace_back(param.antiquant_scale);
    if (param.inputInstance[5] == 1) inputTensorDesc.emplace_back(param.antiquant_offset);
    if (param.inputInstance[6] == 1) inputTensorDesc.emplace_back(param.dequant_scale);
    if (param.inputInstance[7] == 1) inputTensorDesc.emplace_back(param.pertoken_scale);
    if (param.inputInstance[8] == 1) inputTensorDesc.emplace_back(param.comm_quant_scale_1);
    if (param.inputInstance[9] == 1) inputTensorDesc.emplace_back(param.comm_quant_scale_2);
    gert::InfershapeContextPara inferShapeContextPara(
        "MatmulAllReduce",
        inputTensorDesc,
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
        param.inputInstance, param.outputInstance
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", param.ranksize}
    };
    Mc2ExecuteTestCase(inferShapeContextPara, hcomTopologyMockValues, param.expectResult, param.expectOutputShape);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulAllReduce,
    InferShapeTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulAllReduceInferShapeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<MatmulAllReduceInferShapeUtParam>
);

} // namespace MatmulAllReduceUT
