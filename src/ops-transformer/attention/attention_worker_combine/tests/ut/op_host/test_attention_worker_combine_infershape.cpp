/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class AttentionWorkerCombineInfershapeTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AttentionWorkerCombineInfershapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AttentionWorkerCombineInfershapeTest TearDown" << std::endl;
    }
};

TEST_F(AttentionWorkerCombineInfershapeTest, AttentionWorkerCombine_infershape_test01)
{
    gert::StorageShape schedule_context_shape = {{1024}, {1024}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape layer_id_shape = {{1}, {1}};
    gert::InfershapeContextPara infershapeContextPara(
        "AttentionWorkerCombine",
        {// input
            {schedule_context_shape, ge::DT_INT8, ge::FORMAT_ND},
            {expert_scales_shape, ge::DT_FLOAT, ge::FORMAT_ND},
            {layer_id_shape, ge::DT_INT32, ge::FORMAT_ND}
        },
        {// output
            {{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND}
        },
        {
            {"hidden_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(7168)},
            {"token_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"need_schedule", Ops::Transformer::AnyValue::CreateFrom<int64_t>(15)}
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 7168}, {1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}