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
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace AllGatherMatmulUt {

class AllGatherMatmulInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(AllGatherMatmulInferShapeTest, Basic)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulInferShapeTest, EmptyTensorTest)
{
    gert::StorageShape x1Shape = {{8192, 0}, {}};
    gert::StorageShape x2Shape = {{0, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues);
}

TEST_F(AllGatherMatmulInferShapeTest, IsGatherOutFalse)
{
    gert::StorageShape x1Shape = {{8192, 12288}, {}};
    gert::StorageShape x2Shape = {{12288, 3904}, {}};

    gert::InfershapeContextPara infershapeContextPara("AllGatherMatmul",
        {
            {x1Shape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {x2Shape, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND}
        },
        {
            {"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
            {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(false)}
        }
    );
    Mc2Hcom::MockValues hcomTopologyMockValues {
        {"rankNum", 8}
    };

    std::vector<std::vector<int64_t>> expectOutputShape = {{65536, 3904}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(AllGatherMatmulInferShapeTest, InferDatatype)
{
    ge::DataType x1Type = ge::DT_FLOAT16;
    ge::DataType x2Type = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_UNDEFINED;
    ge::DataType gatherOutputType = ge::DT_UNDEFINED;

    auto contextHolder = gert::InferDataTypeContextFaker()
        .NodeIoNum(3, 2)
        .NodeAttrs({{"groupstr", Ops::Transformer::AnyValue::CreateFrom<std::string>("hcclCom")},
                    {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                    {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                    {"gather_index", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                    {"is_gather_out", Ops::Transformer::AnyValue::CreateFrom<int64_t>(true)}})
        .NodeInputTd(0, x1Type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, x2Type, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, biasType, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&x1Type, &x2Type, &biasType})
        .OutputDataTypes({&outputType, &gatherOutputType})
        .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDataTypeFunc = spaceRegistry->GetOpImpl("AllGatherMatmul")->infer_datatype;
    ASSERT_EQ(inferDataTypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), ge::DT_FLOAT16);
    EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), ge::DT_FLOAT16);
}

} // AllGatherMatmulUT