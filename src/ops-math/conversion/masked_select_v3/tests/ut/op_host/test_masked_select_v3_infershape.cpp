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
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "op_infer_shape_range_context_builder.h"
#include "op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"

class MaskedSelectv3InferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaskedSelectv3 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaskedSelectv3 Proto Test TearDown" << std::endl; }
};

TEST_F(MaskedSelectv3InferTest, infershape_1d_fp16)
{
    gert::InfershapeContextPara infershapeContextPara(
        "MaskedSelect", {{{{8}, {8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, {{{8}, {8}}, ge::DT_BOOL, ge::FORMAT_ND}},
        {{{{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MaskedSelectv3InferTest, infershape_4d_fp32)
{
    gert::InfershapeContextPara infershapeContextPara("MaskedSelect",
                                                      {{{{2, 4, 6, 8}, {2, 4, 6, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                       {{{2, 4, 6, 8}, {2, 4, 6, 8}}, ge::DT_BOOL, ge::FORMAT_ND}},
                                                      {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MaskedSelectv3InferTest, infershape_range_base)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MaskedSelect");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape_range, nullptr);

    gert::OpInferShapeRangeContextBuilder builder;
    builder.OpType("MaskedSelect").OpName("MaskedSelect");
    builder.IONum(2, 1);
    builder.OutputTensorDesc(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND);
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->infer_shape_range(context);
    EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(MaskedSelectv3InferTest, tiling_parse_prepare)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("MaskedSelectV3");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    struct MaskedSelectV3CompileInfo {
        uint64_t aivNum = 0;
        uint64_t ubSize = 0;
        uint64_t workSpaceSize = 0;
        bool isRegbase = false;
    } compileInfo = {};

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> socInfos = {{"ai_core_cnt", "48"}, {"core_type_list", "AICore"}};
    platformInfo.SetPlatformRes("SoCInfo", socInfos);
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", "262144"}};
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);

    gert::OpTilingParseContextBuilder builder;
    builder.OpType("MaskedSelectV3").OpName("MaskedSelectV3");
    builder.IONum(2, 1);
    builder.InputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.InputTensorDesc(1, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.OutputTensorDesc(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    builder.CompiledInfo(&compileInfo);
    builder.CompiledJson("{}");
    builder.PlatformInfo(reinterpret_cast<void*>(&platformInfo));
    auto contextHolder = builder.Build();
    auto* context = contextHolder.GetContext();
    ASSERT_NE(context, nullptr);

    auto ret = opImpl->tiling_parse(reinterpret_cast<gert::KernelContext*>(context));
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
    EXPECT_NE(compileInfo.aivNum, 0UL);
    EXPECT_NE(compileInfo.ubSize, 0UL);
    EXPECT_EQ(compileInfo.isRegbase, false);
}
