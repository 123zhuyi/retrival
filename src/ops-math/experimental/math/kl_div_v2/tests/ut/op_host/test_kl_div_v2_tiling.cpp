/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_kernel/kl_div_v2_tiling_data.h"

namespace KLDivV2UT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "KLDivV2";

struct KLDivV2TestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> targetShape;
    ge::DataType targetDtype;
    ge::Format targetFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static KLDivV2TestParam testCases[] = {
    {"kl_div_v2_0",
     {3, 5},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {3, 5},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     0UL,
     "68719476751 4294967297 64424517616 16 1 ",
     {16777220},
     64,
     262144,
     4096},
};

class KLDivV2TilingTest : public testing::TestWithParam<KLDivV2TestParam> {
protected:
    static void SetUpTestCase() { std::cout << "KLDivV2TilingTest SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "KLDivV2TilingTest TearDown." << std::endl; }
};

struct KLDivV2CompileInfo {
} compileInfo;

static void TestOneParamCase(const KLDivV2TestParam& param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape targetShape = {param.targetShape, param.targetShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}, {targetShape, param.targetDtype, param.targetFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(
        gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<std::string>("mean")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("log_target", Ops::Math::AnyValue::CreateFrom<bool>(false)));
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.maxAIVNum, param.ubSize, param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

TEST_P(KLDivV2TilingTest, tiling_test)
{
    const KLDivV2TestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(KLDivV2TilingTests, KLDivV2TilingTest, testing::ValuesIn(testCases));

} // namespace KLDivV2UT
