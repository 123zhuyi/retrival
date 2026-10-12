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
#include "../../../op_kernel/assign_sub_tiling_data.h"

namespace AssignSubUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "AssignSub";

struct AssignSubTestParam {
    std::string caseName;
    std::initializer_list<int64_t> varShape;
    ge::DataType varDtype;
    ge::Format varFormat;
    std::initializer_list<int64_t> valueShape;
    ge::DataType valueDtype;
    ge::Format valueFormat;
    std::initializer_list<int64_t> var_outShape;
    ge::DataType var_outDtype;
    ge::Format var_outFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static AssignSubTestParam testCases[] = {
    {"assign_sub_0",
     {3, 5, 7, 9},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {3, 5, 7, 9},
     ge::DT_INT8,
     ge::FORMAT_ND,
     {3, 5, 7, 9},
     ge::DT_INT8,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     1UL,
     "945 32 32 ",
     {0},
     64,
     262144,
     4096},
};

class AssignSubTilingTest : public testing::TestWithParam<AssignSubTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "AssignSubTilingTest SetUp." << std::endl; }
    static void TearDownTestCase() { std::cout << "AssignSubTilingTest TearDown." << std::endl; }
};

struct AssignSubCompileInfo {
} compileInfo;

static void TestOneParamCase(const AssignSubTestParam& param)
{
    gert::StorageShape varShape = {param.varShape, param.varShape};
    gert::StorageShape valueShape = {param.valueShape, param.valueShape};
    gert::StorageShape var_outShape = {param.var_outShape, param.var_outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{varShape, param.varDtype, param.varFormat}, {valueShape, param.valueDtype, param.valueFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{var_outShape, param.var_outDtype, param.var_outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.maxAIVNum, param.ubSize, param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

TEST_P(AssignSubTilingTest, tiling_test)
{
    const AssignSubTestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(AssignSubTilingTests, AssignSubTilingTest, testing::ValuesIn(testCases));

} // namespace AssignSubUT
