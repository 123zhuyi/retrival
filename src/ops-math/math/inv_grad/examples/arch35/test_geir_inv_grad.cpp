/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"

#include "inv_grad_geir_test_common.h"

namespace {

struct StaticCase {
    const char* tag;
    ge::DataType dtype;
    std::vector<int64_t> shape;
};

bool RunStaticCase(ge::Session& session, uint32_t graphId, const StaticCase& testCase)
{
    const std::string graphName = std::string("inv_grad_static_") + testCase.tag;
    ge::Graph graph(graphName.c_str());
    if (!inv_grad_geir_test::BuildGraph(graph, testCase.shape, testCase.dtype, graphName)) {
        return false;
    }
    std::map<ge::AscendString, ge::AscendString> graphOptions;
    if (session.AddGraph(graphId, graph, graphOptions) != ge::SUCCESS) {
        return false;
    }

    ge::Tensor x;
    ge::Tensor grad;
    if (!inv_grad_geir_test::MakeFilledTensor(testCase.shape, testCase.dtype, 2.0, x) ||
        !inv_grad_geir_test::MakeFilledTensor(testCase.shape, testCase.dtype, 3.0, grad)) {
        return false;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(graphId, {x, grad}, outputs) != ge::SUCCESS) {
        return false;
    }
    return inv_grad_geir_test::ValidateOutput(outputs, testCase.shape, testCase.dtype, -12.0);
}

} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::printf("InvGrad static GEIR initialization failure\n");
        return -1;
    }

    const std::vector<StaticCase> cases = {
        {"float32", ge::DT_FLOAT, {2, 3}}, {"float16", ge::DT_FLOAT16, {2, 3}},     {"bfloat16", ge::DT_BF16, {2, 3}},
        {"int32", ge::DT_INT32, {2, 3}},   {"empty_float32", ge::DT_FLOAT, {0, 3}},
    };

    bool passed = true;
    {
        std::map<ge::AscendString, ge::AscendString> sessionOptions;
        ge::Session session(sessionOptions);
        for (size_t index = 0; index < cases.size(); ++index) {
            const StaticCase& testCase = cases[index];
            std::printf("Static case %s, dtype %s, shape %s\n", testCase.tag,
                        inv_grad_geir_test::DtypeName(testCase.dtype),
                        inv_grad_geir_test::ShapeToJson(testCase.shape).c_str());
            if (!RunStaticCase(session, static_cast<uint32_t>(index + 1U), testCase)) {
                std::printf("InvGrad static GEIR case failure: %s\n", testCase.tag);
                passed = false;
                break;
            }
            std::printf("Shape, dtype and values PASSED for %s\n",
                        inv_grad_geir_test::ShapeToJson(testCase.shape).c_str());
        }
    }

    const ge::Status finalizeStatus = ge::GEFinalize();
    if (!passed || finalizeStatus != ge::SUCCESS) {
        return -1;
    }
    std::printf("InvGrad static GEIR verification PASSED\n");
    return 0;
}
