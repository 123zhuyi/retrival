/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <iostream>
#include <map>
#include <new>
#include <vector>
#include "ge_api.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "../op_graph/bias_proto.h"

using namespace ge;
using std::vector;

#define SUCCESS 0
#define FAILED -1

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

namespace {
uint32_t GetDataTypeSize(DataType dtype)
{
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) {
        return 4;
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return 2;
    }
    return 1;
}

int32_t GenOnesData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, DataType dtype,
                    std::vector<uint8_t>& storage)
{
    desc.SetRealDimCnt(shape.size());
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    const uint32_t dataSize = static_cast<uint32_t>(numel) * GetDataTypeSize(dtype);
    storage.assign(dataSize, 0);
    if (dtype == ge::DT_FLOAT) {
        auto* ptr = reinterpret_cast<float*>(storage.data());
        for (int64_t i = 0; i < numel; ++i) {
            ptr[i] = 1.0F;
        }
    }
    tensor = Tensor(desc, storage.data(), dataSize);
    return SUCCESS;
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {ge::AscendString("ge.exec.deviceId"), ge::AscendString("0")},
        {ge::AscendString("ge.graphRunMode"), ge::AscendString("1")},
        {ge::AscendString("ge.socVersion"), ge::AscendString("Ascend950DT_9592")}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        std::cout << "GEInitialize failed" << std::endl;
        return FAILED;
    }

    Graph graph("bias_geir_example");
    std::vector<ge::Tensor> inputs;
    std::vector<Operator> graphInputs;

    auto x = op::Data("x").set_attr_index(0);
    auto biasTensor = op::Data("bias").set_attr_index(1);
    auto bias = op::Bias("bias_op");

    DataType dtype = ge::DT_FLOAT;
    vector<int64_t> xShape = {2, 3, 4};
    vector<int64_t> biasShape = {3, 4};
    TensorDesc xDesc(ge::Shape(xShape), ge::FORMAT_ND, dtype);
    TensorDesc biasDesc(ge::Shape(biasShape), ge::FORMAT_ND, dtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    biasDesc.SetPlacement(ge::kPlacementHost);

    Tensor xData;
    Tensor biasData;
    std::vector<uint8_t> xStorage;
    std::vector<uint8_t> biasStorage;
    GenOnesData(xShape, xData, xDesc, dtype, xStorage);
    GenOnesData(biasShape, biasData, biasDesc, dtype, biasStorage);
    inputs.push_back(xData);
    inputs.push_back(biasData);

    x.update_input_desc_x(xDesc);
    x.update_output_desc_y(xDesc);
    graph.AddOp(x);
    bias.set_input_x(x);
    bias.update_input_desc_x(xDesc);

    biasTensor.update_input_desc_x(biasDesc);
    biasTensor.update_output_desc_y(biasDesc);
    graph.AddOp(biasTensor);
    bias.set_input_bias(biasTensor);
    bias.update_input_desc_bias(biasDesc);
    bias.set_attr_axis(1);
    bias.set_attr_num_axes(-1);
    bias.set_attr_bias_from_blob(true);
    TensorDesc yDesc(ge::Shape(xShape), ge::FORMAT_ND, dtype);
    bias.update_output_desc_y(yDesc);
    graphInputs.push_back(x);
    graphInputs.push_back(biasTensor);
    std::vector<Operator> graphOutputs = {bias};
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<ge::AscendString, ge::AscendString> buildOptions;
    ge::Session* session = new (std::nothrow) ge::Session(buildOptions);
    if (session == nullptr) {
        std::cout << "Create Session failed" << std::endl;
        ge::GEFinalize();
        return FAILED;
    }

    std::map<ge::AscendString, ge::AscendString> graphOptions = {
        {ge::AscendString("ge.exec.precision_mode"), ge::AscendString("allow_mix_precision")}};
    uint32_t graphId = 0;
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        std::cout << "AddGraph failed" << std::endl;
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    std::vector<ge::Tensor> outputs;
    if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
        std::cout << "RunGraph failed" << std::endl;
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    delete session;

    if (outputs.size() != 1U) {
        std::cout << "Unexpected output count: " << outputs.size() << std::endl;
        ge::GEFinalize();
        return FAILED;
    }
    const auto* result = reinterpret_cast<const float*>(outputs[0].GetData());
    const int64_t elementCount = outputs[0].GetTensorDesc().GetShape().GetShapeSize();
    for (int64_t i = 0; i < elementCount; ++i) {
        if (std::fabs(result[i] - 2.0F) > 1e-6F) {
            std::cout << "Bias output mismatch at index " << i << ", expect 2 but got " << result[i] << std::endl;
            ge::GEFinalize();
            return FAILED;
        }
    }
    std::cout << "Bias GEIR example run success, outputs=" << outputs.size() << std::endl;
    ge::GEFinalize();
    return SUCCESS;
}
