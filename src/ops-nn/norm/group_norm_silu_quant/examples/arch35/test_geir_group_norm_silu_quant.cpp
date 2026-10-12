/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_geir_group_norm_silu_quant.cpp
 * \brief GroupNormSiluQuant GE IR (graph-mode) construction example.
 */

#include <ctime>
#include <cstdint>
#include <iostream>
#include <map>
#include <new>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "array_ops.h"
#include "tensor.h"
#include "types.h"
#include "../../op_graph/group_norm_silu_quant_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::string;
using std::vector;

namespace {
constexpr uint32_t INT8_BYTE_SIZE = 1;
constexpr uint32_t FP16_BYTE_SIZE = 2;
constexpr uint32_t FP32_BYTE_SIZE = 4;

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int64_t GetShapeSize(const vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16) {
        return FP16_BYTE_SIZE;
    }
    if (dt == ge::DT_INT8) {
        return INT8_BYTE_SIZE;
    }
    return FP32_BYTE_SIZE;
}

int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& tensorDesc, DataType dataType)
{
    tensorDesc.SetRealDimCnt(shape.size());
    size_t elementNum = static_cast<size_t>(GetShapeSize(shape));
    size_t dataLen = elementNum * GetDataTypeSize(dataType);
    uint8_t* data = new (std::nothrow) uint8_t[dataLen];
    if (data == nullptr) {
        printf("%s - ERROR - [XIR]: Alloc input data failed\n", GetTime().c_str());
        return FAILED;
    }
    for (size_t i = 0; i < dataLen; ++i) {
        data[i] = static_cast<uint8_t>(i % 23);
    }
    tensor = Tensor(tensorDesc, data, dataLen);
    delete[] data;
    return SUCCESS;
}

struct GroupNormSiluQuantCase {
    const char* name;
    DataType xDtype; // float16 / bfloat16
    int64_t numGroups;
    bool perChannel; // quantScale: false -> {1}, true -> {C}
    bool activateSilu;
};

// x = (N, C, HW); gamma/beta = (C); quantScale = (1) or (C);
// yOut = (N, C, HW) int8; meanOut/rstdOut = (N, num_groups) same dtype as x.
constexpr int64_t N_DIM = 2;
constexpr int64_t C_DIM = 32;
constexpr int64_t HW_DIM = 16;

int CreateGroupNormSiluQuantGraph(const GroupNormSiluQuantCase& tc, Graph& graph, vector<Tensor>& input,
                                  vector<Operator>& inputs, vector<Operator>& outputs)
{
    auto op = ge::op::GroupNormSiluQuant("group_norm_silu_quant");

    vector<int64_t> xShape = {N_DIM, C_DIM, HW_DIM};
    vector<int64_t> gammaShape = {C_DIM};
    vector<int64_t> quantScaleShape = tc.perChannel ? vector<int64_t>{C_DIM} : vector<int64_t>{1};

    // x
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, tc.xDtype);
    xDesc.SetPlacement(ge::kPlacementHost);
    Tensor xTensor;
    if (GenInputData(xShape, xTensor, xDesc, tc.xDtype) != SUCCESS) {
        return FAILED;
    }
    auto xData = op::Data("x").set_attr_index(0);
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);
    op.set_input_x(xData);
    input.push_back(xTensor);
    inputs.push_back(xData);

    // gamma
    TensorDesc gammaDesc(ge::Shape(gammaShape), FORMAT_ND, tc.xDtype);
    gammaDesc.SetPlacement(ge::kPlacementHost);
    Tensor gammaTensor;
    if (GenInputData(gammaShape, gammaTensor, gammaDesc, tc.xDtype) != SUCCESS) {
        return FAILED;
    }
    auto gammaData = op::Data("gamma").set_attr_index(1);
    gammaData.update_input_desc_x(gammaDesc);
    gammaData.update_output_desc_y(gammaDesc);
    graph.AddOp(gammaData);
    op.set_input_gamma(gammaData);
    input.push_back(gammaTensor);
    inputs.push_back(gammaData);

    // beta
    TensorDesc betaDesc(ge::Shape(gammaShape), FORMAT_ND, tc.xDtype);
    betaDesc.SetPlacement(ge::kPlacementHost);
    Tensor betaTensor;
    if (GenInputData(gammaShape, betaTensor, betaDesc, tc.xDtype) != SUCCESS) {
        return FAILED;
    }
    auto betaData = op::Data("beta").set_attr_index(2);
    betaData.update_input_desc_x(betaDesc);
    betaData.update_output_desc_y(betaDesc);
    graph.AddOp(betaData);
    op.set_input_beta(betaData);
    input.push_back(betaTensor);
    inputs.push_back(betaData);

    // quantScale (float32)
    TensorDesc qsDesc(ge::Shape(quantScaleShape), FORMAT_ND, ge::DT_FLOAT);
    qsDesc.SetPlacement(ge::kPlacementHost);
    Tensor qsTensor;
    if (GenInputData(quantScaleShape, qsTensor, qsDesc, ge::DT_FLOAT) != SUCCESS) {
        return FAILED;
    }
    auto qsData = op::Data("quantScale").set_attr_index(3);
    qsData.update_input_desc_x(qsDesc);
    qsData.update_output_desc_y(qsDesc);
    graph.AddOp(qsData);
    op.set_input_quantScale(qsData);
    input.push_back(qsTensor);
    inputs.push_back(qsData);

    // outputs
    vector<int64_t> meanShape = {N_DIM, tc.numGroups};
    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, ge::DT_INT8);
    TensorDesc meanDesc(ge::Shape(meanShape), FORMAT_ND, tc.xDtype);
    TensorDesc rstdDesc(ge::Shape(meanShape), FORMAT_ND, tc.xDtype);
    op.update_output_desc_yOut(yDesc);
    op.update_output_desc_meanOut(meanDesc);
    op.update_output_desc_rstdOut(rstdDesc);

    op.set_attr_num_groups(tc.numGroups);
    op.set_attr_eps(0.00001f);
    op.set_attr_activate_silu(tc.activateSilu);

    outputs.push_back(op);
    return SUCCESS;
}

int RunGroupNormSiluQuantCase(const GroupNormSiluQuantCase& tc)
{
    printf("%s - INFO - [XIR]: Run %s\n", GetTime().c_str(), tc.name);

    Graph graph(tc.name);
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    if (CreateGroupNormSiluQuantGraph(tc, graph, input, inputs, outputs) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }
    graph.SetInputs(inputs).SetOutputs(outputs);

    std::map<AscendString, AscendString> buildOptions = {};
    Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        return FAILED;
    }

    vector<Tensor> output;
    if (session->RunGraph(graphId, input, output) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run %s graph failed\n", GetTime().c_str(), tc.name);
        delete session;
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); ++i) {
        const TensorDesc& desc = output[i].GetTensorDesc();
        std::cout << tc.name << " output " << i << " dtype: " << desc.GetDataType()
                  << " shape size = " << desc.GetShape().GetShapeSize() << std::endl;
    }
    printf("%s - INFO - [XIR]: Run %s graph success\n", GetTime().c_str(), tc.name);
    delete session;
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge success\n", GetTime().c_str());

    vector<GroupNormSiluQuantCase> testCases = {
        {"fp16_per_tensor_silu", ge::DT_FLOAT16, 4, false, true},
        {"bf16_per_channel_silu", ge::DT_BF16, 4, true, true},
    };

    for (const auto& tc : testCases) {
        if (RunGroupNormSiluQuantCase(tc) != SUCCESS) {
            (void)ge::GEFinalize();
            return FAILED;
        }
    }

    if (ge::GEFinalize() != SUCCESS) {
        printf("%s - ERROR - [XIR]: Finalize ge failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ge success\n", GetTime().c_str());
    return SUCCESS;
}
