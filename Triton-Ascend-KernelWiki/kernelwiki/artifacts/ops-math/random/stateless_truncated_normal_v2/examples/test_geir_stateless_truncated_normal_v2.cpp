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
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"
#include "../op_graph/stateless_truncated_normal_v2_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT) return 4;
    if (dt == ge::DT_FLOAT16) return 2;
    if (dt == ge::DT_BF16) return 2;
    if (dt == ge::DT_INT32) return 4;
    if (dt == ge::DT_UINT32) return 4;
    if (dt == ge::DT_INT64) return 8;
    if (dt == ge::DT_UINT64) return 8;
    return 1;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t *inputData)
{
    FILE *fp = fopen(bin_file.c_str(), "w");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor> &input, std::vector<Operator> &inputs,
    std::vector<Operator> &outputs, Graph &graph)
{
    // StatelessTruncatedNormalV2 op
    auto op1 = op::StatelessTruncatedNormalV2("stateless_truncated_normal_v2");

    // Input 0: shape (Const, int32, describes output shape)
    std::vector<int64_t> shapeShape = {2};
    auto shapeNode = op::Const("shape_const");
    TensorDesc shapeDesc(ge::Shape(shapeShape), FORMAT_ND, DT_INT32);
    shapeDesc.SetPlacement(ge::kPlacementHost);
    int32_t shapeData[] = {4, 8};  // output shape: [4, 8]
    Tensor shapeTensor(shapeDesc, reinterpret_cast<uint8_t*>(shapeData), sizeof(shapeData));
    shapeNode.SetAttr("value", shapeTensor);
    shapeNode.update_output_desc_y(shapeDesc);
    graph.AddOp(shapeNode);
    op1.set_input_shape(shapeNode);
    op1.update_input_desc_shape(shapeDesc);
    inputs.push_back(shapeNode);

    // Input 1: key (Data, uint64, shape=[1])
    std::vector<int64_t> keyShape = {1};
    auto keyNode = op::Data("key_data").set_attr_index(1);
    TensorDesc keyDesc(ge::Shape(keyShape), FORMAT_ND, DT_UINT64);
    keyDesc.SetPlacement(ge::kPlacementHost);
    uint64_t keyData[] = {12345};
    Tensor keyTensor(keyDesc, reinterpret_cast<uint8_t*>(keyData), sizeof(keyData));
    keyNode.update_input_desc_x(keyDesc);
    input.push_back(keyTensor);
    graph.AddOp(keyNode);
    op1.set_input_key(keyNode);
    inputs.push_back(keyNode);

    // Input 2: counter (Data, uint64, shape=[2])
    std::vector<int64_t> counterShape = {2};
    auto counterNode = op::Data("counter_data").set_attr_index(2);
    TensorDesc counterDesc(ge::Shape(counterShape), FORMAT_ND, DT_UINT64);
    counterDesc.SetPlacement(ge::kPlacementHost);
    uint64_t counterData[] = {0, 0};
    Tensor counterTensor(counterDesc, reinterpret_cast<uint8_t*>(counterData), sizeof(counterData));
    counterNode.update_input_desc_x(counterDesc);
    input.push_back(counterTensor);
    graph.AddOp(counterNode);
    op1.set_input_counter(counterNode);
    inputs.push_back(counterNode);

    // Input 3: alg (Data, int32, scalar)
    std::vector<int64_t> algShape = {1};
    auto algNode = op::Data("alg_data").set_attr_index(3);
    TensorDesc algDesc(ge::Shape(algShape), FORMAT_ND, DT_INT32);
    algDesc.SetPlacement(ge::kPlacementHost);
    int32_t algData[] = {1};  // 1 = Philox
    Tensor algTensor(algDesc, reinterpret_cast<uint8_t*>(algData), sizeof(algData));
    algNode.update_input_desc_x(algDesc);
    input.push_back(algTensor);
    graph.AddOp(algNode);
    op1.set_input_alg(algNode);
    inputs.push_back(algNode);

    // Attr: dtype
    op1.set_attr_dtype(0);  // 0 = float32

    // Output: y
    std::vector<int64_t> outShape = {4, 8};
    TensorDesc yDesc(ge::Shape(outShape), FORMAT_ND, DT_FLOAT);
    op1.update_output_desc_y(yDesc);

    outputs.push_back(op1);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    const char *graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    printf("%s - INFO - [XIR]: Session add graph success\n", GetTime().c_str());

    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype: " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./stateless_truncated_normal_v2_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "output " << i << " shape size = " << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data_i);
        float *resultData = (float*)output_data_i;
        for (int64_t j = 0; j < output_shape && j < 32; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
        }
    }

    printf("%s - INFO - [XIR]: Start to finalize\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize success\n", GetTime().c_str());
    return SUCCESS;
}
