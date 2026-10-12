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
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <optional>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/truncate_mod_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

namespace ge {
REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

namespace {
string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    struct tm tm_info;
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime_r(&timep, &tm_info));
    return tmp;
}

uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, const uint8_t* input_data)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("Failed to open file: %s\n", bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("Failed to write file: %s\n", bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

template <typename T>
ge::Tensor CreateTensor(vector<int64_t> shape, ge::DataType d_type, const vector<T>& values)
{
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, d_type);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        size *= static_cast<size_t>(shape[i]);
    }
    if (values.size() != size) {
        printf("%s - ERROR - [XIR]: Generate input tensor failed\n", GetTime().c_str());
        return ge::Tensor{};
    }

    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(T));
}

ge::Operator CreateInputNode(size_t i, TensorDesc desc)
{
    auto node = op::Data("placeholder" + std::to_string(i)).set_attr_index(i);
    node.update_input_desc_x(desc);
    node.update_output_desc_y(desc);
    return node;
}

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    ge::DataType d_type = ge::DT_FLOAT;
    auto node_truncate_mod = op::TruncateMod("truncate_mod");
    vector<int64_t> x_shape = {1, 2};
    vector<int64_t> y_shape = {1, 2};

    auto tensor_x1 = CreateTensor<float>(x_shape, d_type, {2.0F, -2.0F});
    if (tensor_x1.GetSize() == 0) {
        return FAILED;
    }
    auto tensor_x2 = CreateTensor<float>(x_shape, d_type, {1.4F, 1.4F});
    if (tensor_x2.GetSize() == 0) {
        return FAILED;
    }
    auto node_x1 = CreateInputNode(0, tensor_x1.GetTensorDesc());
    auto node_x2 = CreateInputNode(1, tensor_x1.GetTensorDesc());

    input.push_back(tensor_x1);
    input.push_back(tensor_x2);
    graph.AddOp(node_x1);
    graph.AddOp(node_x2);
    node_truncate_mod.set_input_x1(node_x1);
    node_truncate_mod.set_input_x2(node_x2);
    inputs.push_back(node_x1);
    inputs.push_back(node_x2);

    TensorDesc y_desc(ge::Shape(y_shape), FORMAT_ND, d_type);
    node_truncate_mod.update_output_desc_y(y_desc);
    outputs.push_back(node_truncate_mod);
    return SUCCESS;
}

int SaveInput(const vector<ge::Tensor>& input)
{
    for (size_t i = 0; i < input.size(); ++i) {
        string input_file = "./tc_ge_irrun_example_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t* input_data = input[i].GetData();
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(shape_size) *
                             ge::GetSizeByDataType(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data) != SUCCESS) {
            return FAILED;
        }
    }
    return SUCCESS;
}

int SaveOutput(const vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < output.size(); ++i) {
        string output_file = "./tc_ge_irrun_example_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* output_data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = static_cast<uint32_t>(shape_size) *
                             ge::GetSizeByDataType(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data) != SUCCESS) {
            return FAILED;
        }

        const float* result = reinterpret_cast<const float*>(output_data);
        for (int64_t j = 0; j < shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, result[j]);
        }
    }
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graph_name = "tc_ge_irrun_example";
    Graph graph(graph_name);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    vector<Operator> inputs;
    vector<Operator> outputs;
    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> build_options;
    std::unique_ptr<ge::Session> session(new (std::nothrow) Session(build_options));
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph success\n", GetTime().c_str());

    ret = SaveInput(input);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Save graph input failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ret = SaveOutput(output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Save graph output failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Run graph over\n", GetTime().c_str());
    return SUCCESS;
}
