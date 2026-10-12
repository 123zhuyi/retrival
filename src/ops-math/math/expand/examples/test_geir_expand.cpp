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
 * \file test_geir_expand.cpp
 * \brief Test Expand via GE IR graph mode
 */

#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/expand_proto.h"

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

uint32_t GetDataTypeSize(DataType data_type)
{
    if (data_type == ge::DT_BOOL || data_type == ge::DT_INT8 || data_type == ge::DT_UINT8) {
        return 1;
    }
    if (data_type == ge::DT_INT16 || data_type == ge::DT_UINT16 || data_type == ge::DT_FLOAT16 ||
        data_type == ge::DT_BF16) {
        return 2;
    }
    if (data_type == ge::DT_INT32 || data_type == ge::DT_UINT32 || data_type == ge::DT_FLOAT) {
        return 4;
    }
    if (data_type == ge::DT_INT64 || data_type == ge::DT_UINT64 || data_type == ge::DT_DOUBLE) {
        return 8;
    }
    return 0;
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

int32_t GenBf16Data(
    const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, const vector<uint16_t>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); ++i) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != size) {
        return FAILED;
    }

    input_tensor =
        Tensor(input_tensor_desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(uint16_t));
    return SUCCESS;
}

int32_t GenInt64Data(
    const vector<int64_t>& shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, const vector<int64_t>& values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); ++i) {
        size *= static_cast<size_t>(shapes[i]);
    }
    if (values.size() != size) {
        return FAILED;
    }

    input_tensor =
        Tensor(input_tensor_desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int64_t));
    return SUCCESS;
}

int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    auto expand = op::Expand("expand_graph");
    vector<int64_t> x_shape = {1, 2};
    vector<int64_t> shape_shape = {2};
    vector<int64_t> y_shape = {3, 2};

    auto x = op::Data("placeholder0").set_attr_index(0);
    TensorDesc x_desc(ge::Shape(x_shape), FORMAT_ND, DT_BF16);
    x_desc.SetPlacement(ge::kPlacementHost);
    Tensor x_tensor;
    int ret = GenBf16Data(x_shape, x_tensor, x_desc, {FloatToBf16Bits(1.0F), FloatToBf16Bits(2.0F)});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input x data failed\n", GetTime().c_str());
        return FAILED;
    }
    x.update_input_desc_x(x_desc);
    x.update_output_desc_y(x_desc);
    input.push_back(x_tensor);
    graph.AddOp(x);
    expand.set_input_x(x);
    inputs.push_back(x);

    auto shape = op::Data("placeholder1").set_attr_index(1);
    TensorDesc shape_desc(ge::Shape(shape_shape), FORMAT_ND, DT_INT64);
    shape_desc.SetPlacement(ge::kPlacementHost);
    Tensor shape_tensor;
    ret = GenInt64Data(shape_shape, shape_tensor, shape_desc, {3, 2});
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input shape data failed\n", GetTime().c_str());
        return FAILED;
    }
    shape.update_input_desc_x(shape_desc);
    shape.update_output_desc_y(shape_desc);
    input.push_back(shape_tensor);
    graph.AddOp(shape);
    expand.set_input_shape(shape);
    expand.update_input_desc_shape(shape_desc);
    inputs.push_back(shape);

    TensorDesc y_desc(ge::Shape(y_shape), FORMAT_ND, DT_BF16);
    expand.update_output_desc_y(y_desc);
    outputs.push_back(expand);
    return SUCCESS;
}

int SaveInputOutput(const vector<ge::Tensor>& input, const vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < input.size(); ++i) {
        string input_file = "./tc_ge_irrun_expand_npu_input_" + std::to_string(i) + ".bin";
        const uint8_t* input_data = input[i].GetData();
        int64_t shape_size = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size =
            static_cast<uint32_t>(shape_size) * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(input_file, data_size, input_data) != SUCCESS) {
            return FAILED;
        }
    }

    const uint16_t expected[] = {FloatToBf16Bits(1.0F), FloatToBf16Bits(2.0F), FloatToBf16Bits(1.0F),
                                 FloatToBf16Bits(2.0F), FloatToBf16Bits(1.0F), FloatToBf16Bits(2.0F)};
    for (size_t i = 0; i < output.size(); ++i) {
        string output_file = "./tc_ge_irrun_expand_npu_output_" + std::to_string(i) + ".bin";
        const uint8_t* output_data = output[i].GetData();
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size =
            static_cast<uint32_t>(shape_size) * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile(output_file, data_size, output_data) != SUCCESS) {
            return FAILED;
        }

        const uint16_t* result = reinterpret_cast<const uint16_t*>(output_data);
        for (int64_t j = 0; j < shape_size; ++j) {
            LOG_PRINT("result[%ld] bf16 bits: 0x%04x\n", j, static_cast<unsigned int>(result[j]));
            if (result[j] != expected[j]) {
                printf("Expand output mismatch at index %ld\n", j);
                return FAILED;
            }
        }
    }
    return SUCCESS;
}
} // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graph_name = "tc_ge_irrun_expand";
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

    map<AscendString, AscendString> graph_options = {{"ge.exec.precision_mode", "allow_mix_precision"}};
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

    printf("Input dtype: %d\n", DT_BF16);
    printf(
        "Hint: if your active OPP also enables AI Core Expand for BF16, temporarily mask the AI Core Expand entry to "
        "probe AICPU dispatch.\n");
    ret = SaveInputOutput(input, output);
    if (ret != SUCCESS) {
        ge::GEFinalize();
        return FAILED;
    }

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GE Finalize failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}