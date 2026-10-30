/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdint.h>
#include <time.h>

#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../op_graph/get_dynamic_dims_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

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
    if (dt == ge::DT_INT64) {
        return sizeof(int64_t);
    }
    return sizeof(int32_t);
}

template <typename T>
int32_t GenTensorData(const vector<T>& data, Tensor& tensor, TensorDesc& tensor_desc)
{
    tensor_desc.SetRealDimCnt(1);
    size_t data_len = data.size() * sizeof(T);
    T* p_data = new (std::nothrow) T[data.size()];
    if (p_data == nullptr) {
        return FAILED;
    }
    std::memcpy(p_data, data.data(), data_len);
    tensor = Tensor(tensor_desc, reinterpret_cast<uint8_t*>(p_data), data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(const string& bin_file, uint64_t data_size, uint8_t* input_data)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr) {
        return FAILED;
    }
    fwrite(input_data, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    auto get_dynamic_dims = op::GetDynamicDims("get_dynamic_dims");
    get_dynamic_dims.set_attr_N(3);
    get_dynamic_dims.set_attr_shape_info({4, 3, 2, -1, 1, 3, 1, 2, 1, 5, 16, -1, -1, 3, 4});
    get_dynamic_dims.create_dynamic_input_input(3);

    vector<vector<int64_t>> shape_data = {{3, 2, 4, 1}, {1, 2, 1}, {16, 112, 112, 3, 4}};
    for (size_t i = 0; i < shape_data.size(); ++i) {
        string name = "placeholder" + std::to_string(i);
        auto placeholder = op::Data(name).set_attr_index(static_cast<int64_t>(i));
        TensorDesc desc = TensorDesc(ge::Shape({static_cast<int64_t>(shape_data[i].size())}), FORMAT_ND, DT_INT64);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);

        Tensor tensor;
        if (GenTensorData(shape_data[i], tensor, desc) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
            return FAILED;
        }
        placeholder.update_input_desc_x(desc);
        placeholder.update_output_desc_y(desc);
        input.push_back(tensor);
        graph.AddOp(placeholder);
        get_dynamic_dims.set_dynamic_input_input(static_cast<uint32_t>(i), placeholder);
        inputs.push_back(placeholder);
    }

    TensorDesc output_desc = TensorDesc(ge::Shape({3}), FORMAT_ND, DT_INT64);
    get_dynamic_dims.update_output_desc_dims(output_desc);
    if (graph.AddOp(get_dynamic_dims) != GRAPH_SUCCESS) {
        printf("%s - ERROR - [XIR]: Add GetDynamicDims op to graph failed\n", GetTime().c_str());
        return FAILED;
    }
    outputs.push_back(get_dynamic_dims);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};
    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    std::map<AscendString, AscendString> graph_options = {};
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        delete session;
        GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    for (size_t i = 0; i < output.size(); i++) {
        string output_file = "./tc_ge_irrun_test_0008_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile(output_file, data_size, output_data);
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        return FAILED;
    }
    return SUCCESS;
}
