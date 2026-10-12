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
#include "nn_other.h"
#include "../op_graph/ragged_bin_count_proto.h"

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
    if (dt == ge::DT_FLOAT || dt == ge::DT_INT32) {
        return 4U;
    }
    if (dt == ge::DT_INT64) {
        return 8U;
    }
    return 1U;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("%s - ERROR - [XIR]: Failed to open file %s\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    if (written != data_size) {
        printf("%s - ERROR - [XIR]: Failed to write data to file %s\n", GetTime().c_str(), bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

/**
 * Build one host-placement Data node carrying the supplied payload.
 *
 * RaggedBinCount declares InputsDataDependency({size}), so the bin count must be
 * readable on host at infer time; splits is kept on host as well so the graph can
 * be dumped and replayed without a device round trip.
 */
template <typename T>
static int32_t AddHostInput(const string& name, int32_t index, const vector<int64_t>& shape, DataType dtype,
                            const vector<T>& payload, op::RaggedBinCount& node, Graph& graph,
                            vector<ge::Tensor>& inputTensors, vector<Operator>& inputs, Tensor& tensorOut)
{
    TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(ge::FORMAT_ND);
    desc.SetRealDimCnt(shape.size());

    auto placeholder = op::Data(name.c_str()).set_attr_index(index);
    placeholder.update_input_desc_x(desc);
    placeholder.update_output_desc_y(desc);

    const uint64_t dataLen = static_cast<uint64_t>(payload.size()) * sizeof(T);
    tensorOut = Tensor(desc, reinterpret_cast<uint8_t*>(const_cast<T*>(payload.data())), dataLen);

    graph.AddOp(placeholder);
    inputTensors.push_back(tensorOut);
    inputs.push_back(placeholder);
    return SUCCESS;
}

/**
 * Two ragged rows over six values, four bins, empty weights (pure counting).
 *
 *   splits  = [0, 3, 6]        -> row0 = values[0:3], row1 = values[3:6]
 *   values  = [0, 1, 2, 1, 3, 0]
 *   size    = [4]              -> M = 4 bins
 *   weights = []               -> empty weights accumulate 1.0f per hit
 *
 * Expected output (shape [2, 4], FP32):
 *   row0 : [1, 1, 1, 0]
 *   row1 : [1, 1, 0, 1]
 */
int CreateOppInGraph(std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
                     Graph& graph)
{
    auto raggedBinCount = op::RaggedBinCount("ragged_bin_count");

    static const vector<int64_t> splitsData = {0, 3, 6};
    static const vector<int32_t> valuesData = {0, 1, 2, 1, 3, 0};
    static const vector<int32_t> sizeData = {4};
    static const vector<float> weightsData = {};

    Tensor splitsTensor;
    Tensor valuesTensor;
    Tensor sizeTensor;
    Tensor weightsTensor;

    if (AddHostInput<int64_t>("splits", 0, {3}, DT_INT64, splitsData, raggedBinCount, graph, input, inputs,
                              splitsTensor) != SUCCESS) {
        return FAILED;
    }
    if (AddHostInput<int32_t>("values", 1, {6}, DT_INT32, valuesData, raggedBinCount, graph, input, inputs,
                              valuesTensor) != SUCCESS) {
        return FAILED;
    }
    if (AddHostInput<int32_t>("size", 2, {1}, DT_INT32, sizeData, raggedBinCount, graph, input, inputs, sizeTensor) !=
        SUCCESS) {
        return FAILED;
    }
    if (AddHostInput<float>("weights", 3, {0}, DT_FLOAT, weightsData, raggedBinCount, graph, input, inputs,
                            weightsTensor) != SUCCESS) {
        return FAILED;
    }

    raggedBinCount.set_input_splits(inputs[0]);
    raggedBinCount.set_input_values(inputs[1]);
    raggedBinCount.set_input_size(inputs[2]);
    raggedBinCount.set_input_weights(inputs[3]);
    raggedBinCount.set_attr_binary_output(false);

    TensorDesc outputDesc(ge::Shape({2, 4}), ge::FORMAT_ND, DT_FLOAT);
    raggedBinCount.update_output_desc_output(outputDesc);

    outputs.push_back(raggedBinCount);
    return SUCCESS;
}

bool InitEnv()
{
    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input)
{
    printf("%s - INFO - [XIR]: Start to CreateAndConfigGraph\n", GetTime().c_str());
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    Status ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id)
{
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {

    };

    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    return true;
}

bool DumpAndRunGraph(ge::Session* session, Graph& graph, std::vector<ge::Tensor>& input,
                     std::vector<ge::Tensor>& output, uint32_t graph_id)
{
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());
    return true;
}

void ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./ragged_bin_count_geir_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        const float* values = reinterpret_cast<const float*>(output_data_i);
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, values[j]);
        }
    }
}

int FinalizeRes()
{
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    // 初始化环境
    if (!InitEnv()) {
        return FAILED;
    }

    // 创建计算图
    const char* graph_name = "ragged_bin_count_geir_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    if (!CreateAndConfigGraph(graph, input)) {
        return FAILED;
    }

    // 创建会话并添加图
    std::map<AscendString, AscendString> build_options = {

    };
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        return FAILED;
    }

    // 执行图
    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        return FAILED;
    }
    // 处理输出数据
    ProcessOutputData(output);

    // 清理资源
    return FinalizeRes();
}
