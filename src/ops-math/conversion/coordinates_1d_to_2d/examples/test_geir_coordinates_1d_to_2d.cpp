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
 * \file test_geir_coordinates_1d_to_2d.cpp
 * \brief GE IR test for Coordinates1DTo2D operator
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

#include "../op_graph/coordinates_1d_to_2d_proto.h"

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
    uint32_t oneByte = 1;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_UINT64) {
        return eightByte;
    }
    return fourByte;
}

int32_t GenTestData(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type, int64_t value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate memory failed\n", GetTime().c_str());
        return FAILED;
    }

    if (data_type == ge::DT_INT32) {
        int32_t* pTypedData = reinterpret_cast<int32_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = static_cast<int32_t>(value);
        }
    } else if (data_type == ge::DT_INT64) {
        int64_t* pTypedData = reinterpret_cast<int64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = value;
        }
    } else if (data_type == ge::DT_UINT64) {
        uint64_t* pTypedData = reinterpret_cast<uint64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = static_cast<uint64_t>(value);
        }
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

int32_t GenShapeData(vector<int64_t> shape_values, Tensor& input_tensor, TensorDesc& input_tensor_desc, DataType data_type)
{
    input_tensor_desc.SetRealDimCnt(1);
    size_t size = shape_values.size();
    uint32_t data_len = size * GetDataTypeSize(data_type);
    uint8_t* pData = new (std::nothrow) uint8_t[data_len];
    if (pData == nullptr) {
        printf("%s - ERROR - [XIR]: Allocate memory failed\n", GetTime().c_str());
        return FAILED;
    }

    if (data_type == ge::DT_INT32) {
        int32_t* pTypedData = reinterpret_cast<int32_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = static_cast<int32_t>(shape_values[i]);
        }
    } else if (data_type == ge::DT_INT64) {
        int64_t* pTypedData = reinterpret_cast<int64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = shape_values[i];
        }
    } else if (data_type == ge::DT_UINT64) {
        uint64_t* pTypedData = reinterpret_cast<uint64_t*>(pData);
        for (size_t i = 0; i < size; ++i) {
            pTypedData[i] = static_cast<uint64_t>(shape_values[i]);
        }
    }

    input_tensor = Tensor(input_tensor_desc, pData, data_len);
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "wb");
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(
    DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs, std::vector<Operator>& outputs,
    Graph& graph)
{
    Status ret = SUCCESS;
    
    std::string name_x = "placeholder_x";
    auto placeholder_x = op::Data(name_x.c_str()).set_attr_index(0);
    TensorDesc placeholder_x_desc = TensorDesc(ge::Shape({1}), FORMAT_ND, inDtype);
    placeholder_x_desc.SetPlacement(ge::kPlacementHost);
    placeholder_x_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder_x;
    ret = GenTestData({1}, tensor_placeholder_x, placeholder_x_desc, inDtype, 5);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder_x.update_input_desc_x(placeholder_x_desc);
    graph.AddOp(placeholder_x);
    input.push_back(tensor_placeholder_x);
    inputs.push_back(placeholder_x);

    std::string name_shape = "placeholder_shape";
    auto placeholder_shape = op::Data(name_shape.c_str()).set_attr_index(1);
    TensorDesc placeholder_shape_desc = TensorDesc(ge::Shape({4}), FORMAT_ND, inDtype);
    placeholder_shape_desc.SetPlacement(ge::kPlacementHost);
    placeholder_shape_desc.SetFormat(FORMAT_ND);
    Tensor tensor_placeholder_shape;
    ret = GenShapeData({1, 1, 1, 10}, tensor_placeholder_shape, placeholder_shape_desc, inDtype);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate shape data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholder_shape.update_input_desc_x(placeholder_shape_desc);
    graph.AddOp(placeholder_shape);
    input.push_back(tensor_placeholder_shape);
    inputs.push_back(placeholder_shape);

    auto coordinates_1d_to_2d_op = op::Coordinates1DTo2D("coordinates_1d_to_2d");
    coordinates_1d_to_2d_op.set_input_x(placeholder_x);
    coordinates_1d_to_2d_op.set_input_shape(placeholder_shape);

    TensorDesc output_row_desc = TensorDesc(ge::Shape({1}), FORMAT_ND, inDtype);
    TensorDesc output_col_desc = TensorDesc(ge::Shape({1}), FORMAT_ND, inDtype);
    TensorDesc output_n_desc = TensorDesc(ge::Shape({1}), FORMAT_ND, inDtype);
    coordinates_1d_to_2d_op.update_output_desc_row(output_row_desc);
    coordinates_1d_to_2d_op.update_output_desc_col(output_col_desc);
    coordinates_1d_to_2d_op.update_output_desc_n(output_n_desc);

    outputs.push_back(coordinates_1d_to_2d_op);
    return SUCCESS;
}

void SaveInputOutput(std::vector<ge::Tensor>& input, std::vector<ge::Tensor>& output)
{
    for (size_t i = 0; i < input.size(); i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        string input_file = "./tc_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i);
    }

    for (size_t i = 0; i < output.size(); i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i);
        int64_t* resultData = (int64_t*)output_data_i;
        for (int64_t j = 0; j < output_shape && j < 10; j++) {
            printf("result[%ld] is: %ld\n", j, resultData[j]);
        }
    }
}

int main(int argc, char* argv[])
{
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
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_INT64;
    if (argc >= 2) {
        std::string dtype_str = argv[1];
        if (dtype_str == "int32") {
            inDtype = DT_INT32;
        } else if (dtype_str == "int64") {
            inDtype = DT_INT64;
        } else if (dtype_str == "uint64") {
            inDtype = DT_UINT64;
        } else {
            std::cout << "Unknown dtype: " << dtype_str << ", using default: int64" << std::endl;
        }
    } else {
        std::cout << "No dtype specified, using default: int64" << std::endl;
    }

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        return FAILED;
    }

    std::map<AscendString, AscendString> graph_options = {
        {"ge.exec.exclude_engines", "AiCore"}
    };
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    SaveInputOutput(input, output);

    delete session;

    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize session success\n", GetTime().c_str());
    return SUCCESS;
}