/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <cmath>
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
#include "../op_graph/stateless_uniform_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;
#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape, value)                   \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesDataFloat32(placeholder##intputIndex##_shape,                               \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        value);                                                                              \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);           \
    input.push_back(tensor_placeholder##intputIndex);                                        \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    inputs.push_back(placeholder##intputIndex)

#define ADD_INT_INPUT(intputIndex, intputName, intputDtype, inputShape, value)               \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesDataInt64(placeholder##intputIndex##_shape,                                 \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        value);                                                                              \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);           \
    input.push_back(tensor_placeholder##intputIndex);                                        \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    inputs.push_back(placeholder##intputIndex)

#define ADD_DOUBLE_INPUT(intputIndex, intputName, inputShape, value)                         \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0); \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, ge::DT_DOUBLE);  \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenOnesDataDouble(placeholder##intputIndex##_shape,                                \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        value);                                                                              \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);           \
    input.push_back(tensor_placeholder##intputIndex);                                        \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    inputs.push_back(placeholder##intputIndex)

#define ADD_INPUT_ATTR(attrName, attrValue)                                                  \
    add1.set_attr_##attrName(attrValue)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                        \
    TensorDesc outputName##outputIndex##_desc =                                              \
        TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype);                          \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define ADD_CONST_INPUT(intputIndex, intputName, intputDtype, inputShape, constValues)       \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                           \
    auto placeholder##intputIndex = op::Const("placeholder" + intputIndex);                  \
    TensorDesc placeholder##intputIndex##_desc =                                             \
        TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, intputDtype);     \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                        \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                    \
    Tensor tensor_placeholder##intputIndex;                                                  \
    ret = GenConstDataInt64(placeholder##intputIndex##_shape,                                \
        tensor_placeholder##intputIndex,                                                     \
        placeholder##intputIndex##_desc,                                                     \
        constValues);                                                                        \
    if (ret != SUCCESS) {                                                                    \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());       \
        return FAILED;                                                                       \
    }                                                                                        \
    placeholder##intputIndex.SetAttr("value", tensor_placeholder##intputIndex);              \
    placeholder##intputIndex.update_output_desc_y(placeholder##intputIndex##_desc);          \
    graph.AddOp(placeholder##intputIndex);                                                   \
    add1.set_input_##intputName(placeholder##intputIndex);                                   \
    add1.update_input_desc_##intputName(placeholder##intputIndex##_desc);                    \
    inputs.push_back(placeholder##intputIndex)

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
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        return twoByte;
    } else if (dt == ge::DT_BF16) {
        return twoByte;
    } else if (dt == ge::DT_DOUBLE) {
        return eightByte;
    } else if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_UINT64) {
        return eightByte;
    } else if (dt == ge::DT_INT8) {
        return oneByte;
    }
    return oneByte;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, float value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(float);
    float *pData = new (std::nothrow) float[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, (uint8_t *)pData, data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenOnesDataInt64(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, int64_t value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(int64_t);
    int64_t *pData = new (std::nothrow) int64_t[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenOnesDataDouble(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc, double value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(double);
    double *pData = new (std::nothrow) double[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenConstDataInt64(vector<int64_t> shapes, Tensor &input_tensor, TensorDesc &input_tensor_desc,
    const vector<int64_t> &values)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    uint32_t data_len = size * sizeof(int64_t);
    int64_t *pData = new (std::nothrow) int64_t[size];
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = (i < values.size()) ? values[i] : 0;
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t *>(pData), data_len);
    delete[] pData;
    return SUCCESS;
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
    Status ret = SUCCESS;
    // StatelessUniform 算子定义
    auto add1 = op::StatelessUniform("add1");

    // shape: 1-D tensor, 输出形状为 {4, 8} => shape tensor 内容是 [4, 8], tensor shape 为 {2}
    std::vector<int64_t> shapeInputShape = {2};
    std::vector<int64_t> shapeValues = {4, 8};

    // seed, offset: 0-D scalar (用 {1} 表示)
    std::vector<int64_t> scalarShape = {1};

    // 输出形状
    std::vector<int64_t> outShape = {4, 8};

    // Input 1: shape (Const, DT_INT64)
    ADD_CONST_INPUT(1, shape, ge::DT_INT64, shapeInputShape, shapeValues);

    // Input 2: seed (Data, DT_INT64), value = 12345
    ADD_INT_INPUT(2, seed, ge::DT_INT64, scalarShape, 12345);

    // Input 3: offset (Data, DT_INT64), value = 0
    ADD_INT_INPUT(3, offset, ge::DT_INT64, scalarShape, 0);

    // Input 4: from (Data, DT_DOUBLE), value = 0.0
    ADD_DOUBLE_INPUT(4, from, scalarShape, 0.0);

    // Input 5: to (Data, DT_DOUBLE), value = 1.0
    ADD_DOUBLE_INPUT(5, to, scalarShape, 1.0);

    // Attr: dtype = DT_FLOAT (0)
    ADD_INPUT_ATTR(dtype, 0);

    // Output: y (DT_FLOAT)
    ADD_OUTPUT(1, y, ge::DT_FLOAT, outShape);

    outputs.push_back(add1);
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

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create op in graph failed\n", GetTime().c_str());
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session *session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    // ==================== Input Summary ====================
    printf("\n========== INPUT SUMMARY ==========\n");
    printf("Total inputs: %zu\n", input.size());
    int input_num = input.size();
    const char* dtypeNames[] = {
        "FLOAT(0)", "FLOAT16(1)", "INT8(2)", "INT32(3)", "UINT8(4)", "",
        "INT16(6)", "UINT16(7)", "UINT32(8)", "INT64(9)", "UINT64(10)",
        "DOUBLE(11)", "BOOL(12)", "", "UINT1(14)", "", "", "", "", "", "",
        "", "", "", "", "", "", "BF16(27)"
    };
    for (int i = 0; i < input_num; i++) {
        printf("---------- Input %d ----------\n", i);
        DataType dt = input[i].GetTensorDesc().GetDataType();
        ge::Shape inShape = input[i].GetTensorDesc().GetShape();
        size_t inDimNum = inShape.GetDimNum();
        int64_t inShapeSize = inShape.GetShapeSize();
        uint32_t elemSize = GetDataTypeSize(dt);
        uint32_t dataBytes = inShapeSize * elemSize;

        // dtype
        printf("  dtype     : %d", dt);
        if (dt >= 0 && dt <= 27 && strlen(dtypeNames[dt]) > 0) {
            printf(" (%s)", dtypeNames[dt]);
        }
        printf("\n");

        // shape dims
        printf("  shape     : [");
        for (size_t d = 0; d < inDimNum; d++) {
            printf("%ld", inShape.GetDim(d));
            if (d + 1 < inDimNum) printf(", ");
        }
        printf("]  (dims=%zu, elements=%ld)\n", inDimNum, inShapeSize);

        // format
        printf("  format    : %d\n", input[i].GetTensorDesc().GetFormat());

        // data bytes
        printf("  data size : %u bytes (%ld elems * %u bytes/elem)\n", dataBytes, inShapeSize, elemSize);

        // print actual values
        uint8_t *inData = input[i].GetData();
        if (inData != nullptr && inShapeSize > 0) {
            printf("  values    : ");
            if (dt == ge::DT_INT64) {
                int64_t *vals = (int64_t*)inData;
                for (int64_t j = 0; j < inShapeSize && j < 16; j++) {
                    printf("%ld", vals[j]);
                    if (j + 1 < inShapeSize && j + 1 < 16) printf(", ");
                }
            } else if (dt == ge::DT_DOUBLE) {
                double *vals = (double*)inData;
                for (int64_t j = 0; j < inShapeSize && j < 16; j++) {
                    printf("%.6f", vals[j]);
                    if (j + 1 < inShapeSize && j + 1 < 16) printf(", ");
                }
            } else if (dt == ge::DT_FLOAT) {
                float *vals = (float*)inData;
                for (int64_t j = 0; j < inShapeSize && j < 16; j++) {
                    printf("%.6f", vals[j]);
                    if (j + 1 < inShapeSize && j + 1 < 16) printf(", ");
                }
            } else if (dt == ge::DT_INT32) {
                int32_t *vals = (int32_t*)inData;
                for (int64_t j = 0; j < inShapeSize && j < 16; j++) {
                    printf("%d", vals[j]);
                    if (j + 1 < inShapeSize && j + 1 < 16) printf(", ");
                }
            }
            if (inShapeSize > 16) printf(" ... (%ld more)", inShapeSize - 16);
            printf("\n");
        }

        // write to file
        string input_file = "./tc_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        WriteDataToFile((const char *)input_file.c_str(), dataBytes, inData);
        printf("  saved to  : %s\n", input_file.c_str());
    }

    // ==================== Output Summary ====================
    printf("\n========== OUTPUT SUMMARY ==========\n");
    printf("Total outputs: %zu\n", output.size());
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        printf("---------- Output %d ----------\n", i);
        DataType dt = output[i].GetTensorDesc().GetDataType();
        ge::Shape outShape = output[i].GetTensorDesc().GetShape();
        size_t outDimNum = outShape.GetDimNum();
        int64_t outShapeSize = outShape.GetShapeSize();
        uint32_t elemSize = GetDataTypeSize(dt);
        uint32_t dataBytes = outShapeSize * elemSize;

        // dtype
        printf("  dtype          : %d", dt);
        if (dt >= 0 && dt <= 27 && strlen(dtypeNames[dt]) > 0) {
            printf(" (%s)", dtypeNames[dt]);
        }
        printf("\n");

        // inferred shape (from InferShape)
        printf("  inferred shape : [");
        for (size_t d = 0; d < outDimNum; d++) {
            printf("%ld", outShape.GetDim(d));
            if (d + 1 < outDimNum) printf(", ");
        }
        printf("]  (dims=%zu, elements=%ld)\n", outDimNum, outShapeSize);

        // format
        printf("  format         : %d\n", output[i].GetTensorDesc().GetFormat());

        // data bytes
        printf("  data size      : %u bytes (%ld elems * %u bytes/elem)\n", dataBytes, outShapeSize, elemSize);

        // write to file
        string output_file = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t *output_data_i = output[i].GetData();
        WriteDataToFile((const char *)output_file.c_str(), dataBytes, output_data_i);
        printf("  saved to       : %s\n", output_file.c_str());

        // print values with statistics
        if (dt == ge::DT_FLOAT && output_data_i != nullptr && outShapeSize > 0) {
            float *resultData = (float*)output_data_i;
            float minVal = resultData[0], maxVal = resultData[0];
            double sum = 0.0;
            int nanCount = 0, infCount = 0;
            for (int64_t j = 0; j < outShapeSize; j++) {
                float v = resultData[j];
                if (std::isnan(v)) { nanCount++; continue; }
                if (std::isinf(v)) { infCount++; continue; }
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
                sum += v;
            }
            printf("  --- Statistics ---\n");
            printf("    min          : %.6f\n", minVal);
            printf("    max          : %.6f\n", maxVal);
            printf("    mean         : %.6f\n", sum / outShapeSize);
            printf("    range check  : all in [0, 1) ? %s\n",
                   (minVal >= 0.0f && maxVal < 1.0f && nanCount == 0) ? "YES" : "NO");
            if (nanCount > 0) printf("    NaN count    : %d\n", nanCount);
            if (infCount > 0) printf("    Inf count    : %d\n", infCount);

            printf("  --- Values ---\n");
            for (int64_t j = 0; j < outShapeSize && j < 64; j++) {
                printf("    result[%ld] = %.8f\n", j, resultData[j]);
            }
            if (outShapeSize > 64) {
                printf("    ... (%ld more values, see .bin file)\n", outShapeSize - 64);
            }
        } else if (dt == ge::DT_FLOAT16 && output_data_i != nullptr && outShapeSize > 0) {
            printf("  --- Values (fp16 raw hex) ---\n");
            uint16_t *fp16Data = (uint16_t*)output_data_i;
            for (int64_t j = 0; j < outShapeSize && j < 32; j++) {
                printf("    result[%ld] = 0x%04X\n", j, fp16Data[j]);
            }
        }
    }

    // ==================== Error/Warning ====================
    printf("\n========== DIAGNOSTICS ==========\n");
    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    printf("Error message  : %s\n", error_str.empty() ? "(none)" : error_str.c_str());
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    printf("Warning message: %s\n", warning_str.empty() ? "(none)" : warning_str.c_str());
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
