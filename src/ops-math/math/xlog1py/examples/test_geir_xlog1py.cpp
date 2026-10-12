/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/**
 * @file test_geir_xlog1py.cpp
 * @brief GE IR 图模式调用示例 - Xlog1py 算子
 *
 * z = x * log1p(y), x==0 -> z=0
 * 测试流程: GEInitialize -> 建图 -> AddGraph -> RunGraph -> 比对精度 -> GEFinalize
 */

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "array_ops.h"

#include "../op_graph/xlog1py_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

static std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

static uint32_t GetDtypeSize(DataType dt)
{
    switch (dt) {
    case ge::DT_FLOAT:  case ge::DT_INT32:  case ge::DT_UINT32:  return 4;
    case ge::DT_FLOAT16: case ge::DT_BF16:  case ge::DT_INT16:   case ge::DT_UINT16: return 2;
    case ge::DT_INT64:  case ge::DT_UINT64: case ge::DT_DOUBLE:  return 8;
    default: return 1;
    }
}

static Status GenFloatData(const vector<int64_t>& shapes, Tensor& tensor,
                            TensorDesc& desc, const vector<float>& values)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto s : shapes) size *= s;
    uint32_t dataLen = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (int64_t i = 0; i < size; i++) pData[i] = values[i % values.size()];
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), dataLen);
    return SUCCESS;
}

static Status GenScalarData(const vector<int64_t>& shapes, Tensor& tensor,
                             TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shapes.size());
    int64_t size = 1;
    for (auto s : shapes) size *= s;
    uint32_t dataLen = size * sizeof(float);
    float* pData = new (std::nothrow) float[size];
    for (int64_t i = 0; i < size; i++) pData[i] = value;
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), dataLen);
    return SUCCESS;
}

// CPU Golden: z = x * log1p(y), x==0 -> z=0
static std::vector<float> ComputeGolden(const vector<float>& x, const vector<int64_t>& shapeX,
    const vector<float>& y, const vector<int64_t>& shapeY, const vector<int64_t>& outShape)
{
    auto broadcastIdx = [](int64_t flat, const vector<int64_t>& inShape, const vector<int64_t>& outShape) {
        int inRank = (int)inShape.size(), outRank = (int)outShape.size();
        int64_t outIdx = 0, outStride = 1;
        for (int d = 0; d < outRank; d++) {
            int dimIdx = outRank - 1 - d;
            int64_t dim = outShape[dimIdx];
            int64_t coord = (flat / outStride) % dim;
            int inDimIdx = dimIdx - (outRank - inRank);
            int64_t inDim = (inDimIdx >= 0) ? inShape[inDimIdx] : 1;
            int64_t inCoord = (inDim == 1) ? 0 : coord;
            int inStride = 1;
            for (int dd = inRank - 1; dd > inDimIdx; dd--) inStride *= inShape[dd];
            outIdx += inCoord * inStride;
            outStride *= dim;
        }
        return outIdx;
    };
    int64_t n = 1;
    for (auto d : outShape) n *= d;
    vector<float> result(n);
    for (int64_t i = 0; i < n; i++) {
        int64_t ix = broadcastIdx(i, shapeX, outShape);
        int64_t iy = broadcastIdx(i, shapeY, outShape);
        float fx = x[ix], fy = y[iy];
        result[i] = (fx == 0.0f) ? 0.0f : fx * std::log1p(fy);
    }
    return result;
}

static bool CompareOutput(const vector<float>& golden, const vector<float>& output,
                           int64_t n, const string& tag)
{
    bool allPass = true;
    double maxMere = 0.0;
    for (int64_t i = 0; i < n; i++) {
        float g = golden[i], r = output[i];
        double mere = (std::fabs(g) > 1e-6) ? std::fabs(r - g) / std::fabs(g) : std::fabs(r - g);
        if (mere > maxMere) maxMere = mere;
        if (mere > 0.001) {
            printf("  [FAIL][%s][%ld] golden=%.6f geir=%.6f mere=%.6e\n",
                   tag.c_str(), i, g, r, mere);
            allPass = false;
        }
    }
    if (allPass) {
        printf("  [PASS][%s] all %ld elems OK, max_mere=%.6e\n", tag.c_str(), n, maxMere);
    }
    return allPass;
}

static int RunGEIRTest(const string& tag,
    const vector<int64_t>& shapeX, const vector<float>& dataX,
    const vector<int64_t>& shapeY, const vector<float>& dataY)
{
    printf("--- Test GEIR %s ---\n", tag.c_str());

    // Compute output shape (broadcast)
    int rank = std::max((int)shapeX.size(), (int)shapeY.size());
    vector<int64_t> outShape(rank);
    for (int d = 0; d < rank; d++) {
        int dx = d - (rank - shapeX.size());
        int dy = d - (rank - shapeY.size());
        outShape[d] = std::max(dx >= 0 ? shapeX[dx] : 1, dy >= 0 ? shapeY[dy] : 1);
    }
    int64_t outSize = 1;
    for (auto d : outShape) outSize *= d;
    auto golden = ComputeGolden(dataX, shapeX, dataY, shapeY, outShape);

    // Build graph
    Graph graph("tc_ge_irrun_xlog1py");
    vector<ge::Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    Status ret;

    // Input X — Data node with index 0
    auto opX = op::Data("inputX").set_attr_index(0);
    TensorDesc descX(ge::Shape(shapeX), FORMAT_ND, DT_FLOAT);
    descX.SetPlacement(ge::kPlacementHost);
    descX.SetFormat(FORMAT_ND);
    Tensor tensorX;
    ret = GenFloatData(shapeX, tensorX, descX, dataX);
    if (ret != SUCCESS) { printf("Gen input X data failed\n"); return FAILED; }
    opX.update_input_desc_x(descX);
    input.push_back(tensorX);
    graph.AddOp(opX);
    inputs.push_back(opX);

    // Input Y — Data node with index 1
    auto opY = op::Data("inputY").set_attr_index(1);
    TensorDesc descY(ge::Shape(shapeY), FORMAT_ND, DT_FLOAT);
    descY.SetPlacement(ge::kPlacementHost);
    descY.SetFormat(FORMAT_ND);
    Tensor tensorY;
    ret = GenFloatData(shapeY, tensorY, descY, dataY);
    if (ret != SUCCESS) { printf("Gen input Y data failed\n"); return FAILED; }
    opY.update_input_desc_x(descY);
    input.push_back(tensorY);
    graph.AddOp(opY);
    inputs.push_back(opY);

    // Xlog1py operator
    auto xlog1py = op::Xlog1py("xlog1py_op");
    xlog1py.set_input_x(opX);
    xlog1py.set_input_y(opY);

    TensorDesc descZ(ge::Shape(outShape), FORMAT_ND, DT_FLOAT);
    xlog1py.update_output_desc_z(descZ);
    outputs.push_back(xlog1py);

    graph.SetInputs(inputs).SetOutputs(outputs);

    // Run graph
    map<AscendString, AscendString> build_options;
    ge::Session* session = new Session(build_options);
    if (session == nullptr) { printf("Create session failed\n"); return FAILED; }

    map<AscendString, AscendString> graph_options;
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graph_options);
    if (ret != SUCCESS) {
        printf("AddGraph failed, ret=%d\n", (int)ret);
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        std::string errStr(errMsg.GetString());
        printf("GE Error: %s\n", errStr.c_str());
        delete session;
        return FAILED;
    }

    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        ge::AscendString errMsg = ge::GEGetErrorMsgV2();
        printf("RunGraph failed, ret=%d\n", (int)ret);
        printf("GE Error: %s\n", errMsg.GetString());
        delete session;
        return FAILED;
    }
    printf("RunGraph success\n");

    // Read output
    if (output.empty()) { printf("No output\n"); delete session; return FAILED; }
    int64_t outputShapeSize = output[0].GetTensorDesc().GetShape().GetShapeSize();
    uint32_t dataSize = outputShapeSize * GetDtypeSize(output[0].GetTensorDesc().GetDataType());
    uint8_t* outputData = output[0].GetData();
    vector<float> npuResult(outputShapeSize);
    memcpy(npuResult.data(), outputData, dataSize);

    // Compare
    bool pass = CompareOutput(golden, npuResult, outSize, tag);

    delete session;
    return pass ? SUCCESS : FAILED;
}

int main(int argc, char* argv[])
{
    printf("%s - INFO - [XIR]: Start to initialize GE\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"}
    };
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    int numPass = 0, numFail = 0;

    // Test 1: same shape
    {
        vector<int64_t> shape = {1, 2, 4, 4};
        vector<float> x(32, 2.0f), y(32, 1.0f);
        if (RunGEIRTest("same_shape", shape, x, shape, y) == SUCCESS) numPass++; else numFail++;
    }

    // Test 2: broadcast
    {
        vector<int64_t> shapeX = {1, 2, 1, 4};
        vector<int64_t> shapeY = {1, 2, 4, 4};
        vector<float> x(8, 3.0f), y(32, 2.0f);
        if (RunGEIRTest("broadcast", shapeX, x, shapeY, y) == SUCCESS) numPass++; else numFail++;
    }

    // Test 3: x == 0 boundary
    {
        vector<int64_t> shape = {1, 1, 8, 8};
        vector<float> x(64, 0.0f), y(64, 100.0f);
        if (RunGEIRTest("x_eq_0", shape, x, shape, y) == SUCCESS) numPass++; else numFail++;
    }

    // Test 4: scalar broadcast
    {
        vector<int64_t> shapeX = {1};
        vector<int64_t> shapeY = {4, 8, 16, 16};
        vector<float> x(1, 2.5f);
        int64_t n = 8192;
        vector<float> y(n);
        for (int64_t i = 0; i < n; i++) y[i] = 1.0f + 0.1f * (i % 5);
        if (RunGEIRTest("scalar_broadcast", shapeX, x, shapeY, y) == SUCCESS) numPass++; else numFail++;
    }

    printf("========================================\n");
    printf("GEIR Xlog1py NPU results: PASS=%d  FAIL=%d\n", numPass, numFail);
    printf("========================================\n");

    printf("%s - INFO - [XIR]: Finalize GE\n", GetTime().c_str());
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return (numFail == 0) ? SUCCESS : FAILED;
}
