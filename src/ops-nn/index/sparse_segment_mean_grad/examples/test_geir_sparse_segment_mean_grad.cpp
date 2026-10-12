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
 * \file test_geir_sparse_segment_mean_grad.cpp
 * \brief GEIR graph-mode example for SparseSegmentMeanGrad.
 *        y[indices[j]] += x[segment_ids[j]] / count(segment_ids[j])
 */

#include <cmath>
#include <cstdint>
#include <ctime>
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

#include "../op_graph/sparse_segment_mean_grad_proto.h"

namespace {
constexpr int kFailed = -1;
constexpr int kSuccess = 0;
constexpr uint32_t kDeviceId = 0;
constexpr double kTolerance = 1e-6;

// x is viewed as [kNumSegments, kColumn], y as [kOutputDim0, kColumn].
constexpr int64_t kNumSegments = 2;
constexpr int64_t kColumn = 3;
constexpr int64_t kIndicesNum = 3;
constexpr int32_t kOutputDim0 = 3;
constexpr int64_t kYElementNum = static_cast<int64_t>(kOutputDim0) * kColumn;

std::string GetTime()
{
    time_t now;
    time(&now);
    char buf[64] = {0};
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S,000", localtime(&now));
    return buf;
}

std::string DataTypeToString(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
            return "DT_FLOAT";
        case ge::DT_FLOAT16:
            return "DT_FLOAT16";
        case ge::DT_DOUBLE:
            return "DT_DOUBLE";
        case ge::DT_INT32:
            return "DT_INT32";
        case ge::DT_INT64:
            return "DT_INT64";
        default:
            return "DTYPE(" + std::to_string(static_cast<int32_t>(dtype)) + ")";
    }
}

ge::Tensor BuildDoubleTensor(const std::vector<int64_t>& shape, const std::vector<double>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_DOUBLE);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(double));
}

ge::Tensor BuildInt32Tensor(const std::vector<int64_t>& shape, const std::vector<int32_t>& values)
{
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_INT32);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetRealDimCnt(shape.size());
    return ge::Tensor(desc, reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(int32_t));
}

int CreateGraph(std::vector<ge::Tensor>& inputs, std::vector<ge::Operator>& graphInputs,
                std::vector<ge::Operator>& graphOutputs, ge::Graph& graph)
{
    // x: Data input, double, shape [kNumSegments, kColumn] -> [[1,2,3],[4,5,6]]
    auto xData = ge::op::Data("x").set_attr_index(0);
    const std::vector<int64_t> xShape = {kNumSegments, kColumn};
    const std::vector<double> xValues = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    ge::Tensor xTensor = BuildDoubleTensor(xShape, xValues);
    ge::TensorDesc xDesc = xTensor.GetTensorDesc();
    xData.update_input_desc_x(xDesc);
    xData.update_output_desc_y(xDesc);
    graph.AddOp(xData);

    // indices: Const input, int32, shape [kIndicesNum] -> {0, 2, 1}
    auto indicesConst = ge::op::Const("indices");
    const std::vector<int64_t> indicesShape = {kIndicesNum};
    ge::Tensor indicesTensor = BuildInt32Tensor(indicesShape, {0, 2, 1});
    ge::TensorDesc indicesDesc = indicesTensor.GetTensorDesc();
    indicesConst.set_attr_value(indicesTensor);
    indicesConst.update_output_desc_y(indicesDesc);
    graph.AddOp(indicesConst);

    // segment_ids: Const input, int32, shape [kIndicesNum] -> {0, 0, 1}
    auto segmentIdsConst = ge::op::Const("segment_ids");
    ge::Tensor segmentIdsTensor = BuildInt32Tensor(indicesShape, {0, 0, 1});
    ge::TensorDesc segmentIdsDesc = segmentIdsTensor.GetTensorDesc();
    segmentIdsConst.set_attr_value(segmentIdsTensor);
    segmentIdsConst.update_output_desc_y(segmentIdsDesc);
    graph.AddOp(segmentIdsConst);

    // output_dim0: Const input, int32 scalar (rank 0) -> kOutputDim0
    auto outputDim0Const = ge::op::Const("output_dim0");
    const std::vector<int64_t> scalarShape = {};
    ge::Tensor outputDim0Tensor = BuildInt32Tensor(scalarShape, {kOutputDim0});
    ge::TensorDesc outputDim0Desc = outputDim0Tensor.GetTensorDesc();
    outputDim0Const.set_attr_value(outputDim0Tensor);
    outputDim0Const.update_output_desc_y(outputDim0Desc);
    graph.AddOp(outputDim0Const);

    auto sparseSegmentMeanGrad = ge::op::SparseSegmentMeanGrad("sparse_segment_mean_grad");
    sparseSegmentMeanGrad.set_input_x(xData);
    sparseSegmentMeanGrad.set_input_indices(indicesConst);
    sparseSegmentMeanGrad.set_input_segment_ids(segmentIdsConst);
    sparseSegmentMeanGrad.set_input_output_dim0(outputDim0Const);
    sparseSegmentMeanGrad.update_input_desc_x(xDesc);
    sparseSegmentMeanGrad.update_input_desc_indices(indicesDesc);
    sparseSegmentMeanGrad.update_input_desc_segment_ids(segmentIdsDesc);
    sparseSegmentMeanGrad.update_input_desc_output_dim0(outputDim0Desc);

    inputs.push_back(xTensor);
    graphInputs.push_back(xData);
    graphOutputs.push_back(sparseSegmentMeanGrad);
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    if (outputs.size() != 1) {
        std::cerr << "Unexpected output count: " << outputs.size() << std::endl;
        return kFailed;
    }

    const auto& tensor = outputs[0];
    std::cout << "sparse_segment_mean_grad output dtype = " << DataTypeToString(tensor.GetTensorDesc().GetDataType())
              << std::endl;
    const int64_t elemCount = tensor.GetTensorDesc().GetShape().GetShapeSize();
    const auto* data = reinterpret_cast<const double*>(tensor.GetData());
    for (int64_t i = 0; i < elemCount; ++i) {
        std::cout << "sparse_segment_mean_grad output[" << i << "] = " << data[i] << std::endl;
    }

    if (elemCount != kYElementNum) {
        std::cerr << "Unexpected output element count: " << elemCount << ", expected " << kYElementNum << std::endl;
        return kFailed;
    }
    // segment 0 is referenced twice (scale 0.5), segment 1 once (scale 1.0);
    // indices {0, 2, 1} scatter x[0]*0.5 to y[0], x[0]*0.5 to y[2] and x[1]*1.0 to y[1].
    const double expected[kYElementNum] = {0.5, 1.0, 1.5, 4.0, 5.0, 6.0, 0.5, 1.0, 1.5};
    for (int64_t i = 0; i < elemCount; ++i) {
        if (std::fabs(data[i] - expected[i]) > kTolerance) {
            std::cerr << "Unexpected sparse_segment_mean_grad result at index " << i << ": " << data[i] << ", expected "
                      << expected[i] << std::endl;
            return kFailed;
        }
    }
    return kSuccess;
}

void PrintGeMessages()
{
    const std::string errorMsg = ge::GEGetErrorMsgV2().GetString();
    if (!errorMsg.empty()) {
        std::cout << "Error message: " << errorMsg << std::endl;
    }
    const std::string warningMsg = ge::GEGetWarningMsgV2().GetString();
    if (!warningMsg.empty()) {
        std::cout << "Warning message: " << warningMsg << std::endl;
    }
}
} // namespace

int main()
{
    std::cout << GetTime() << " - INFO - Start SparseSegmentMeanGrad GEIR example" << std::endl;
    std::map<ge::AscendString, ge::AscendString> globalOptions = {
        {"ge.exec.deviceId", std::to_string(kDeviceId).c_str()},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != ge::SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return kFailed;
    }

    ge::Graph graph("sparse_segment_mean_grad_graph");
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Operator> graphInputs;
    std::vector<ge::Operator> graphOutputs;
    if (CreateGraph(inputs, graphInputs, graphOutputs, graph) != kSuccess) {
        ge::GEFinalize();
        return kFailed;
    }
    graph.SetInputs(graphInputs).SetOutputs(graphOutputs);

    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    ge::Session session(sessionOptions);
    if (session.AddGraph(0, graph) != ge::SUCCESS) {
        std::cerr << "AddGraph failed" << std::endl;
        PrintGeMessages();
        ge::GEFinalize();
        return kFailed;
    }

    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::SUCCESS) {
        std::cerr << "RunGraph failed" << std::endl;
        PrintGeMessages();
        ge::GEFinalize();
        return kFailed;
    }

    const int ret = CheckOutput(outputs);
    PrintGeMessages();
    ge::GEFinalize();
    if (ret != kSuccess) {
        return kFailed;
    }
    std::cout << GetTime() << " - INFO - SparseSegmentMeanGrad GEIR example success" << std::endl;
    return kSuccess;
}
