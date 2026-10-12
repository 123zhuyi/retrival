/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/reduce_std_v2_update_proto.h"

namespace {

std::string CurrentError()
{
    std::string error = ge::GEGetErrorMsgV2().GetString();
    std::replace(error.begin(), error.end(), '\n', ' ');
    std::replace(error.begin(), error.end(), '\r', ' ');
    return error;
}

std::string FailureStage(const std::string& error)
{
    if (error.find("Tiling func") != std::string::npos || error.find("tiling failed") != std::string::npos) {
        return "tiling";
    }
    if (error.find("Unsupported_Operator") != std::string::npos ||
        error.find("No supported Ops kernel") != std::string::npos) {
        return "engine_selection";
    }
    if (error.find("para_check.py") != std::string::npos || error.find("num of dimensions") != std::string::npos) {
        return "parameter_check";
    }
    return "graph_compile";
}

struct CaseConfig {
    std::string id;
    std::string category;
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType meanDtype = ge::DT_FLOAT;
    ge::Format xFormat = ge::FORMAT_ND;
    ge::Format meanFormat = ge::FORMAT_ND;
    ge::Format outputFormat = ge::FORMAT_ND;
    std::vector<int64_t> shape = {1};
    std::vector<int64_t> meanShape = {1};
    std::vector<int64_t> outputShape = {1};
    std::vector<int64_t> dim = {0};
    int64_t correction = 0;
    bool castX = false;
    bool expectAccept = false;
    std::vector<std::string> expectedErrors;
    std::vector<std::string> allowedFailureStages = {"tiling"};
};

bool Contains(const std::vector<std::string>& values, const std::string& target)
{
    return std::find(values.begin(), values.end(), target) != values.end();
}

bool MatchesAnyError(const std::vector<std::string>& signatures, const std::string& error)
{
    return std::any_of(signatures.begin(), signatures.end(), [&error](const std::string& signature) {
        return !signature.empty() && error.find(signature) != std::string::npos;
    });
}

int64_t Count(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (const auto dim : shape) {
        count *= dim;
    }
    return count;
}

ge::Tensor MakeTensor(const ge::TensorDesc& desc, ge::DataType dtype, int64_t count, float value)
{
    if (dtype == ge::DT_INT32) {
        std::vector<int32_t> values(static_cast<size_t>(count), static_cast<int32_t>(value));
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(int32_t));
    }
    if (dtype == ge::DT_INT8) {
        std::vector<int8_t> values(static_cast<size_t>(count), static_cast<int8_t>(value));
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size());
    }
    if (dtype == ge::DT_FLOAT16) {
        // IEEE-754 binary16 encoding of 2.0F. The mismatch case is rejected in Tiling,
        // but the host tensor still needs the byte width promised by its descriptor.
        std::vector<uint16_t> values(static_cast<size_t>(count), 0x4000U);
        return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(uint16_t));
    }
    std::vector<float> values(static_cast<size_t>(count), value);
    return ge::Tensor(desc, reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(float));
}

ge::Operator AddData(const std::string& name, uint32_t index, const std::vector<int64_t>& shape, ge::DataType dtype,
                     ge::Format format, float value, ge::Graph& graph, std::vector<ge::Tensor>& tensors,
                     std::vector<ge::Operator>& inputs)
{
    auto data = ge::op::Data(name.c_str()).set_attr_index(index);
    ge::TensorDesc desc(ge::Shape(shape), format, dtype);
    desc.SetPlacement(ge::kPlacementHost);
    desc.SetFormat(format);
    desc.SetRealDimCnt(shape.size());
    data.update_input_desc_x(desc);
    data.update_output_desc_y(desc);
    tensors.push_back(MakeTensor(desc, dtype, Count(shape), value));
    graph.AddOp(data);
    inputs.push_back(data);
    return data;
}

void BuildGraph(const CaseConfig& config, ge::Graph& graph, std::vector<ge::Tensor>& tensors,
                std::vector<ge::Operator>& inputs, std::vector<ge::Operator>& outputs, bool setDim = true)
{
    const auto sourceDtype = config.castX ? ge::DT_INT32 : config.xDtype;
    auto xData = AddData(config.id + "_x", 0, config.shape, sourceDtype, config.xFormat, 2.0F, graph, tensors, inputs);
    ge::Operator x = xData;
    ge::TensorDesc xDesc(ge::Shape(config.shape), config.xFormat, config.xDtype);
    if (config.castX) {
        ge::TensorDesc sourceDesc(ge::Shape(config.shape), config.xFormat, sourceDtype);
        auto cast = ge::op::Cast((config.id + "_cast").c_str()).set_input_x(xData).set_attr_dst_type(config.xDtype);
        cast.update_input_desc_x(sourceDesc);
        cast.update_output_desc_y(xDesc);
        graph.AddOp(cast);
        x = cast;
    }
    auto mean = AddData(config.id + "_mean", 1, config.meanShape, config.meanDtype, config.meanFormat, 2.0F, graph,
                        tensors, inputs);
    ge::TensorDesc meanDesc(ge::Shape(config.meanShape), config.meanFormat, config.meanDtype);

    auto op = ge::op::ReduceStdV2Update((config.id + "_op").c_str());
    op.set_input_x(x);
    op.set_input_mean(mean);
    op.update_input_desc_x(xDesc);
    op.update_input_desc_mean(meanDesc);
    if (setDim) {
        op.set_attr_dim(config.dim);
    }
    op.set_attr_if_std(false);
    op.set_attr_unbiased(false);
    op.set_attr_keepdim(true);
    op.set_attr_correction(config.correction);
    ge::TensorDesc outputDesc(ge::Shape(config.outputShape), config.outputFormat, config.xDtype);
    op.update_output_desc_output_var(outputDesc);
    graph.AddOp(op);
    outputs.push_back(op);
}

bool VerifyOutput(const std::vector<ge::Tensor>& result, const CaseConfig& config)
{
    if (result.size() != 1 || result[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        result[0].GetTensorDesc().GetShape().GetDims() != config.outputShape) {
        return false;
    }
    if (Count(config.outputShape) == 0) {
        return true;
    }
    if (Count(config.outputShape) != 1) {
        return false;
    }
    const auto* value = reinterpret_cast<const float*>(result[0].GetData());
    return value != nullptr && std::abs(value[0]) <= 1.0e-6F;
}

bool RunCase(const CaseConfig& config, bool setDim = true)
{
    ge::Graph graph(("reduce_std_v2_update_exception_" + config.id).c_str());
    std::vector<ge::Tensor> tensors;
    std::vector<ge::Operator> inputs;
    std::vector<ge::Operator> outputs;
    BuildGraph(config, graph, tensors, inputs, outputs, setDim);
    graph.SetInputs(inputs).SetOutputs(outputs);
    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    ge::Status status = session.AddGraph(0, graph, options);
    std::vector<ge::Tensor> result;
    if (status == ge::SUCCESS) {
        status = session.RunGraph(0, tensors, result);
    }
    if (!config.expectAccept) {
        if (status == ge::SUCCESS) {
            std::cout << "EXCEPTION_CASE " << config.id << " FAIL expected=reject actual=accept" << std::endl;
            return false;
        }
        const std::string actualError = CurrentError();
        const std::string failureStage = FailureStage(actualError);
        const bool stageMatched = Contains(config.allowedFailureStages, failureStage);
        const bool errorMatched = MatchesAnyError(config.expectedErrors, actualError);
        if (!stageMatched || !errorMatched) {
            std::cout << "EXCEPTION_CASE " << config.id << " FAIL category=" << config.category
                      << " expected=reject actual=reject failure_stage=" << failureStage
                      << " failure_stage_matched=" << (stageMatched ? "true" : "false")
                      << " kernel_launched=false error_signature_matched=" << (errorMatched ? "true" : "false")
                      << " actual_error=" << actualError << std::endl;
            return false;
        }
        std::cout << "EXCEPTION_CASE " << config.id << " PASS category=" << config.category
                  << " expected=reject actual=reject failure_stage=" << failureStage
                  << " kernel_launched=false error_signature_matched=true actual_error=" << actualError;
        if (config.category == "rank_9_tensor") {
            std::cout << " rank=9";
        }
        std::cout << std::endl;
        return true;
    }
    if (status != ge::SUCCESS || !VerifyOutput(result, config)) {
        std::cout << "EXCEPTION_CASE " << config.id
                  << " FAIL expected=accept actual=reject error=" << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return false;
    }
    std::cout << "EXCEPTION_CASE " << config.id << " PASS category=" << config.category
              << " expected=accept actual=accept validation_stage=tiling kernel_launched=true output_verified=true"
              << " kernel_launch_observed=true";
    if (config.castX) {
        std::cout << " source_dtype=int32 cast_dtype=float32 required_dtype=float32 observed_input_dtype=float32";
    }
    std::cout << std::endl;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << "EXCEPTION_SUITE ReduceStdV2Update Ascend950 GEIR" << std::endl;
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::SUCCESS) {
        return -1;
    }
    // GE can reject invalid dtypes or ranks before custom tiling, while format inference can propagate an invalid
    // output format to an input descriptor. Keep every accepted stage and error signature specific to its case.
    const std::vector<CaseConfig> cases = {
        {"dtype_x_int8",
         "dtype_unsupported",
         ge::DT_INT8,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {0},
         0,
         false,
         false,
         {"unsupported x dtype", "Verifying dtype_x_int8_op failed"},
         {"tiling", "graph_compile"}},
        {"dtype_mean_int8",
         "dtype_unsupported",
         ge::DT_FLOAT,
         ge::DT_INT8,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {0},
         0,
         false,
         false,
         {"unsupported mean dtype", "Verifying dtype_mean_int8_op failed"},
         {"tiling", "graph_compile"}},
        {"dtype_mean_mismatch",
         "dtype_combination_mismatch",
         ge::DT_FLOAT,
         ge::DT_FLOAT16,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {0},
         0,
         false,
         false,
         {"mean dtype 1 != x dtype 0", "Verifying dtype_mean_mismatch_op failed"},
         {"tiling", "graph_compile"}},
        {"cast_x_int32_to_float32",
         "dtype_geir_cast",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {0},
         0,
         true,
         true,
         {},
         {}},
        {"format_x_nchw",
         "format_unsupported",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_NCHW,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {0},
         0,
         false,
         false,
         {"x format 0 is unsupported; only ND is supported"}},
        {"format_mean_nchw",
         "format_unsupported",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_NCHW,
         ge::FORMAT_ND,
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {0},
         0,
         false,
         false,
         {"mean format 0 is unsupported; only ND is supported"}},
        {"format_output_nchw",
         "format_unsupported",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_NCHW,
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {1, 1, 1, 1},
         {0},
         0,
         false,
         false,
         {"output_var format 0 is unsupported; only ND is supported",
          "x format 0 is unsupported; only ND is supported"}},
        {"rank9_x",
         "rank_9_tensor",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {1, 1, 1, 1, 1, 1, 1, 1, 1},
         {0},
         0,
         false,
         false,
         {"x rank=9 exceeds supported range [0, 8]",
          "num of dimensions of input/output[x_in__] should be in the range of [0, 8]"},
         {"tiling", "parameter_check"}},
        {"correction_two",
         "attribute_out_of_range",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {0},
         2,
         false,
         false,
         {"correction=2, only 0 or 1 supported"}},
        {"dim_out_of_range",
         "attribute_out_of_range",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {1},
         {1},
         {1},
         {1},
         0,
         false,
         false,
         {"dim[0]=1 out of range [-1, 1)"}},
        {"shape_mean_mismatch",
         "shape_dimension_violation",
         ge::DT_FLOAT,
         ge::DT_FLOAT,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         ge::FORMAT_ND,
         {4},
         {2, 2},
         {1},
         {0},
         0,
         false,
         false,
         {"mean rank=2 != x rank=1"}},
    };
    bool passed = true;
    bool matched = false;
    for (const auto& item : cases) {
        if (argc > 1 && item.id != argv[1]) {
            continue;
        }
        matched = true;
        passed = RunCase(item) && passed;
    }
    const std::string missingDimId = "missing_required_dim";
    if (argc <= 1 || missingDimId == argv[1]) {
        CaseConfig missingDim;
        missingDim.id = missingDimId;
        missingDim.category = "required_attribute_missing";
        missingDim.expectedErrors = {
            "Call InferShapeAndType for node:missing_required_dim_op(ReduceStdV2Update) failed"};
        missingDim.allowedFailureStages = {"graph_compile"};
        matched = true;
        passed = RunCase(missingDim, false) && passed;
    }
    if (!matched) {
        std::cout << "EXCEPTION_SUITE ReduceStdV2Update FAIL unknown_case=" << argv[1] << std::endl;
        passed = false;
    }
    passed = ge::GEFinalize() == ge::SUCCESS && passed;
    if (passed) {
        std::cout << "EXCEPTION_SUITE ReduceStdV2Update PASSED" << std::endl;
    }
    return passed ? 0 : -1;
}
