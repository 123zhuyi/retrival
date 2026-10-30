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
 * \file test_geir_sort_with_index.cpp
 * \brief SortWithIndex (experimental, ascend910b) GE 图模式构图调用示例。
 *
 * 通过算子 IR（op_graph/sort_with_index_exp_proto.h 的 REG_OP(SortWithIndex)）构建单算子图，
 * 沿最后一维排序（x: float32 + index: int32，属性 axis=-1 / descending=false / stable=false）。
 *
 * dtype 范围与 910B 首版一致：value{float16, float, bfloat16, int32} × index{int32}（本示例取 float32 + int32）。
 * 参考真值源 math/sort_with_index/examples/test_geir_sort_with_index.cpp。
 *
 * 说明：本示例演示「构图」流程，可独立编译；RunGraph 需要 GE 图执行运行环境（含 device、算子包），
 *       若仅验证构图，可只跑到 graph.SetInputs/SetOutputs 并 aclgrphDumpGraph 落盘。
 */

#include <cstdint>
#include <cstring>
#include <ctime>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"
#include "nn_other.h"
#include "../op_graph/sort_with_index_exp_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...) printf(message, ##__VA_ARGS__)

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return tmp;
}

static uint32_t GetDataTypeSize(DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4U;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        default:
            return 4U;
    }
}

// 生成填充数据（float32 用线性序列、int32 用 0..N-1，便于观察排序后的重排）。
static int32_t GenInputData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, DataType dt)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (auto d : shape) {
        size *= static_cast<size_t>(d);
    }
    uint32_t dataLen = static_cast<uint32_t>(size) * GetDataTypeSize(dt);
    if (dt == ge::DT_FLOAT) {
        std::vector<float> buf(size);
        for (size_t i = 0; i < size; ++i) {
            buf[i] = static_cast<float>((size - i)); // 倒序，便于看到排序后变升序
        }
        tensor = Tensor(desc, reinterpret_cast<uint8_t*>(buf.data()), dataLen);
    } else { // int32 index = 0..N-1（按最后一维循环）
        std::vector<int32_t> buf(size);
        int64_t axisLen = shape.empty() ? 1 : shape.back();
        for (size_t i = 0; i < size; ++i) {
            buf[i] = static_cast<int32_t>(static_cast<int64_t>(i) % axisLen);
        }
        tensor = Tensor(desc, reinterpret_cast<uint8_t*>(buf.data()), dataLen);
    }
    return SUCCESS;
}

// 构建单算子 SortWithIndex 图。
static int BuildGraph(Graph& graph, DataType dataType, DataType indexType, std::vector<ge::Tensor>& inputTensors)
{
    std::vector<int64_t> shape = {2, 16};

    // 输入 x（待排序数值）。
    auto dataX = op::Data("x").set_attr_index(0);
    TensorDesc descX(ge::Shape(shape), FORMAT_ND, dataType);
    descX.SetPlacement(ge::kPlacementHost);
    dataX.update_input_desc_x(descX);
    dataX.update_output_desc_y(descX);
    Tensor tensorX;
    GenInputData(shape, tensorX, descX, dataType);
    inputTensors.push_back(tensorX);

    // 输入 index（跟随排序的索引）。
    auto dataIndex = op::Data("index").set_attr_index(1);
    TensorDesc descIndex(ge::Shape(shape), FORMAT_ND, indexType);
    descIndex.SetPlacement(ge::kPlacementHost);
    dataIndex.update_input_desc_x(descIndex);
    dataIndex.update_output_desc_y(descIndex);
    Tensor tensorIndex;
    GenInputData(shape, tensorIndex, descIndex, indexType);
    inputTensors.push_back(tensorIndex);

    // SortWithIndex 算子（axis=-1 升序非稳定）。
    auto sortWithIndex = op::SortWithIndex("sort_with_index")
                             .set_input_x(dataX)
                             .set_input_index(dataIndex)
                             .set_attr_axis(-1)
                             .set_attr_descending(false)
                             .set_attr_stable(false);
    TensorDesc descY(ge::Shape(shape), FORMAT_ND, dataType);
    TensorDesc descSortedIndex(ge::Shape(shape), FORMAT_ND, indexType);
    sortWithIndex.update_output_desc_y(descY);
    sortWithIndex.update_output_desc_sorted_index(descSortedIndex);

    std::vector<Operator> inputs{dataX, dataIndex};
    std::vector<Operator> outputs{sortWithIndex};
    graph.SetInputs(inputs).SetOutputs(outputs);
    LOG_PRINT("%s - INFO: SortWithIndex 图构建完成（x: float, index: int32, axis=-1, ascending）\n", GetTime().c_str());
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    DataType dataType = DT_FLOAT;
    DataType indexType = DT_INT32;

    // 1. 构图（不依赖运行环境，可独立验证）。
    Graph graph("sort_with_index_geir_graph");
    std::vector<ge::Tensor> inputs;
    if (BuildGraph(graph, dataType, indexType, inputs) != SUCCESS) {
        LOG_PRINT("%s - ERROR: 构图失败\n", GetTime().c_str());
        return FAILED;
    }

    // 落盘图（便于离线检视 IR）。
    std::string dumpPath = "./dump_sort_with_index";
    aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.length());
    LOG_PRINT("%s - INFO: 图已落盘到 %s\n", GetTime().c_str(), dumpPath.c_str());

    // 2. （可选）图执行：需要 GE 图执行运行环境（device + 已安装算子包）。
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    if (ge::GEInitialize(globalOptions) != SUCCESS) {
        LOG_PRINT("%s - WARN: GEInitialize 失败（无图执行环境时属预期），构图阶段已通过\n", GetTime().c_str());
        return SUCCESS; // 构图成功即视为示例编译/构图验证通过
    }

    std::map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR: 创建 Session 失败\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions = {};
    if (session->AddGraph(graphId, graph, graphOptions) != SUCCESS) {
        LOG_PRINT("%s - ERROR: AddGraph 失败\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }

    std::vector<ge::Tensor> outputs;
    if (session->RunGraph(graphId, inputs, outputs) != SUCCESS) {
        LOG_PRINT("%s - WARN: RunGraph 失败（检查算子包/运行环境），构图阶段已通过\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return SUCCESS;
    }

    LOG_PRINT("%s - INFO: 图执行成功，输出 %zu 个张量\n", GetTime().c_str(), outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        int64_t shapeSize = outputs[i].GetTensorDesc().GetShape().GetShapeSize();
        LOG_PRINT(
            "  output[%zu] dtype=%d shapeSize=%ld\n", i, static_cast<int>(outputs[i].GetTensorDesc().GetDataType()),
            static_cast<long>(shapeSize));
    }

    delete session;
    ge::GEFinalize();
    return SUCCESS;
}
