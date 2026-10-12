/**
 * @file test_geir_bitwise_not.cpp
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ============================================================================
 * BitwiseNot (experimental/math/bitwise_not) graph (GE IR) 调用示例。
 *
 * 本示例构造一个单算子图 BitwiseNot（对齐 op_graph proto REG_OP(BitwiseNot)），
 * 经 GE Session AddGraph/RunGraph 在图模式下拉起本 experimental native kernel。
 * OpType "BitwiseNot" 与 op_graph/bitwise_not_proto.h 的 REG_OP(BitwiseNot) 一致；
 * 该算子单输入 x / 单输出 y，dtype/shape 相等（此处用 INT32），ND，无属性。
 *
 * 图模式经 GE 直接调度 OpType "BitwiseNot" 的 kernel（与 aclnn 入口为同一份 native kernel）。
 *
 * 运行环境说明:
 *   - 编译：依赖 GE / graph 头与库（libgraph / libge_runner / libge_compiler / libgraph_base），
 *     与 `bash build.sh --run_example bitwise_not graph --experimental` 的 compile_graph_example 一致。
 *   - 运行：需完整的 GE 图执行运行环境（GEInitialize/Session/RunGraph）+ 已安装本算子自定义包
 *     （ASCEND_CUSTOM_OPP_PATH 指向安装路径）。若仅做编译验收，
 *     可用配套 run.sh --geir-build-only 仅编译本可执行（不实跑图）。
 * ============================================================================
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
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

// op::Data（图输入占位 placeholder）来自 GE 内置算子原型头。
#include "array_ops.h"
// 本算子自定义 proto（REG_OP(BitwiseNot)），与 op_graph 注册一致 —— 提供 op::BitwiseNot。
#include "../op_graph/bitwise_not_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

static std::string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64] = {0};
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S", localtime(&timep));
    return std::string(tmp);
}

// 生成 INT32 输入数据：x[i] = baseValue + i（确保多元素有区分度）。
static int32_t GenInt32Data(const std::vector<int64_t> &shapes, Tensor &tensor, TensorDesc &desc, int32_t baseValue)
{
    desc.SetRealDimCnt(static_cast<uint32_t>(shapes.size()));
    size_t size = 1;
    for (auto d : shapes) {
        size *= static_cast<size_t>(d);
    }
    std::vector<int32_t> buf(size);
    for (size_t i = 0; i < size; ++i) {
        buf[i] = baseValue + static_cast<int32_t>(i);
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t *>(buf.data()), size * sizeof(int32_t));
    return SUCCESS;
}

// 构造单算子图：placeholder(Data) -> BitwiseNot -> y。
static int BuildBitwiseNotGraph(DataType dtype, std::vector<Tensor> &graphInputs, std::vector<Operator> &inputs,
                                  std::vector<Operator> &outputs, Graph &graph)
{
    Status ret = SUCCESS;

    // 输入 placeholder（图运行期外部喂数据）。
    std::vector<int64_t> xShape = {2, 4, 4, 4};
    TensorDesc xDesc(ge::Shape(xShape), FORMAT_ND, dtype);
    xDesc.SetPlacement(ge::kPlacementHost);

    auto placeholder = op::Data("x_placeholder").set_attr_index(0);
    placeholder.update_input_desc_x(xDesc);
    placeholder.update_output_desc_y(xDesc);

    // 喂入图运行期数据（INT32）。
    Tensor xTensor;
    ret = GenInt32Data(xShape, xTensor, xDesc, 1);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());
        return FAILED;
    }
    graphInputs.push_back(xTensor);

    // BitwiseNot 节点（OpType 与 REG_OP(BitwiseNot) 一致）。
    auto bitwiseNot = op::BitwiseNot("bitwise_not_0");
    bitwiseNot.set_input_x(placeholder);

    TensorDesc yDesc(ge::Shape(xShape), FORMAT_ND, dtype);
    bitwiseNot.update_input_desc_x(xDesc);
    bitwiseNot.update_output_desc_y(yDesc);

    graph.AddOp(placeholder);
    graph.AddOp(bitwiseNot);

    inputs.push_back(placeholder);
    outputs.push_back(bitwiseNot);
    return SUCCESS;
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    const char *graphName = "bitwise_not_geir_test";
    Graph graph(graphName);
    std::vector<Tensor> graphInputs;
    std::vector<Operator> inputs;
    std::vector<Operator> outputs;

    DataType dtype = DT_INT32;  // 6 dtype 之一；改 dtype 同时改 GenXxxData 即可

    LOG_PRINT("%s - INFO - [XIR]: GEInitialize\n", GetTime().c_str());
    std::map<AscendString, AscendString> globalOptions = {
        {"ge.exec.deviceId", "0"},
        {"ge.graphRunMode", "1"},
    };
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }

    ret = BuildBitwiseNotGraph(dtype, graphInputs, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Build graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    LOG_PRINT("%s - INFO - [XIR]: create Session\n", GetTime().c_str());
    std::map<AscendString, AscendString> buildOptions;
    ge::Session *session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        LOG_PRINT("%s - ERROR - [XIR]: create Session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graphId = 0;
    std::map<AscendString, AscendString> graphOptions;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    LOG_PRINT("%s - INFO - [XIR]: RunGraph\n", GetTime().c_str());
    std::vector<Tensor> graphOutputs;
    ret = session->RunGraph(graphId, graphInputs, graphOutputs);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: RunGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    // 打印部分输出并与 CPU golden(~x) 对拍。
    for (size_t i = 0; i < graphOutputs.size(); ++i) {
        const int32_t *out = reinterpret_cast<const int32_t *>(graphOutputs[i].GetData());
        int64_t numel = graphOutputs[i].GetTensorDesc().GetShape().GetShapeSize();
        int64_t showN = numel < 8 ? numel : 8;
        for (int64_t j = 0; j < showN; ++j) {
            int32_t golden = static_cast<int32_t>(~(static_cast<int32_t>(1 + j)));  // x[j]=1+j
            LOG_PRINT("    out[%ld]=%d  golden(~x)=%d  %s\n", j, out[j], golden, (out[j] == golden) ? "OK" : "MISMATCH");
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    LOG_PRINT("%s - INFO - [XIR]: done\n", GetTime().c_str());
    return SUCCESS;
}
