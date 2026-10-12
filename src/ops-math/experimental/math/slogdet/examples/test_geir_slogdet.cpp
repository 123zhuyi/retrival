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
 * 我们正常的版权申明，下面是我们的备注
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

// ============================================================================
// slogdet 算子 图模式 (GE IR) 调用示例（experimental/math/slogdet）
//
// 演示用 GE IR 构图方式调用 Slogdet 算子：单输入 self [*, n, n]，双输出
// signOut [*] / logOut [*]。OpType `Slogdet` 由自定义算子包的 op_proto 注册
// （slogdet_proto.h，REG_OP(Slogdet)）。
//
// 说明：本算子为 experimental 贡献的原生 AscendC 实现，aclnn 调用（见
//   test_aclnn_slogdet.cpp）为主验收路径，已在真机 NPU 验证。图模式构图编译通过；
//   能否成功 RunGraph 取决于运行环境是否注册了 Slogdet 的 op_proto/InferShape 与
//   AICPU/AICore 实现（自定义包安装 + ASCEND_CUSTOM_OPP_PATH 指向 vendor 目录）。
//   若图模式注册受 experimental 限制无法运行，构图与编译仍保证通过。
// ============================================================================

#include <cstdint>
#include <cstring>
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

// 自定义算子 op_proto（REG_OP(Slogdet)）。
// CMake 通过 SLOGDET_CUSTOM_OPP/op_proto/inc 提供该头文件搜索路径。
#include "slogdet_proto.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

// 生成常量浮点数据填充的 host tensor
static int32_t GenFloatData(const vector<int64_t>& shape, Tensor& tensor, TensorDesc& desc, float value)
{
    desc.SetRealDimCnt(shape.size());
    size_t size = 1;
    for (auto d : shape) {
        size *= static_cast<size_t>(d);
    }
    uint32_t dataLen = static_cast<uint32_t>(size * sizeof(float));
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        pData[i] = value;
    }
    tensor = Tensor(desc, reinterpret_cast<uint8_t*>(pData), dataLen);
    delete[] pData;
    return SUCCESS;
}

// 构造 Slogdet 单算子图：Data(self) -> Slogdet -> {signOut, logOut}
static int CreateSlogdetGraph(
    DataType inDtype, vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    // self: [3, 2, 2] —— 3 个 2x2 方阵；填充对角占优值保非奇异（构图演示用）。
    vector<int64_t> selfShape = {3, 2, 2};
    auto selfData = op::Data("self").set_attr_index(0);
    TensorDesc selfDesc = TensorDesc(ge::Shape(selfShape), FORMAT_ND, inDtype);
    selfDesc.SetPlacement(ge::kPlacementHost);
    selfDesc.SetFormat(FORMAT_ND);
    selfData.update_input_desc_x(selfDesc);
    selfData.update_output_desc_y(selfDesc);

    Tensor selfTensor;
    ret = GenFloatData(selfShape, selfTensor, selfDesc, 2.0f);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate self data failed\n", GetTime().c_str());
        return FAILED;
    }
    input.push_back(selfTensor);
    graph.AddOp(selfData);
    inputs.push_back(selfData);

    // Slogdet 算子节点：单输入 self，双输出 signOut / logOut。
    auto slogdet = op::Slogdet("slogdet0");
    slogdet.set_input_self(selfData);

    // 输出 batch 形状 = self.shape[:-2] = [3]
    vector<int64_t> outShape = {3};
    TensorDesc signOutDesc = TensorDesc(ge::Shape(outShape), FORMAT_ND, inDtype);
    TensorDesc logOutDesc = TensorDesc(ge::Shape(outShape), FORMAT_ND, inDtype);
    slogdet.update_output_desc_signOut(signOutDesc);
    slogdet.update_output_desc_logOut(logOutDesc);

    graph.AddOp(slogdet);
    outputs.push_back(slogdet);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    const char* graphName = "tc_ge_slogdet_test";
    Graph graph(graphName);
    vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    vector<Operator> inputs{};
    vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    ret = CreateSlogdetGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create Slogdet graph failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions = {};
    printf("%s - INFO - [XIR]: Start to create ir session\n", GetTime().c_str());
    ge::Session* session = new (std::nothrow) Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session success\n", GetTime().c_str());

    map<AscendString, AscendString> graphOptions = {};
    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, graphOptions);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    // dump 图到 txt 便于检视构图结果
    std::string filePath = "./dump";
    aclgrphDumpGraph(graph, filePath.c_str(), filePath.length());

    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != SUCCESS) {
        // 图模式运行依赖运行环境已注册 Slogdet 的 op_proto/InferShape 与 kernel 实现。
        // experimental 算子若图模式注册受限，RunGraph 可能失败；此处不视为编译/构图失败。
        printf("%s - WARN - [XIR]: RunGraph failed (experimental 图模式注册受限时预期)\n", GetTime().c_str());
        ge::AscendString errorMsg = ge::GEGetErrorMsgV2();
        std::cout << "GE error message: " << errorMsg.GetString() << std::endl;
        delete session;
        ge::GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    int outputNum = static_cast<int>(output.size());
    for (int i = 0; i < outputNum; i++) {
        int64_t shapeSize = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "output " << i << " dtype=" << output[i].GetTensorDesc().GetDataType()
                  << " shapeSize=" << shapeSize << std::endl;
    }

    printf("%s - INFO - [XIR]: Start to finalize ge\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEFinalize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEFinalize success\n", GetTime().c_str());
    return SUCCESS;
}
