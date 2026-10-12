/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random_standard_normal_fusion_pass.cpp
 * \brief RandomStandardNormal fusion pass (RandomStandardNormal --> RandomStandardNormalV2)
 *
 * Pattern:
 *            shape                                       shape          offset(const 0)
 *              |                                           |            /
 *       RandomStandardNormal          ==>         RandomStandardNormalV2
 *              |                                           |            \
 *            output                                      output         offset
 *
 * The key transformation:
 * - RandomStandardNormal has 1 input: shape, 1 output: y
 * - RandomStandardNormalV2 has 2 inputs: shape + offset, 2 outputs: y + offset
 * - An offset constant (value 0, dtype int64) is created as the additional input
 * - Attributes seed, seed2, dtype are transferred to the new node
 */

#include <vector>
#include <string>
#include <set>
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "ge/ge_utils.h"
#include "log/log.h"
#include "version/ge-compiler_version.h"
#include "random_standard_normal_fusion_pass.h"
#include "common/inc/op_graph/fusion_pass/fusion_pass_common.h"

using namespace ge;
using namespace ge::fusion;
using namespace fe;

namespace ops {

// D1 scenario: uses kCompatibleInherited stage (9.0.0+).
// Strategy: compile-time macro guard + runtime version check + overall silence.
#define GE_COMPILER_VERSION_900 90000000

namespace {
#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
CustomPassStage GetRandomStandardNormalFusionPassStage()
{
    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version >= GE_COMPILER_VERSION_900) {
        return CustomPassStage::kCompatibleInherited;
    }
    return CustomPassStage::kBeforeInferShape;
}
#endif
} // namespace

static const std::string kPassName = "RandomStandardNormalFusionPass";
static const int64_t kCaptureIdx = 0L;

static const std::set<ge::DataType> kSupportedDtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static void InitOffsetVariable(const es::EsTensorHolder& rOffset, const TensorDesc& offsetDesc)
{
    auto varNode = rOffset.GetProducer();
    if (varNode != nullptr) {
        varNode->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNode->SetAttr("init_value", initTensor);
    }
}

static void GetInputsInfo(const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
                          std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

std::vector<PatternUniqPtr> RandomStandardNormalFusionPass::Patterns()
{
    OP_LOGD(kPassName.c_str(), "Enter Patterns for RandomStandardNormalFusionPass");
    std::vector<PatternUniqPtr> patternGraphs;

    if (!IsTargetVersion()) {
        OP_LOGD(kPassName.c_str(), "GE runtime version < %d, skip pass.", GE_COMPILER_VERSION_910);
        return patternGraphs;
    }

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto shape = graphBuilder.CreateInput(0);
    auto output = es::RandomStandardNormal(shape, DT_FLOAT, 0, 0);
    auto graph = graphBuilder.BuildAndReset({output});

    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*output.GetProducer(), 0});

    patternGraphs.emplace_back(std::move(pattern));
    return patternGraphs;
}

bool RandomStandardNormalFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter MeetRequirements for RandomStandardNormalFusionPass");

    int32_t version = 0;
    if (aclsysGetVersionNum) {
        aclsysGetVersionNum(const_cast<char*>("ge_compiler"), &version);
    }
    if (version < GE_COMPILER_VERSION_900) {
        OP_LOGD(kPassName.c_str(), "GE runtime version %d < 90000000, skip pass.", version);
        return false;
    }

    // 1. Platform check - regbase architectures only
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend950") {
        OP_LOGD(kPassName.c_str(), "Platform %s is not supported, skip.", soc.c_str());
        return false;
    }

    // 2. Get captured RandomStandardNormal node
    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdx, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get captured tensor.");
        return false;
    }

    // 3. Check dtype attribute
    ge::DataType dtype = ge::DT_FLOAT;
    if (nodeIo.node.GetAttr("dtype", dtype) != GRAPH_SUCCESS) {
        OP_LOGD(kPassName.c_str(), "Failed to get dtype attribute, using default DT_FLOAT.");
    }
    OP_LOGI(kPassName.c_str(), "[MeetReq] V1 dtype from attr: %d (0=float32, 1=float16, 27=bf16, 11=double)",
            static_cast<int32_t>(dtype));
    if (kSupportedDtypes.count(dtype) == 0) {
        OP_LOGD(kPassName.c_str(), "RandomStandardNormalV2 dtype only supports float32/float16/bfloat16, got %d, skip.",
                static_cast<int32_t>(dtype));
        return false;
    }

    return true;
}

std::unique_ptr<Graph> RandomStandardNormalFusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGD(kPassName.c_str(), "Enter Replacement for RandomStandardNormalFusionPass");

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    if (subgraphInputs.empty()) {
        OP_LOGE(kPassName.c_str(), "SubgraphInputs is empty, cannot get input info.");
        return nullptr;
    }

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    NodeIo nodeIo;
    if (matchResult->GetCapturedTensor(kCaptureIdx, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to GetCaptured tensor in Replacement.");
        return nullptr;
    }

    DataType dtype = DT_FLOAT;
    nodeIo.node.GetAttr("dtype", dtype);
    int64_t dtypeInt = static_cast<int64_t>(dtype);
    OP_LOGI(kPassName.c_str(), "[Replacement] V1 dtype enum: %d, converted to V2 dtypeInt: %ld",
            static_cast<int32_t>(dtype), dtypeInt);

    int64_t seed = 0;
    nodeIo.node.GetAttr("seed", seed);

    int64_t seed2 = 0;
    nodeIo.node.GetAttr("seed2", seed2);

    auto replaceGraphBuilder = es::EsGraphBuilder("replacement");

    auto rShape = replaceGraphBuilder.CreateInput(0, "shape", inputDtypes[0], inputFormats[0],
                                                  inputShapes[0].GetDims());

    AscendString nodeName;
    if (nodeIo.node.GetName(nodeName) != GRAPH_SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Failed to get node name.");
        return nullptr;
    }
    std::string varName = std::string(nodeName.GetString()) + "/offsetVariable";

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);

    auto rOffset = replaceGraphBuilder.CreateVariable(1, varName.c_str());
    InitOffsetVariable(rOffset, offsetDesc);

    auto v2Output = es::RandomStandardNormalV2(rShape, rOffset, dtypeInt, seed, seed2);
    GNode v2NodePtr = *v2Output.y.GetProducer();

    TensorDesc shapeInputDesc(inputShapes[0], inputFormats[0], inputDtypes[0]);
    v2NodePtr.UpdateInputDesc(0, shapeInputDesc);
    v2NodePtr.UpdateInputDesc(1, offsetDesc);

    TensorDesc outputYDesc;
    nodeIo.node.GetOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(0, outputYDesc);
    v2NodePtr.UpdateOutputDesc(1, offsetDesc);

    GraphUniqPtr replaceGraph = replaceGraphBuilder.BuildAndReset({v2Output.y});
    return replaceGraph;
}

#if GE_COMPILER_VERSION_NUM >= GE_COMPILER_VERSION_900
REG_FUSION_PASS(RandomStandardNormalFusionPass).Stage(GetRandomStandardNormalFusionPassStage());
#else
REG_FUSION_PASS(RandomStandardNormalFusionPass).Stage(CustomPassStage::kBeforeInferShape);
#endif

} // namespace ops
