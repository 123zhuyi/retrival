/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// HistogramV2 tiling UT helper — runs tiling while faking a chosen SoC per case.
//
// HistogramV2 selects its tiling template at runtime via IsRegbaseSocVersion() (RegBase ->
// HistogramV2SimtTiling, non-RegBase -> HistogramV2MembaseTiling), reading the platform NpuArch.
// The shared tiling_case_executor bakes the SoC from the compile-time BUILD_SOC_VERSION, so it can
// only exercise one template per build. This helper builds the tiling context the same way but sets
// the platform SoC from a per-case argument, so a single build (any BUILD_SOC_VERSION) can cover
// both the SIMT (arch35) and MemBase (arch32) tiling sources. It is a local copy of the shared
// DO_TILING flow — kept entirely inside histogram_v2 so no common test file is modified.

#ifndef HISTOGRAM_V2_TESTS_UT_OP_HOST_HISTOGRAM_V2_TILING_UT_UTIL_H
#define HISTOGRAM_V2_TESTS_UT_OP_HOST_HISTOGRAM_V2_TILING_UT_UTIL_H

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace histogram_v2_ut {

// Run HistogramV2 tiling, faking the given short SoC (lowercase, e.g. "ascend950" for RegBase or
// "ascend910b" for non-RegBase) so the template selection is independent of BUILD_SOC_VERSION.
// Returns true on GRAPH_SUCCESS and writes the resolved tiling key.
inline bool RunTilingWithSoc(const gert::TilingContextPara& para, const std::string& shortSocLower, uint64_t& tilingKey)
{
    tilingKey = 0;
    auto contextFaker = gert::TilingContextFaker();
    size_t inputNum = para.inputTensorDesc_.size();
    size_t outputNum = para.outputTensorDesc_.size();
    if (para.inputInstanceNum_.size() != 0 || para.outputInstanceNum_.size() != 0) {
        contextFaker.IrInstanceNum(para.inputInstanceNum_, para.outputInstanceNum_);
    } else {
        contextFaker.NodeIoNum(inputNum, outputNum);
    }

    std::vector<gert::Tensor*> inputTensors = {};
    std::vector<gert::Tensor*> outputTensors = {};
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive = {};
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorsKeepAlive = {};
    for (size_t index = 0; index < inputNum; index++) {
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
            para.inputTensorDesc_[index].shape_,
            gert::StorageFormat(para.inputTensorDesc_[index].format_, para.inputTensorDesc_[index].format_,
                                gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, para.inputTensorDesc_[index].dtype_,
            para.inputTensorDesc_[index].isConst_ ? para.inputTensorDesc_[index].constValue_ : nullptr);
        inputTensors.push_back(curTensor.get());
        inputTensorsKeepAlive.push_back(std::move(curTensor));
    }
    for (size_t index = 0; index < outputNum; index++) {
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(
            para.outputTensorDesc_[index].shape_,
            gert::StorageFormat(para.outputTensorDesc_[index].format_, para.outputTensorDesc_[index].format_,
                                gert::ExpandDimsType()),
            gert::TensorPlacement::kOnHost, para.outputTensorDesc_[index].dtype_,
            para.outputTensorDesc_[index].isConst_ ? para.outputTensorDesc_[index].constValue_ : nullptr);
        outputTensors.push_back(curTensor.get());
        outputTensorsKeepAlive.push_back(std::move(curTensor));
    }
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);
    for (auto& attrInfo : para.attrs_) {
        switch (attrInfo.attr_.type_) {
            case Ops::Math::AnyValue::ValueType::VT_BOOL:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Math::AnyValue::ValueType::VT_INT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));
                break;
            case Ops::Math::AnyValue::ValueType::VT_FLOAT:
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));
                break;
            default:
                break;
        }
    }

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto tilingData = gert::TilingData::CreateCap(para.tilingDataSize_);
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);
    auto contextHolder = contextFaker.SetOpType(para.opName_.c_str())
                             .CompileInfo(para.compileInfo_)
                             .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                             .TilingData(tilingData.get())
                             .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()))
                             .Build();

    // Resolve the faked SoC to its upper-case short name + NpuArch, then populate the platform
    // resources directly (no JSON parsing) so this header needs no extra include dirs. Values match
    // what the shared harness derives from its hardware_info string for the same ubSize/coreNum.
    std::map<std::string, std::string> socToUpper = {{"ascend910b", "Ascend910B"}, {"ascend910_93", "Ascend910_93"},
                                                     {"ascend950", "Ascend950"},   {"ascend310p", "Ascend310P"},
                                                     {"ascend910", "Ascend910"},   {"ascend310b", "Ascend310B"}};
    std::map<std::string, std::string> socToArch = {{"Ascend310P", "2002"},
                                                    {"Ascend910B", "2201"},
                                                    {"Ascend910_93", "2201"},
                                                    {"Ascend950", "3510"},
                                                    {"Ascend910", "1001"}};
    std::string upperSoc = socToUpper.count(shortSocLower) ? socToUpper[shortSocLower] : shortSocLower;

    std::map<std::string, std::string> socInfos = {
        {"ai_core_cnt", std::to_string(para.coreNum_)}, {"l2_size", "33554432"}, {"core_type_list", "AICore"}};
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", std::to_string(para.ubSize_)},
                                                     {"l0_a_size", "65536"},
                                                     {"l0_b_size", "65536"},
                                                     {"l0_c_size", "131072"},
                                                     {"l1_size", "524288"},
                                                     {"bt_size", "0"},
                                                     {"load3d_constraints", "1"},
                                                     {"cube_freq", "cube_freq"}};
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socversions = {{"NpuArch", socToArch.count(upperSoc) ? socToArch[upperSoc] : ""},
                                                      {"Short_SoC_version", upperSoc}};

    auto tilingContext = contextHolder.GetContext();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    if (spaceRegistry == nullptr) {
        throw std::invalid_argument("not found spaceRegistry");
    }
    auto functionStruct = spaceRegistry->GetOpImpl(para.opName_.c_str());
    if (functionStruct == nullptr) {
        throw std::invalid_argument("not found " + para.opName_);
    }
    auto tilingRet = functionStruct->tiling(tilingContext);
    if (tilingRet != ge::GRAPH_SUCCESS) {
        return false;
    }
    tilingKey = tilingContext->GetTilingKey();
    return true;
}

} // namespace histogram_v2_ut

#endif // HISTOGRAM_V2_TESTS_UT_OP_HOST_HISTOGRAM_V2_TILING_UT_UTIL_H
