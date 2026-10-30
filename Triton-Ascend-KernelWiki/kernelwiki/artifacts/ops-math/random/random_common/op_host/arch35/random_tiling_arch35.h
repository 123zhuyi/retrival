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
 * \file random_tiling_arch35.h
 * \brief
 */
#ifndef RANDOM_TILING_ARCH35_H
#define RANDOM_TILING_ARCH35_H

#include <unordered_map>
#include <vector>
#include <string>
#include <set>
#include "random/random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

namespace optiling {

// Tensor slice state constants for large tensor partitioning
constexpr uint32_t MAX_TENSOR_DIMS = 8;
constexpr uint64_t INDEX_32BIT_LIMIT = 2147483647;
constexpr uint32_t SIMT_THREAD_GROUP_SIZE = 256;
constexpr uint32_t MAX_THREADS_PER_AIC = 2048;
constexpr uint32_t AIC_CLUSTER_COUNT = 78;
constexpr uint32_t MAX_PRNG_COUNTER_INCR = 4;

struct RandomOperatorCompileInfo {
    int64_t totalCoreNum = 0;
    int64_t ubSize = 0;
};

ge::graphStatus ExtractTensorValue(const gert::TilingContext* context, const int64_t constIdx, gert::Shape& constShape);

ge::graphStatus RandomTilingParseArch35(gert::TilingParseContext* context, const std::string& operatorName);

struct TensorSliceState {
    int64_t shape[MAX_TENSOR_DIMS] = {0};
    int64_t strides[MAX_TENSOR_DIMS] = {0};
    int64_t ndim = 0;
    int64_t numel = 0;
    int64_t elementSize = 0;
    int64_t gmOffset = 0;

    int64_t GetMaxOffsetBytes() const;
    bool Is32bitIndexable() const;
    int64_t GetDimToSplit() const;
    void ReduceDimExtent(int64_t dim, int64_t start, int64_t size);
    void PartitionDim(int64_t dim, TensorSliceState& other);
};

ge::graphStatus InitTensorSliceState(
    TensorSliceState& state,
    const gert::Shape& outputTensor,
    int64_t outputSize,
    ge::DataType outputDtype);

ge::graphStatus CalcSplitBlocks(
    TensorSliceState& state,
    RandomUnifiedSimtTilingDataStruct& simtTilingData);

ge::graphStatus CalcExecutionPoliciesForBlocks(
    RandomUnifiedSimtTilingDataStruct& simtTilingData,
    uint32_t unrollFactor);

// 输入输出Tensor校验规则配置
struct TensorCheckRule {
    // 输入-1表示不校验
    std::set<ge::DataType> dtypeSet; // 允许的dtype列表
    int64_t shapeSize = -1;          // 要求的shapeSize
    std::set<int64_t> dimNumSet;     // 允许的维度数，为空时不做检查
    // 扩展校验（可按需添加）
    std::function<bool(gert::TilingContext*)> customCheck; // 自定义校验逻辑
};

// 算子私有配置
enum class RandomKernelMode {SIMD, SIMT};

struct OpTilingConfig {
    std::unordered_map<int32_t, TensorCheckRule> inputCheckRules;
    std::unordered_map<int32_t, TensorCheckRule> optionalInputCheckRules;
    std::unordered_map<int32_t, TensorCheckRule> outputCheckRules;
    // // 属性校验规则（key: 属性名，value: 自定义校验函数）
    std::unordered_map<int32_t, std::function<bool(gert::TilingContext*)>> attrCheckRules;

    // 字段映射函数（算子仅需实现这些函数，优先查看TilingUtils中公共代码是否可复用）
    std::function<ge::graphStatus(gert::TilingContext*, int64_t&)> getOutputSize;
    std::function<ge::graphStatus(gert::TilingContext*, uint32_t[2], uint32_t[4])> getKeyAndCounter;
    std::function<ge::graphStatus(gert::TilingContext*, int64_t&)> getBufferNum;
    std::function<ge::graphStatus(gert::TilingContext*, int64_t&, int64_t&)> getSeedAndOffset;

    // 启动相关
    bool isNeedSyncAll = false;

    // Dcache相关  默认为0 表示kernel代码没有simtvf
    int64_t DcacheSize = 0;
    int64_t sharedTmpBufSize = 0;
    int64_t coreAlignSize = 4;
    int64_t ubAlignSize = 0;
    int64_t keepProbNum = 0;
    uint32_t unrollFactor = 4;
    bool enableSplitBlocks = false;
    uint32_t splitOutputIndex = 0;

    RandomKernelMode kernelMode = RandomKernelMode::SIMD;
};

class RandomTilingArch35
{
public:
    explicit RandomTilingArch35(gert::TilingContext* context, const OpTilingConfig& config);
    virtual ~RandomTilingArch35() = default;

    ge::graphStatus DoTiling();

protected:
    virtual ge::graphStatus UniqueProcess()
    {
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus BeforeProcess()
    {
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckInputsOutputsAndAttrs();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus FillUnifiedTilingData();
    ge::graphStatus FillUnifiedSimtTilingData();
    ge::graphStatus DoBlockTiling();
    virtual ge::graphStatus DoSimtBlockTiling();
    ge::graphStatus DoUbTiling();
    ge::graphStatus CalcTilingKeyAndWorkspace();
    ge::graphStatus WriteBackToContext();

    ge::graphStatus CheckTensor(const gert::CompileTimeTensorDesc* tensorDesc, const gert::Shape& tensorShape, const TensorCheckRule& rule, const std::string& tensorName);

    gert::TilingContext* context_ = nullptr;
    OpTilingConfig config_;
    RandomUnifiedTilingDataStruct tilingData_;
    RandomUnifiedSimtTilingDataStruct simtTilingData_;

    std::string opName_;
    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t bufNum_= 0;
};

} // namespace optiling
#endif // RANDOM_TILING_ARCH35_H