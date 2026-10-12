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
 * \file stateless_randperm_tiling_arch35.cpp
 * \brief
 */

#include <vector>
#include <string>
#include <stack>
#include "base/context_builder/op_tiling_context_builder.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_host/tiling_base_util.h"
#include "stateless_randperm_tiling_for_sort.h"
#include "stateless_randperm_tiling_arch35.h"

namespace optiling {
using namespace ge;
using namespace Ops::Base;

static constexpr int64_t SIMT_THREAD_NUM_512 = 512;    // SIMT启用的线程数
static constexpr int64_t SIMT_THREAD_NUM_2048 = 2048;  // SIMT启用的线程数
static constexpr int64_t SIMT_DCACHE_SIZE = 32 * 1024; // 预留32KB作为SIMT的dcache
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int32_t BITS_8 = 8;
static constexpr int32_t BITS_16 = 16;
static constexpr int32_t BITS_32 = 32;
static constexpr int32_t BITS_64 = 64;
static constexpr double CONST_12 = 12;
static constexpr double CONST_6 = 6;

static const std::set<ge::DataType> OUTPUT_DTYPE = {
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_UINT8, ge::DT_INT8, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
};

static const std::map<ge::DataType, int64_t> maxValueMap = {
    {ge::DT_INT64, std::numeric_limits<int64_t>::max()},
    {ge::DT_INT32, std::numeric_limits<int32_t>::max()},
    {ge::DT_INT16, std::numeric_limits<int16_t>::max()},
    {ge::DT_UINT8, std::numeric_limits<uint8_t>::max()},
    {ge::DT_INT8, std::numeric_limits<int8_t>::max()},
    {ge::DT_FLOAT, std::numeric_limits<float>::max()},
    {ge::DT_FLOAT16, static_cast<int64_t>((2.0 - std::exp2(-10.0)) * std::exp2(15.0))},
    {ge::DT_BF16, static_cast<int64_t>((2.0 - std::exp2(-7.0)) * std::exp2(127.0))},
};

// 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
ge::graphStatus StatelessRandpermTiling::GetPlatformInfo()
{
    auto compileInfo = static_cast<const StatelessRandpermCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);

    totalCoreNum_ = static_cast<int64_t>(compileInfo->totalCoreNum);
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(opName_, "UB size is invalid."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= SIMT_DCACHE_SIZE,
                OP_LOGE(opName_, "UB size %ld bytes must be greater than simt dcache size %ld bytes, please check.",
                        ubSize_, SIMT_DCACHE_SIZE),
                return ge::GRAPH_FAILED);
    ubSize_ -= SIMT_DCACHE_SIZE;
    OP_LOGI(opName_,
            "StatelessRandpermTiling::GetPlatformInfo ubSize_= %ld, simt dcache size = %ld, totalCoreNum_= %ld",
            ubSize_, SIMT_DCACHE_SIZE, totalCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandpermTiling::GetAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    // layout is not used currently

    // dtype
    auto dtypePtr = attrs->GetAttrPointer<ge::DataType>(ATTR_IDX_DTYPE);
    if (dtypePtr == nullptr) {
        attrOutDtype_ = ge::DT_INT64;
        OP_LOGW(opName_, "[attr]dtype is not provided, set to default value %s.",
                Ops::Base::ToString(attrOutDtype_).c_str());
    } else {
        attrOutDtype_ = *dtypePtr;
    }
    if (OUTPUT_DTYPE.find(attrOutDtype_) == OUTPUT_DTYPE.end()) {
        std::string valueStr = ToString(attrOutDtype_);
        std::string reasonMsg = "[attr]dtype only support int64, int32, int16, int8, float32, float16, bfloat16";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "attr dtype", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// INPUT
ge::graphStatus StatelessRandpermTiling::GetInputN()
{
    // 校验 n 的类型
    auto nDesc = context_->GetInputDesc(INPUT_IDX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nDesc);
    ge::DataType nDtype = nDesc->GetDataType();
    if (nDtype != ge::DT_INT64) {
        std::string valueStr = ToString(nDtype);
        std::string reasonMsg = "input n dtype should be int64";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto nShape = context_->GetInputShape(INPUT_IDX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nShape);
    auto nStorageShape = Ops::Base::EnsureNotScalar(nShape->GetStorageShape());
    int64_t nShapeSize = nStorageShape.GetShapeSize();
    if (nShapeSize != 1) {
        std::string valueStr = std::to_string(nShapeSize);
        std::string reasonMsg = "input n shapeSize should equal 1";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto nTensor = context_->GetRequiredInputTensor(INPUT_IDX_N);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nTensor);
    auto nPtr = nTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, nPtr);
    n_ = *(nPtr);

    // 值校验
    auto maxIter = maxValueMap.find(attrOutDtype_);
    if (maxIter == maxValueMap.end()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "attr dtype", ToString(attrOutDtype_).c_str(),
                                              "attr dtype not supported, must be in [DT_INT64, DT_INT32, DT_INT16, "
                                              "DT_UINT8, DT_INT8, DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return ge::GRAPH_FAILED;
    }
    if (n_ > maxIter->second) {
        std::string valueStr = std::to_string(n_);
        std::string reasonMsg = "input n must not be greater than max value of [attr]dtype";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (n_ < 0) {
        std::string valueStr = std::to_string(n_);
        std::string reasonMsg = "input n must be non-negative";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    nIsInt32_ = (n_ <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
    if (nIsInt32_ != true) {
        std::string valueStr = std::to_string(n_);
        std::string reasonMsg = "n currently only supports values <= int32 max";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandpermTiling::GetInputSeed()
{
    // 校验 seed 类型
    auto seedDesc = context_->GetInputDesc(INPUT_IDX_SEED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedDesc);
    ge::DataType seedDtype = seedDesc->GetDataType();
    if (seedDtype != ge::DT_INT64) {
        std::string valueStr = ToString(seedDtype);
        std::string reasonMsg = "input seed dtype should be int64";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input tensor seed", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto seedShape = context_->GetInputShape(INPUT_IDX_SEED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedShape);
    auto seedStorageShape = Ops::Base::EnsureNotScalar(seedShape->GetStorageShape());
    int64_t seedShapeSize = seedStorageShape.GetShapeSize();
    if (seedShapeSize != 1) {
        std::string valueStr = std::to_string(seedShapeSize);
        std::string reasonMsg = "input seed shapeSize should equal 1";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "input tensor seed", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto seedTensor = context_->GetInputTensor(INPUT_IDX_SEED);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedTensor);
    const uint64_t* seedPtr = seedTensor->GetData<uint64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, seedPtr);
    uint64_t seed = *(seedPtr);
    key_[0] = static_cast<uint32_t>(seed);
    key_[1] = static_cast<uint32_t>(seed >> BITS_32);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandpermTiling::GetInputOffset()
{
    // 校验 offset 类型
    auto offsetDesc = context_->GetInputDesc(INPUT_IDX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetDesc);
    ge::DataType offsetDtype_ = offsetDesc->GetDataType();
    if (offsetDtype_ != ge::DT_INT64) {
        std::string valueStr = ToString(offsetDtype_);
        std::string reasonMsg = "input offset dtype should be int64";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "input tensor offset", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto offsetShape = context_->GetInputShape(INPUT_IDX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetShape);
    auto offsetStorageShape = Ops::Base::EnsureNotScalar(offsetShape->GetStorageShape());
    int64_t offsetShapeSize = offsetStorageShape.GetShapeSize();
    if (offsetShapeSize != 1) {
        std::string valueStr = std::to_string(offsetShapeSize);
        std::string reasonMsg = "input offset shapeSize should equal 1";
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "input tensor offset", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    auto offsetTensor = context_->GetInputTensor(INPUT_IDX_OFFSET);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetTensor);
    const uint64_t* offsetPtr = offsetTensor->GetData<uint64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetPtr);
    offset_ = *(offsetPtr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus StatelessRandpermTiling::GetOutputY()
{
    auto outDesc = context_->GetOutputDesc(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outDesc);
    auto outDtype = outDesc->GetDataType();
    if (OUTPUT_DTYPE.count(outDtype) == 0) {
        std::string valueStr = ToString(outDtype);
        std::string reasonMsg = "output y dtype should be in int64, int32, int16, int8, float32, float16, bfloat16";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "output tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // shape校验
    auto yShape = context_->GetOutputShape(OUTPUT_IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yShape);
    auto yStorageShape = Ops::Base::EnsureNotScalar(yShape->GetStorageShape());
    size_t yDimNum = yStorageShape.GetDimNum();
    if (yDimNum != 1) {
        std::string valueStr = std::to_string(yDimNum);
        std::string reasonMsg = "output y dim must be 1";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "output tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // 值校验
    int64_t yLen = yStorageShape.GetDim(0);
    if (yLen != n_) {
        std::string valueStr = std::to_string(yLen) + " and " + std::to_string(n_);
        std::string reasonMsg = "output y shape[0] must be equal to n";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "output tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    auto iter = maxValueMap.find(outDtype);
    OP_CHECK_IF(iter == maxValueMap.end(),
                OP_LOGE(opName_, "max value of y.dtype does not exist in the map, please check."),
                return ge::GRAPH_FAILED);
    const double nMax = iter->second;
    if (static_cast<double>(yLen) > nMax) {
        std::string valueStr = std::to_string(yLen);
        std::string reasonMsg = "output y shape[0] is over the max value of y dtype";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "output tensor y", valueStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 2、获取INPUT/OUTPUT/ATTR信息
ge::graphStatus StatelessRandpermTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(GetAttrs(), OP_LOGE(opName_, "GetAttrs failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputN() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "GetInputN failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeed() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "GetInputSeed failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOffset() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "GetInputOffset failed!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetOutputY() != ge::GRAPH_SUCCESS, OP_LOGE(opName_, "GetOutputY failed!"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void StatelessRandpermTiling::PhiloxRandomComputeBits()
{
    // compute bits 固定公式
    const double log_threshold_12 = std::log(0.9) * 12;
    double nd = static_cast<double>(n_);
    randomBits_ = std::min(BITS_64,
                           static_cast<int>(std::ceil(std::log2(nd - (CONST_6 * nd * nd + 1) / log_threshold_12))));
    randomIsInt32_ = randomBits_ <= BITS_32;
    // sort优化措施，低于32bit的场景，random数据类型可以更细化，减少基数排序轮数
    if (randomBits_ <= BITS_8) {
        randomType_ = DTYPE_UINT8;
        randomDtype_ = ge::DT_UINT8;
    } else if (randomBits_ <= BITS_16) {
        randomType_ = DTYPE_UINT16;
        randomDtype_ = ge::DT_UINT16;
    } else if (randomBits_ <= BITS_32) {
        randomType_ = DTYPE_INT32;
        randomDtype_ = ge::DT_INT32;
    } else {
        randomType_ = DTYPE_INT64;
        randomDtype_ = ge::DT_INT64;
    }
    return;
}

/*
 * sort-tiling桥接
 * 构造TilingContext，调用sort-tiling接口
 */
ge::graphStatus StatelessRandpermTiling::SortTilingBridge()
{
    auto indexDtype = (nIsInt32_ == 1) ? ge::DT_INT32 : ge::DT_INT64;
    gert::StorageShape storageShape({n_}, {n_});
    gert::StorageFormat storageFormat({ge::FORMAT_ND, ge::FORMAT_RESERVED, gert::ExpandDimsType()});
    gert::Tensor xTensor(storageShape, storageFormat, randomDtype_); // 根据randomBits选择
    gert::Tensor yTensor(storageShape, storageFormat, randomDtype_);
    gert::Tensor indexTensor(storageShape, storageFormat, indexDtype); // 性能优化：小于int32最大值可以用int32计算

    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(1);
    auto wsPtr = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    uint8_t fakeCompileInfo[] = {0};
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);

    gert::OpTilingContextBuilder tilingBuilder;
    auto holder = tilingBuilder.OpName("StatelessRandpermTiling-Sort")
                      .OpType("Sort")
                      .IONum(1, 2)
                      .AppendAttr(int64_t(-1))                      // [attr]axis
                      .AppendAttr(false)                            // [attr]descending
                      .AppendAttr(true)                             // [attr]stable
                      .AppendAttr(static_cast<int64_t>(indexDtype)) // [attr]y2_dtype
                      .TilingDataSize(sizeof(SortRegBaseTilingData))
                      .Workspace(wsPtr)
                      .CompileInfo(fakeCompileInfo) // sort-tiling未使用
                      .Deterministic(1)             // sort-tiling未使用
                      .PlatformInfo(platformInfo)
                      .InputTensors({&xTensor})
                      .OutputTensors({&yTensor, &indexTensor})
                      .Build();
    auto sortTilingContext = holder.GetContext();
    OP_CHECK_IF(sortTilingContext == nullptr, OP_LOGE(opName_, "tilingBuilder failed."), return GRAPH_FAILED);
    auto ret = statelessRandpermTiling::SortTilingSimt(sortTilingContext, totalCoreNum_);
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(opName_, "SortTilingSimt failed."), return GRAPH_FAILED);
    sortTilingData_ = reinterpret_cast<SortRegBaseTilingDataForRandperm*>(
        sortTilingContext->GetTilingData<SortRegBaseTilingData>());
    OP_CHECK_IF(sortTilingData_ == nullptr, OP_LOGE(opName_, "sortTilingData is nullptr."), return GRAPH_FAILED);
    needCoreNumForSort_ = sortTilingContext->GetBlockDim();
    tilingKeyForSort_ = sortTilingContext->GetTilingKey();
    size_t* userWorkSpaceSize = sortTilingContext->GetWorkspaceSizes(1);
    OP_CHECK_IF(userWorkSpaceSize == nullptr, OP_LOGE(opName_, "userWorkSpaceSize is nullptr."), return GRAPH_FAILED);
    workSpaceSizeForSort_ = userWorkSpaceSize[0];
    return GRAPH_SUCCESS;
}

// 3、是否使能该tiling模板
bool StatelessRandpermTiling::IsCapable() { return true; }

// 判断长度为n的tensor的地址偏移是否能被int32表示
bool StatelessRandpermTiling::canUse32bitIndexing(int64_t len)
{
    int64_t maxVal = std::numeric_limits<int32_t>::max();
    if (len > maxVal) {
        return false;
    }

    int64_t maxOffset = 1;
    if (randomIsInt32_ == 1) {
        maxOffset += (len - 1) * sizeof(int32_t);
    } else {
        maxOffset += (len - 1) * sizeof(int64_t);
    }

    if (maxOffset > maxVal) {
        return false;
    }
    return true;
}

// 地址偏移不能被int32表示的Tensor（长度为n），需要不断对半切分，直到可以被int32表示
void StatelessRandpermTiling::Int32IndexingSplit(int64_t len, std::vector<int32_t>& subBlocks, uint32_t& splitCount)
{
    std::stack<int64_t> tmpStack;
    tmpStack.push(len);

    while (!tmpStack.empty()) {
        int64_t current = tmpStack.top();
        tmpStack.pop();

        if (canUse32bitIndexing(current)) {
            subBlocks.push_back(static_cast<int32_t>(current));
        } else {
            int64_t left = current / 2;
            int64_t right = current - left;
            tmpStack.push(right);
            tmpStack.push(left);
        }
    }

    splitCount = subBlocks.size();
    return;
}

void StatelessRandpermTiling::ThreadBlockNumCalc(uint32_t threadNum, uint32_t& factor, uint32_t& factorTail)
{
    uint32_t baseTile = threadNum;
    uint32_t blockCount = CeilDiv(static_cast<uint32_t>(n_), baseTile);
    factor = CeilDiv(blockCount, needCoreNumForSort_); // 受sort限制
    uint32_t needCoreNum = CeilDiv(blockCount, factor);
    factorTail = blockCount - (needCoreNum - 1) * factor;
    return;
}

// 4、计算数据切分TilingData
ge::graphStatus StatelessRandpermTiling::DoOpTiling()
{
    OP_LOGD(opName_, "StatelessRandpermTiling DoOpTiling.");
    tilingData_ = context_->GetTilingData<StatelessRandpermTilingData>();
    // 1、计算randombits，判断数据类型，供sort-tiling使用
    PhiloxRandomComputeBits();

    // 2、sort-tiling
    auto ret = SortTilingBridge();
    OP_CHECK_IF(ret != GRAPH_SUCCESS, OP_LOGE(opName_, "SortTilingBridge failed."), return GRAPH_FAILED);

    // 3、n值切分
    Int32IndexingSplit(n_, subNs_, subNSize_);
    if (subNSize_ > SUB_N_TILE_COUNT) {
        std::string valueStr = std::to_string(subNSize_);
        std::string reasonMsg = "After n splits, the number of blocks should not exceed " +
                                std::to_string(SUB_N_TILE_COUNT);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "input tensor n", valueStr.c_str(), reasonMsg.c_str());
        return GRAPH_FAILED;
    }

    // 4.1、Fisher-Yates部分，线程块参数计算
    ThreadBlockNumCalc(SIMT_THREAD_NUM_512, islandFactor_, islandFactorTail_);
    // 4.2、cast部分，线程块参数计算
    ThreadBlockNumCalc(SIMT_THREAD_NUM_2048, castFactor_, castFactorTail_);

    // 5、tilingkey赋值（sort和本算子的key合并）
    tilingKey_ = (tilingKeyForSort_ << BITS_16) | (nIsInt32_ << BITS_8) | (randomType_);
    return ge::GRAPH_SUCCESS;
}

// 5、计算高阶API的TilingData
ge::graphStatus StatelessRandpermTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

// 6、计算TilingKey
uint64_t StatelessRandpermTiling::GetTilingKey() const { return tilingKey_; }

// 7、计算Workspace 大小
ge::graphStatus StatelessRandpermTiling::GetWorkspaceSize()
{
    size_t indexWorkSpace = ((nIsInt32_ == 1) ? sizeof(int32_t) : sizeof(int64_t)) * n_;
    size_t randWorkSpace = ge::GetSizeByDataType(randomDtype_) * n_;
    size_t arangeWorkSpace = indexWorkSpace;
    size_t y1WorkSpace = randWorkSpace;
    workSpaceSizeForRandom_ = arangeWorkSpace + randWorkSpace + y1WorkSpace + indexWorkSpace;
    workSpaceSize_ = workSpaceSizeForSort_ + workSpaceSizeForRandom_;
    return ge::GRAPH_SUCCESS;
}

void StatelessRandpermTiling::SetTilingData()
{
    tilingData_->sortTilingData = *sortTilingData_;
    tilingData_->n = n_;
    tilingData_->randomBits = randomBits_;
    tilingData_->islandFactor = islandFactor_;
    tilingData_->islandFactorTail = islandFactorTail_;
    tilingData_->castFactor = castFactor_;
    tilingData_->castFactorTail = castFactorTail_;
    tilingData_->randomWkSizeByte = workSpaceSizeForRandom_;
    for (size_t i = 0; i < PHILOX_KEY_SIZE; i++) {
        tilingData_->philoxKey[i] = key_[i];
    }
    tilingData_->philoxOffset = offset_;
    tilingData_->realCoreNum = needCoreNumForSort_;
    tilingData_->subNTileCount = subNs_.size();
    for (size_t i = 0; i < subNs_.size(); i++) {
        tilingData_->subNTile[i] = subNs_[i]; // 目前最大支持n为int32最大值的序列
    }
}

// 7、保存Tiling数据
ge::graphStatus StatelessRandpermTiling::PostTiling()
{
    SetTilingData();
    DumpTilingInfo();

    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workSpaceSize_;
    context_->SetBlockDim(needCoreNumForSort_);
    context_->SetLocalMemorySize(ubSize_);
    context_->SetScheduleMode(1); // 设置为batch mode模式，所有核同时启动
    return ge::GRAPH_SUCCESS;
}

// 输出 Tiling 信息
void StatelessRandpermTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "ubSize: " << ubSize_;
    info << ", totalCoreNum: " << totalCoreNum_;
    info << ", needCoreNumForSort: " << needCoreNumForSort_;
    info << ", randomBits: " << randomBits_;
    for (size_t i = 0; i < PHILOX_KEY_SIZE; i++) {
        info << ", philoxKey[" << i << "]: " << key_[i];
    }
    info << ", philoxOffset: " << offset_;
    info << ", islandFactor: " << islandFactor_;
    info << ", islandFactorTail: " << islandFactorTail_;
    info << ", castFactor: " << castFactor_;
    info << ", castFactorTail: " << castFactorTail_;
    info << ", randomWkSizeByte: " << workSpaceSizeForRandom_;
    info << ", subNTileCount: " << subNs_.size();
    for (size_t i = 0; i < subNs_.size(); i++) {
        info << ", subNTile[" << i << "]: " << subNs_[i];
    }
    OP_LOGI(opName_, "%s", info.str().c_str());
}

static ge::graphStatus TilingStatelessRandperm(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TilingStatelessRandperm", "Tiling context is nullptr!"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingStatelessRandperm enter");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    StatelessRandpermTiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepareForStatelessRandperm(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TilingPrepareForStatelessRandperm", "Tiling context is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForStatelessRandperm start");

    auto compileInfo = context->GetCompiledInfo<StatelessRandpermCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = static_cast<int64_t>(ubSizePlatForm);

    OP_LOGD(context->GetNodeName(), "TilingPrepareForStatelessRandperm end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(StatelessRandperm)
    .Tiling(TilingStatelessRandperm)
    .TilingParse<StatelessRandpermCompileInfo>(TilingPrepareForStatelessRandperm)
    .TilingInputsDataDependency({INPUT_IDX_N, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
