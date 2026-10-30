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
 * \file gather_pa_kv_cache_tiling_arch35.cpp
 * \brief
 */

#include <cmath>
#include <array>
#include "log/log.h"
#include "platform/platform_info_def.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_util.h"
#include "util/math_util.h"
#include "gather_pa_kv_cache_tiling.h"

using namespace AscendC;
using namespace ge;
using namespace Ops::Transformer::OpTiling;

namespace optiling {

constexpr int64_t INDEX_ATTR_CACHE_MODE = 0;
constexpr int64_t INDEX_ATTR_IS_SEQ_LENS_CUMSUM = 1;
constexpr int64_t INDEX_INPUT_KEY_CACHE = 0;
constexpr int64_t INDEX_INPUT_VALUE_CACHE = 1;
constexpr int64_t INDEX_INPUT_BLOCK_TABLES = 2;
constexpr int64_t INDEX_INPUT_SEQ_LENS = 3;
constexpr int64_t INDEX_INPUT_KEY = 4;
constexpr int64_t INDEX_INPUT_VALUE = 5;
constexpr int64_t INDEX_OPT_INPUT_SEQ_OFFSETS = 6;
constexpr int64_t INDEX_OUTPUT_KEY = 0;
constexpr int64_t INDEX_OUTPUT_VALUE = 1;
constexpr int64_t DIM_ONE = 1;
constexpr int64_t DIM_TWO = 2;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t WORKSPACE_SIZE = 32;
constexpr uint32_t UB_REVERSE = 1024;
constexpr uint32_t SEQ_LEN_ACC_SPACE = 1024;
constexpr uint32_t DOUBLE_BUFFER = 2;

static const std::map<ge::DataType, uint32_t> tilingDataTypeByteTable = {
    {ge::DT_INT64, 8},    {ge::DT_INT32, 4},       {ge::DT_UINT32, 4},        {ge::DT_FLOAT, 4}, {ge::DT_INT16, 2},
    {ge::DT_UINT16, 2},   {ge::DT_FLOAT16, 2},     {ge::DT_BF16, 2},          {ge::DT_INT8, 1},  {ge::DT_UINT8, 1},
    {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E5M2, 1}, {ge::DT_FLOAT8_E4M3FN, 1},
};

static const std::set<ge::DataType> KV_SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT16, ge::DT_UINT16,   ge::DT_INT8,        ge::DT_UINT8};

static const std::set<ge::DataType> INDEX_SUPPORT_DTYPE = {ge::DT_INT32, ge::DT_INT64};


ge::graphStatus GatherPaKvCacheTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const GatherPaKvCacheCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetAttrs()
{
    auto *attrs = context_->GetAttrs();
    cacheMode_ = std::string(attrs->GetAttrPointer<char>(INDEX_ATTR_CACHE_MODE));
    if (cacheMode_.empty()) {
        cacheMode_ = "Norm";
    } else {
        std::set<std::string> checkList = {"Norm", "PA_NZ"};
        OP_CHECK_IF(checkList.find(cacheMode_) == checkList.end(),
                    OP_LOGE(context_, "[attr]cache_mode only support ['Norm', 'PA_NZ'], please check."),
                    return ge::GRAPH_FAILED);
    }
    isCacheModeNorm_ = (cacheMode_ == "Norm");

    const bool *isSeqLensCumsumPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_IS_SEQ_LENS_CUMSUM);
    if (isSeqLensCumsumPtr == nullptr) {
        isSeqLenCumSum_ = true;
    } else {
        isSeqLenCumSum_ = *isSeqLensCumsumPtr;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputKeyCache()
{
    auto kCacheDesc = context_->GetInputDesc(INDEX_INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheDesc);
    ge::DataType kCacheDType_ = kCacheDesc->GetDataType();
    ge::Format kCacheFormat = kCacheDesc->GetFormat().GetStorageFormat();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(kCacheDType_) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_,
                        "key_cache dtype only support [float32, float16, bf16,"
                        " hf8, fp8_e5m2, fp8_e4m3fn, int32, uint32, int16, uint16, int8, uint8], please check."),
                return ge::GRAPH_FAILED);

    cacheDTypeByteSize_ = tilingDataTypeByteTable.find(kCacheDType_)->second;

    auto kCacheStoreShape = context_->GetInputShape(INDEX_INPUT_KEY_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheStoreShape);
    kCacheShape_ = EnsureNotScalar(kCacheStoreShape->GetStorageShape());
    kCacheDimNum_ = kCacheShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(kCacheDimNum_ != 4,
                OP_LOGE(context_, "key_cache dimension must be 4, but got %zu. Please check.", kCacheDimNum_),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < kCacheDimNum_; i++) {
        OP_CHECK_IF(kCacheShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "key_cache.shape[%zu] must be positive, Please check.", i),
                    return ge::GRAPH_FAILED);
    }

    numBlocks_ = kCacheShape_.GetDim(0);
    blockSize_ = kCacheShape_.GetDim(1);
    // 当数据格式为NZ时
    if (!isCacheModeNorm_) {
        uint32_t kCacheByteAlign = BLOCK_SIZE / cacheDTypeByteSize_;
        OP_CHECK_IF(kCacheShape_.GetDim(kCacheDimNum_ - 1) != kCacheByteAlign,
                    OP_LOGE(context_, "key_cache.shape[3](%ld) must align and equal to 32B, please check.",
                            kCacheShape_.GetDim(kCacheDimNum_ - 1)),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(kCacheFormat != ge::Format::FORMAT_FRACTAL_NZ,
                    OP_LOGE(context_, "key_cache format should be FRACTAL_NZ when cache_mode is PA_NZ, please check."),
                    return ge::GRAPH_FAILED);
        // 2 is blockSize_ dim
        blockSize_ = kCacheShape_.GetDim(2);
    } else {
        OP_CHECK_IF(kCacheFormat != ge::Format::FORMAT_ND,
                    OP_LOGE(context_, "key_cache format should be ND when cache_mode is Norm, please check."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputValueCache()
{
    auto vCacheDesc = context_->GetInputDesc(INDEX_INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheDesc);
    ge::DataType vCacheDType = vCacheDesc->GetDataType();
    ge::Format vCacheFormat = vCacheDesc->GetFormat().GetStorageFormat();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(vCacheDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_,
                        "value_cache dtype only support [float32, float16, bf16,"
                        " hf8, fp8_e5m2, fp8_e4m3fn, int32, uint32, int16, uint16, int8, uint8], please check."),
                return ge::GRAPH_FAILED);

    auto vCacheStoreShape = context_->GetInputShape(INDEX_INPUT_VALUE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheStoreShape);
    vCacheShape_ = EnsureNotScalar(vCacheStoreShape->GetStorageShape());
    vCacheDimNum_ = vCacheShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(vCacheDimNum_ != 4,
                OP_LOGE(context_, "value_cache dimension must be 4, but got %zu. Please check.", vCacheDimNum_),
                return ge::GRAPH_FAILED);

    // 当数据格式为NZ时，需要检查尾轴是否与32B对齐。kcache和vcache除第1维，其他轴必须相等。
    // 当数据格式为ND时，kcache和vcache的shape的非尾轴必须相等。
    size_t skipAxis = vCacheDimNum_ - 1;
    if (!isCacheModeNorm_) {
        skipAxis = 1;
        uint32_t vCacheDtypeSize = static_cast<uint32_t>(tilingDataTypeByteTable.find(vCacheDType)->second);
        uint32_t vCacheByteAlign = BLOCK_SIZE / vCacheDtypeSize;
        OP_CHECK_IF(vCacheShape_.GetDim(vCacheDimNum_ - 1) != vCacheByteAlign,
                    OP_LOGE(context_, "value_cache last dimension must align and equal to 32B, please check."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            vCacheFormat != ge::Format::FORMAT_FRACTAL_NZ,
            OP_LOGE(context_, "value_cache format should be FRACTAL_NZ when cache_mode is PA_NZ, please check."),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(vCacheFormat != ge::Format::FORMAT_ND,
                    OP_LOGE(context_, "value_cache format should be ND when cache_mode is Norm, please check."),
                    return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < vCacheDimNum_; i++) {
        if (i == skipAxis) {
            continue;
        }
        OP_CHECK_IF(vCacheShape_.GetDim(i) != kCacheShape_.GetDim(i),
                    OP_LOGE(context_,
                            "value_cache.shape[%zu] %ld is not equal to key_cache.shape[%zu] %ld, "
                            "please check.",
                            i, vCacheShape_.GetDim(i), i, kCacheShape_.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputBlockTables()
{
    auto blockTablesDesc = context_->GetInputDesc(INDEX_INPUT_BLOCK_TABLES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTablesDesc);
    ge::DataType blockTablesDType = blockTablesDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(blockTablesDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "block_tables dtype only support [int32, int64], please check."),
                return ge::GRAPH_FAILED);
    indexByteSize_ = tilingDataTypeByteTable.find(blockTablesDType)->second;

    auto blockTableStoreShape = context_->GetInputShape(INDEX_INPUT_BLOCK_TABLES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockTableStoreShape);
    blockTableShape_ = EnsureNotScalar(blockTableStoreShape->GetStorageShape());
    size_t blockTableDimNum = blockTableShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(blockTableDimNum != 2,
                OP_LOGE(context_, "block_tables dimension must be 2, but got %zu. Please check.", blockTableDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < blockTableDimNum; i++) {
        OP_CHECK_IF(blockTableShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "block_tables.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }
    batchCount_ = blockTableShape_.GetDim(0);
    blockTableWidth_ = blockTableShape_.GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputSeqLens()
{
    auto seqLensDesc = context_->GetInputDesc(INDEX_INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLensDesc);
    ge::DataType seqLensDType = seqLensDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(seqLensDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "seq_lens dtype only support [int32, int64], please check."),
                return ge::GRAPH_FAILED);

    auto seqLenStoreShape = context_->GetInputShape(INDEX_INPUT_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqLenStoreShape);
    seqLenShape_ = EnsureNotScalar(seqLenStoreShape->GetStorageShape());
    uint32_t seqLenDimNum = seqLenShape_.GetDimNum();

    // 检查形状是否合法
    OP_CHECK_IF(seqLenDimNum != 1,
                OP_LOGE(context_, "seq_lens dimension must be 1, but got %u. Please check.", seqLenDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < seqLenDimNum; i++) {
        OP_CHECK_IF(seqLenShape_.GetDim(i) <= 0,
                    OP_LOGE(context_, "seq_lens.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }
    // seq_lens的第0维要和block_tables的第0维相等
    if (isSeqLenCumSum_) {
        OP_CHECK_IF(seqLenShape_.GetDim(0) != (blockTableShape_.GetDim(0) + 1),
                    OP_LOGE(context_, "When [attr]is_seq_lens_cumsum is true, seq_lens.shape[0] must equal "
                                      "block_tables.shape[0] + 1, please check."),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(seqLenShape_.GetDim(0) != blockTableShape_.GetDim(0),
                    OP_LOGE(context_, "When [attr]is_seq_lens_cumsum is false, seq_lens.shape[0] must equal "
                                      "block_tables.shape[0], please check."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputOutputKey()
{
    auto keyDesc = context_->GetInputDesc(INDEX_INPUT_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyDesc);
    ge::DataType keyDType = keyDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(keyDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_,
                        "key dtype only support [float32, float16, bf16, "
                        "hf8, fp8_e5m2, fp8_e4m3fn, int32, uint32, int16, uint16, int8, uint8], please check."),
                return ge::GRAPH_FAILED);

    auto keyStoreShape = context_->GetInputShape(INDEX_INPUT_KEY);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyStoreShape);
    keyShape_ = EnsureNotScalar(keyStoreShape->GetStorageShape());
    size_t keyDimNum = keyShape_.GetDimNum();

    uint32_t keyDimExpect = (isCacheModeNorm_) ? uint32_t(3) : uint32_t(2);
    OP_CHECK_IF(keyDimNum != keyDimExpect,
                OP_LOGE(context_, "key dimension must be %u, but got %zu. Please check.", keyDimExpect, keyDimNum),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i < keyDimNum; i++) {
        OP_CHECK_IF(keyShape_.GetDim(i) <= 0, OP_LOGE(context_, "key.shape[%zu] must be positive, please check.", i),
                    return ge::GRAPH_FAILED);
    }

    numTokens_ = keyShape_.GetDim(0);

    if (isCacheModeNorm_) {
        // ND
        hiddenSizeK_ = keyShape_.GetDim(DIM_ONE) * keyShape_.GetDim(DIM_TWO);
        for (size_t i = 1; i < keyDimNum; i++) {
            OP_CHECK_IF(keyShape_.GetDim(i) != kCacheShape_.GetDim(i + 1),
                        OP_LOGE(context_,
                                "key.shape[%zu] %ld is not equal to key_cache.shape[%zu] %ld, "
                                "please check.",
                                i, keyShape_.GetDim(i), i + 1, kCacheShape_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // NZ
        hiddenSizeK_ = keyShape_.GetDim(1);
        int64_t kCacheShape1 = kCacheShape_.GetDim(1);
        int64_t kCacheShape3 = kCacheShape_.GetDim(3);
        uint64_t hiddenSizeKCache = static_cast<uint64_t>(kCacheShape1) * kCacheShape3;
        OP_CHECK_IF(hiddenSizeK_ != hiddenSizeKCache,
                    OP_LOGE(context_,
                            "key.shape[1] (%lu) is not equal to "
                            "the product of key_cache.shape[1] and key_cache.shape[3] (%ld * %ld = %lu), "
                            "please check.",
                            hiddenSizeK_, kCacheShape1, kCacheShape3, hiddenSizeKCache),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputOutputValue()
{
    auto valueDesc = context_->GetInputDesc(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueDesc);
    ge::DataType valueDType = valueDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((KV_SUPPORT_DTYPE.find(valueDType) == KV_SUPPORT_DTYPE.end()),
                OP_LOGE(context_,
                        "value dtype only support [float32, float16, bf16, "
                        "hf8, fp8_e5m2, fp8_e4m3fn, int32, uint32, int16, uint16, int8, uint8], please check."),
                return ge::GRAPH_FAILED);

    auto valueStoreShape = context_->GetInputShape(INDEX_INPUT_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueStoreShape);
    valueShape_ = EnsureNotScalar(valueStoreShape->GetStorageShape());
    size_t valueDimNum = valueShape_.GetDimNum();

    // 检查形状是否合法
    uint32_t valueDimExpect = (isCacheModeNorm_) ? uint32_t(3) : uint32_t(2);
    OP_CHECK_IF(
        valueDimNum != valueDimExpect,
        OP_LOGE(context_, "value dimension must be %u, but got %zu. Please check.", valueDimExpect, valueDimNum),
        return ge::GRAPH_FAILED);
    for (size_t i = 0; i < valueDimNum - 1; i++) {
        OP_CHECK_IF(valueShape_.GetDim(i) != keyShape_.GetDim(i),
                    OP_LOGE(context_, "value.shape[%zu] %ld is not equal to key.shape[%zu] %ld, please check.", i,
                            valueShape_.GetDim(i), i, keyShape_.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }

    if (isCacheModeNorm_) {
        // ND
        hiddenSizeV_ = valueShape_.GetDim(DIM_ONE) * valueShape_.GetDim(DIM_TWO);
        for (size_t i = 1; i < valueDimNum; i++) {
            OP_CHECK_IF(valueShape_.GetDim(i) != vCacheShape_.GetDim(i + 1),
                        OP_LOGE(context_,
                                "value.shape[%zu] %ld is not equal to value_cache.shape[%zu] %ld, "
                                "please check.",
                                i, valueShape_.GetDim(i), i + 1, vCacheShape_.GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        // NZ
        hiddenSizeV_ = valueShape_.GetDim(1);
        int64_t vCacheShape1 = vCacheShape_.GetDim(1);
        int64_t vCacheShape3 = vCacheShape_.GetDim(3);
        uint64_t hiddenSizeVCache = static_cast<uint64_t>(vCacheShape1) * vCacheShape3;
        OP_CHECK_IF(hiddenSizeV_ != hiddenSizeVCache,
                    OP_LOGE(context_,
                            "value.shape[1] (%lu) is not equal to "
                            "the product of value_cache.shape[1] and value_cache.shape[3] (%ld * %ld = %lu), "
                            "please check.",
                            hiddenSizeV_, vCacheShape1, vCacheShape3, hiddenSizeVCache),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetInputSeqOffset()
{
    auto seqOffsetDesc = context_->GetOptionalInputDesc(INDEX_OPT_INPUT_SEQ_OFFSETS);
    if (seqOffsetDesc == nullptr) {
        OP_LOGD(context_, "[attr]seq_offset is not been set.");
        hasSeqOffset_ = false;
        return ge::GRAPH_SUCCESS;
    }
    hasSeqOffset_ = true;
    ge::DataType seqOffsetDType = seqOffsetDesc->GetDataType();

    // 校验数据类型是否合法
    OP_CHECK_IF((INDEX_SUPPORT_DTYPE.find(seqOffsetDType) == INDEX_SUPPORT_DTYPE.end()),
                OP_LOGE(context_, "seq_offset dtype only support [int64, int32], please check."),
                return ge::GRAPH_FAILED);

    auto seqOffsetStoreShape = context_->GetOptionalInputShape(INDEX_OPT_INPUT_SEQ_OFFSETS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, seqOffsetStoreShape);
    seqOffsetShape_ = EnsureNotScalar(seqOffsetStoreShape->GetStorageShape());
    uint32_t seqOffsetDimNum = seqOffsetShape_.GetDimNum();
    // 检查形状是否合法
    OP_CHECK_IF(seqOffsetDimNum != 1,
                OP_LOGE(context_, "seq_offset dimension must be 1, but got %u. Please check.", seqOffsetDimNum),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(seqOffsetShape_.GetDim(0) != blockTableShape_.GetDim(0),
                OP_LOGE(context_, "seqOffset.shape[%zu] %ld is not equal to block_tables.shape[%zu] %ld, please check.",
                        size_t(0), seqOffsetShape_.GetDim(0), size_t(0), blockTableShape_.GetDim(0)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("GatherPaKvCacheTiling", "context is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetAttrs() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get attrs failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputKeyCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input key_cache failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputValueCache() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input value_cache failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputBlockTables() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input block_tables failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqLens() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input seq_lens failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputKey() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output key failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputOutputValue() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get input/output value failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetInputSeqOffset() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "get optional input seq_offset failed."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool GatherPaKvCacheTiling::IsCapable()
{
    return true;
}

ge::graphStatus GatherPaKvCacheTiling::DoOpTiling()
{
    // batch分核
    int64_t batchPerCore = Ops::Base::CeilDiv(batchCount_, coreNum_);
    needCoreNum_ = static_cast<uint32_t>(std::min(Ops::Base::CeilDiv(batchCount_, batchPerCore), coreNum_));
    // uint32_t batchTail = batchCount_ - batchPerCore * (needCoreNum - 1);
    uint32_t tileBase = BLOCK_SIZE / cacheDTypeByteSize_;

    // 计算UB最大能放下的KV Cache大小
    uint32_t seqLenAccumSize = 1024;
    uint32_t factor = (ubSize_ - UB_REVERSE - seqLenAccumSize * DOUBLE_BUFFER * indexByteSize_ - BLOCK_SIZE) /
                      (tileBase * DOUBLE_BUFFER * cacheDTypeByteSize_);
    uint64_t cacheBlockK = static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(hiddenSizeK_);
    uint64_t cacheBlockV = static_cast<uint64_t>(blockSize_) * static_cast<uint64_t>(hiddenSizeV_);
    uint64_t maxUbHiddenSizeK =
        std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockK); // 最大不超过1个cacheBlock
    uint64_t maxUbHiddenSizeV =
        std::min(static_cast<uint64_t>(factor) * tileBase, cacheBlockV); // 最大不超过1个cacheBlock
    uint64_t maxUbHiddenSize = std::max(maxUbHiddenSizeK, maxUbHiddenSizeV);
    maxUbHiddenSize = Ops::Base::CeilAlign(maxUbHiddenSize, static_cast<uint64_t>(tileBase)); // 保证maxUbHiddenSize和32B对齐

    // 动态调整: 如果有多余空间，就用于累加和的计算
    if (maxUbHiddenSizeK == cacheBlockK || maxUbHiddenSizeV == cacheBlockV) {
        uint32_t spareBuffer =
            ubSize_ - UB_REVERSE - maxUbHiddenSize * DOUBLE_BUFFER * cacheDTypeByteSize_ - BLOCK_SIZE;
        seqLenAccumSize = Ops::Base::CeilDiv(spareBuffer / DOUBLE_BUFFER, BLOCK_SIZE) * BLOCK_SIZE / indexByteSize_;
    }

    // 配置tilingdata
    tilingData_.set_batchCount(batchCount_);
    tilingData_.set_batchPerCore(batchPerCore);
    tilingData_.set_needCoreNum(needCoreNum_);
    tilingData_.set_seqLenAccumSize(seqLenAccumSize);
    tilingData_.set_blockTableWidth(blockTableWidth_);
    tilingData_.set_numBlocks(numBlocks_);
    tilingData_.set_hiddenSizeK(hiddenSizeK_);
    tilingData_.set_hiddenSizeV(hiddenSizeV_);
    tilingData_.set_numTokens(numTokens_);
    tilingData_.set_maxUbHiddenSizeK(maxUbHiddenSizeK);
    tilingData_.set_maxUbHiddenSizeV(maxUbHiddenSizeV);
    tilingData_.set_maxUbHiddenSize(maxUbHiddenSize);
    tilingData_.set_kvCacheBlockSize(blockSize_);

    // 根据属性设置tilingkey
    for (const auto &item : tilingKeyTable) {
        if (item.isCacheModeNorm == isCacheModeNorm_ && item.isSeqLenCumSum == isSeqLenCumSum_ &&
            item.hasSeqOffset == hasSeqOffset_) {
            tilingKey_ = item.tilingKey;
            break;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GatherPaKvCacheTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus GatherPaKvCacheTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherPaKvCacheTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(needCoreNum_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForGatherPaKvCache(gert::TilingContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingForGatherPaKvCache", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingForGatherPaKvCache enter.");

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGD(context, "Tiling4GatherPaKvCache enter.");
        return Tiling4GatherPaKvCache(context);
    }

    GatherPaKvCacheTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForGatherPaKvCache(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TilingPrepareForGatherPaKvCache", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context, "TilingPrepareForGatherPaKvCache enter");

    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        return TilingPrepare4GatherPaKvCache(context);
    }

    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GatherPaKvCache op.
IMPL_OP_OPTILING(GatherPaKvCache)
    .Tiling(TilingForGatherPaKvCache)
    .TilingParse<GatherPaKvCacheCompileInfo>(TilingPrepareForGatherPaKvCache);

} // namespace optiling
