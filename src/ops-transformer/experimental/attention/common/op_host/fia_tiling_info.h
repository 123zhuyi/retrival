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
 * \file fia_tiling_info.h
 * \brief
 */

#ifndef FIA_TILING_INFO_H
#define FIA_TILING_INFO_H

#include <vector>
#include "fia_tiling_base.h"

namespace optiling {
const std::string KEY_NAME = "key";
const std::string KEY_ROPE_NAME = "key_rope";
const std::string KV_HEADS_NUM_NAME = "the key/value's heads num";
const std::string QUERY_NAME = "query";
const std::string QUERY_HEADS_NUM_NAME = "the query's heads num";
const std::string QUERY_ROPE_NAME = "query_rope";
const std::string VALUE_NAME = "value";

enum class FiaLayout : uint32_t {
    // stardard
    BSND = 1,
    BNSD = 2,
};

enum class FiaAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    TENSOR_LIST = 1,
    PAGE_ATTENTION = 2
};

enum class RopeMode : uint32_t {
    NO_ROPE = 0,
    ROPE_SPLIT = 1,
    ROPE_COMBINE = 2
};

enum class FiaTilingInOutMode : uint32_t {
    FP16_FP16 = 4,
    BF16_BF16 = 5,
};

enum class TilingKeyLayout : uint32_t {
    BSH_BSND = 0,
    BNSD = 1,
};

enum class FiaTemplateId : uint32_t {
    EMPTY_TENSOR = 0,
    HIGH_PERFORMANCE_GQA = 3,
    GENERAL_GQA = 4,
    HIGH_PERFORMANCE_MLA = 5
};

std::string LayoutToSerialString(FiaLayout layout);
std::string AxisToSerialString(FiaAxis axis);
std::string SituationToSerialString(RopeMode ropeMode);

struct FIARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FIAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct FIAParaInfo {
    FIARequiredParaInfo query = {nullptr, nullptr};
    FIARequiredParaInfo key = {nullptr, nullptr};
    FIARequiredParaInfo value = {nullptr, nullptr};

    FIAOptionalParaInfo actualSeqLengthsQ = {nullptr, nullptr};
    FIAOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    FIAOptionalParaInfo queryRope = {nullptr, nullptr};
    FIAOptionalParaInfo keyRope = {nullptr, nullptr};

    FIARequiredParaInfo attenOut = {nullptr, nullptr};

    const int32_t *numHeads = nullptr;
    const int64_t *preToken = nullptr;
    const int64_t *nextToken = nullptr;
    const float *scaleValue = nullptr;
    const int32_t *kvHeadNums = nullptr;
    const char *layOut = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *innerPrecise = nullptr;
    const int64_t *antiquantMode = nullptr;
    const bool *softmaxLseFlag = nullptr;
    const int64_t *keyAntiquantMode = nullptr;
    const int64_t *valueAntiquantMode = nullptr;
    const int32_t *sparseMode = nullptr;
    const int64_t *queryQuantMode = nullptr;
};

class FiaTilingInfo : public TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FIAParaInfo opParamInfo;

    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0; // 仅TND/NTD时生效
    uint32_t kTSize = 0;
    float scaleValue = 0;
    int32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;

    uint64_t totalOutputSize = 0;
    uint32_t totalBlockNum = 0;

    // Q actual_seq_lens
    uint32_t actualLenQDims = 0;
    int64_t maxActualseq = 0;
    bool isAccumQSeq = false;

    // KV actual_seq_lens
    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDims = 0;
    std::vector<int64_t> kvListSeqLens {};
    bool isAccumKVSeq = false;

    // Others Flag
    bool batchContinuousFlag = true;
    bool kvPaddingSizeFlag = false;
    bool qPaddingSizeFlag = false;
    bool softmaxLseFlag = false;
    bool quantFlag = false;
    bool isMaxWorkspace = false;
    bool isLegacyIfa = false;
    bool needInit = false;
    bool slidingFlag = false;
    bool learnableSinkFlag = false;
    // DType
    FiaTilingInOutMode inOutMode = FiaTilingInOutMode::FP16_FP16;
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    // Layout
    TilingKeyLayout inputKvLayout = TilingKeyLayout::BSH_BSND;
    TilingKeyLayout inputLayout = TilingKeyLayout::BSH_BSND;
    TilingKeyLayout outputLayout = TilingKeyLayout::BSH_BSND;

    // BaseParams
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode = RopeMode::NO_ROPE;

    // Layout
    FiaLayout qLayout = FiaLayout::BSND;
    FiaLayout outLayout = FiaLayout::BSND;
    FiaLayout kvLayout = FiaLayout::BSND;

    ge::DataType inputQRopeType = ge::DT_FLOAT16;
    ge::DataType inputKRopeType = ge::DT_FLOAT16;

    uint64_t l2CacheSize = 0;
    std::vector<gert::StorageShape *> kCache = {};
    std::vector<gert::StorageShape *> vCache = {};
};
} // optiling
#endif // FIA_TILING_INFO_H