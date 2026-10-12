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
 * \file fused_infer_attention_score_tiling_info_parser.h
 * \brief
 */

#pragma once

#include "../../../common/op_host/fia_tiling_info.h"
#include "../../../common/op_host/fia_tiling_shape.h"

namespace optiling {
class FiaInfoParser {
public:
    explicit FiaInfoParser(const gert::TilingContext *context) : context_(context) {}
    ~FiaInfoParser() = default;

    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();

    void GetOptionalInputParaRopeInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetLegacyIfaFlag();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQkHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetKvStorageMode();
    ge::graphStatus GetKvLayout();
    void SetFiaShape();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetRopeMode();
    ge::graphStatus GetRopeHeadDim();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    TilingKeyLayout MapStringToLayout(FiaLayout &layoutString) const;
    void GenerateAxisInfo(FiaTilingInfo &fiaInfo);
    void GenerateDtypeInfo(FiaTilingInfo &fiaInfo);
    void GenerateLayoutInfo(FiaTilingInfo &fiaInfo);
    void GenerateInfo(FiaTilingInfo &fiaInfo);
    ge::graphStatus ParseAxisInfo();
    ge::graphStatus Parse(FiaTilingInfo &fiaInfo);

public:
    const gert::TilingContext *context_ = nullptr;

    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    FIAParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qkHeadDim_ = 0;
    uint32_t vHeadDim_ = 0;
    uint32_t ropeHeadDim_ = 0;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    RopeMode ropeMode_ = RopeMode::NO_ROPE;

    // Layout
    FiaLayout qLayout_ = FiaLayout::BSND;
    FiaLayout outLayout_ = FiaLayout::BSND;
    FiaLayout kvLayout_ = FiaLayout::BSND;

    // 局部参数, 暂存
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType inputQRopeType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;

    uint64_t l2CacheSize_ = 0;
    std::vector<gert::StorageShape *> kCache_ = {};
    std::vector<gert::StorageShape *> vCache_ = {};

    bool needInit_ = false;

    bool isLegacyIfa_ = false;      

    uint32_t actualLenQDims_ = 0;
    uint32_t actualLenDims_ = 0;

    std::shared_ptr<FiaTilingShape> queryShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> valueShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> queryRopeShape_ = nullptr;
    std::shared_ptr<FiaTilingShape> keyRopeShape_ = nullptr;
};
} // optiling
