/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_score_tiling_regbase.h"
#include "../flash_attention_score_tiling_common.h"

namespace optiling {
namespace FA {
ge::graphStatus FlashAttentionScoreTilingRegbase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    size_t *workspaces = context_->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValuePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keepProbPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, preTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, nextTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, n1SizePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputLayoutPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context_, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context_, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attenOutShape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingRegbase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                   return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        socVersion = compileInfoPtr->socVersion;
        npuArch = compileInfoPtr->npuArch;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        socVersion = ascendcPlatform.GetSocVersion();
        npuArch = ascendcPlatform.GetCurNpuArch();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
    }
    OP_LOGI(context_, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
              aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            calcTypeSize = ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                        ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(context_, "get high precision flag: %d.", isHighPercision);
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto keepProbPtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    inputLayout = attrs->GetAttrPointer<char>(idx++);

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    keepProb = *keepProbPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    if (preTokens > std::numeric_limits<int32_t>::max()) {
        OP_LOGW(context_, "preTokens[%ld] config error, should not greater than max int value."
            "preTokens will be reset max int value.", preTokens);
        preTokens = std::numeric_limits<int32_t>::max();
    } else if (preTokens < std::numeric_limits<int32_t>::min()) {
        OP_LOGW(context_, "preTokens[%ld] config error, should not less than min int value."
            "preTokens will be reset min int value.", preTokens);
        preTokens = std::numeric_limits<int32_t>::min();
    }
    if (nextTokens > std::numeric_limits<int32_t>::max()) {
        OP_LOGW(context_, "nextTokens[%ld] config error, should not greater than max int value."
            "nextTokens will be reset max int value.", nextTokens);
        nextTokens = std::numeric_limits<int32_t>::max();
    } else if (nextTokens < std::numeric_limits<int32_t>::min()) {
        OP_LOGW(context_, "nextTokens[%ld] config error, should not less than min int value."
            "nextTokens will be reset min int value.", nextTokens);
        nextTokens = std::numeric_limits<int32_t>::min();
    }
    OP_CHECK_IF(n1Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "head num is zero."), return false);
    OP_CHECK_IF(keepProb <= 0.0 || keepProb > 1.0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "keepProb value must be in range of (0, 1]."), return false);
    keepProbUint8 = static_cast<int64_t>(keepProb * UINT8_MAX);
    hasDropOut = (keepProb < 1.0f);

    implMode = ImplMode::AA_HIGH_PRECISION;
    if (attrs->GetAttrNum() > idx) {
        auto implModePtr = attrs->GetAttrPointer<uint8_t>(idx++);
        if (static_cast<ImplMode>(*implModePtr) == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        }
        isHighPercision = true; // use default value
    }

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_BAND_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::BAND_LEFT_UP_CAUSAL_MODE);
        } else if (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            attenMaskCompressMode = static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE);
        }
        OP_LOGD(context_, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }
    if (attrs->GetAttrNum() > idx) {
        auto pseTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        pseType = *pseTypePtr;
        OP_CHECK_IF(pseType < 0 || pseType >= static_cast<uint8_t>(PseType::PSE_INVALID_TYPE),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "pseType value is out of range"), return false);
    }
    if (attrs->GetAttrNum() > idx) {
        auto seedPtr = attrs->GetAttrPointer<int64_t>(idx++);
        seed = *seedPtr;
    }
    if (attrs->GetAttrNum() > idx) {
        auto offsetPtr = attrs->GetAttrPointer<int64_t>(idx++);
        offset = *offsetPtr;
    }
    if (attrs->GetAttrNum() > idx) {
        auto outDtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        outDtype = *outDtypePtr;
        OP_CHECK_IF(outDtype < 0 || outDtype >= 2,
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "outDtype value is out of range"), return false);
        outDtype = outDtype + 1; // 外部合法是0或1, 内部对应使用1和2,如果没有量化参数, 后面会刷成0, 1表示fp16, 2表示bf16
    }
    idx++; // 跳过softmax_out_layout属性
    OP_LOGD(context_, "attrs: scale_value[%f] keep_prob[%f] pre_tockens[%ld] next_tockens[%ld] head_num[%ld] input_layout[%s]"
                      "inner_precise[%d] sparse_mode[%ld] pseType[%ld] seed[%ld] offset[%ld] outDtype[%ld].",
              scaleValue, keepProb, preTokens, nextTokens, n1Size, inputLayout, static_cast<int>(implMode), sparseMode, pseType,
              seed, offset, outDtype);
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeLayout()
{
    auto &queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto &keyShape = context_->GetInputShape(1)->GetStorageShape();
    auto &valueShape = context_->GetInputShape(2)->GetStorageShape();

    auto queryRope = context_->GetOptionalInputShape(QUERY_ROPE_INDEX);
    bool hasQueryRope = queryRope != nullptr && queryRope->GetStorageShape().GetDimNum() != 0;
    auto keyRope = context_->GetOptionalInputShape(KEY_ROPE_INDEX);
    bool hasKeyRope = keyRope != nullptr && keyRope->GetStorageShape().GetDimNum() != 0;
    if (hasQueryRope ^ hasKeyRope) {
        OP_LOGE(opName, "query_rope and key_rope should be present or absent at the same time, check this.");
        return false;
    }
    dSizeRope = 0; // init dSizeRope
    hasRope = hasQueryRope && hasKeyRope;
    const gert::Shape *queryRopeShape = hasQueryRope ? &queryRope->GetStorageShape() : nullptr;

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(context_, "get input_layout [%s].", inputLayout);
    OP_CHECK_IF(queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen ||
               valueShape.GetDimNum() != layoutLen,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid layout[%s].", inputLayout), return false);
    OP_CHECK_IF(!Analyze3DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape)||
               !Analyze4DimLayout(queryShape, keyShape, valueShape, layoutLen, queryRopeShape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "get unsupported layout: %s", inputLayout), return false);
    if (s1Size > std::numeric_limits<int32_t>::max() || s2Size > std::numeric_limits<int32_t>::max()) {
        OP_LOGE(context_, "s1Size[%ld] and s2Size[%ld] config error, both should not greater than max int value.",
            s1Size, s2Size);
        return false;
    }
    OP_CHECK_IF(gSize == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "gSize is zero"), return false);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "n2Size is zero"), return false);
    OP_CHECK_IF(dSize <= 0L,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not support <= 0"), return false);
    OP_CHECK_IF(dSizeV <= 0L,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "value's dSize is not support <= 0"), return false);
    OP_CHECK_IF(dSizeV > dSize,
            OPS_REPORT_VECTOR_INNER_ERR(opName, "value's dSize is larger than query's dSize"), return false);
    OP_CHECK_IF(n1Size % n2Size != 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "n1Size [%ld] should be a multiple of n2Size [%ld]", n1Size, n2Size),
               return false);
    return true;
}

bool FlashAttentionScoreTilingRegbase::GetActualSeqLenData(
    int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const
{
    auto actualSeqLenTensor = context_->GetOptionalInputTensor(inputIdx);
    if (actualSeqLenTensor == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor is null pointer", templateName);
        return true;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s]actualSeqLenShape is invalid %lu %ld", templateName, actualSeqLenShape.GetDimNum(),
                  actualSeqLenShape.GetDim(0));
        return true;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGW(context_, "[%s]actualSeqLenTensor data is null pointer", templateName);
        return true;
    }
    int64_t seqLen = actualSeqLenShape.GetDim(0);
    try {
        res.reserve(seqLen);
    } catch (...) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "Init actual_seq_len failed, array is too long.");
        return false;
    }
    res.emplace_back(value[0]);
    actualLen++;
    for (auto i = 1; i < seqLen; ++i) {
        auto qLen = value[i] - value[i - 1];
        res.emplace_back(qLen < 0 ? 0 : qLen);
        actualLen++;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeTndLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                      const gert::Shape &valueShape)
{
    int64_t actualSeqQLen = 0;
    int64_t actualSeqKVLen = 0;
    int64_t t1Size = queryShape.GetDim(0);
    int64_t t2Size = keyShape.GetDim(0);
    realT1Size = t1Size;
    std::fill(actualSeqLenData.begin(), actualSeqLenData.end(), 0);
    std::fill(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0);
    if (!GetActualSeqLenData(ACTUAL_SEQ_LENGTH_INPUT_INDEX, actualSeqLenData, actualSeqQLen)) {
        OP_LOGE(opName, "Get actual_seq_qlen failed.");
        return false;
    }
    if (!GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_INPUT_INDEX, actualSeqLenKvData, actualSeqKVLen)) {
        OP_LOGE(opName, "Get actual_seq_kvlen failed.");
        return false;
    }
    OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "VarLen scene, q is not equal kv."), return false);
    bSize = actualSeqQLen;
    accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
    accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
    OP_CHECK_IF(
        t1Size < accumS1 || t2Size < accumS2,
        OPS_REPORT_VECTOR_INNER_ERR(
            opName,
            "Query T(%ld) and key T(%ld) need larger than respectively sum of seqLen(%ld) and sekvLen(%ld).",
            t1Size, t2Size, accumS1, accumS2),
        return false);
    uint32_t firstValidIndex = 0;
    uint32_t lastValidIndex = static_cast<uint32_t>(bSize - 1);
    for (int64_t i = 0; i < bSize; ++i) {
        if (actualSeqLenData[i] != 0) {
            firstValidIndex = static_cast<uint32_t>(i);
            break;
        }
    }
    for (auto i = bSize - 1; i >= 0; --i) {
        if (actualSeqLenData[i] != 0) {
            lastValidIndex = static_cast<uint32_t>(i);
            break;
        }
    }
    if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL_BAND)) {
        bandIndex = static_cast<int64_t>(lastValidIndex);
        inputParamsRegbase_->set_bandIndex(lastValidIndex);
    }
    if (sparseMode == static_cast<int64_t>(SparseMode::BAND_LEFT_UP_CAUSAL)) {
        bandIndex = static_cast<int64_t>(firstValidIndex);
        inputParamsRegbase_->set_bandIndex(firstValidIndex);
    }
    s1Size = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
    s2Size = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
    OP_CHECK_IF(s1Size <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "s1Size should be larger than 0."), return false);
    OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                            queryShape.GetDim(1)),
                return false);
    n2Size = keyShape.GetDim(1);
    OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
    gSize = queryShape.GetDim(1) / n2Size;
    dSize = queryShape.GetDim(DIM_NUM_2);
    dSizeV = valueShape.GetDim(DIM_NUM_2);
    return true;
}

bool FlashAttentionScoreTilingRegbase::Analyze3DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                       const gert::Shape &valueShape, size_t layoutLen,
                                                       const gert::Shape *queryRopeShape)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    int64_t h3 = 0;
    int64_t hRope = 0;
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[DIM_NUM_2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(1);
            bSize = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2);   // 2: H idx
            if (hasRope) {
                hRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            s1StrideSize = h1;
            s2StrideSize = h2;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BSH));
            tilingKeyLayout = LayoutType::LAYOUT_BSH;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[DIM_NUM_2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            h3 = valueShape.GetDim(2);   // 2: H idx
            if (hasRope) {
                hRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_SBH));
            tilingKeyLayout = LayoutType::LAYOUT_SBH;
        } else if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[DIM_NUM_2] == 'D') {
            OP_CHECK_IF(!AnalyzeTndLayout(queryShape, keyShape, valueShape),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze tnd layout error."), return false);
            h1 = n1Size * dSize;
            h2 = n2Size * dSize;
            h3 = n2Size * dSizeV;
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_2);
            }
            hRope = n1Size * dSizeRope;
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_TND));
            tilingKeyLayout = LayoutType::LAYOUT_TND;
        } else {
            return false;
        }
        OP_CHECK_IF(h1 == 0 || h2 == 0 || h3 == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "H is zero."), return false);
        OP_CHECK_IF(h1 % n1Size != 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "h1 [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        OP_CHECK_IF(hRope % n1Size != 0,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "hRope [%ld] should be a multiple of n1Size [%ld].", h1, n1Size),
                   return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        dSizeRope = hRope / n1Size;
        n2Size = h2 / dSize;
        dSizeV = h3 / n2Size;
    }

    return true;
}

bool FlashAttentionScoreTilingRegbase::Analyze4DimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                                                       const gert::Shape &valueShape, size_t layoutLen,
                                                       const gert::Shape *queryRopeShape)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2); // 2: N idx
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(2),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim2 [%ld].", n1Size,
                                                   queryShape.GetDim(2)),
                       return false);
            gSize = queryShape.GetDim(2) / n2Size; // 2: N idx
            dSize = queryShape.GetDim(3);          // 3: D idx
            dSizeV = valueShape.GetDim(3);          // 3: D idx
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_3);
            }
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BSND));
            tilingKeyLayout = LayoutType::LAYOUT_BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1); // 1: N idx
            OP_CHECK_IF(n2Size == 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "N2 is zero."), return false);
            OP_CHECK_IF(n1Size != queryShape.GetDim(1),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "head_num is [%ld], but got query dim1 [%ld].", n1Size,
                                                   queryShape.GetDim(1)),
                       return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2); // 2: S idx
            s2Size = keyShape.GetDim(2);   // 2: S idx
            dSize = queryShape.GetDim(3);  // 3: D idx
            dSizeV = valueShape.GetDim(3);  // 3: D idx
            if (hasRope) {
                dSizeRope = queryRopeShape->GetDim(DIM_NUM_3);
            }
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            inputParamsRegbase_->set_layoutType(static_cast<uint8_t>(LayoutType::LAYOUT_BNSD));
            tilingKeyLayout = LayoutType::LAYOUT_BNSD;
        } else {
            return false;
        }
    }

    return true;
}

ge::graphStatus FlashAttentionScoreTilingRegbase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputDtype == ge::DT_HIFLOAT8) && (hasAttenMask || hasPse || hasDropOut || hasRope ||
                tilingKeyLayout != LayoutType::LAYOUT_BSND ||
                bSize != 1 || n1Size != n2Size || dSize != 128 || dSizeV != 128 ||
                !((s1Size == 57600 && s2Size == 57600 && n1Size == 5) || (s1Size == 7200 && s2Size == 512 && n1Size == 40))),
                OPS_REPORT_VECTOR_INNER_ERR(opName, "HIFLOAT8 can only support layout:BSND without any optional inputs, "
                "and the input shape must be: "
                "query:[1, 57600, 5, 128] key:[1, 57600, 5, 128] value:[1, 57600, 5, 128] or "
                "query:[1, 7200, 40, 128] key:[1, 512, 40, 128] value:[1, 512, 40, 128]."
                "[hasAttenMask:%d, hasPse:%d, hasDropOut:%d, hasRope:%d, input_layout:%s, bSize:%d, s1Size:%d, s2Size:%d, "
                "n1Size:%d, n2Size:%d, dSize:%d, dSizeV:%d]", hasAttenMask, hasPse, hasDropOut, hasRope, inputLayout,
                bSize, s1Size, s2Size, n1Size, n2Size, dSize, dSizeV), return ge::GRAPH_FAILED);

    if (hasRope && (dSize != 128 || dSizeRope != 64)) {
        OPS_REPORT_VECTOR_INNER_ERR(opName, "MLA concat only support dSize=128, dSizeRope=64.");
        return ge::GRAPH_FAILED;
    }

    inputParamsRegbase_->set_bSize(bSize);
    inputParamsRegbase_->set_n2Size(n2Size);
    inputParamsRegbase_->set_gSize(gSize);
    inputParamsRegbase_->set_s1Size(s1Size);
    inputParamsRegbase_->set_s2Size(s2Size);
    inputParamsRegbase_->set_dSize(dSize);
    inputParamsRegbase_->set_dSizeV(dSizeV);
    inputParamsRegbase_->set_dSizeRope(dSizeRope);
    inputParamsRegbase_->set_keepProb(keepProb);
    inputParamsRegbase_->set_scaleValue(scaleValue);
    inputParamsRegbase_->set_pseType(static_cast<uint32_t>(pseType));
    inputParamsRegbase_->set_keepProbUint8(keepProbUint8);
    inputParamsRegbase_->set_seed(seed);
    inputParamsRegbase_->set_offset(offset);

    OP_LOGD(context_, "input ParamsRegbase: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], keepProb[%f], scaleValue[%f],"
                        "pseType:%ld.", bSize, n2Size, gSize, s1Size, s2Size, dSize, keepProb, scaleValue, pseType);

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeTndPseOptionalInput(PseShapeType &pseShapeType,
                                                                const gert::Shape &pseShapeDims,
                                                                size_t pseDimNum, int64_t pseBSize)
{
    int64_t accumS1S2 = 0;
    for (auto i = 0; i < bSize; i++) {
        accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
    }
    if (pseBSize == accumS2 * n1Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    } else if (pseBSize == accumS1S2 * n1Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else if (pseDimNum == PSE_DIM_NUM && (pseShapeDims.GetDim(0) == 1 || pseShapeDims.GetDim(0) == bSize) &&
               pseShapeDims.GetDim(1) == n1Size && pseShapeDims.GetDim(DIM_NUM_2) == PSE_ALIBI_S_SIZE &&
               pseShapeDims.GetDim(DIM_NUM_3) == s2Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else {
        OP_LOGE(context_, "get unsupported pse shape");
        return false;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeGeneralPseOptionalInput(PseShapeType &pseShapeType,
                                                                    const gert::Shape &pseShapeDims,
                                                                    size_t pseDimNum, int64_t pseBSize)
{
    if (pseDimNum != PSE_DIM_NUM) {
        OP_LOGE(context_, "pse dim should be 4, but got %zu", pseDimNum);
        return false;
    }
    if (pseBSize != bSize && pseBSize != 1) {
        OP_LOGE(context_, "pse batchsize should be 1 or %ld, but got %ld", bSize, pseBSize);
        return false;
    }

    int64_t pseDim1Size = pseShapeDims.GetDim(1);
    int64_t pseDim2Size = pseShapeDims.GetDim(DIM_NUM_2);
    int64_t pseDim3Size = pseShapeDims.GetDim(DIM_NUM_3);
    if (pseDim1Size == n1Size && pseDim2Size == s1Size && pseDim3Size == s2Size) { // 2: pre last axiss
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else if (pseDim1Size == n1Size && pseDim2Size == 1 && pseDim3Size == s2Size) {
        pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    } else if (pseDim1Size == n1Size && pseDim2Size == static_cast<int64_t>(PSE_ALIBI_S_SIZE) &&
               pseDim3Size == s2Size) {
        if (s1Size < pseDim2Size) {
            OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                        pseDim2Size, pseDim3Size);
            return false;
        }
        pseShapeType = PseShapeType::PSE_B_N2_G_S1_S2;
    } else {
        OP_LOGE(opName, "get unsupported pse shape, the shape is [%ld, %ld, %ld, %ld]", pseBSize, pseDim1Size,
                    pseDim2Size, pseDim3Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzePseOptionalInput()
{
    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PseShapeType pseShapeType = PseShapeType::PSE_B_N2_G_1_S2;
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        hasPse = true;
        auto &pseShapeDims = pseShape->GetStorageShape();
        size_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = pseShapeDims.GetDim(0);
        if (pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            if (pseDimNum != SLOPE_BN_DIM_NUM && pseDimNum != SLOPE_N_DIM_NUM) {
                OP_LOGE(context_, "pse inner mode, unsupported pse shape");
                return false;
            }
            pseShapeType = PseShapeType::PSE_B_N2_G_SLOPE;
            if (pseDimNum == 1) {
                pseShapeType = PseShapeType::PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        } else if (inputParamsRegbase_->get_layoutType() == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            OP_CHECK_IF(!AnalyzeTndPseOptionalInput(pseShapeType, pseShapeDims, pseDimNum, pseBSize),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze tnd mode pse shape fail."),
                       return false);
        } else {
            OP_CHECK_IF(!AnalyzeGeneralPseOptionalInput(pseShapeType, pseShapeDims, pseDimNum, pseBSize),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze general pse shape fail."),
                       return false);
        }
        inputParamsRegbase_->set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    inputParamsRegbase_->set_pseShapeType(static_cast<uint8_t>(pseShapeType));
    return true;
}

bool FlashAttentionScoreTilingRegbase::Analyze4DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                                                   const gert::Shape &attenMaskStorageShape)
{
    int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
    int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
    int64_t attenMaskDim2Size = attenMaskStorageShape.GetDim(DIM_NUM_2);
    int64_t attenMaskDim3Size = attenMaskStorageShape.GetDim(DIM_NUM_3);
    if (attenMaskDim0Size == 1 && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
        attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_S1_S2;
    } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == 1 && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_B_1_1_S1_S2;
    } else if (attenMaskDim0Size == bSize && attenMaskDim1Size == n1Size && attenMaskDim2Size == s1Size &&
                attenMaskDim3Size == s2Size) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_B_N2_G_S1_S2;
    } else {
        OP_LOGE(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld, %ld, %ld]",
                    attenMaskDim0Size, attenMaskDim1Size, attenMaskDim2Size, attenMaskDim3Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::Analyze2DimAttenOptionalInput(AttenMaskShapeType &attenMaskShapeType,
                                                                   const gert::Shape &attenMaskStorageShape)
{
    int64_t attenMaskDim0Size = attenMaskStorageShape.GetDim(0);
    int64_t attenMaskDim1Size = attenMaskStorageShape.GetDim(1);
    if ((attenMaskDim0Size == s1Size && attenMaskDim1Size == s2Size) ||
        (attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE))) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_S1_S2; // maybe [S1, S2]
    } else if (attenMaskDim0Size == accumS1 && attenMaskDim1Size == accumS2) {
        attenMaskShapeType = AttenMaskShapeType::ATTEN_1_1_1_T_T;
    } else {
        OP_LOGE(context_, "get unsupported atten_mask shape, the shape is [%ld, %ld]", attenMaskDim0Size,
                    attenMaskDim1Size);
        return false;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeAttenOptionalInputDimNumLimit(const gert::Shape &attenMaskStorageShape,
                                                                          size_t attenMaskDimNum)
{
    if ((attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE) &&
         attenMaskCompressMode != static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE)) &&
        ((attenMaskStorageShape.GetDim(attenMaskDimNum - ATTEN_MASK_S1_REV_INDEX) != ATTEN_MASK_COMPRESS_LIMIT) ||
         (attenMaskStorageShape.GetDim(attenMaskDimNum - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
        OP_LOGE(context_, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
        return false;
    }
    if (attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::PREFIX_MODE) &&
        ((attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) !=
          ATTEN_MASK_COMPRESS_PREFIX_LIMIT) ||
         (attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
        OP_LOGE(context_, "In the prefix attenmask compression, please set the atten_mask_shape to [3072,2048].");
        return false;
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeAttenOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);
    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        hasAttenMask = true;
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid attenMask dtype[%s], only support bool or uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                   return false);

        inputParamsRegbase_->set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = AttenMaskShapeType::ATTEN_B_N2_G_S1_S2;
        auto &attenMaskStorageShape = attenMaskShape->GetStorageShape();
        size_t attenMaskDimNum = attenMaskStorageShape.GetDimNum();
        if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_4) {
            OP_CHECK_IF(!Analyze4DimAttenOptionalInput(attenMaskShapeType, attenMaskStorageShape),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze dim 4 attenmask shape fail."),
                       return false);
        } else if (attenMaskDimNum == ATTENTION_MASK_DIM_NUM_2) {
            OP_CHECK_IF(!Analyze2DimAttenOptionalInput(attenMaskShapeType, attenMaskStorageShape),
                       OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze dim 2 attenmask shape fail."),
                       return false);
        } else {
            OP_LOGE(context_, "atten mask dim should be 2 or 4, but got %zu", attenMaskDimNum);
            return false;
        }

        inputParamsRegbase_->set_attenMaskShapeType(static_cast<uint8_t>(attenMaskShapeType));
        OP_CHECK_IF(!AnalyzeAttenOptionalInputDimNumLimit(attenMaskStorageShape, attenMaskDimNum),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze attenmask dim num limit error."),
                   return false);
        inputParamsRegbase_->set_attenMaskS2Size(attenMaskStorageShape.GetDim(attenMaskDimNum - 1));
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeDropOptionalInput()
{
    auto dropMaskShape = context_->GetOptionalInputShape(DROP_MASK_INPUT_INDEX);
    auto dropMaskInput = context_->GetOptionalInputDesc(DROP_MASK_INPUT_INDEX);
    if (dropMaskInput != nullptr && dropMaskShape != nullptr && dropMaskShape->GetStorageShape().GetDimNum() != 0) {
        if (!hasDropOut) {
            OP_LOGE(context_, "Dropmask parameter is invalid, please check keepProb.");
            return false;
        }
        auto dropMaskDtype = dropMaskInput->GetDataType();
        OP_CHECK_IF(dropMaskDtype != ge::DT_UINT8,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dropMask dtype[%s], only support uint8.",
                                               ge::TypeUtils::DataTypeToSerialString(dropMaskDtype).c_str()),
                   return false);
        int64_t dimNum = dropMaskShape->GetStorageShape().GetDimNum();
        int64_t dropMaskShapeSize = 1;
        int64_t shapeSize = 0;
        for (int64_t i = 0; i < dimNum; ++i) {
            int64_t dimValue = dropMaskShape->GetStorageShape().GetDim(i);
            dropMaskShapeSize *= dimValue;
        }
        if (inputParamsRegbase_->get_layoutType() == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            int64_t accumS1S2 = 0;
            for (auto i = 0; i < bSize; i++) {
                accumS1S2 += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeSize = accumS1S2 * n1Size;
        } else {
            shapeSize = bSize * n1Size * s1Size * s2Size;
        }
        shapeSize = AlignUp(shapeSize, BYTE_BIT_NUM) / BYTE_BIT_NUM;
        if (dropMaskShapeSize < shapeSize) {
            OP_LOGE(context_, "Input dropMask shapeSize is invalid, it should not be less than %ld, but got %ld",
                      shapeSize, dropMaskShapeSize);
            return false;
        }
        dropMaskOuter = true;
    }

    // if s2Size algined to 8, then no need dropMaskOp to transfer dropMask from bit to byte format
    inputParamsRegbase_->set_dropMaskOuter(static_cast<uint8_t>(dropMaskOuter));
    inputParamsRegbase_->set_needDropMaskOp(static_cast<uint8_t>(dropMaskOuter && s2Size % ALIGNED_NUM_8 != 0));
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        auto needDropMaskOp = (dropMaskOuter) && (s2Size % ALIGNED_NUM_8 != 0 || bSize > 1);
        inputParamsRegbase_->set_needDropMaskOp(static_cast<uint8_t>(needDropMaskOp));
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeFp8OptionalInput()
{
    auto dScaleQShape = context_->GetOptionalInputShape(D_Q_SCALE_INDEX);
    auto dScaleQInput = context_->GetOptionalInputDesc(D_Q_SCALE_INDEX);
    if (dScaleQInput != nullptr && dScaleQShape != nullptr && dScaleQShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleQDtype = dScaleQInput->GetDataType();
        OP_CHECK_IF(dScaleQDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleQDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleQShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleQShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n1Size ||
        (dimValue2 != (s1Size + QUANT_BLOCK_SIZE - 1) / QUANT_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleQ dimNump[%ld][%ld][%ld][%ld], only support [B, N1, ceil(S1/128), 1]",
                dimValue0, dimValue1, dimValue2, dimValue3),
                return false);
    }

    auto dScaleKShape = context_->GetOptionalInputShape(D_K_SCALE_INDEX);
    auto dScaleKInput = context_->GetOptionalInputDesc(D_K_SCALE_INDEX);
    if (dScaleKInput != nullptr && dScaleKShape != nullptr && dScaleKShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleKDtype = dScaleKInput->GetDataType();
        OP_CHECK_IF(dScaleKDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleKDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleKShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleKShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
            (dimValue2 != (s2Size + QUANT_K_BLOCK_SIZE  - 1) / QUANT_K_BLOCK_SIZE ) || dimValue3 != D_SCALE_DIM_NUM_1,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleK dimNump[%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/256), 1]",
                    dimValue0, dimValue1, dimValue2, dimValue3),
                    return false);
    }

    auto dScaleVShape = context_->GetOptionalInputShape(D_V_SCALE_INDEX);
    auto dScaleVInput = context_->GetOptionalInputDesc(D_V_SCALE_INDEX);
    if (dScaleVInput != nullptr && dScaleVShape != nullptr && dScaleVShape->GetStorageShape().GetDimNum() != 0) {
        auto dScaleVDtype = dScaleVInput->GetDataType();
        OP_CHECK_IF(dScaleVDtype != ge::DT_FLOAT,
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dtype[%s], only support float32.",
                                               ge::TypeUtils::DataTypeToSerialString(dScaleVDtype).c_str()),
                   return false);
        int64_t dimNum = dScaleVShape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum != D_SCALE_DIM_NUM_4,
                OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dimNum [%ld], only support 4 dims.", dimNum),
                return false);
        int64_t dimValue0 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_0);
        int64_t dimValue1 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_1);
        int64_t dimValue2 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_2);
        int64_t dimValue3 = dScaleVShape->GetStorageShape().GetDim(D_SCALE_DIM_NUM_3);
        OP_CHECK_IF(dimValue0 != bSize || dimValue1 != n2Size ||
            (dimValue2 != (s2Size + QUANT_V_BLOCK_SIZE - 1) / QUANT_V_BLOCK_SIZE) || dimValue3 != D_SCALE_DIM_NUM_1,
                    OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid dScaleV dimNum [%ld][%ld][%ld][%ld], only support [B, N2, ceil(S2/512), 1]",
                    dimValue0, dimValue1, dimValue2, dimValue3),
                    return false);
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::AnalyzeOptionalInput()
{
    OP_CHECK_IF(!AnalyzePseOptionalInput() || !AnalyzeAttenOptionalInput() || !AnalyzeDropOptionalInput() ||
               !AnalyzeFp8OptionalInput(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Analyze Optional Input error."), return false);
    OP_LOGD(context_, "hasPse: %d, hasAttenMask: %d, hasDropOut: %d, dropMaskouter %d.",
              hasPse, hasAttenMask, hasDropOut, dropMaskOuter);
    return true;
}

void FlashAttentionScoreTilingRegbase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(coreNum));
    multiCoreParamsRegbase_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    multiCoreParamsRegbase_->set_totalSize(totalSize);
    multiCoreParamsRegbase_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    multiCoreParamsRegbase_->set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParamsRegbase_->get_splitFactorSize()));
    multiCoreParamsRegbase_->set_splitCoreMode(static_cast<uint8_t>(splitCoreMode));
    multiCoreParamsRegbase_->set_firstFullLoadS1OuterIdx(firstFullLoadS1OuterIdx);
}

void FlashAttentionScoreTilingRegbase::SetSparseParamsRegbase(int64_t maxCoreNum)
{
    if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    inputParamsRegbase_->set_s1SparseValidSize(s1SparseValidSize);
    inputParamsRegbase_->set_s2SparseValidSize(s2SparseValidSize);

    if (inputParamsRegbase_->get_sparseType() == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        std::vector<std::vector<int64_t>> sparseValidArray;
        for (int64_t bIdx = 0; bIdx < bSize; bIdx++) {
            sparseValidArray.emplace_back(std::vector<int64_t>(multiCoreParamsRegbase_->get_s1OuterSize(), 0));
            InitSparseValidArray(sparseValidArray.back(), bIdx);
        }
        SetPrefixSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);
    } else {
        std::vector<int64_t> sparseValidArray(multiCoreParamsRegbase_->get_s1OuterSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, *multiCoreParamsRegbase_, maxCoreNum);
    }
}

ge::graphStatus FlashAttentionScoreTilingRegbase::DoOpTiling()
{
    OP_LOGD(context_, "try template[%s]", templateName);
    OP_CHECK_IF(dSize > HEAD_DIM_MAX_VALUE,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "query or key dSize is not in range:(0, 768]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dSizeV > HEAD_DIM_MAX_VALUE,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "value dSize is not in range:(0, 768]"), return ge::GRAPH_FAILED);
    CalcDBasicBlock();
    CalcDVBasicBlock();
    CalcS1S2BasicBlock();
    SparseEnum sparseType = SparseEnum::ALL;
    OP_CHECK_IF(!GetSparseInfo(sparseType), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to get sparse info."),
               return ge::GRAPH_FAILED);
    SetSparseTilingInfo(sparseType);
    inputParamsRegbase_->set_implMode(static_cast<uint8_t>(implMode));
    implMode = (hasAttenMask && inputDtypeBytes != DATA_TYPE_FP32) ? implMode : ImplMode::AA_HIGH_PRECISION;
    if (!isSparseValidSizeAligned) {
        s1SparseValidSize = preTokens;
        s2SparseValidSize = nextTokens;
    }
    if (SetQKVStartIdx() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetOutputDtype();
    multiCoreParamsRegbase_->set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    int64_t totalSize = CalcTotalSize();
    SetSplitCoreModeParam();
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    SetSparseParamsRegbase(static_cast<int64_t>(aicNum));
    OP_CHECK_IF(!SetPseAlibiParamsRegbase(), OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to set pse alibi info."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingRegbase::CalcThresholdForS2Size() {
    int64_t l2CacheSizeForKandV = L2_CACHE_SIZE * NUM_1024 * NUM_1024;
    if (bSize == 0 || n2Size == 0 || dBasicBlock == 0 || dVBasicBlock == 0) {
        OP_LOGE(context_, "The product of bSize[%ld], nSize[%ld], dSize[%ld] and dVSzie[%ld] cannot be zero.",
            bSize, n1Size, dBasicBlock, dVBasicBlock);
        thresholdS2Size = std::numeric_limits<int64_t>::max();
        return;
    }

    int64_t typeSize = ge::GetSizeByDataType(inputDtype);
    // 当bn1比较大时，部分s比较小的场景，性能受限并非在L2cache资源的复用；
    int64_t bnSize = std::min(bSize * n2Size, static_cast<int64_t>(aicNum));
    thresholdS2Size = l2CacheSizeForKandV / (bnSize * (dBasicBlock + dVBasicBlock) * typeSize);
}

bool FlashAttentionScoreTilingRegbase::IsUseSpliteCoreMode(SparseMode inputSparseMode) {
    if (inputSparseMode == SparseMode::LEFT_UP_CAUSAL) {
        return std::min(s1Size, s2Size) >= thresholdS2Size;
    }

    if (inputSparseMode == SparseMode::RIGHT_DOWN_CAUSAL) {
        if (s1Size <= s2Size) {
            return s2Size >= thresholdS2Size;
        }
    }

    return false;
}

void FlashAttentionScoreTilingRegbase::SetSplitCoreModeParam()
{
    CalcThresholdForS2Size();

    // 索引从0开始，需要将基本块个数减1
    if ((sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) &&
        IsUseSpliteCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
        firstFullLoadS1OuterIdx = CeilDivision(std::min(s1Size, s2Size), s1BasicBlock) - 1;
        splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
    } else if ((sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) &&
        IsUseSpliteCoreMode(SparseMode::RIGHT_DOWN_CAUSAL)) {
        firstFullLoadS1OuterIdx = multiCoreParamsRegbase_->get_s1OuterSize() - 1;
        splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
    } else if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
        if (!hasAttenMask && s2Size >= thresholdS2Size) {
            firstFullLoadS1OuterIdx = -1;
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        } else if (preTokens >= s1Size && nextTokens == 0 && IsUseSpliteCoreMode(SparseMode::LEFT_UP_CAUSAL)) {
            firstFullLoadS1OuterIdx = CeilDivision(std::min(s1Size, s2Size), s1BasicBlock) - 1;
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        } else if (inputDtype == ge::DT_HIFLOAT8) {
            firstFullLoadS1OuterIdx = -1;
            splitCoreMode = SplitCoreMode::SQ_MULTI_CORE_FIRST;
        }
    }

    OP_LOGD(context_, "sparseMode: %ld, firstFullLoadS1OuterIdx: %ld, splitCoreMode: %d, s2SizeThreshold: %d.",
        sparseMode, firstFullLoadS1OuterIdx, splitCoreMode, thresholdS2Size);
}

void FlashAttentionScoreTilingRegbase::SetSparseTilingInfo(SparseEnum &sparseType)
{
    inputParamsRegbase_->set_attenMaskCompressMode(attenMaskCompressMode);
    inputParamsRegbase_->set_sparseType(static_cast<uint8_t>(sparseType));
    inputParamsRegbase_->set_preTokens(preTokens);
    inputParamsRegbase_->set_nextTokens(nextTokens);
}

void FlashAttentionScoreTilingRegbase::EnableBandInvalidLineImplMode()
{
    if (implMode == ImplMode::AA_INVALID_LINE_HIGH_PRECISION) {
        return;
    }
    // pretoken and nexttoken are already valid values (leftup vertex) after adjusted
    if (preTokens < (s1Size - s2Size) || nextTokens < 0) {
        implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
        OP_LOGI(context_, "Enable invalid line impl mode.");
        return;
    }
}

int64_t FlashAttentionScoreTilingRegbase::CalcTotalSize()
{
    int64_t totalSize = bSize * n2Size * gSize * multiCoreParamsRegbase_->get_s1OuterSize();
    return totalSize;
}

bool FlashAttentionScoreTilingRegbase::PretokenAndNexttokenAdjustment(SparseEnum &sparseType)
{
    if (sparseMode == static_cast<int64_t>(SparseMode::ALL_MASK) ||
        sparseMode == static_cast<int64_t>(SparseMode::PREFIX) ||
        sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens and nextTokens will be reset max int value.",
                      preTokens, nextTokens, sparseMode);
            preTokens = std::numeric_limits<int32_t>::max();
            nextTokens = std::numeric_limits<int32_t>::max();
        }
        sparseType = (sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) ?
                     static_cast<SparseEnum>(static_cast<uint8_t>(SparseMode::PREFIX)) :
                     static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
    } else if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
        if (preTokens != s1Size || nextTokens != 0) {
            OP_LOGW(context_,
                      "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                      "preTokens will be reset max int value and nextTokens will be reset 0.",
                      preTokens, nextTokens, sparseMode);
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
        }
        sparseType = SparseEnum::CAUSAL;
    } else if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        if (s1Size == s2Size) {
            if (preTokens != s1Size || nextTokens != 0) {
                OP_LOGW(context_,
                          "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                          "preTokens will be reset max int value and nextTokens will be reset 0.",
                          preTokens, nextTokens, sparseMode);
                preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
                nextTokens = 0;
            }
            sparseType = SparseEnum::CAUSAL;
        } else {
            // unequal S, change to band
            preTokens = s1Size;
            nextTokens = s2Size - s1Size;
            OP_LOGD(context_,
                      "Unequal s, sparseType rightDownCasual reset to band, and reset preTokens[%ld] "
                      "and nextTokens[%ld].",
                      preTokens, nextTokens);
            sparseType = SparseEnum::BAND;
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = preTokens;
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
        }
    } else if (sparseMode == static_cast<int64_t>(SparseMode::BAND)) {
        // unequal s, pretoken and nexttoken count from rigthDown vertex, need to change to leftUp vertex
        if (s1Size != s2Size) {
            preTokens = s1Size - s2Size + preTokens;
            nextTokens = s2Size - s1Size + nextTokens;
        }
        if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::SparseBandModeCheck(int64_t maxS1Val, int64_t maxS2Val, int64_t minS1Val,
                                                         int64_t minS2Val, SparseEnum &sparseType)
{
    int64_t oriPreTokens = (sparseMode == static_cast<int64_t>(SparseMode::BAND)) ?
                           (preTokens + s2Size - s1Size) : preTokens;
    int64_t oriNextTokens = (sparseMode == static_cast<int64_t>(SparseMode::BAND)) ?
                            (nextTokens + s1Size - s2Size) : nextTokens;
    if (preTokens >= 0 && nextTokens >= 0) {
        if (preTokens >= maxS1Val && nextTokens >= maxS2Val) {
            OP_LOGW(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxS1Val[%ld] "
                      "maxS2Val[%ld].",
                      oriPreTokens, oriNextTokens, maxS1Val, maxS2Val);
            return true;
        }
        s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
        s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
        isSparseValidSizeAligned = true;
        sparseType = SparseEnum::BAND;
        // check need to enable AA_INVALID_LINE_HIGH_PRECISION
        EnableBandInvalidLineImplMode();
        return true;
    }

    if (preTokens < 0 && nextTokens < 0) {
        OP_LOGE(context_, "PreTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.",
                  oriPreTokens, oriNextTokens);
        return false;
    }

    if (preTokens < 0 && nextTokens >= 0) {
        int64_t absPreTokens = std::abs(preTokens);
        if (nextTokens >= absPreTokens && absPreTokens < minS2Val) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S1[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS1Val);
            return false;
        }
    }

    if (preTokens >= 0 && nextTokens < 0) {
        int64_t absNextTokens = std::abs(nextTokens);
        if (absNextTokens <= preTokens && absNextTokens < minS1Val) {
            // check need to enable AA_INVALID_LINE_HIGH_PRECISION
            EnableBandInvalidLineImplMode();
            s1SparseValidSize = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            s2SparseValidSize = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            isSparseValidSizeAligned = true;
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(context_,
                      "PreTokens[%ld] and nextTokens[%ld] config error with S2[%ld], there is no valid data block.",
                      oriPreTokens, oriNextTokens, minS2Val);
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::SparseModeProcess(SparseEnum &sparseType)
{
    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    if (sparseMode == static_cast<int64_t>(SparseEnum::PREFIX) || sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        std::ostringstream failReason;
        sparseType = GetPrefixNList(failReason);
        if (sparseType != SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
            OP_LOGE(context_, "[%s] %s.", templateName, failReason.str().c_str());
            return false;
        }

        if (sparseType == SparseEnum::PREFIX && sparseMode == static_cast<int64_t>(SparseMode::PREFIX) &&
            inputParamsRegbase_->get_attenMaskShapeType() !=
                static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_N2_G_S1_S2) &&
            inputParamsRegbase_->get_attenMaskShapeType() !=
                static_cast<uint8_t>(AttenMaskShapeType::ATTEN_B_1_1_S1_S2) && bSize != 1) {
            OP_LOGE(context_, "Prefix mode get invalid atten_mask shape, should be [BNSS] or [B1SS].");
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::GetSparseInfo(SparseEnum &sparseType)
{
    OP_LOGD(context_, "check sparse info: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], hasAttenMask[%d].",
              preTokens, nextTokens, s1Size, s2Size, hasAttenMask);
    if (sparseMode > static_cast<int64_t>(SparseMode::PREFIX_COMPRESS)) {
        OP_LOGE(context_, "Not support sparse mode of %ld.", sparseMode);
        return false;
    }

    if (!hasAttenMask) {
        return true;
    }

    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return true;
    }

    if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
        if (preTokens >= s1Size && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
        } else {
            if (preTokens >= s1Size && nextTokens >= s2Size) {
                return true;
            }
            if (!SparseBandModeCheck(s1Size, s2Size, s1Size, s2Size, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}


bool FlashAttentionScoreTilingRegbase::InitSparseValidArray(std::vector<int64_t> &sparseValidArray, int64_t bIdx)
{
    OP_CHECK_IF(sparseValidArray.empty(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "Sparse valid array size should be larger than 0."), return false);
    uint8_t sparseType = inputParamsRegbase_->get_sparseType();
    if (sparseType == static_cast<uint8_t>(SparseEnum::PREFIX)) {
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t s2IgnoredEndLen =
                inputParamsRegbase_->get_s1Size() - s1BasicBlock * (i + 1);
            int64_t s2EndLen = 0;
            s2IgnoredEndLen = std::max(static_cast<int64_t>(0), s2IgnoredEndLen);
            if (inputParamsRegbase_->get_s2Size() > s2IgnoredEndLen) {
                s2EndLen = inputParamsRegbase_->get_s2Size() - s2IgnoredEndLen;
                s2EndLen = std::max(s2EndLen, prefixNData[bIdx]);
            } else {
                s2EndLen = inputParamsRegbase_->get_s2Size();
                s2EndLen = std::min(s2EndLen, prefixNData[bIdx]);
            }

            s2EndLen = std::min(s2EndLen, inputParamsRegbase_->get_s2Size());
            sparseValidArray[i] = CeilDivision(s2EndLen, s2BasicBlock);
        }
    } else {
        int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
        int64_t validS1Size = CeilDivision(s1SparseValidSize, s1BasicBlock);
        int64_t validS2Size = CeilDivision(s2SparseValidSize, s2BasicBlock);
        for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
            int64_t reduceBlk =
                (i < validS1Size) ? 0 : (CeilDivision((i + 1) * s1BasicBlock - s1SparseValidSize, s2BasicBlock) - 1);
            int64_t addBlk =
                std::min(s2BlkNum - validS2Size,
                         CeilDivision((i + 1) * s1BasicBlock + s2SparseValidSize, s2BasicBlock) - validS2Size);
            int64_t validBlockNum = validS2Size - reduceBlk + addBlk;
            sparseValidArray[i] = validBlockNum > 0 ? validBlockNum : INVALID_ROW_SPARSE_RATIO;
            maxValidS2Len = std::max(sparseValidArray[i] * s1BasicBlock, maxValidS2Len);
        }
    }
    return true;
}

bool FlashAttentionScoreTilingRegbase::PartitionSparseData(const std::vector<int64_t> &sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t> &partitionResult)
{
    OP_CHECK_IF(partitionResult.empty(),
               OPS_REPORT_VECTOR_INNER_ERR(opName, "partitionResult size should be larger than 0."), return false);

    OP_CHECK_IF(sparseRollingArraySum <= 0,
               OPS_REPORT_VECTOR_INNER_ERR(opName, "sparseRollingArraySum should be larger than 0."), return false);
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }

        loadSize += loadNext;
    }

    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

void FlashAttentionScoreTilingRegbase::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>> &sparseValidArray,
                                                             MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum)
{
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()), maxCoreNum);
    int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();
    for (int64_t idx = 0; idx < maxCoreNum; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    if (totalSize <= validAivNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validAivNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0LL);
    }
    int64_t n2G = n2Size * gSize;
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validAivNum);
    int64_t s1OuterSize = multiCoreParamsRegbase_->get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index + 1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < maxCoreNum; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreTilingRegbase::SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray,
                                                       MultiCoreParamsRegbase &multiCoreParamsRegbase, int64_t maxCoreNum)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParamsRegbase.get_coreNum()), maxCoreNum);
    int64_t totalSize = multiCoreParamsRegbase.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParamsRegbase.get_sparseStartIdxPtr();
    for (int64_t idx = 0; idx < maxCoreNum; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validAivNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validAivNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validAivNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OP_CHECK_IF(validAivNum <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName, "validAivNum should be larger than 0."),
               return false);
    int64_t loadEachCoreLowerBound = loadTotal / validAivNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validAivNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validAivNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validAivNum,
                                     CeilDivision(loadTotal, validAivNum));
    }
    return true;
}

void FlashAttentionScoreTilingRegbase::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t> &sparseValidArray,
                                                                 int64_t *sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = multiCoreParamsRegbase_->get_totalSize();
    int64_t s1OuterSize = multiCoreParamsRegbase_->get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0L);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0L);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0L);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0L);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OP_LOGD(context_, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
              avgLoadSize);
}


SparseEnum FlashAttentionScoreTilingRegbase::GetPrefixNList(std::ostringstream &failReason)
{
    auto prefixN = context_->GetOptionalInputTensor(PREFIX_INPUT_INDEX);
    if (prefixN == nullptr) {
        OP_LOGW(context_, "[%s]prefixN is null pointer while sparse mode is prefix", templateName);
        failReason << "prefixN is null pointer while sparse mode is prefix";
        return SparseEnum::ALL;
    }

    auto &prefixShape = prefixN->GetShape().GetStorageShape();
    if (prefixShape.GetDimNum() != 1) {
        OP_LOGW(context_, "[%s] prefixN shape is invalid, DimNum should be 1, but it is %lu.", templateName,
                  prefixShape.GetDimNum());
        failReason << "prefixN shape is invalid, DimNum should be 1, but it is " << prefixShape.GetDimNum();
        return SparseEnum::ALL;
    }
    if (prefixShape.GetDim(0) != bSize) {
        OP_LOGW(context_, "[%s] prefixN is invalid, it should be the same size as bSize[%ld], but it is %ld.",
                  templateName, bSize, prefixShape.GetDim(0));
        failReason << "prefixN is invalid, it should be the same size as bSize[" << bSize
                   << "], but it is " << prefixShape.GetDim(0);
        return SparseEnum::ALL;
    }
    /* Get Data from tensor. */
    prefixNData = prefixN->GetData<int64_t>();
    if (prefixNData == nullptr) {
        OP_LOGW(context_, "[%s]prefixN data is null pointer", templateName);
        failReason << "prefixN data is null pointer";
        return SparseEnum::ALL;
    }

    int64_t nMin = ((s2Size - s1Size) > 0) ? (s2Size - s1Size) : 0;
    for (int64_t i = 0; i < bSize; ++i) {
        if (prefixNData[i] < nMin || prefixNData[i] > s2Size) {
            OP_LOGW(context_, "[%s] batch[%ld] prefixN=%ld is invalid, should be in range of [%ld, %ld]",
                      templateName, i, prefixNData[i], nMin, s2Size);
            failReason << "batch[" << i << "] prefixN=" << prefixNData[i] << " is invalid, should be in range of ["
                       << nMin << ", " << s2Size << "]";
            return SparseEnum::ALL;
        }

        if (s1Size > s2Size && prefixNData[i] == 0) {
            implMode = ImplMode::AA_INVALID_LINE_HIGH_PRECISION;
            OP_LOGD(context_, "Enable invalid line impl mode.");
        }
    }

    return SparseEnum::PREFIX;
}
void FlashAttentionScoreTilingRegbase::SetOutputDtype() {
    if (inputDtype != ge::DT_FLOAT8_E5M2 && inputDtype != ge::DT_FLOAT8_E4M3FN && inputDtype != ge::DT_HIFLOAT8) {
        outDtype = 0;
    }
}

ge::graphStatus FlashAttentionScoreTilingRegbase::SetQKVStartIdx() {
    inputParamsRegbase_->set_qStartIdx(0);
    inputParamsRegbase_->set_kvStartIdx(0);
    auto qStartIdxTensor = context_->GetOptionalInputTensor(Q_START_IDX_INPUT_INDEX);
    if (qStartIdxTensor != nullptr) {
        auto &qStartIdxShape = qStartIdxTensor->GetShape().GetStorageShape();
        if (qStartIdxShape.GetDimNum() >= 1 && qStartIdxShape.GetDim(0) != 0) {
            const int64_t *value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx = value[0];
                inputParamsRegbase_->set_qStartIdx(qStartIdx);
                OP_LOGD(context_, "[%s] SetQKVStartIdx qStartIdx:%ld", templateName, qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = context_->GetOptionalInputTensor(KV_START_IDX_INPUT_INDEX);
    if (kvStartIdxTensor != nullptr) {
        auto &kvStartIdxShape = kvStartIdxTensor->GetShape().GetStorageShape();
        if (kvStartIdxShape.GetDimNum() >= 1 && kvStartIdxShape.GetDim(0) != 0) {
            const int64_t *kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx = kvValue[0];
                inputParamsRegbase_->set_kvStartIdx(kvStartIdx);
                OP_LOGD(context_, "[%s] SetQKVStartIdx kvStartIdx:%ld", templateName, kvStartIdx);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，由于编译器不支持大数值类型转换，kernel侧int_64转float类型时可能发生截断。
    OP_CHECK_IF(kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN, OPS_REPORT_VECTOR_INNER_ERR(opName,
        "kvStartIdx - qStartIdx should >= %d and <= %d, but qStartIdx = %ld, kvStartIdx = %ld.", INT32_MIN, INT32_MAX, qStartIdx, kvStartIdx),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingRegbase::SetPseAlibiParamsRegbase()
{
    inputParamsRegbase_->set_pseEncodeType(static_cast<uint8_t>(PseEncodeType::PSE_ENCODE_NONE));
    auto pseShape = context_->GetOptionalInputShape(PSE_INPUT_INDEX);
    if (pseShape == nullptr) {
        return true;
    }
    auto pseS1Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 2);
    auto pseS2Size = pseShape->GetStorageShape().GetDim(pseShape->GetStorageShape().GetDimNum() - 1);
    if (pseS1Size == PSE_ALIBI_S_SIZE && s1Size > PSE_ALIBI_S_SIZE && pseS2Size == s2Size) {
        if (s1Size != s2Size) {
            OPS_REPORT_VECTOR_INNER_ERR(opName, "Pse alibi only support same S1 S2 when S1 lager than 1024");
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreTilingRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingRegbase::CalcDVBasicBlock() {
    dVBasicBlock = AlignUp(dSizeV, D_TEMPLATE_SPLIT_SIZE);
    if (dTemplateType == DTemplateType::ALIGNED_192 && (hasRope || dVBasicBlock == NUM_128)) {
        dVTemplateType = DTemplateType::ALIGNED_128;
    } else {
        dVTemplateType = dTemplateType;
    }
}


ge::graphStatus FlashAttentionScoreTilingRegbase::PostTiling()
{
    auto numBlocks = CalcTschBlockDim(multiCoreParamsRegbase_->get_coreNum() * 2, aicNum, aivNum);
    context_->SetBlockDim(numBlocks);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    if (inputParamsRegbase_->get_needDropMaskOp() == 1) {
        numBlocks = CalcTschBlockDim(aivNum, aicNum, aivNum);
        context_->SetBlockDim(numBlocks);

        int64_t shapeTotalSize = bSize * n2Size * gSize * s1Size * s2Size;
        auto layoutType = inputParamsRegbase_->get_layoutType();
        if (layoutType == static_cast<uint8_t>(LayoutType::LAYOUT_TND)) {
            for (auto i = 0; i < bSize; i++) {
                dropTotalSize += (actualSeqLenData[i] * actualSeqLenKvData[i]);
            }
            shapeTotalSize = n2Size * gSize * dropTotalSize;
        }
        dropmaskParamsRegbase_->dropMaskAddrOffset = workspaces[0];
        shapeTotalSize = AlignUp(shapeTotalSize, GM_ALIGN);
        workspaces[0] += static_cast<size_t>(shapeTotalSize);
    }
    workspaces[0] += WORK_SPACE_RESERVE_SIZE;
    OP_LOGD(opName, "[%s]tiling data: %s", templateName, GetTilingDataDebugStr().c_str());
    return ge::GRAPH_SUCCESS;
}
} // namespace FA
} // namespace optiling
