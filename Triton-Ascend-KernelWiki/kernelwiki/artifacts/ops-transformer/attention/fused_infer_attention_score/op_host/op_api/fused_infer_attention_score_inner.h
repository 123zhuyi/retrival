/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale, const aclTensor *keyAntiquantOffset, const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset, const aclTensor *keySharedPrefix, const aclTensor *valueSharedPrefix,
    const aclIntArray *actualSharedPrefixLen, const aclTensor *query_rope,
    const aclTensor *key_rope, const aclTensor *keyRopeAntiquantScale, const aclTensor *dequantScaleQuery,
    const aclTensor *learnableSink, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, int64_t queryQuantMode,  int64_t pseType, int64_t outType,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize(
    const aclTensor *query, const aclTensorList *key, const aclTensorList *value, const aclTensor *pseShift,
    const aclTensor *attenMask, const aclTensor *actualSeqLengths, const aclTensor *actualSeqLengthsKv,
    const aclTensor *deqScale1, const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale, const aclTensor *keyAntiquantOffset, const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset, const aclTensor *keySharedPrefix, const aclTensor *valueSharedPrefix,
    const aclTensor *actualSharedPrefixLen, const aclTensor *query_rope,
    const aclTensor *key_rope, const aclTensor *keyRopeAntiquantScale, const aclTensor *dequantScaleQuery,
    const aclTensor *learnableSink, const aclTensor *qStartIdxOptional, const aclTensor *kvStartIdxOptional,
    int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, int64_t queryQuantMode,  int64_t pseType, int64_t outType,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      const aclrtStream stream);

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue);
void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue);
aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outArray);

void FusedInferAttentionScoreProcessSoftmaxLse(bool softmaxLseFlag, const aclTensor *softmaxLse,
                                               const aclTensor *&tempTensor, const aclTensor *&placeHolder);


#ifdef __cplusplus
}
#endif

#endif