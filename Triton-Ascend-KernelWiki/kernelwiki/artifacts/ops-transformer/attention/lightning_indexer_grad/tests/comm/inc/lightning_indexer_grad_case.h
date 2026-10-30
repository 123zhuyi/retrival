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
 * \file lightning_indexer_grad_case.h
 * \brief LightningIndexerGrad 测试用例.
 */
#ifndef LIGHTNING_INDEXER_GRAD_CASE_H
#define LIGHTNING_INDEXER_GRAD_CASE_H

#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "tests/utils/case_with_socversion.h"
#include "tests/utils/op_info_with_socversion.h"
#include "tests/utils/context_with_template_tilingkey.h"
#include "tests/utils/context.h"
#include "lightning_indexer_grad_param.h"

#define LIGHTNING_INDEXER_GRAD_PARAM                                                                                                           \
    (uint8_t * query, uint8_t * key, uint8_t * dy, uint8_t * sparse_indices, uint8_t * weights, uint8_t * actual_seq_lengths_query,            \
     uint8_t * actual_seq_lengths_key, uint8_t * dq, uint8_t * dk, uint8_t * dweights, uint8_t * workspace, uint8_t * tiling)

#define LIGHTNING_INDEXER_GRAD_PARAM_                                                                                                          \
    uint8_t * query, uint8_t * key, uint8_t * dy, uint8_t * sparse_indices, uint8_t * weights, uint8_t * actual_seq_lengths_query,             \
    uint8_t * actual_seq_lengths_key, uint8_t * dq, uint8_t * dk, uint8_t * dweights, uint8_t * workspace, uint8_t * tiling

#define LIG_INPUT_DTYPE                                                                                                                        \
    uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *

#define LIG_INPUT_PARAMS                                                                                                                       \                                  
    query, key, dy, sparse_indices, weights, actual_seq_lengths_query, actual_seq_lengths_key, dq, dk, dweights, workspace, tiling

namespace ops::adv::tests::LIG {
class LightningIndexerGradCase : public ops::adv::tests::utils::CaseWithSocversion {
public:
    using OpInfoWithSocversion = ops::adv::tests::utils::OpInfoWithSocversion;
    using ContextWithTemplateTilingKey = ops::adv::tests::utils::ContextWithTemplateTilingKey<LIG_INPUT_DTYPE>;
    using Tensor = ops::adv::tests::utils::Tensor;

    typedef void(*LightningIndexerGradKernelFunc) LIGHTNING_INDEXER_GRAD_PARAM;

    class DoTilingParam {
    public:
        gert::TilingContext *ctx = nullptr;
        ge::graphStatus ret = ge::GRAPH_SUCCESS;
        gert::Tensor *actSeqLenQTensor = nullptr;
        gert::Tensor *actSeqLenKTensor = nullptr;

    };
    /**
     * 执行 Tiling 前, 修改 TilingContext 回调函数.
     *
     * \attention: 一般用于异常用例.
     */
    typedef void (*PreTilingRunCbf)(DoTilingParam &tilingParam);
public:
    /* 算子控制信息 */
    OpInfoWithSocversion mOpInfo;
    ContextWithTemplateTilingKey mCtx;

    /* 输入/输出 参数 */
    LightningIndexerGradParam mParam;

    gert::OpImplRegisterV2::TilingKernelFunc mLightningIndexerGradOriginTilingFunc = nullptr;
    PreTilingRunCbf mPreTilingRunCbf = nullptr;

    LightningIndexerGradCase();
    LightningIndexerGradCase(const std::function<void(LIG_INPUT_DTYPE)>& templatekeyKernelFunc);
    LightningIndexerGradCase(const char *name, bool enable, const char *dbgInfo, OpInfoWithSocversion incre,
        LightningIndexerGradParam param, int32_t tilingTemplatePriority = kTilingTemplatePriority_Invalid);
    bool Run() override;
    bool DoOpTiling(DoTilingParam &tilingParam);

    static void PreTilingRunCbf_SetPlatformInfoNull(LightningIndexerGradCase::DoTilingParam &tilingParam);
protected:
    std::string mLightningIndexerGradOriginTilingFuncName;
    std::function<void(LIG_INPUT_DTYPE)> mLIGKernelFunc = nullptr;

    bool InitParam() override;
    bool InitOpInfo() override;
    bool InitOpInfoCtx();
    bool InitOriginTilingFunc();
    bool InitCurrentCasePtr() override;
};
} // namespace ops::adv::tests::LIG
#endif // LIGHTNING_INDEXER_GRAD_CASE_H