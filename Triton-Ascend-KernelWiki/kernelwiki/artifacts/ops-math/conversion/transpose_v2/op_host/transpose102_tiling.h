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
 * \file transpose102_tiling.h
 * \brief
 */

#ifndef TRANSPOSE102_TILING_H
#define TRANSPOSE102_TILING_H
#include "transpose_v2_tiling.h"

namespace optiling {

struct Transpose102Params {
    uint64_t dim0Len{1};
    uint64_t dim1Len{0};
    uint64_t dim2Len{0};
    uint64_t dim3Len{1};
    uint64_t dim3LenAlign{1};
    uint64_t tasksPerCore{0};
    uint64_t tailCore{0};
    uint64_t dim1OnceMax{0};
    uint64_t dim2OnceMax{0};
    uint64_t doubleBuffer{2};

    uint64_t subMode{0};
    uint64_t typeSize{0};
    uint64_t block{0};
    uint64_t tasksPerCoreLarge{0};

    uint64_t tilingKey{0};
    uint64_t ubSize{0};
    uint64_t sysWorkspaceSize{0};
    uint64_t workspaceSize{0};
    uint32_t coreNum{0};
};

class Transpose102Tiling {
public:
    explicit Transpose102Tiling(gert::TilingContext* ctx) : context(ctx) {};
    virtual ~Transpose102Tiling() = default;

    ge::graphStatus GetPlatformInfo() {
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        params.coreNum = ascendcPlatform.GetCoreNumAiv();
        params.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, params.ubSize);
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoTiling() {
        // 合轴后都是0213
        int32_t transDim0 = 0;
        int32_t transDim1 = 1;
        const gert::StorageShape* permShape = context->GetInputShape(1);
        // 4 is permSize
        if (permShape->GetStorageShape().GetDim(0) == 4) {
            transDim0 = 1;
            transDim1 = 2; // 2 is one of dim to transpose
        }
        const gert::StorageShape* xShape = context->GetInputShape(0);
        int32_t dimNum = xShape->GetStorageShape().GetDimNum();
        for (auto i = 0; i < transDim0; i++) {
            params.dim0Len *= xShape->GetStorageShape().GetDim(i);
        }
        params.dim1Len = xShape->GetStorageShape().GetDim(transDim0);
        params.dim2Len = xShape->GetStorageShape().GetDim(transDim1);
        for (auto i = transDim1 + 1; i < dimNum; i++) {
            params.dim3Len *= xShape->GetStorageShape().GetDim(i);
        }

        if (params.dim1Len >= params.dim2Len) {
            params.subMode = 0U;
        } else {
            params.subMode = 1U;
        }

        // 分核
        SubCore();
        // 分块
        auto xDataType = context->GetInputDesc(0)->GetDataType();
        if (xDataType == ge::DataType::DT_FLOAT16 || xDataType == ge::DataType::DT_BF16){
            params.typeSize = sizeof(uint16_t);
        } else if (xDataType == ge::DataType::DT_FLOAT){
            params.typeSize = sizeof(float);
        } else {
            OP_LOGE(context->GetNodeName(), "Unsupport type.");
            return ge::GRAPH_FAILED;
        }
        params.block = BLOCK_SIZE / params.typeSize;
        params.dim3LenAlign = GetAlign(params.dim3Len, params.block);
        SubBlock();
        return ge::GRAPH_SUCCESS;
    }

    void SubCore() {
        uint64_t subTasks = params.dim1Len;
        if (params.subMode == 1U) {
            subTasks = params.dim2Len;
        }
        if (params.dim0Len == 1U) {
            params.tasksPerCore = subTasks / params.coreNum;
            params.tailCore = subTasks % params.coreNum;
            params.tasksPerCoreLarge = params.tailCore == 0U ? params.tasksPerCore : params.tasksPerCore + 1U;
        } else {
            params.tasksPerCore = params.dim0Len / params.coreNum;
            params.tailCore = params.dim0Len % params.coreNum;
        }
        params.coreNum = params.tasksPerCore == 0U ? params.tailCore : params.coreNum;
    }
    void SubBlock() {
        const gert::StorageShape* permShape = context->GetInputShape(1);
        // 4 is limit of copy mode, 3 is perm limit
        if (params.dim3Len < params.block * 4U && permShape->GetStorageShape().GetDim(0) == 3) {
            // 2 is bufferNum
            params.dim2OnceMax = params.ubSize / 2U / params.doubleBuffer / params.dim3LenAlign / params.typeSize;
            if (params.subMode == 0U) {
                // 2 is double
                params.dim1OnceMax = params.dim2OnceMax / GetAlign(params.dim2Len, TRANS_BLOCK) < (TRANS_BLOCK * 2) ?
                    TRANS_BLOCK : (params.dim2OnceMax / GetAlign(params.dim2Len, TRANS_BLOCK)) / TRANS_BLOCK * TRANS_BLOCK;
                params.dim2OnceMax = (params.dim2OnceMax / params.dim1OnceMax) / TRANS_BLOCK * TRANS_BLOCK;
            } else if (params.subMode == 1U) {
                // 2 is double
                params.dim1OnceMax = params.dim2OnceMax / GetAlign(params.tasksPerCoreLarge, TRANS_BLOCK) < (TRANS_BLOCK * 2) ?
                    TRANS_BLOCK : (params.dim2OnceMax / GetAlign(params.tasksPerCoreLarge, TRANS_BLOCK)) / TRANS_BLOCK * TRANS_BLOCK;
                params.dim2OnceMax = (params.dim2OnceMax / params.dim1OnceMax) / TRANS_BLOCK * TRANS_BLOCK;
            }
        } else {
            if (params.subMode == 0U) {
                // 2 is bufferNum
                params.dim2OnceMax = params.ubSize / 2U / params.doubleBuffer / params.dim3LenAlign / params.typeSize;
                params.dim1OnceMax = params.dim2OnceMax / params.dim2Len == 0U ? 1 : params.dim2OnceMax / params.dim2Len;
                params.dim1OnceMax = params.dim3LenAlign == params.dim3Len ? params.dim1OnceMax : 1;
                params.dim2OnceMax = params.dim2OnceMax / params.dim1OnceMax;
            } else if (params.subMode == 1U) {
                // 2 is bufferNum
                params.dim1OnceMax = params.ubSize / 2U / params.doubleBuffer / params.dim3LenAlign / params.typeSize;
                params.dim2OnceMax = params.dim1OnceMax / params.dim1Len == 0U ? 1 : params.dim1OnceMax / params.dim1Len;
                params.dim2OnceMax = params.dim3LenAlign == params.dim3Len ? params.dim2OnceMax : 1;
                params.dim1OnceMax = params.dim1OnceMax / params.dim2OnceMax;
            }
        }
    }

    void ComputeTilingKey() {
        const gert::StorageShape* permShape = context->GetInputShape(1);
        // 3 is permSize
        if (permShape->GetStorageShape().GetDim(0) == 3) {
            params.tilingKey = PERM_KEY;
        } else {
            // 2 is 0213 perm
            params.tilingKey = PERM_KEY * 2U;
        }
        params.tilingKey += params.typeSize * TYPE_KEY;
        // 4 is limit of copy mode, 4 is perm limit
        if (params.dim3Len >= params.block * 4U || permShape->GetStorageShape().GetDim(0) == 4) {
            params.tilingKey += COPY_KEY;
        }
    }
    void SetTiling() {
        tilingData.set_dim1Len(params.dim1Len);
        tilingData.set_dim2Len(params.dim2Len);
        tilingData.set_dim3Len(params.dim3Len);
        tilingData.set_dim3LenAlign(params.dim3LenAlign);
        tilingData.set_tasksPerCore(params.tasksPerCore);
        tilingData.set_tailCore(params.tailCore);
        tilingData.set_subMode(params.subMode);
        tilingData.set_dim1OnceMax(params.dim1OnceMax);
        tilingData.set_dim2OnceMax(params.dim2OnceMax);
        tilingData.set_doubleBuffer(params.doubleBuffer);
        size_t* workspaceSize = context->GetWorkspaceSizes(1);
        *workspaceSize = params.workspaceSize + params.sysWorkspaceSize;
        context->SetTilingKey(params.tilingKey);
        context->SetBlockDim(params.coreNum);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }
    void PrintTilingData() {
        OP_LOGD(context->GetNodeName(), "Start TransposeCTilingData printing"         );
        OP_LOGD(context->GetNodeName(), "-----------------------------------------------" );
        OP_LOGD(context->GetNodeName(), "dim1Len is:%lu "   , params.dim1Len       );
        OP_LOGD(context->GetNodeName(), "dim2Len is:%lu "        , params.dim2Len            );
        OP_LOGD(context->GetNodeName(), "dim3Len is:%lu "        , params.dim3Len            );
        OP_LOGD(context->GetNodeName(), "dim3LenAlign is:%lu "  , params.dim3LenAlign      );
        OP_LOGD(context->GetNodeName(), "tasksPerCore is:%lu "     , params.tasksPerCore         );
        OP_LOGD(context->GetNodeName(), "tailCore is:%lu "       , params.tailCore           );
        OP_LOGD(context->GetNodeName(), "subMode is:%lu "         , params.subMode             );
        OP_LOGD(context->GetNodeName(), "dim1OnceMax is:%lu "       , params.dim1OnceMax           );
        OP_LOGD(context->GetNodeName(), "dim2OnceMax is:%lu "  , params.dim2OnceMax      );
        OP_LOGD(context->GetNodeName(), "doubleBuffer is:%lu "  , params.doubleBuffer      );
        OP_LOGD(context->GetNodeName(), "numBlocks is:%u "      , context->GetBlockDim()   );
        OP_LOGD(context->GetNodeName(), "tilingKey is:%lu "     , context->GetTilingKey()  );
        OP_LOGD(context->GetNodeName(), "-----------------------------------------------" );
        OP_LOGD(context->GetNodeName(), "End TransposeCTilingData printing"           );
    }
    uint64_t GetAlign(uint64_t len, uint64_t size) {
        return size == 0U ? 0U : (len + size - 1U) / size * size;
    }
private:
    gert::TilingContext *context = nullptr;
    Transpose102Params params;
    TransposeV2TilingData tilingData;
};
}
#endif
