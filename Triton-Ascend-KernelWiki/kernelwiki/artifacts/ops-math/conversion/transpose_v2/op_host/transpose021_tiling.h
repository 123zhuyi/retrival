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
 * \file transpose021_tiling.h
 * \brief
 */

#ifndef TRANSPOSE021_TILING_H
#define TRANSPOSE021_TILING_H
#include "transpose_v2_tiling.h"

namespace optiling {
struct Transpose021Params {
    uint64_t tasksPerCore{0};
    uint64_t tasksTail{0};
    uint64_t inputH{0};
    uint64_t inputW{0};
    uint64_t inputH16Align{0};
    uint64_t inputWAlign{0};
    uint64_t hOnce{0};
    uint64_t tasksOnceMax{0};
    uint64_t repeatH{0};
    uint64_t transLoop{0};
    uint64_t doubleBuffer{2};

    uint64_t inputNC{1};
    uint64_t typeSize{0};
    uint64_t ubSize{0};
    uint64_t tilingKey{0};
    uint64_t sysWorkspaceSize{0};
    uint64_t workspaceSize{0};
    uint32_t coreNum{0};
};

class Transpose021Tiling {
public:
    explicit Transpose021Tiling(gert::TilingContext* ctx) : context(ctx) {};
    virtual ~Transpose021Tiling() = default;

    ge::graphStatus GetPlatformInfo() {
        auto platformInfo = context->GetPlatformInfo();
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        params.coreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, params.ubSize);
        params.sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoTiling() {
        const gert::StorageShape* x_shape = context->GetInputShape(0);
        auto dimNum = x_shape->GetStorageShape().GetDimNum();
        OP_CHECK_IF(dimNum < 2,  // 2 is x's dims min
            OP_LOGE(context->GetNodeName(), "x's dims shoulu be greater than 2"),
            return ge::GRAPH_FAILED);
        for (uint64_t i = 0; i < dimNum - 2; i++) { // 2 is x's dims min
            params.inputNC *= x_shape->GetStorageShape().GetDim(i);
        }
        params.inputH = x_shape->GetStorageShape().GetDim(dimNum - 2); // dimNum - 2 is h dim index
        params.inputW = x_shape->GetStorageShape().GetDim(dimNum - 1); // dimNum - 1 is w dim index

        // 分核
        params.tasksPerCore = params.inputNC / params.coreNum;
        params.tasksTail = params.inputNC % params.coreNum;
        params.coreNum = params.tasksPerCore == 0U ? params.tasksTail : params.coreNum;

        // 分块
        auto xDataType = context->GetInputDesc(0)->GetDataType();
        if (xDataType == ge::DataType::DT_FLOAT16 || xDataType == ge::DataType::DT_BF16) {
            params.typeSize = sizeof(uint16_t);
        } else if (xDataType == ge::DataType::DT_FLOAT){
            params.typeSize = sizeof(float);
        } else {
            OP_LOGE(context->GetNodeName(), "Unsupport type.");
            return ge::GRAPH_FAILED;
        }

        uint64_t block = BLOCK_SIZE / params.typeSize;
        if (params.inputH == 1U || params.inputW == 1U) {
            params.inputH16Align = params.inputH;
            params.inputWAlign = params.inputW;
        } else {
            params.inputH16Align = GetAlign(params.inputH, TRANS_BLOCK);
            params.inputWAlign = GetAlign(params.inputW, block);
            params.repeatH = params.inputH16Align / TRANS_BLOCK;
        }
        
        if (params.inputH16Align > LIMIT_H || params.inputWAlign > LIMIT_W) {
            OP_LOGE(context->GetNodeName(), "Unsupport shape.");
            return ge::GRAPH_FAILED;
        }
        params.hOnce = params.inputWAlign == params.inputW ? params.inputH16Align : TRANS_BLOCK * block;
        // db on
        uint64_t hOnceMax = (params.ubSize / BUFFER_NUM / params.inputWAlign / params.typeSize) / params.hOnce * params.hOnce;
        if (hOnceMax == 0U) {
            // db off
            hOnceMax = params.hOnce;
            params.doubleBuffer = 1U;
        }
        
        params.tasksOnceMax = hOnceMax / params.inputH16Align;
        params.transLoop = GetAlign(params.tasksOnceMax * params.inputH, params.hOnce) / params.hOnce;
        return ge::GRAPH_SUCCESS;
    }
    void ComputeTilingKey() {
        params.tilingKey = params.typeSize * TYPE_KEY;
    }
    void SetTiling() {
        tilingData.set_tasksPerCore(params.tasksPerCore);
        tilingData.set_tasksTail(params.tasksTail);
        tilingData.set_inputH(params.inputH);
        tilingData.set_inputW(params.inputW);
        tilingData.set_inputH16Align(params.inputH16Align);
        tilingData.set_inputWAlign(params.inputWAlign);
        tilingData.set_hOnce(params.hOnce);
        tilingData.set_tasksOnceMax(params.tasksOnceMax);
        tilingData.set_transLoop(params.transLoop);
        tilingData.set_repeatH(params.repeatH);
        tilingData.set_doubleBuffer(params.doubleBuffer);

        size_t* workspaceSize = context->GetWorkspaceSizes(1);
        *workspaceSize = params.workspaceSize + params.sysWorkspaceSize;
        context->SetTilingKey(params.tilingKey);
        context->SetBlockDim(params.coreNum);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }
    void PrintTilingData() {
        OP_LOGD(context->GetNodeName(), "Start TransposeV2TilingData printing"             );
        OP_LOGD(context->GetNodeName(), "-----------------------------------------------" );
        OP_LOGD(context->GetNodeName(), "tasksPerCore is:%lu"  , params.tasksPerCore      );
        OP_LOGD(context->GetNodeName(), "tasksTail is: %lu"    , params.tasksTail         );
        OP_LOGD(context->GetNodeName(), "inputH is: %lu"       , params.inputH            );
        OP_LOGD(context->GetNodeName(), "inputW is: %lu"       , params.inputW            );
        OP_LOGD(context->GetNodeName(), "inputH16Align is: %lu", params.inputH16Align     );
        OP_LOGD(context->GetNodeName(), "inputWAlign is: %lu"  , params.inputWAlign       );
        OP_LOGD(context->GetNodeName(), "hOnce is: %lu"        , params.hOnce             );
        OP_LOGD(context->GetNodeName(), "tasksOnceMax is: %lu" , params.tasksOnceMax      );
        OP_LOGD(context->GetNodeName(), "transLoop is: %lu"    , params.transLoop         );
        OP_LOGD(context->GetNodeName(), "repeatH is: %lu"      , params.repeatH           );
        OP_LOGD(context->GetNodeName(), "doubleBuffer is: %lu" , params.doubleBuffer      );
        OP_LOGD(context->GetNodeName(), "numBlocks is: %u"      , context->GetBlockDim()   );
        OP_LOGD(context->GetNodeName(), "tilingKey is: %lu"    , context->GetTilingKey()  );
        OP_LOGD(context->GetNodeName(), "-----------------------------------------------" );
        OP_LOGD(context->GetNodeName(), "End TransposeV2TilingData printing"               );
    }
private:
    uint64_t GetAlign(uint64_t len, uint64_t size) {
        return size == 0U ? 0U : (len + size - 1U) / size * size;
    }
private:
    gert::TilingContext *context = nullptr;
    Transpose021Params params;
    TransposeV2TilingData tilingData;
};

}
#endif
