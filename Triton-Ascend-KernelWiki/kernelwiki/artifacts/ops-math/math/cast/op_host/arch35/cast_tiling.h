/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cast_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_TILING_H

#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "log/log.h"

namespace optiling {

struct CastCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

BEGIN_TILING_DATA_DEF(CastTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockNum);    // 启动多少核处理
TILING_DATA_FIELD_DEF(int64_t, ubFormer);    // 一次ub处理的个数，开db后ub按照一半算
TILING_DATA_FIELD_DEF(int64_t, blockFormer); // 整核处理的个数

TILING_DATA_FIELD_DEF(int64_t, ubLoopOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubLoopOfTailBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerBlock);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailBlock);

TILING_DATA_FIELD_DEF(int64_t, regCopyInStep);  // ub搬入reg，ub的步长
TILING_DATA_FIELD_DEF(int64_t, regCopyOutStep); // reg搬出到ub，ub的步长
TILING_DATA_FIELD_DEF(int64_t, ubFormerRegLoop);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfFormerRegLoop);
TILING_DATA_FIELD_DEF(int64_t, ubTailOfTailRegLoop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cast, CastTilingData);

struct CastMapSt {
    ge::DataType srcType_; // key1
    ge::DataType dstType_; // key2

    uint8_t id_;
    uint8_t srcMapType_;
    uint8_t dstMapType_;
    uint8_t midType_;

    uint8_t castMode1_;
    uint8_t castMode2_;
    uint8_t regCopyInMode_;
    uint8_t regCopyOutMode_;

    CastMapSt() {}
    CastMapSt(ge::DataType srcType, ge::DataType dstType, uint8_t id,
        uint8_t srcMapType, uint8_t dstMapType, uint8_t midType,
        uint8_t castMode1, uint8_t castMode2,
        uint8_t regCopyInMode, uint8_t regCopyOutMode)
        : srcType_(srcType), dstType_(dstType), id_(id), srcMapType_(srcMapType),
        dstMapType_(dstMapType), midType_(midType), castMode1_(castMode1), castMode2_(castMode2),
        regCopyInMode_(regCopyInMode), regCopyOutMode_(regCopyOutMode) {}
};

class CastTiling : public Ops::Base::TilingBaseClass {
public:
    explicit CastTiling(gert::TilingContext *context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    int64_t GetUbCopyStep(uint8_t inType, uint8_t outType,
        uint8_t copyType, int64_t &oneLoopCopyInBitSize);
    int64_t GetDtypeBitSize(uint8_t dtype);
    int64_t GetGeDtypeBitSize(ge::DataType dtype);
    int64_t GetUbFormer(int64_t inputTypeBitSize, int64_t outputTypeBitSize);
    bool IsSimt();
    ge::DataType TransAclToGeDataType(int32_t aclType);

    int64_t coreNum_{ 0 };      // syscfg
    int64_t ubSize_{ 0 };       // syscfg unit: Byte
    int64_t vlBitSize_{2048};       // 2048 unit: bit
    int64_t shapeSize_ {0};

    CastTilingData tilingData_;
    CastMapSt policy_;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_TILING_H