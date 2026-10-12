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
#include "op_host/tiling_base_class.h"
#include "log/log.h"
#include "math/cast/op_kernel/arch35/cast_tiling_data.h"

namespace optiling {

struct CastCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

struct CastMapSt {
    ge::DataType srcType_ = ge::DT_UNDEFINED; // key1
    ge::DataType dstType_ = ge::DT_UNDEFINED; // key2

    uint8_t id_ = 0;
    uint8_t srcMapType_ = 0;
    uint8_t dstMapType_ = 0;
    uint8_t midType_ = 0;

    uint8_t castMode1_ = 0;
    uint8_t castMode2_ = 0;
    uint8_t regCopyInMode_ = 0;
    uint8_t regCopyOutMode_ = 0;

    CastMapSt() {}
    CastMapSt(ge::DataType srcType, ge::DataType dstType, uint8_t id, uint8_t srcMapType, uint8_t dstMapType,
              uint8_t midType, uint8_t castMode1, uint8_t castMode2, uint8_t regCopyInMode, uint8_t regCopyOutMode)
        : srcType_(srcType),
          dstType_(dstType),
          id_(id),
          srcMapType_(srcMapType),
          dstMapType_(dstMapType),
          midType_(midType),
          castMode1_(castMode1),
          castMode2_(castMode2),
          regCopyInMode_(regCopyInMode),
          regCopyOutMode_(regCopyOutMode)
    {}
};

class CastTiling : public Ops::Base::TilingBaseClass {
public:
    explicit CastTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

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
    int64_t GetDtypeBitSize(uint8_t dtype) const;
    int64_t GetGeDtypeBitSize(ge::DataType dtype) const;
    int64_t GetUbFormer(int64_t inputTypeBitSize, int64_t outputTypeBitSize);
    bool IsSimt() const;
    ge::DataType TransAclToGeDataType(int32_t aclType) const;

    int64_t coreNum_{0};      // syscfg
    int64_t ubSize_{0};       // syscfg unit: Byte
    int64_t vlBitSize_{2048}; // 2048 unit: bit
    int64_t shapeSize_{0};
    int64_t usedCoreNum_{0}; // computed core num to use
    int64_t ubFormer_{0};    // computed ub former

    CastMapSt policy_;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_CAST_TILING_H
