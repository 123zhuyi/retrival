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
 * \file addr_tiling.h
 * \brief addr_tiling head file
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_ADDR_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_ADDR_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/arch35/addr_dag.h"
#include "platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

struct AddrCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class AddrTiling : public Ops::Base::TilingBaseClass {
public:
    explicit AddrTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;

private:
    uint64_t tilingKey_ = 0;
    ge::DataType x1Dtype_;
    ge::DataType x2Dtype_;
    ge::DataType x3Dtype_;
    ge::DataType betaDtype_;
    ge::DataType alphaDtype_;
    ge::DataType yDtype_;

    gert::Shape x1Shape_;
    gert::Shape x2Shape_;
    gert::Shape x3Shape_;
    gert::Shape betaShape_;
    gert::Shape alphaShape_;
    gert::Shape yShape_;
    gert::Shape x2ReShape_;

    bool CheckDtype();
    bool CheckShapes();
    bool CheckBroadcast();

    template <typename T>
    ge::graphStatus GetConstData(uint32_t inputIdx, bool isEmpty, T emptyValue, T& data);

    template <typename OpDag1, typename OpDag2, typename OpDag3, typename T>
    ge::graphStatus BroadcastTiling(T beta, T alpha);

    ge::graphStatus DoOpTiling4Float(bool betaEmpty, bool alphaEmpty);
    ge::graphStatus DoOpTiling4Half(bool betaEmpty, bool alphaEmpty);
};

} // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_ADDR_TILING_H
