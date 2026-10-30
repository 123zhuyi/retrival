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
 * \file select_simt_tiling.h
 * \brief
 */

#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SELECT_SIMT_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SELECT_SIMT_TILING_H

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base_class.h"

namespace optiling {

class SelectSimtTiling : public Ops::Base::TilingBaseClass {
    public:
        explicit SelectSimtTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context) {}

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
        bool IsMatchAB();
        bool XDtypeImprove();

        uint64_t tilingKey = 0;
        int64_t aivNum_;
        int64_t ubSize_ = 0;
        int64_t needCoreNum_ = 0;
        gert::Shape conditionShape_;
        gert::Shape x1Shape_;
        gert::Shape x2Shape_;
        int64_t aSize_ = 1;
        int64_t bSize_ = 1;
        int64_t ySize_ = 1;
#ifdef DAVID_FPGA
        int64_t threadNum_ = 128;
#else
        int64_t threadNum_ = 2048;
#endif
};

}   // namespace optiling

#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SELECT_SIMT_TILING_H