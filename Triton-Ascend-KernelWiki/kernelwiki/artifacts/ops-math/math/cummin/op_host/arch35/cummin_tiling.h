/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cumsum.h
 * \brief calc corenum and threadnum for AscendC kernel
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CUMMIN_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CUMMIN_H_
#include <cstdint>
#include "register/tilingdata_base.h"
#include "op_host/tiling_base_class.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"
#include "register/op_def_registry.h"
#include "op_host/math_tiling_templates_registry.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/cummin_struct.h"
#include "../../op_kernel/arch35/cummin_tiling_key.h"
namespace optiling {

struct CumminCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    uint32_t clSize = 0;
    uint32_t vRegSize = 0;
};

class CumminTiling : public Ops::Base::TilingBaseClass {
public:
    explicit CumminTiling(gert::TilingContext* context) : Ops::Base::TilingBaseClass(context)
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
    const CumminCompileInfo* compileInfo_;
    CumminRegbaseTilingData tilingData_{};

    uint32_t blockDim_{1};
    int64_t M{1}; // 合轴后
    int64_t R{1}; // 合轴后
    int64_t N{1}; // 合轴后
    int64_t alignedN_{1};
    int64_t dSize_{1};
    int64_t argminDSize_{1};
    int64_t ubSize_{1};
    int64_t tilingKey_{0};
    int64_t perBlockNum_{0};
    ge::graphStatus DoUBSliceForM(int64_t M);
    ge::graphStatus DoUBSliceForR(CumminSplitInfo& info, int64_t length);
    ge::graphStatus DoUBSliceForN(CumminSplitInfo& info, int64_t length);
    ge::graphStatus DoUBSlice(CumminSplitInfo& info, int64_t length);
    ge::graphStatus DoComputeWithSimt();
    void PrintCumminAll();
    void PrintCumminSplitInfo(const CumminSplitInfo& info);
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_CUMMIN_H_
