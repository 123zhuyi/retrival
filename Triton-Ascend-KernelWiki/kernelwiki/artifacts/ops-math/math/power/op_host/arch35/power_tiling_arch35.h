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
 * \file power_tiling_arch35.h
 * \brief Power 算子 ascend950 平台 Tiling 头文件。
 *        在 Tiling 层根据 scale / power / shift 三个标量属性完成分支决策，
 *        以 culType 路由到 kernel 端对应的 DAG 实现。
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_POWER_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_POWER_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "math/power/op_kernel/arch35/power_tiling_struct.h"

namespace optiling {
using namespace Ops::Base;
using namespace PowerOp;

// Power 算子计算路径枚举。
// 取值与 op_kernel/arch35/power_struct.h 中的 POWER_TPL_CUL_* 宏一一对应，
// 完全在 Tiling 层依据属性决策，kernel 端根据 tilingKey 直接选用对应 DAG，
// 无需运行时分支。
enum class CulTypeEnum : int32_t {
    ALL_ZEROS         = 0,   // y 全 0：power=0 或 (shift=0 且 power>0 且 scale*power=0)
    BROADCAST_SCALAR  = 1,   // y 全为同一标量 bcastVal：在 host 端预计算 pow(shift, power) 或异常值 +inf
    LINEAR            = 2,   // power == 1：y = x * scale + shift
    SQUARE            = 3,   // power == 2：base = x*scale + shift; y = base * base
    CUBE              = 4,   // power == 3：base = x*scale + shift; y = base^3
    GENERIC_POW_POS   = 5,   // 通用幂运算，power > 0（base==0 时输出 0）
    GENERIC_POW_NEG   = 6,   // 通用幂运算，power < 0（base==0 时输出 +inf 异常值）
};

class PowerTiling {
public:
    explicit PowerTiling(gert::TilingContext* context) : tilingContext(context) {};
    // Tiling 主入口：依次完成 dtype/shape 校验 → 读取属性 → 决策 culType
    //                → 调用 elementwise 通用 tiling → 写 scalar/tilingKey/blockDim。
    ge::graphStatus RunTiling();

protected:
    // 校验输入张量 dtype 是否在 {fp16, bf16, fp32} 内。
    ge::graphStatus CalcInputDtype();   
    // 校验输出 dtype 与输入一致。
    ge::graphStatus CalcOutputDtype();
    // 校验输入与输出 shape 完全相同（elementwise 算子）。
    ge::graphStatus CheckShape();
    // 读取 power / scale / shift 三个属性到成员变量。
    ge::graphStatus SetAttr();
    // 根据 power/scale/shift 标量计算决定 culType 与对应的 kernel 标量列表；
    // 返回 false 表示校验或路由失败。
    bool DecideCulType();

    // 浮点近似相等判断，复用 math/is_close 算子的判等公式：
    //     |a - b| <= atol + rtol * |b|
    // 用于在 host 端容忍浮点误差地比较 power/scale/shift 是否等于 0、1、2、3 等关键值。
    static bool IsCloseScalar(float a, float b);
    // 判断浮点数 v 是否为整数（v == floor(v) 且为有限值）。
    // 用于决定负底数 + 整数幂能否落到 LINEAR/SQUARE/CUBE 或 (-1)^power 规则。
    static bool IsInteger(float v);

    // 根据 culType 分发到对应的 DAG 进行 tiling，内部处理 dtype 映射。
    ge::graphStatus DispatchTilingByCulType(ElewiseBaseTiling& tiling, PowerOp::PowerTilingData* data);

private:
    ge::graphStatus PerformValidationChecks();
    ge::graphStatus MapOutputDtypeToTplKey();
    ge::graphStatus SetTilingResults(PowerOp::PowerTilingData* powerTilingData);
    
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;

    // ----- 算子属性 -----
    float attrPower = 1.0f;
    float attrScale = 1.0f;
    float attrShift = 0.0f;

    // ----- 路由结果 -----
    CulTypeEnum culType = CulTypeEnum::ALL_ZEROS;
    uint64_t culTypeKey = 0;   // culType 转 uint64_t，参与 GET_TPL_TILING_KEY 组合
    uint64_t dType = 0;        // dtype 对应的模板键值（POWER_TPL_DTYPE_*）
    uint64_t schMode = 0;      // elementwise 模板的 schedule mode

    // ----- 传递到 kernel 的标量（写入 EleBaseTilingData32B::scalarData[16]） -----
    // 不同 culType 下的语义（详见 docs/DESIGN.md）：
    //   ALL_ZEROS         : 不使用
    //   BROADCAST_SCALAR  : [0]=bcastVal
    //   LINEAR/SQUARE/CUBE: [0]=scale, [1]=shift
    //   GENERIC_POW_*     : [0]=scale, [1]=shift, [2]=power, [3]=negScalar
    //                       （negScalar：底数为负时的修正系数，整数幂 ±1、非整数幂 +inf）
    float scalar0 = 0.0f;
    float scalar1 = 0.0f;
    float scalar2 = 0.0f;
    float scalar3 = 0.0f;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_POWER_TILING_H
