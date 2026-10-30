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
 * \file sign_tiling_arch35.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_SIGN_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_SIGN_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../op_kernel/sign_struct.h"

namespace optiling {
using namespace Ops::Base;
using namespace SignNs;

class SignTiling {
public:
    explicit SignTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus CalcOutputDtype();
    ge::graphStatus CheckOutputDtype();
    ge::graphStatus CalcInputDtype();
    ge::graphStatus CheckOutputShape();
    std::string DataTypeToSerialString(ge::DataType type);
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext;
    ge::DataType outputDtype = ge::DT_UNDEFINED;
    ge::DataType inputDtype = ge::DT_UNDEFINED;
    SignTilingData* tiling = nullptr;
};

} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_SIGN_TILING_H