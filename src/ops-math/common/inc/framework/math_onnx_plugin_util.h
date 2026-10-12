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
 * \file math_onnx_plugin_util.h
 * \brief Protobuf-free helpers shared by ONNX plugins.
 *
 * Shared by ONNX plugins that decouple from protobuf (ParseParamsByOperatorFn) and
 * therefore must not include onnx_common.h (which pulls ge_onnx.pb.h). Keep this
 * header free of any protobuf dependency so it can be reused by both protobuf-coupled
 * and protobuf-decoupled plugins.
 */

#ifndef MATH_COMMON_MATH_ONNX_PLUGIN_UTIL_H
#define MATH_COMMON_MATH_ONNX_PLUGIN_UTIL_H

#include <string>

#include "graph/operator.h"

namespace domi {
template <typename T>
inline std::string GetOpName(const T& op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}
} // namespace domi

#endif // MATH_COMMON_MATH_ONNX_PLUGIN_UTIL_H
