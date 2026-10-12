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
 * \file add_v2_tiling_arch35.h
 * \brief add_v2 tiling header for ascend950 (arch35)
 */

#ifndef ADD_V2_TILING_ARCH35_H
#define ADD_V2_TILING_ARCH35_H

#include <cstdint>

namespace optiling {

struct AddV2CompileInfoArch35 {
    uint64_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling

#endif // ADD_V2_TILING_ARCH35_H
