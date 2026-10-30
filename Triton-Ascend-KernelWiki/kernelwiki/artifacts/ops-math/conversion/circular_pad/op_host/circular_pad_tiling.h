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
 * \file circular_pad_tiling.h
 * \brief
 */
#ifndef CIRCULAR_PAD_TILING_H
#define CIRCULAR_PAD_TILING_H
#include "circular_pad_common_tiling.h"

namespace optiling {

class CircularPadTiling : public CircularPadCommonTiling {
public:
    explicit CircularPadTiling(gert::TilingContext* context) : CircularPadCommonTiling(context) {};
    ~CircularPadTiling() override = default;

    ge::graphStatus DoOpTiling();
    ge::graphStatus CheckLeftAndRight() override;
    ge::graphStatus CheckTopAndBottom() override;
    ge::graphStatus CheckFrontAndBack() override;
    ge::graphStatus CheckInput() override;
    ge::graphStatus CheckDtype() override;
};
} // namespace optiling
#endif // CIRCULAR_PAD_TILING_H
