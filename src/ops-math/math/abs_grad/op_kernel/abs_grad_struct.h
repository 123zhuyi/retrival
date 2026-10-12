/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file abs_grad_struct.h
 * \brief
 */

#ifndef ABS_GRAD_STRUCT_H
#define ABS_GRAD_STRUCT_H

#include "atvoss/elewise/elewise_base_struct.h"

namespace AbsGradNs {
using namespace Ops::Base;
struct AbsGradTilingData {
    EleBaseTilingData baseTiling;
};

} // namespace AbsGradNs
#endif // ABS_GRAD_STRUCT_H
