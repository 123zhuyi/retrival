/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file trilu.cc
 * \brief
 */

#include "conversion/triu/op_host/arch35/triangulator_tiling.h"
#include "conversion/triu/op_host/arch35/triu_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "platform/platform_info.h"

namespace optiling {
IMPL_OP_OPTILING(Tril).Tiling(Tiling4Trilu<false, false>).TilingParse<TriluCompileInfo>(TilingPrepare4Trilu);
} // namespace optiling
