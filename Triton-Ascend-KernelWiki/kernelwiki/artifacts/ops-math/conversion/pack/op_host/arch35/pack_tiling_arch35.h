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
 * \file pack_tiling_arch35.h
 * \brief pack tiling arch35 header
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_PACK_TILING_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_PACK_TILING_ARCH35_H_

#include "conversion/concat/op_host/arch35/concat_tiling_arch35.h"
#include "register/tilingdata_base.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(Pack, ConcatTilingData)

REGISTER_TILING_DATA_CLASS(Pack_12111, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12112, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12114, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12118, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12211, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12212, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12214, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12311, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12312, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_12314, ConcatTilingDataNoArray)
REGISTER_TILING_DATA_CLASS(Pack_20001, ConcatTilingDataNoArray)

REGISTER_TILING_DATA_CLASS(Pack_30001, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Pack_30002, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Pack_30004, ConcatTilingDataForSimt)
REGISTER_TILING_DATA_CLASS(Pack_30008, ConcatTilingDataForSimt)

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_PACK_TILING_ARCH35_H_