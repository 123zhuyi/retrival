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
 * \file is_close_struct.h
 * \brief is_close_struct
 */
#ifndef IS_CLOSE_STRUCT_H_
#define IS_CLOSE_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "atvoss/broadcast/broadcast_base_struct.h"

#define IS_CLOSE_TPL_NOT_EQUAL_NAN 0
#define IS_CLOSE_TPL_EQUAL_NAN 1

namespace IsCloseOp

{
ASCENDC_TPL_ARGS_DECL(IsClose, BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
                            ASCENDC_TPL_UINT_DECL(equalNan, 1, ASCENDC_TPL_UI_LIST, IS_CLOSE_TPL_NOT_EQUAL_NAN, IS_CLOSE_TPL_EQUAL_NAN));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode),ASCENDC_TPL_UINT_SEL(equalNan, ASCENDC_TPL_UI_LIST, IS_CLOSE_TPL_NOT_EQUAL_NAN, IS_CLOSE_TPL_EQUAL_NAN)),
    );

}
#endif // IS_CLOSE_STRUCT_H_