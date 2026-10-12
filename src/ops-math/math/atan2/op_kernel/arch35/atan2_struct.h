/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file atan2_struct.h
 * \brief atan2_struct
 */
#ifndef ATAN2_STRUCT_H_
#define ATAN2_STRUCT_H_

#include "atvoss/broadcast/broadcast_base_struct.h"


// 算子自定义的tiling key字段
ASCENDC_TPL_ARGS_DECL(Atan2, BRC_TEMP_SCH_MODE_KEY_DECL(schMode));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(BRC_TEMP_SCH_MODE_KEY_SEL(schMode))
);
#endif // ATAN2_STRUCT_H_