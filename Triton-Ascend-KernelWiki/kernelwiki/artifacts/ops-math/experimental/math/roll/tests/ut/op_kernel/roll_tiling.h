/*!
 * \file roll_tiling.h
 * \brief Roll tiling data for kernel UT
 */

#ifndef ROLL_TILING_UT_H_
#define ROLL_TILING_UT_H_

#include <cstdint>
#include <cstring>

#include "../../../op_kernel/roll_tiling_data.h"
#include "tikicpulib.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "graph/c_types.h"
#include "ascendc/host_api/tiling/template_argument.h"

#ifndef __aicore__
#define __aicore__
#endif

#ifndef __gm__
#define __gm__
#endif

#ifndef __ubuf__
#define __ubuf__
#endif

inline void InitTilingData(uint8_t* tiling, RollTilingData* constData)
{
    memcpy(constData, tiling, sizeof(RollTilingData));
}

#define GET_TILING_DATA_WITH_STRUCT(tilingStruct, tilingData, tilingArg) \
    tilingStruct tilingData;                                              \
    InitTilingData(tilingArg, &tilingData)

#define GET_TILING_DATA(tilingData, tilingArg) \
    RollTilingData tilingData;                 \
    InitTilingData(tilingArg, &tilingData)

#endif // ROLL_TILING_UT_H_
