/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROD_ENV_MAT_A_CALC_RIJ_PROTO_H
#define PROD_ENV_MAT_A_CALC_RIJ_PROTO_H

#include "graph/operator.h"
#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Calculate ProdEnvMatACalRij.
 * Use type, natoms, sel_a, and rcut_r as constraints, find the central element in
 * the corresponding coord through mesh, output the index of the central element
 * and the distance between the central element and each neighbor. \n
 *
 * @par Inputs:
 * @li coord: A Tensor. Must be one of the following types: float32, float64.
 * @li type: A Tensor. Must be one of the following types: int32.
 * @li natoms: A Tensor. Must be one of the following types: int32.
 * @li box: A Tensor. Must be one of the following types: float32, float64.
 * @li mesh: A Tensor. Must be one of the following types: int32. \n
 *
 * @par Outputs:
 * rij: A Tensor. Must be one of the following types: float32, float64.
 * nlist: A Tensor. Must be one of the following types: int32.
 * distance: A Tensor. Must be one of the following types: float32, float64.
 * rij_x: A Tensor. Must be one of the following types: float32, float64.
 * rij_y: A Tensor. Must be one of the following types: float32, float64.
 * rij_z: A Tensor. Must be one of the following types: float32, float64. \n
 *
 * @par Attributes:
 * @li rcut_a: A Float.
 * @li rcut_r: A Float.
 * @li rcut_r_smth: A Float.
 * @li sel_a: A ListInt.
 * @li sel_r: A ListInt. \n
 *
 * @par Restrictions:
 * Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use.
 */
REG_OP(ProdEnvMatACalcRij)
    .INPUT(coord, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(type, TensorType({DT_INT32}))
    .INPUT(natoms, TensorType({DT_INT32}))
    .INPUT(box, TensorType({DT_FLOAT, DT_DOUBLE}))
    .INPUT(mesh, TensorType({DT_INT32}))
    .OUTPUT(rij, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(nlist, TensorType({DT_INT32}))
    .OUTPUT(distance, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_x, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_y, TensorType({DT_FLOAT, DT_DOUBLE}))
    .OUTPUT(rij_z, TensorType({DT_FLOAT, DT_DOUBLE}))
    .ATTR(rcut_a, Float, 1.0)
    .ATTR(rcut_r, Float, 1.0)
    .ATTR(rcut_r_smth, Float, 1.0)
    .ATTR(sel_a, ListInt, {})
    .ATTR(sel_r, ListInt, {})
    .OP_END_FACTORY_REG(ProdEnvMatACalcRij)
} // namespace ge

#endif
