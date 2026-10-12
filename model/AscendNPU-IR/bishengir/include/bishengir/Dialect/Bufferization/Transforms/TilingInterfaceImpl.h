//===- TilingInterfaceImpl.h - Implementation of TilingInterface ----------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements Tiling interface for Bufferization Dialect Ops with
// ExternalModel.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H
#define BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H

#include "mlir/IR/Dialect.h"

namespace bishengir {
namespace bufferization {

/// Registers external models for Tiling interface for bufferization ops.
/// Currently, it registers:
///
/// * TilingInterface for `bufferization.to_tensor`.
void registerTilingInterfaceExternalModels(mlir::DialectRegistry &registry);

} // namespace bufferization
} // namespace bishengir

#endif // BISHENGIR_DIALECT_BUFFERIZATION_TRANSFORMS_TILINGINTERFACEIMPL_H
