//===- CopyOpInterface.h ---------------------------------------------------===//
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

#ifndef BISHENGIR_INTERFACES_COPYOPINTERFACE_H
#define BISHENGIR_INTERFACES_COPYOPINTERFACE_H

// On LLVM 22+ MLIR removed CopyOpInterface (PR #157711); use the vendored
// definition generated from bishengir/Interfaces/CopyOpInterface.td. On
// LLVM < 22 MLIR still ships it (pulled in transitively via MemRef.h etc.),
// so this wrapper is a no-op there.
#ifdef __LLVM_MAJOR_VERSION_22_COMPATIBLE__
#include "mlir/IR/OpDefinition.h"
#include "bishengir/Interfaces/CopyOpInterface.h.inc"
#endif

#endif // BISHENGIR_INTERFACES_COPYOPINTERFACE_H
