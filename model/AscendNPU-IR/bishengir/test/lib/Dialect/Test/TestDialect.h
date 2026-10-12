//===----TestDialect.h - MLIR Dialect for Testing BiShengIR  ----*- C++ -*-===//
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
// This file defines a fake 'test' dialect that can be used for
// testing things that do not have a respective counterpart in the
// main source directories.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_TESTDIALECT_H
#define TEST_TESTDIALECT_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/SmallVector.h"

//===----------------------------------------------------------------------===//
// TestDialect
//===----------------------------------------------------------------------===//

#include "TestOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "TestOps.h.inc"

namespace bishengir_test {
void registerTestDialect(::mlir::DialectRegistry &registry);
} // namespace bishengir_test

#endif // TEST_TESTDIALECT_H
