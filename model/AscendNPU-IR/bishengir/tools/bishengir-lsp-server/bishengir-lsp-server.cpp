//===- bishengir-lsp-server.cpp - BiShengIR Language Server -----*- C++ -*-===//
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

#include "bishengir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

#ifdef MLIR_INCLUDE_TESTS
namespace bishengir_test {
void registerTestDialect(::mlir::DialectRegistry &registry);
} // namespace bishengir_test

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test
#endif

int main(int argc, char **argv) {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  bishengir::registerAllDialects(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::test::registerTestDialect(registry);
  ::bishengir_test::registerTestDialect(registry);
#endif

  return failed(MlirLspServerMain(argc, argv, registry));
}
