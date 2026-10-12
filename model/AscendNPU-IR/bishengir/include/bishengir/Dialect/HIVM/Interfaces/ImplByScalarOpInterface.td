//===- ImplByScalarOpInterface.td - Ops to decompose -------*- tablegen -*-===//
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
// This is the definition file for the ImplByScalarOpInterface.
//
//===----------------------------------------------------------------------===//

#ifndef BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD
#define BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def ImplByScalarOpInterface : OpInterface<"ImplByScalarOpInterface"> {
  let description = [{
    Interface for operations that needs to be lowered to scalar loops.
  }];
  let cppNamespace = "::mlir::hivm";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Method to lower the operation to scalar loops.
      }],
      /*retType=*/"mlir::FailureOr<mlir::SmallVector<mlir::Value>>",
      /*methodName=*/"lowerToLoops",
      /*args=*/(ins
          "RewriterBase &":$b),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return whether the operation should be lowered to scalar loops.
      }],
      /*retType=*/"bool",
      /*methodName=*/"shouldLowerToScalarLoops",
      /*args=*/(ins),
      /*methodBody=*/""
    >
  ];
}

#endif // BISHENGIR_DIALECT_HIVM_INTERFACES_IMPLBYSCALAROPINTERFACE_TD
