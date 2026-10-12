set(LLVM_TARGET_DEFINITIONS Passes.td)

mlir_tablegen(Passes.h.inc -gen-pass-decls -name BiShengIRTransform)
mlir_tablegen(BiShengIRTransform.capi.h.inc -gen-pass-capi-header --prefix BiShengIRTransform)
mlir_tablegen(BiShengIRTransform.capi.cpp.inc -gen-pass-capi-impl --prefix BiShengIRTransform)

add_public_tablegen_target(BiShengIRTransformsPassIncGen)
add_dependencies(mlir-headers BiShengIRTransformsPassIncGen)
