add_bishengir_conversion_library(BiShengIRHFusionToVector
  HFusionToVector.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/HFusionToVector

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  MLIRArithDialect
)
