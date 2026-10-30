set("OPERATOR_DEFINITION" ${CMAKE_CURRENT_SOURCE_DIR}/init.cpp)
message(STATUS "OPERATOR_DEFINITION: ${OPERATOR_DEFINITION}")
message(STATUS "Final TORCH_EXTENSION_OPERATOR_TARGETS: ${TORCH_EXTENSION_OPERATOR_TARGETS}")
unset(CMAKE_CXX_FLAGS)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type(Release/Debug)" FORCE)
endif()
set(CMAKE_C_COMPILER ${BISHENG})
set(CMAKE_CXX_COMPILER ${BISHENG})
set(CMAKE_LINKER ${BISHENG})
if(NOT TORCH_EXTENSION_OPERATOR_TARGETS)
    message(WARNING "TORCH_EXTENSION_OPERATOR_TARGETS is empty.")
    add_library(_C SHARED ${OPERATOR_DEFINITION})
    target_include_directories(_C PRIVATE ${TORCH_EXTENSION_INCLUDE_DIRS})
    set_target_properties(_C PROPERTIES
        PREFIX ""
        SUFFIX ".abi3.so"
        OUTPUT_NAME "_C"
    )
else()
    add_library(_C SHARED
        ${OPERATOR_DEFINITION}
        ${TORCH_EXTENSION_OPERATOR_TARGETS}
    )
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
    set_target_properties(_C PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LINK_FLAGS "--cce-fatobj-link"
        PREFIX ""
        SUFFIX ".abi3.so"
        OUTPUT_NAME "_C"
    )
    target_compile_options(_C PRIVATE ${TORCH_EXTENSION_COMPILE_OPTIONS})
    target_include_directories(_C PRIVATE ${TORCH_EXTENSION_INCLUDE_DIRS})
    target_link_directories(_C PRIVATE ${TORCH_EXTENSION_LINK_DIRS})
    target_link_libraries(_C PRIVATE ${TORCH_EXTENSION_LINK_LIBS})
endif()

add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:_C>
    ${CMAKE_CURRENT_SOURCE_DIR}/npu_math_extension/$<TARGET_FILE_NAME:_C>
    COMMENT "Copying compiled extension $<TARGET_FILE_NAME:_C> to ${CMAKE_CURRENT_SOURCE_DIR}/npu_math_extension/"
)
