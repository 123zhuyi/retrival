if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(pdist_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/pdist_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/pdist_infershape.cpp
        )
endif()
