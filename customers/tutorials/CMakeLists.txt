add_customer_libraries(tutorials
    SOURCES
        src/decode_tu6a.cc
        src/decode_tu6b.cc
        src/decode_tu7.cc
        src/decode_tu9.cc
        src/inplace_tu9.cc
        src/decode_superpoint.cc
        src/decode_edgepoint.cc
    INCLUDE_DIRECTORIES
        include
)

if(OpenMP_CXX_FOUND)
    target_compile_options(decode_edgepoint PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(decode_edgepoint PRIVATE OpenMP::OpenMP_CXX)
endif()
