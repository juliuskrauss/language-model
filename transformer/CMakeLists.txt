# Transformer library
add_library(transformer_lib STATIC
    src/transformer.cpp
    include/transformer.h
)

target_include_directories(transformer_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(transformer_lib PUBLIC
    ${TORCH_LIBRARIES}
)

# Set the output directory for the library
set_target_properties(transformer_lib PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)