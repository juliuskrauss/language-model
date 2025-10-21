# Language Model executable
add_executable(lm_train
    src/main.cpp
)

target_include_directories(lm_train PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/transformer/include
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(lm_train PRIVATE
    transformer_lib
    ${TORCH_LIBRARIES}
)

# Set the output directory for the executable
set_target_properties(lm_train PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Set RPATH for finding libtorch libraries at runtime
set_target_properties(lm_train PROPERTIES
    BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)