#!/bin/bash

# Create build directory
mkdir -p build
cd build

# Configure and build
cmake ..
cmake --build . --config Release -j$(nproc)

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Run with: ./build/bin/lm_train"
echo "Or from build directory: ./bin/lm_train"
echo "=========================================="