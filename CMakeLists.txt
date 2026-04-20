cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

# Set CUDA architecturesto 86 for Ampere (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Common CUDA flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find all .cu files recursively
file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/kernels/**/*.cu")

# Optional: Group sources by directory
foreach(source IN LISTS CUDA_SOURCES)
    get_filename_component(parent_dir "${source}" DIRECTORY)
    string(REPLACE "${CMAKE_SOURCE_DIR}/" "" group "${parent_dir}")
    string(REPLACE "/" "\\" group "${group}")
    source_group("${group}" FILES "${source}")
endforeach()

# Kernel targets
add_executable(naive_matmul kernels/matmul/naive_matmul.cu)
add_executable(shared_memory_matmul kernels/matmul/shared_memory_matmul.cu)
add_executable(thread_tile_matmul kernels/matmul/thread_tile_matmul.cu)
add_executable(double_buffering_matmul kernels/matmul/double_buffering_matmul.cu)
add_executable(matrix_transpose kernels/transpose/matrix_transpose.cu)
add_executable(histogram_shared kernels/histogram/histogram_shared.cu)
add_executable(RMSNorm kernels/RMSNorm/RMSNorm.cu)
# Print found sources
message(STATUS "Found ${CUDA_SOURCES}")
