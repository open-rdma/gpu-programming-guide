cmake_minimum_required(VERSION 3.19)
project(SGEMM_Labs LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

# Set your GPU compute capability (default: 86 for A6000/A40/A10 Ampere)
#   A100: 80, A6000: 86, RTX 3090: 86, RTX 4090: 89, H100: 90
set(CUDA_COMPUTE_CAPABILITY 86 CACHE STRING "GPU Compute Capability")
# For Ampere (A6000): use 86
# For Ampere (A100): use 80

# Release flags for maximum performance
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -Xptxas=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=compute_${CUDA_COMPUTE_CAPABILITY} -code=sm_${CUDA_COMPUTE_CAPABILITY}")

# Add debug symbols in debug mode
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# ============================================================================
# Include directories
# ============================================================================
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================================
# Build all lab solutions
# ============================================================================

function(add_lab LAB_NUM)
  add_executable(lab${LAB_NUM}_solution lab${LAB_NUM}_solution.cu)
  set_target_properties(lab${LAB_NUM}_solution PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
  target_link_libraries(lab${LAB_NUM}_solution
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

  add_executable(lab${LAB_NUM}_start lab${LAB_NUM}_start.cu)
  set_target_properties(lab${LAB_NUM}_start PROPERTIES
    CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
  target_link_libraries(lab${LAB_NUM}_start
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endfunction()

add_lab(1)
add_lab(2)
add_lab(3)
add_lab(4)
add_lab(5)
add_lab(6)
add_lab(7)
add_lab(8)

# ============================================================================
# Optional: build all solutions into a single benchmark runner
# ============================================================================
# add_executable(bench_all benchmark_all.cu)
# set_target_properties(bench_all PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
# target_link_libraries(bench_all ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
