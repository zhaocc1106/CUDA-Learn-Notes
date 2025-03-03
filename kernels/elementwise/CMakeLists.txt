cmake_minimum_required(VERSION 3.18.5)
project(elementwise LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_C_FLAGS_RELWITHDEBINFO})
message(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

# cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
list(APPEND CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include/")
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
message(STATUS "CUDA_CUBLAS_LIBRARIES: ${CUDA_CUBLAS_LIBRARIES}")
set(CUDA_CUBLASLT_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux-gnu/libcublasLt.so")
message(STATUS "CUDA_CUBLASLT_LIBRARIES: ${CUDA_CUBLASLT_LIBRARIES}")
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-std=c++17"
        "-Xcompiler=-fno-gnu-unique"
        "-U__CUDA_NO_HALF_OPERATORS__"
        "-U__CUDA_NO_HALF_CONVERSIONS__"
        "-U__CUDA_NO_HALF2_OPERATORS__"
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--use_fast_math")
set(CMAKE_CUDA_STANDARD 17)

# CUDA targets
IF (DEFINED GPU_ARCHS)
    message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
    SEPARATE_ARGUMENTS(GPU_ARCHS)
ELSE ()
    list(APPEND GPU_ARCHS 75)

    IF (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
        # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
        list(APPEND GPU_ARCHS 80 86 89)
    ELSE ()
        message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
    ENDIF ()

    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
ENDIF ()
# Generate SASS for each architecture
FOREACH (arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES}" "-gencode;arch=compute_${arch},code=sm_${arch}")
ENDFOREACH ()
set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "${GENCODES}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

# torch
set(LIBTORCH_PATH /usr/local/lib/python3.12/dist-packages/torch)
if (NOT LIBTORCH_PATH)
    message("Not defined LIBTORCH_PATH, use default /usr/local/libtorch")
    set(LIBTORCH_PATH /usr/local/libtorch)
endif ()
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})
list(APPEND TORCH_LIBRARIES "${LIBTORCH_PATH}/lib/libtorch_python.so")
include_directories(
        ${TORCH_INCLUDE_DIRS}
)
message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

# python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
include_directories(
        ${Python_INCLUDE_DIRS}
)

ADD_EXECUTABLE(elementwise elementwise.cu)
set_TARGET_PROPERTIES(elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
TARGET_LINK_LIBRARIES(elementwise ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
        ${TORCH_LIBRARIES} ${Python_LIBRARIES})