cmake_minimum_required(VERSION 3.22...3.28)
project(alpaqa-complex-lasso LANGUAGES CXX)
include(CMakeDependentOption)

option(WITH_PYTHON "Support accepting Python arguments directly" Off)
option(WITH_CUDA "GPU support" Off)
cmake_dependent_option(WITH_CUDA_STATIC "Link the CUDA runtime statically" On "WITH_CUDA" Off)

# Compiler warnings
include(cmake/Warnings.cmake)
add_warnings_target(warnings Off)

# Find the alpaqa library, OpenMP (for parallelization) and CUDA (for GPU)
find_package(alpaqa REQUIRED COMPONENTS Core Dl)
find_package(OpenMP REQUIRED COMPONENTS CXX)
if (WITH_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 REQUIRED)
endif()
if (WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Compile complex lasso problem module
alpaqa_add_dl_problem_module("complex-multi-lasso" LINK_ALPAQA
    FILES "src/complex-multi-lasso.cpp" "src/problem.cpp")
target_link_libraries(complex-multi-lasso PRIVATE warnings)
target_link_libraries(complex-multi-lasso PRIVATE OpenMP::OpenMP_CXX)
if (WITH_PYTHON)
    target_link_libraries(complex-multi-lasso PRIVATE
        pybind11::pybind11 Python3::Python)
    target_compile_definitions(complex-multi-lasso PRIVATE WITH_PYTHON)
endif()
if (WITH_CUDA)
    target_sources(complex-multi-lasso PRIVATE
        "src/eval_f_cuda.cpp" "src/eval_f_grad_f_cuda.cpp")
    if (WITH_CUDA_STATIC)
        target_link_libraries(complex-multi-lasso PRIVATE
            CUDA::cublas_static CUDA::cudart_static)
    else()
        target_link_libraries(complex-multi-lasso PRIVATE
            CUDA::cublas CUDA::cudart)
    endif()
    target_compile_definitions(complex-multi-lasso PRIVATE ACL_WITH_CUDA)
endif()