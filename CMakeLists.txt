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
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Compile the lasso problem module
alpaqa_add_dl_problem_module("lasso" LINK_ALPAQA FILES
    "src/lasso.cpp"
    "src/problem.cpp"
    "src/openmp/omp-problem.cpp"
    "src/openmp/f.cpp"
    "src/openmp/grad-f.cpp"
    "src/openmp/f-grad-f.cpp"
    "src/openmp/hess-L.cpp"
    "src/openmp/hess-L-prod.cpp"
    "src/openmp/prox-grad-step.cpp"
    "src/complex-openmp/complex-omp-problem.cpp"
    "src/complex-openmp/f.cpp"
    "src/complex-openmp/grad-f.cpp"
    "src/complex-openmp/f-grad-f.cpp"
    "src/complex-openmp/hess-L-prod.cpp"
    "src/complex-openmp/prox-grad-step.cpp"
)
set_property(TARGET lasso PROPERTY OUTPUT_NAME alpaqa-lasso)
target_link_libraries(lasso PRIVATE warnings)
target_link_libraries(lasso PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(lasso PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
if (WITH_PYTHON)
    target_link_libraries(lasso PRIVATE pybind11::module)
    target_compile_definitions(lasso PRIVATE WITH_PYTHON)
endif()
if (WITH_CUDA)
    target_sources(lasso PRIVATE
        "src/cuda/cuda-problem.cpp"
        "src/cuda/f.cpp"
        "src/cuda/grad-f.cpp"
        "src/cuda/f-grad-f.cpp"
        "src/cuda/hess-L-prod.cpp"
        "src/cuda/prox-grad-step.cpp"
        "src/cuda/prox-grad-step.cu"
    )
    if (WITH_CUDA_STATIC)
        target_link_libraries(lasso PRIVATE
            CUDA::cublas_static CUDA::cudart_static)
    else()
        target_link_libraries(lasso PRIVATE
            CUDA::cublas CUDA::cudart)
    endif()
    target_compile_definitions(lasso PRIVATE ACL_WITH_CUDA)
endif()

include(GNUInstallDirs)
# Install the Python module (py-build-cmake)
install(TARGETS lasso
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_MODULE_NAME}/problems/lasso)
# Standard installation
install(TARGETS lasso
        COMPONENT lib
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
