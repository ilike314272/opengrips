cmake_minimum_required(VERSION 3.28) # Requires 3.28+ for robust C++23 Module support
project(QuantEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create library target
add_library(quant_core)

# Define your C++23 Module Interfaces and Implementations explicitly
target_sources(quant_core
    PUBLIC
        FILE_SET CXX_MODULES FILES
            src/core/diff_eq.ixx
            src/core/matrix_ops.ixx
            src/acceleration/distributed/rdma_backend.ixx
    PRIVATE
        src/core/diff_eq.cpp
        src/core/matrix_ops.cpp
        src/acceleration/distributed/rdma_backend.cpp
)

# Conditionally compile Windows vs. Linux RDMA / Schedulers
if(WIN32)
    # Windows NetworkDirect RDMA Dependencies
    find_package(NetworkDirect REQUIRED) 
    
    target_sources(quant_core PUBLIC FILE_SET CXX_MODULES FILES src/platform/win32/nd_verbs.ixx)
    target_sources(quant_core PRIVATE src/platform/win32/win_threads.cpp src/platform/win32/nd_verbs.cpp)
    target_link_libraries(quant_core PRIVATE NetworkDirect::NetworkDirect)
else()
    # Linux libibverbs (rdma-core) Dependencies
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(IBVERBS REQUIRED libibverbs librdmacm)
    
    target_sources(quant_core PUBLIC FILE_SET CXX_MODULES FILES src/platform/linux/ib_verbs.ixx)
    target_sources(quant_core PRIVATE src/platform/linux/rt_scheduler.cpp src/platform/linux/ib_verbs.cpp)
    target_link_targets(quant_core PRIVATE ${IBVERBS_LIBRARIES})
endif()
