#This project will have optimized CUDA functions

cmake_minimum_required(VERSION 2.8)
project(cuFun)

#Additional libraries
include_directories(${CMAKE_SOURCE_DIR}/cuFun/header/)


#Include cuda libraries
find_package(CUDA REQUIRED)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include/)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib)

#Include eigen
include_directories(${CMAKE_SOURCE_DIR}/include/Eigen/)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_50)


add_subdirectory(cuFun)
add_subdirectory(TEST)


#ADD_EXECUTABLE(test_cuMat test_cuMat.cpp)




#target_link_libraries(test_cuMat cuFun_lib cudart )

