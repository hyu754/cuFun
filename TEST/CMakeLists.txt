#This project will have optimized CUDA functions

cmake_minimum_required(VERSION 2.8)
project(TESTaa)




ADD_EXECUTABLE(test_cuMat test_cuMat.cpp)




target_link_libraries(test_cuMat cuFun_lib cudart )

