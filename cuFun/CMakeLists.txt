#header files 
file(GLOB CUDA_HEADER header/*.h header/*.cuh )
#source files
file(GLOB CUDA_SOURCE source/*.cu source/*.cpp)



source_group("Headers" FILES ${CUDA_HEADER})

CUDA_ADD_LIBRARY(cuFun_lib ${CUDA_SOURCE}  ${CUDA_HEADER})


target_link_libraries( cuFun_lib cudart cublas cusparse)
