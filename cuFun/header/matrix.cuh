#ifndef H_MATRIX_HEADER
#define H_MATRIX_HEADER
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.cuh"

//template<typename T>
//void cuda_vectAdd(T * a, T * b, T * c, int blocks, int threads, int n);
////Cuda add for different data types
//void cuda_vectAdd(int * a, int * b, int * c, int blocks, int threads, int n);
//void cuda_vectAdd(float * a, float * b, float * c, int blocks, int threads, int n);
//void cuda_vectAdd(double * a, double * b, double * c, int blocks, int threads, int n);
//



//template<typename T>
//cuFun::matFun::vect<T> cudaVectAdd(cuFun::matFun::vect<T> vec_b, cuFun::matFun::vect<T> vec_c, int n);


//template<typename T>
//__global__ vectAdd(T * a, T * b, T* c, int n){
//	int tid = blockId.x + blockIdx.x * blockDim.x;
//	if (tid < n){
//		c[tid] = a[tid] + b[tid];
//	}
//}
#endif #H_MATRIX_HEADER
