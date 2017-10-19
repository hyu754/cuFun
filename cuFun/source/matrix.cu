#include "matrix.cuh"

//template<typename T>
//void __global__ 
//cuda_vectAdd_global(T * a, T * b, T* c, int n){
//	int tid = blockIdx.x + blockIdx.x * blockDim.x;
//	if (tid < n){
//		c[tid]  = a[tid] +b[tid];
//	}
//}

//template<typename T>
//void cuda_vectAdd(T * a, T * b, T * c, int blocks, int threads, int n){
//	cuda_vectAdd_global<T> << < blocks, threads >> > (a, b, c, n);
//}
//
