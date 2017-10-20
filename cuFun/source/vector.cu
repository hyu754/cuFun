#include "vector.cuh"


//Cuda addition function: a = b+c
template<typename T>
void __global__
cuda_vectAdd_global(T * a, const T * b, const T* c, size_t n){
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n){
		a[tid] =b[tid]+c[tid];
		
	}
}


//Cuda addition function: a = a+b
template<typename T>
void __global__
cuda_vectAdd_global(T * a, const T * b, size_t n){
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n){
		a[tid] =a[tid] + b[tid];

	}
}

//Wrapper function for cuda_vectAdd_global
void cuda_vectAdd(float * a, const float * b, const float * c, size_t blocks, size_t threads, size_t n){
	cuda_vectAdd_global<float> << < blocks, threads >> > (a, b, c, n);

}

//Wrapper function for cuda_vectAdd_global
void cuda_vectAdd(float * a, const float * b, size_t blocks, size_t threads, size_t n){
	cuda_vectAdd_global<float> << < blocks, threads >> > (a, b, n);

}

