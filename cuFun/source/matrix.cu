#include "matrix.cuh"

template<typename T>
void __global__
cuda_matAdd_global(T * a, const T * b, const T* c, size_t num_rows,size_t num_cols){

	size_t row = threadIdx.x + blockIdx.x * blockDim.x;
	if (row < num_rows){
		for (size_t col = 0; col < num_cols; col++)
			a[IJ2MAT2D(row, col, num_cols)] = b[IJ2MAT2D(row, col, num_cols)]+ c[IJ2MAT2D(row, col, num_cols)];
	}
}

template<typename T>
void __global__
cuda_matAdd_global2(T * a, const T * b, const T* c, size_t num_rows, size_t num_cols){

	size_t row = threadIdx.x + blockIdx.x * blockDim.x;
	size_t col = threadIdx.y + blockIdx.y * blockDim.y;
	if (row < num_rows  && col < num_cols){
		
		a[IJ2MAT2D(row, col, num_cols)] = 1;// b[IJ2MAT2D(row, col, num_cols)] + c[IJ2MAT2D(row, col, num_cols)];
	}
}


void cuda_MatAdd(float * a, const float * b, const float * c, size_t blocks, size_t threads, size_t num_rows, size_t num_cols){

	dim3 dimBlock(CUDA_THREADS, CUDA_THREADS);
	dim3 dimGrid((CUDA_THREADS + num_rows + 1) / CUDA_THREADS, (CUDA_THREADS + num_cols + 1) / CUDA_THREADS);
	cuda_matAdd_global2 << <dimGrid, dimBlock >> >(a, b, c, num_rows, num_cols);
}