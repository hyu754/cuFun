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

	dim3 dimBlock(512, 512);
	dim3 dimGrid((512 + num_rows + 1) / 512, (512 + num_cols + 1) / 512);
	//cuda_matAdd_global2 << <dimGrid, dimBlock >> >(a, b, c, num_rows, num_cols);
	cuda_matAdd_global << <threads, blocks >> >(a, b, c, num_rows, num_cols);
}