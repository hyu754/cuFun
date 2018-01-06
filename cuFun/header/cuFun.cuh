#ifndef H_CUFUN
#define H_CUFUN
#include <cuda.h>
#include <cuda_runtime.h>

/*
Preprocessors
*/

//Matrix
//Preprocessor function to convert (i,j) to matrix index
//i- rows, j - cols. id = (i-1) * num_cols + j 
#define IJ2MAT2D(i,j,num_cols) (i * num_cols + j )

#define CUDA_THREADS 1024
namespace cuFun
{
	
	//size_t cuda_threads = CUDA_THREADS;
	//set_cuda_threads(size_t t_in){ threads_ = t_in; }*/
}

#endif #H_CUFUN
