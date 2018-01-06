#ifndef H_MATRIX_HEADER
#define H_MATRIX_HEADER
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.cuh"
#include "cuFun.cuh"

namespace cuFun
{
	
	namespace cuMat
	{

		//Initialization , zero, or random,
		enum initial_type{
			ZERO, // 0
			RAND, // [0- 1]
		};

		//Matrix data struct
		template<typename T>
		struct _mat{
			size_t rows_;
			size_t cols_;
			T *elements_;
			size_t total_elements(){ return rows_*cols_; }
			size_t total_size(){ return sizeof(T)*total_elements(); }
		};

		

	
		//Matrix template class
		template<class T>
		class mat
		{
		public:
			//Default constructor
			mat(void){ };

			//Constructor with no input, but initialized with zeros, rands, etc
			mat(size_t in_rows, size_t in_cols,initial_type in_type){
				matrix_host.rows_ = in_rows;
				matrix_host.cols_ = in_cols;
				matrix_host.elements_ = new T[in_rows*in_cols];


				if (in_type == initial_type::ZERO){
					for (size_t _row_ = 0; _row_ < in_rows; _row_++){
						for (size_t _col_ = 0; _col_ < in_cols; _col_++){

							matrix_host.elements_[IJ2MAT2D(_row_, _col_, in_cols)] = (T) 0;
						}
					}
				}
				else if (in_type == initial_type::RAND){
					for (size_t _row_ = 0; _row_ < in_rows; _row_++){
						for (size_t _col_ = 0; _col_ < in_cols; _col_++){

							matrix_host.elements_[IJ2MAT2D(_row_, _col_, in_cols)] = (T)rand() /RAND_MAX;
						}
					}
				}
			}
			
			mat(size_t in_rows, size_t in_cols, T* input_matrix){
				matrix_host.elements = new T[in_rows*in_cols];
				for (size_t _row_ = 0; _row_ < in_rows; _row_++){
					for (size_t _col_ = 0; _col_ < in_cols; _col_++){
						matrix_host.elements_[IJ2MAT(_row_, _col_, in_cols)] = input_matrix[IJ2MAT(_row_, _col_, in_cols)];
					}
				}
			}


			//Get the value of result at (i_th row, j_th col)
			T at(size_t i_th, size_t j_th){ return matrix_host.elements_[IJ2MAT2D(i_th, j_th, matrix_host.cols_)]; }

			//Allocate cuda memory
			void host_to_device();

			//Allocate cuda memory
			void allocate_cuda_memory();

			//Device to host memory
			void device_to_host();

			//Return device pointer
			T* return_device_pointer(){ return array_device; }

			
			//array for host 
			_mat<T> matrix_host;


		private:

			//Matrix array for device
			T* array_device;

			
			//Print out operator
			friend std::ostream& operator <<(std::ostream& out, const mat<T> &object){
				out << "[" << std::endl;
				for (size_t _row_ = 0; _row_ < object.matrix_host.rows_; _row_++){
					for (size_t _col_ = 0; _col_ < object.matrix_host.cols_; _col_++){
						out << object.matrix_host.elements_[IJ2MAT2D(_row_, _col_, object.matrix_host.cols_)] << ",";
						if (_col_ > 10)
							break;
					}
					if (_row_ > 10)
						break;
					out << std::endl;
				}
				out << "]" << std::endl;

				return out;
			}
		};

	}
}



template<class T>
void cuFun::cuMat::mat<T>::allocate_cuda_memory(){
	
	cudaMalloc((void**)&array_device, matrix_host.total_size());
}

template<class T>
void cuFun::cuMat::mat<T>::host_to_device(){
	allocate_cuda_memory();
	cudaMemcpy(array_device, matrix_host.elements_, matrix_host.total_size(), cudaMemcpyHostToDevice);
}

template<class T>
void cuFun::cuMat::mat<T>::device_to_host(){

	cudaMemcpy(matrix_host.elements_, array_device, matrix_host.total_size(), cudaMemcpyDeviceToHost);

}

template<typename T>
void cudaMatAdd(cuFun::cuMat::mat<T> &mat_a, cuFun::cuMat::mat<T> mat_b, cuFun::cuMat::mat<T> mat_c ){

	size_t cols_ = mat_b.matrix_host.cols_;
	size_t rows_ = mat_b.matrix_host.rows_;
	size_t threads = CUDA_THREADS;
	size_t blocks = (threads +rows_ + 1) / threads;



	T * a_ptr = mat_a.return_device_pointer();
	T * b_ptr = mat_b.return_device_pointer();

	T * c_ptr = mat_c.return_device_pointer();




	cuda_MatAdd(a_ptr, b_ptr, c_ptr, blocks,threads,rows_,cols_);

	//std::cout << blocks << threads << rows_ << cols_ << std::endl;


}



//Matrix multiplication: for A*B, where [A] = a,b     [B] = b,c
template<typename T>
void cpuMatMul(cuFun::cuMat::mat<T> &a, cuFun::cuMat::mat<T> b, cuFun::cuMat::mat<T> c){


	
	if (b.matrix_host.cols_ != c.matrix_host.rows_){
		std::cerr<< "Matrix must have correct sizes, check matrix for dimensions. " << std::endl;
	}

	size_t num_rows = b.matrix_host.rows_;
	size_t num_cols = c.matrix_host.cols_;
	size_t num_k = b.matrix_host.cols_;

	for (size_t _row_ = 0; _row_ < num_rows; _row_++){
		for (size_t _col_ = 0; _col_ < num_cols; _col_++){
			T sum_;
			for (size_t k = 0; k < num_k; k++){
				sum_ = b.at(_row_, k) *c.at(k, _col_);
			}

			a.matrix_host.elements_[IJ2MAT2D(_row_, _col_, num_cols)] = sum_;

		}
	}

	
}

//Matrix addition for C = A+B; where [A]=[B] (same size)
template<typename T>
void cpuMatAdd(cuFun::cuMat::mat<T> &a, cuFun::cuMat::mat<T> b, cuFun::cuMat::mat<T> c){



	if (b.matrix_host.cols_ != c.matrix_host.cols_){
		std::cerr << "Matrix must have correct sizes, check matrix for dimensions. " << std::endl;
	}
	else if (b.matrix_host.rows_ != c.matrix_host.rows_){
		std::cerr << "Matrix must have correct sizes, check matrix for dimensions. " << std::endl;
	}

	size_t num_rows = b.matrix_host.rows_;
	size_t num_cols = c.matrix_host.cols_;
	size_t num_k = b.matrix_host.cols_;

	for (size_t _row_ = 0; _row_ < num_rows; _row_++){
		for (size_t _col_ = 0; _col_ < num_cols; _col_++){
			T sum_;
			/*for (size_t k = 0; k < num_k; k++){
				sum_ = b.at(_row_, k) *c.at(k, _col_);
			}*/
			sum_ = b.at(_row_, _col_) + c.at(_row_, _col_);
			a.matrix_host.elements_[IJ2MAT2D(_row_, _col_, num_cols)] = sum_;

		}
	}


}

//
template<typename T>
void __global__
cuda_matAdd_global(T * a, const T * b, const T* c, size_t num_rows,size_t num_cols);

//Wrappers
void cuda_MatAdd(float * a, const float * b, const float * c,size_t blocks, size_t threads, size_t num_rows,size_t num_cols);

#endif #H_MATRIX_HEADER
