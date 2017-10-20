#ifndef H_MATRIX_HEADER
#define H_MATRIX_HEADER
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector.cuh"

//Preprocessor function to convert (i,j) to matrix index
//i- rows, j - cols. id = (i-1) * num_cols + j 
#define IJ2MAT2D(i,j,num_cols) (i * num_cols + j )

namespace cuFun
{
	
	namespace cuMat
	{

		//Initialization , zero, or random,
		enum initial_type{
			ZERO,
			RAND,
		};

		//Matrix struct
		template<typename T>
		struct _mat{
			size_t rows_;
			size_t cols_;
			T *elements_;
		};

		


		//Matrix template class
		template<class T>
		class mat
		{
		public:
			mat(void){ };

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
					}
					out << std::endl;
				}
				out << "]" << std::endl;

				return out;
			}
		};

	}
}


template<typename T>
void cpuMatMul(cuFun::cuMat::mat<T> &a, cuFun::cuMat::mat<T> b, cuFun::cuMat::mat<T> c){

	if (b.matrix_host.cols_ != c.matrix_host.rows_){
		throw "Matrix must have correct sizes, check matrix for dimensions. ";// << std::endl;
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
#endif #H_MATRIX_HEADER
