#ifndef H_VECTOR_HEADER
#define H_VECTOR_HEADER
#include <iostream>
#include <ostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "matrix.cuh"



namespace cuFun
{
	namespace matFun
	{
		
		

		//Shape of vector
		typedef enum{
			ROW_VEC,
			COLUMN_VEC
		} _vector_shape;



		template<typename T>
		struct _vect{
			int length;
			T *elements;

		};

		template<class T>
		class vect
		{
		public:
			vect(void){};
			vect(int size){ 
				length_ = size; 
				vector_host.length = size; 
				vector_host.elements = new T[size]; 
				for (int _i = 0; _i < size; _i++){
					vector_host.elements[_i] = (T)0;
				}
			};
			vect(int size, T *input_vector){//Initialize host vector
				vector_host.elements = new T[size];

				for (int i = 0; i < size; i++){
					vector_host.elements[i] = input_vector[i];
				}
				vector_host.length = size;

				length_ = size;
			}
			~vect();


			//initializes the vector
			void initialize_vector(int size, T *input_vector);

			//Allocate cuda memory and copy host to device
			void host_to_device();

			//Allocate cuda memory
			void allocate_cuda_memory();

			//Download from device to host
			void device_to_host();

			//Get length
			int return_length(){ return length_; }
			
			//Cuda add, both inputs will be vector vect<T> class objects
			//vect<T> cudaVectAdd(vect<T> const vec_a, vect<T> const vect_b, int n);
	
			//Return device pointer
			T* return_device_pointer(){ return array_device; }
		
			
			//Array for host
			_vect<T> vector_host;


			//Returns the value at position indicated, note only for host 
			T at(int _idx){ if (_idx < length_){ return vector_host.elements[_idx]; } else{ return 0; } }
		private:
			//
			int length_;


			//Array pointer for device
			T* array_device;

			//Print out operator
			friend std::ostream& operator << (std::ostream& out, const  vect<T> & object){
				out << "[";
				int print_to = object.vector_host.length;

				//Set print limit
				if (print_to > 10){

					print_to = 10;
				}
				for (int i = 0; i < print_to; i++){

					out << object.vector_host.elements[i] << " ";
				}
				if (object.vector_host.length > 10){
					out << " ... " << std::endl;
				}
				out << "]" << std::endl;
				return out;
			}

			//friend void cpu_VectAdd(vect<T> &a, const vect<T> b, const vect<T> c);
			
		


		};





	}

}

template<class T>
void cuFun::matFun::vect<T>::allocate_cuda_memory(){
	size_t length_ = vector_host.length;
	//first allocate cuda vector
	cudaMalloc((void**)&array_device, sizeof(T)*length_);

	cudaMemset(array_device,0,sizeof(T)*length_);
}

template<class T>
void cuFun::matFun::vect<T>::host_to_device(){
	size_t length_ = vector_host.length;
	//first allocate cuda vector
	allocate_cuda_memory();
	T * temp_array = new T[length_];
	for (size_t _i = 0; _i < length_; _i++){
		temp_array[_i] = vector_host.elements[_i];
	}
	cudaMemcpy(array_device,vector_host.elements, sizeof(T)*length_, cudaMemcpyHostToDevice);

	delete[] temp_array;
}

template<class T>
void cuFun::matFun::vect<T>::device_to_host(){

	size_t length_ = vector_host.length;

	cudaMemcpy(vector_host.elements, array_device, sizeof(T)*length_, cudaMemcpyDeviceToHost);

	
}

template<class T>
cuFun::matFun::vect<T>::~vect(){

	if (array_device != NULL)
		cudaFree(array_device);
}


template<typename T>
void cpu_VectAdd(cuFun::matFun::vect<T>  &a, cuFun::matFun::vect<T>   b, cuFun::matFun::vect<T>    c){
	/*if (b.vector_host.length != c.vector_host.length){
	std::cout << "Error vectors must be the same size " << std::endl;
	abort();
	}*/

	int  _flength = b.return_length();
	for (int i = 0; i < _flength; i++){
		a.vector_host.elements[i] = b.vector_host.elements[i] + c.vector_host.elements[i];
	}

}


template<typename T>
void cudaVectAdd(cuFun::matFun::vect<T> &vec_a, cuFun::matFun::vect<T> vec_b, int n){

	int threads = 1024;
	int blocks = (threads + n + 1) / threads;



	T * a_ptr = vec_a.return_device_pointer();
	T * b_ptr = vec_b.return_device_pointer();


	cuda_vectAdd(a_ptr, b_ptr, blocks, threads, n);




}

template<typename T>
void cudaVectAdd(cuFun::matFun::vect<T> &vec_a, cuFun::matFun::vect<T> vec_b, cuFun::matFun::vect<T> vec_c, int n){

	int threads = 1024;
	int blocks = (threads + n +1 ) / threads;



	T * a_ptr = vec_a.return_device_pointer();
	T * b_ptr = vec_b.return_device_pointer();

	T * c_ptr = vec_b.return_device_pointer();


	cuda_vectAdd(a_ptr, b_ptr, c_ptr, blocks, threads, n);
	
	

	
}



template<typename T>
void cudaVectAdd_cublas(cublasHandle_t handle, cuFun::matFun::vect<T> &vec_a, cuFun::matFun::vect<T> vec_b, int n){

	T * a_ptr = vec_a.return_device_pointer();
	T * b_ptr = vec_b.return_device_pointer();

	const float alpha = 1.0f;
	
	cublasSaxpy(handle, n, &alpha, b_ptr, 1,  a_ptr ,1);
	

}


/*
Vector add routines
*/
//CUDA
template<typename T>
void __global__
cuda_vectAdd_global(T * a, const T * b, int n);

template<typename T>
void __global__
cuda_vectAdd_global(T * a, const T * b, const T* c, int n);

//Wrappers
void cuda_vectAdd(float * a, const float * b, int blocks, int threads, int n);
void cuda_vectAdd(float * a, const float * b, const float * c, int blocks, int threads, int n);


#endif #H_VECTOR_HEADER