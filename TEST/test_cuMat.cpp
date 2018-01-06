#include <iostream>
#include <stdlib.h>
#include "vector.cuh"
#include "matrix.cuh"
#include <ctime>



//Range of test functions
bool test_vector(size_t n,size_t iterations);
bool test_matrix(size_t n, size_t iterations);
int main(){

	/*
	CHECK VECTOR
	*/
	bool vector_result = test_vector(
		100000,//Length
		10000 // Number of iteration
		);

	if (vector_result == true){
		std::cout << "Vector test CPU = GPU: TRUE"<< std::endl;
	}
	else{
		std::cout << "Vector test CPU = GPU: FALSE" << std::endl;
	}
	
	/*
	CHECK MATRIX
	*/
	bool matrix_result = test_matrix(100,200);
	
	return 0;
}



bool test_vector(size_t n, size_t num_iterations){
	//set random seed
	srand(1);

	//Make dummy vector
	float *rand_vec;
	

	rand_vec = new float[n];
	for (size_t i = 0; i < n; i++){
		rand_vec[i] = rand() / 32.0f;
	}


	cuFun::cuMat::vect<float> vect_f(n, rand_vec);
	cuFun::cuMat::vect<float> original = vect_f;

	//std::cout <<(vect_f);

	
	//CUDA SPEED 
	std::cout << "Starting cuda test ... " << std::endl;

	vect_f.host_to_device();
	cuFun::cuMat::vect<float> answer_GPU(n);
	answer_GPU.allocate_cuda_memory();
	clock_t start_cuda = std::clock();
	for (int i = 0; i < num_iterations; i++){
		//std::cout << "cuda : "<<i << std::endl;
		cudaVectAdd(answer_GPU, vect_f, n);
		//cudaVectAdd(answer_GPU, vect_f, vect_f, n);

	}
	clock_t end_cuda = std::clock();
	answer_GPU.device_to_host();



	//CPU SPEED
	std::cout << "Starting cpu test ... " << std::endl;

	cuFun::cuMat::vect<float> answer_CPU(n);
	clock_t start_CPU = std::clock();
	for (int i = 0; i < num_iterations; i++){
		//std::cout << "cpu : " <<i << std::endl;
		cpuVectAdd<float>(answer_CPU, answer_CPU, vect_f);
	}

	clock_t end_CPU = std::clock();

#if 0
	//nvidia cublas
	std::cout << "Starting cublas test ... " << std::endl;

	vect_f.host_to_device();
	cuFun::matFun::vect<float> answer_CUBLAS(n);
	answer_CUBLAS.allocate_cuda_memory();

	cublasHandle_t handle;
	cublasCreate(&handle);
	clock_t start_cublas = std::clock();
	for (int i = 0; i < num_iterations; i++){
		//std::cout << "cuda : "<<i << std::endl;
		cudaVectAdd_cublas(handle, answer_CUBLAS, vect_f, n);
		//cudaVectAdd(answer_GPU, vect_f, vect_f, n);

	}
	clock_t end_cublas = std::clock();
	cublasDestroy(handle);

	answer_CUBLAS.device_to_host();
#endif // 0




	//std::cout << "CUBLAS time : " << (float)(end_cublas - start_cublas) / CLOCKS_PER_SEC << std::endl;
	std::cout << "CUDA time : " << (float)(end_cuda - start_cuda) / CLOCKS_PER_SEC << std::endl;
	std::cout << "CPU time : " << (float)(end_CPU - start_CPU) / CLOCKS_PER_SEC << std::endl;


	//Check if CPU == GPU, one necessary assumption for correct implementation
	bool vector_correct = true;
	for (size_t i = 0; i < n; i++){
		if (answer_CPU.at(i) != answer_GPU.at(i)){
			vector_correct = false;
		}
	}

	delete[] rand_vec;
	return vector_correct;
}

bool test_matrix(size_t n = 100 ,size_t iter = 200){

	cuFun::cuMat::mat<float> cpu_result(n, n, cuFun::cuMat::initial_type::ZERO);
	cuFun::cuMat::mat<float> gpu_result(n, n, cuFun::cuMat::initial_type::ZERO);
	cuFun::cuMat::mat<float> a(n, n, cuFun::cuMat::initial_type::RAND);
	cuFun::cuMat::mat<float> b(n, n, cuFun::cuMat::initial_type::RAND);
	a.host_to_device();
	b.host_to_device();
	


	clock_t start_CPU = std::clock();
	for (int _i = 0; _i < iter; _i++)
		cpuMatAdd(cpu_result, cpu_result, b);
	clock_t  end_CPU = std::clock();

	clock_t  start_cuda = std::clock();
	gpu_result.allocate_cuda_memory();
	for (int _i = 0; _i < iter; _i++)
		cudaMatAdd(gpu_result, gpu_result, b);
	gpu_result.device_to_host();
	clock_t  end_cuda = std::clock();

	std::cout << "CPU RESULT: " << std::endl << cpu_result;
	std::cout << "GPU RESULT: " << std::endl << gpu_result;


	//std::cout << "CUBLAS time : " << (float)(end_cublas - start_cublas) / CLOCKS_PER_SEC << std::endl;
	std::cout << "CUDA time : " << (float)(end_cuda - start_cuda) / CLOCKS_PER_SEC << std::endl;
	std::cout << "CPU time : " << (float)(end_CPU - start_CPU) / CLOCKS_PER_SEC << std::endl;



	return 0;
}