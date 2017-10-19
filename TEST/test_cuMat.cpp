#include <iostream>
#include <stdlib.h>
#include "vector.cuh"
#include "matrix.cuh"

#include <ctime>

int main(){
	//set random seed
	srand(1);

	//Make dummy vector
	float *rand_vec;
	size_t n = 10000;

	rand_vec = new float[n];
	for (size_t i = 0; i < n; i++){
		rand_vec[i] = rand() /32.0f;
	}

	
	cuFun::matFun::vect<float> vect_f(n,rand_vec);
	cuFun::matFun::vect<float> original = vect_f;

	//std::cout <<(vect_f);
	
	int num_iterations = 1;
	//CUDA SPEED 
	std::cout << "Starting cuda test ... " << std::endl;
	
	vect_f.host_to_device();
	cuFun::matFun::vect<float> answer_GPU(n);
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

	cuFun::matFun::vect<float> answer_CPU(n);
	clock_t start_CPU = std::clock();
	for (int i = 0; i < num_iterations; i++){
		//std::cout << "cpu : " <<i << std::endl;
		cpu_VectAdd<float>(answer_CPU, answer_CPU, vect_f);
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
	//std::cout << answer_CPU.at(0) << std::endl;
	std::cout << answer_CPU << std::endl;
	std::cout << answer_GPU << std::endl;



	//Check if CPU == GPU, I assume this to be accurate check
	bool vector_correct = true;
	for (size_t i= 0; i < n; i++){
		if (answer_CPU.at(i) != answer_GPU.at(i)){
			vector_correct = false;
		}
	}

	std::cout << "Print answer is : " << vector_correct << float(32.2) << std::endl;
	//std::cout << answer_CUBLAS << std::endl;

	/*for (int i = 0; i < 10; i++){
		std::cout << i << " ";
		original = (original + vect_f);
		std::cout << (original);
	}
*/

	
	
	//std::cout << (answer);

	

	

	delete[] rand_vec;

	return 0;
}