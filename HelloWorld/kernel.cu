
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 10
#define ELEMENT_COUNT (MATRIX_SIZE * MATRIX_SIZE)

#define ALAPH 0.8
#define BETA  0.2

void doCPUStencil(float * M, int matrixSize);
void printMatrix(float * M, int matrixSize);
float getTopElement(float * M, int index, int matrixSize);
float getRightElement(float * M, int index, int matrixSize);
float getBottomElement(float * M, int index, int matrixSize);
float getLeftElement(float * M, int index, int matrixSize);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	float *M = (float *)malloc(sizeof(float) * ELEMENT_COUNT);
	for (int i = 0; i < ELEMENT_COUNT; i++) {
		M[i] = (float)i;
	}

	printMatrix(M, MATRIX_SIZE);

	doCPUStencil(M, MATRIX_SIZE);

	printMatrix(M, MATRIX_SIZE);

	return 0;
}

void printMatrix(float * M, int matrixSize) {
	for (int i = 0; i < matrixSize * matrixSize; i++) {
		printf("%10.0f ", M[i]);

		if (i % matrixSize + 1 == matrixSize)
		{
			printf("\n");
		}
	}

	printf("\n");
}

void doCPUStencil(float * M, int matrixSize) {
	for (int i = 0; i < matrixSize * matrixSize; i++) {
		M[i] = ALAPH * M[i] + BETA * (
			getTopElement(M, i, matrixSize) +
			getRightElement(M, i, matrixSize) +
			getBottomElement(M, i, matrixSize) +
			getLeftElement(M, i, matrixSize));
	}
}


float getTopElement(float * M, int index, int matrixSize) {
	if (index < matrixSize) {
		return 0.0;
	}
	else {
		return M[index - matrixSize];
	}
}
float getRightElement(float * M, int index, int matrixSize) {
	if ((index + 1) % matrixSize == 0) {
		return 0.0;
	}
	else {
		return M[index + 1];
	}
}
float getBottomElement(float * M, int index, int matrixSize) {
	if (index >= matrixSize * (matrixSize - 1)) {
		return 0.0;
	}
	else {
		return M[index + matrixSize];
	}
}
float getLeftElement(float * M, int index, int matrixSize) {
	if (index % matrixSize == 0) {
		return 0.0;
	}
	else {
		return M[index - 1];
	}
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
