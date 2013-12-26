#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void reduce(int *g_idata, int *g_odata)
{
}

int main(int argc, char *argv[])
{
	// We assume that the element number is the power of 2 for simplification.
	const int elemNum = 1 << 22;
	int arraySize = elemNum * sizeof(int);
	// host memory
	int *h_idata;
	int sum; // final output value
	// device memory
	int *d_idata; // input data ptr
	int *d_odata; // output data ptr

	// initialize input data from file
	// use the first argument as the file name
	h_idata = (int *) malloc(arraySize);
	FILE *fp;
	if((fp = fopen(argv[1], "rb")) == NULL)
	{
		printf("Can not open input file!\n");
		exit(0);
	}
	for (int i = 0; i < elemNum; ++i)
	{
		fscanf(fp, "%d", &h_idata[i]);
	}
	fclose(fp);

	// TODO: copy input data from CPU to GPU
	// Hint: use cudaMalloc and cudaMemcpy

	int threadNum = 0;
	int blockNum = 0;
	
	// TODO: malloc GPU output memory to store the outcome from each kernel.
	// Hint: use cudaMalloc

	cudaEvent_t start, stop;
	float stepTime;
	float totalTime = 0;
	// create event for recording GPU execution time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute the first kernel and set the GPU timer
	cudaEventRecord(start, 0);
	// TODO: set grid and block size
	// threadNum = ?
	// blockNum = ?
	// parameters for the first kernel
	int sMemSize = 1024 * sizeof(int);
	reduce<<<threadNum, blockNum, sMemSize>>>(d_idata, d_odata);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// calculate the execution time of the first kernel
	cudaEventElapsedTime(&stepTime, start, stop);
	totalTime += stepTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// create event for recording GPU execution time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute the second kernel and set the GPU timer
	cudaEventRecord(start, 0);
	// TODO: set grid and block size
	// threadNum = ?
	// blockNum = ?
	sMemSize = threadNum * sizeof(int);
	reduce<<<threadNum, blockNum, sMemSize>>>(d_odata, d_odata);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// calculate the execution time of the current kernel
	cudaEventElapsedTime(&stepTime, start, stop);
	totalTime += stepTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* 
	 * Hint: for "first add during global load" optimization, the third kernel is unnecessary. 
	 */
	// create event for recording GPU execution time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute the third kernel and set the GPU timer
	cudaEventRecord(start, 0);
	// TODO: set grid and block size
	// threadNum = ?
	// blockNum = ?
	sMemSize = threadNum * sizeof(int);
	reduce<<<threadNum, blockNum, sMemSize>>>(d_odata, d_odata);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// calculate the execution time of the current kernel
	cudaEventElapsedTime(&stepTime, start, stop);
	totalTime += stepTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// TODO: copy result and free device memory
	// Hint: use cudaMemcpy and cudaFree
		
	float bandwidth = elemNum * sizeof(int) / (totalTime / 1000) / 1024 / 1024 / 1024;
	printf("%d %fms %fGB/s\n", sum, totalTime, bandwidth);
	return 0;
}
