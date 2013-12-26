#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void reduce(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
	}
}

int main(int argc, char *argv[])
{
	// We assume that the element number is the power of 2 for simplification.
	const int elemNum = 1 << 22;
	int arraySize = elemNum * sizeof(int);
	// host memory
	int *h_idata;
	int sum;
	// device memory
	int *d_idata;
	int *d_odata;

	// initialize input data
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

	// copy input data from CPU to GPU
	cudaMalloc((void **) &d_idata, arraySize);
	cudaMemcpy(d_idata, h_idata, arraySize, cudaMemcpyHostToDevice);

	int threadNum = 0;
	int blockNum = 0;
	// calculate the threadNum and blockNum for the first kernel
	cudaDeviceProp deviceProperties;
	cudaGetDeviceProperties(&deviceProperties, 0);
	int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock; // maxThreadsPerBlock = 1024 on K20X
	threadNum = (elemNum > maxThreadsPerBlock)? maxThreadsPerBlock: elemNum;
	blockNum = (int) ceil((double) elemNum / threadNum); // blockNum = 4096
	
	// the number of output elements of the first kernel is blockNum 
	cudaMalloc((void **) &d_odata, blockNum * sizeof(int));

	// use GPU of id=0
	cudaSetDevice(0);

	// parameters for the first kernel
	dim3 gridDim(blockNum, 1, 1);
	dim3 blockDim(threadNum, 1, 1);
	int sMemSize = threadNum * sizeof(int);
	
	cudaEvent_t start, stop;
	float stepTime;
	float totalTime = 0;
	// create event for recording GPU execution time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// execute the first kernel and set the GPU timer
	cudaEventRecord(start, 0);
	reduce<<<gridDim, blockDim, sMemSize>>>(d_idata, d_odata);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// calculate the execution time of the first kernel
	cudaEventElapsedTime(&stepTime, start, stop);
	totalTime += stepTime;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// calculate the threadNum and blockNum for the next kernel
	threadNum = (blockNum > maxThreadsPerBlock)? maxThreadsPerBlock: blockNum;
	blockNum = (int) ceil((double) blockNum / threadNum);
	while(blockNum >= 1) {
		// parameters for the current kernel
		dim3 gridDim(blockNum, 1, 1);
		dim3 blockDim(threadNum, 1, 1);
		sMemSize = threadNum * sizeof(int);

		// create event for recording GPU execution time
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// execute the current kernel and set the GPU timer
		cudaEventRecord(start, 0);
		reduce<<<gridDim, blockDim, sMemSize>>>(d_odata, d_odata);
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		// calculate the execution time of the current kernel
		cudaEventElapsedTime(&stepTime, start, stop);
		totalTime += stepTime;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		if (blockNum == 1) break;
	
		// calculate the threadNum and blockNum for the next kernel
		threadNum = (blockNum > maxThreadsPerBlock)? maxThreadsPerBlock: blockNum;
		blockNum = (int) ceil((double) blockNum / threadNum);
	}

	// copy result back to CPU
	cudaMemcpy(&sum, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_idata);
	cudaFree(d_odata);
	
	float bandwidth = elemNum * sizeof(int) / (totalTime / 1000) / 1024 / 1024 / 1024;
	printf("%d %fms %fGB/s\n", sum, totalTime, bandwidth);
	return 0;
}
