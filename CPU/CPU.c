#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int getSum(const int *array, const int elemNum)
{
	int i;
	int sum = 0;
	for (i = 0; i < elemNum; ++i)
	{
		sum += array[i];
	}
	return sum;
}

int main(int argc, char *argv[])
{
	// We assume that the element number is the power of 2 for simplification.
	const int elemNum = 1 << 22;
	int arraySize = elemNum * sizeof(int);
	// host memory
	int *h_idata;

	h_idata = malloc(arraySize);
	
	FILE *fp;
	if((fp = fopen(argv[1], "rb")) == NULL)
	{
		printf("Can not open input file!\n");
		exit(0);
	}
	int i;
	for (i = 0; i < elemNum; ++i)
	{
		fscanf(fp, "%d", &h_idata[i]);
	}
	fclose(fp);

	struct timespec start, end;
	double totalTime;
	clock_gettime(CLOCK_REALTIME,&start);
	int sum = getSum(h_idata, elemNum);
	clock_gettime(CLOCK_REALTIME,&end);
	totalTime = (double)(end.tv_sec-start.tv_sec)*1000+(double)(end.tv_nsec-start.tv_nsec)/(double)1000000L;

	float bandwidth = elemNum * sizeof(int) / (totalTime / 1000) / 1024 / 1024 / 1024;
	printf("%d %fms %fGB/s\n", sum, totalTime, bandwidth);
	return 0;
}
