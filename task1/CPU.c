#include <stdio.h>
#include <stdlib.h>

void initArray(int *array, const int elemNum)
{
	int i;
	for (i = 0; i < elemNum; ++i)
	{
		array[i] = i;
	}
}

// increase one for all the elements
void incrOneForAll(int *array, const int elemNum)
{
	int i;
	for (i = 0; i < elemNum; ++i)
	{
		array[i] ++;
	}
}

void printArray(const int *array, const int elemNum)
{
	int i;
	for (i = 0; i < elemNum; ++i)
	{
		printf("%d ", array[i]);
	}
}

int main(int argc, char *argv[])
{
	const int elemNum = 1024;
	// host memory
	int h_data[elemNum];
	
	initArray(h_data, elemNum);

	incrOneForAll(h_data, elemNum);

	printArray(h_data, elemNum);

	return 0;
}

