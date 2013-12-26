#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{
	// We assume that the element number is the power of 2 for simplification.
	const int elemNum = 1 << 22;
	int arraySize = elemNum * sizeof(int);
	// host memory
	int array[arraySize];

	int i;
	for (i = 0; i < elemNum; ++i)
	{
		array[i] = rand() % 1024;
		printf("%d ", array[i]);
	}

	return 0;
}
