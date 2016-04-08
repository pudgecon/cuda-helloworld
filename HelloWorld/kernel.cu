
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define ALPHA 1 // 0.8
#define BETA  1 // 0.2

void printMatrix(float ** M, int rowSize, int columnSize);
void doCPUStencil(float ** M, int rowSize, int columnSize);
float getTopElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize);
float getRightElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize);
float getBottomElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize);
float getLeftElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize);

int main()
{
	int const rowSize    = 1024;
	int const columnSize = 1024;

	float **M = (float **)malloc(sizeof(float *) * rowSize);

	for (int i = 0; i < rowSize; i++) {
		// 行优先（Row Major）
		M[i] = (float *)malloc(sizeof(float) * columnSize);

		for (int j = 0; j < columnSize; j++) {
			M[i][j] = (float) i * 10 + j;
		}
	}

	/*printf("Origin Matrix:\n");
	printMatrix(M, rowSize, columnSize);*/

	doCPUStencil(M, rowSize, columnSize);

	/*printf("Transformed Matrix:\n");
	printMatrix(M, rowSize, columnSize);*/
	

	for (int i = 0; i < rowSize; i++) {
		free(M[i]);
	}
	free(M);

	return 0;
}

void printMatrix(float ** M, int rowSize, int columnSize) {
	for (int i = 0; i < rowSize; i++) {
		for (int j = 0; j < columnSize; j++) {
			printf("%5.0f ", M[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void doCPUStencil(float ** M, int rowSize, int columnSize) {
	LARGE_INTEGER m_liPerfFreq = { 0 };
	//获取每秒多少CPU Performance Tick
	QueryPerformanceFrequency(&m_liPerfFreq);
	LARGE_INTEGER m_liPerfStart = { 0 };
	QueryPerformanceCounter(&m_liPerfStart);

	float * tmp = (float *)malloc(sizeof(float) * columnSize);

	for (int i = 0; i < rowSize; i++) {
		for (int j = 0; j < columnSize; j++) {
            tmp[j] = ALPHA * M[i][j] + BETA * (
				getTopElement(M, i, j, rowSize, columnSize) +
				getRightElement(M, i, j, rowSize, columnSize) +
				getBottomElement(M, i, j, rowSize, columnSize) +
				getLeftElement(M, i, j, rowSize, columnSize));
		}
		for (int j = 0; j < columnSize; j++) {
			M[i][j] = tmp[j];
		}
	}

	free(tmp);

	LARGE_INTEGER liPerfNow = { 0 };
	// 计算CPU运行到现在的时间
	QueryPerformanceCounter(&liPerfNow);
	int time = (((liPerfNow.QuadPart - m_liPerfStart.QuadPart) * 1000) / m_liPerfFreq.QuadPart);
	char buffer[100];
	sprintf(buffer, "执行时间： %d millisecond.\n", time);
	printf(buffer);
}

float getTopElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize) {
	if (indexOfRow == 0) {
		return 0.0;
	}
	else {
		return M[indexOfRow - 1][indexOfColumn];
	}
}
float getRightElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize) {
	if (indexOfColumn == columnSize - 1) {
		return 0.0;
	}
	else {
		return M[indexOfRow][indexOfColumn + 1];
	}
}
float getBottomElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize) {
	if (indexOfRow == rowSize - 1) {
		return 0.0;
	}
	else {
		return M[indexOfRow + 1][indexOfColumn];
	}
}
float getLeftElement(float ** M, int indexOfRow, int indexOfColumn, int rowSize, int columnSize) {
	if (indexOfColumn == 0) {
		return 0.0;
	}
	else {
		return M[indexOfRow][indexOfColumn - 1];
	}
}
