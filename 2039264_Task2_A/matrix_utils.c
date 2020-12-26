/*
 * matrix_utils.c
 */

#include <stdio.h>
#include <stdlib.h>

void print_matrix(float** matrix, int n_rows, int n_columns)
{
	for (int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_columns; j++)
		{
			printf("%2.2f ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

float** random_matrix(const int n_rows, const int n_columns)
{
	float** matrix = (float**)malloc(sizeof(float*) * n_rows);

	for(int i = 0; i < n_rows; i++)
	{
		matrix[i] = (float*)malloc(sizeof(float) * n_columns);
		for(int j = 0; j < n_columns; j++)
		{
			matrix[i][j] = rand() % 10;
		}
	}

	return matrix;
}

float** zero_matrix(const int n_rows, const int n_columns)
{
	float** matrix = (float**)malloc(sizeof(float*) * n_rows);

	for(int i = 0; i < n_rows; i++)
	{
		matrix[i] = (float*)malloc(sizeof(float) * n_columns);
		for(int j = 0; j < n_columns; j++)
		{
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

void remove_matrix(float** matrix, const int n_rows, const int n_columns)
{
	for(int i = 0; i < n_rows; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}

float** add(float** A, float** B, const int n_rows, int n_columns)
{
	float** C = zero_matrix(n_rows, n_columns);

	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_columns; j++)
		{
			C[i][j] = A[i][j] + B[i][j];
		}
	}

	return C;
}

float** subtract(float** A, float** B, const int n_rows, int n_columns)
{
	float** C = zero_matrix(n_rows, n_columns);

	for(int i = 0; i < n_rows; i++)
	{
		for(int j = 0; j < n_columns; j++)
		{
			C[i][j] = A[i][j] - B[i][j];
		}
	}

	return C;
}
