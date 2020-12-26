
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "config.h"
#include "matrix_utils.h"

void single_threaded_matrix_multiplication(float** C, float** A, int n, int p, int m, float** D)
{
	for (int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			for(int k = 0; k < p; k++)
			{
				D[i][j] += C[i][k] * A[k][j];
			}
		}
	}
}

int main(int argc, char** argv)
{


	printf("Matrix Multiplication - Single Threaded\n");
	printf(" N : %d\n M : %d\n P : %d\n No Number of Loops : %d \n", N, M, P, MATRIX_MULTIPLICATION_TEST_COUNT);
	printf("%10s %10s \n", "Loop", "Time(seconds)");

	float** C = matrix_random(N, P);
	float** A = matrix_random(P, M);
	float** D = matrix_zero(N, M);


	struct timespec start, finish;
	double test_time[MATRIX_MULTIPLICATION_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < MATRIX_MULTIPLICATION_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		single_threaded_matrix_multiplication(C, A, N, P, M, D);
		clock_gettime(CLOCK_REALTIME, &finish);

		long seconds = finish.tv_sec - start.tv_sec;
	    long ns = finish.tv_nsec - start.tv_nsec;

	    if (start.tv_nsec > finish.tv_nsec)
	    {
	    	--seconds;
	    	ns += 1000000000;
	    }

	    double time_elapsed = (double)seconds + (double)ns/(double)1000000000;

		test_time[i] = time_elapsed;
		total_time += time_elapsed;
		printf("%5d %15.3f \n", (i + 1), time_elapsed);
		fflush(stdout);
	}

	average_time = total_time / MATRIX_MULTIPLICATION_TEST_COUNT;

	for(int i = 0; i < MATRIX_MULTIPLICATION_TEST_COUNT; i++)
	{
		total_square_time += pow(test_time[i] - average_time, 2);
	}

	variance_time = sqrt(total_square_time / MATRIX_MULTIPLICATION_TEST_COUNT);
	printf("\n Average Time %5.3f +/- %5.3f seconds \n", average_time, variance_time);

#ifdef PRINT_RESULT
	matrix_print(C, N, P);
	matrix_print(A, P, M);
	matrix_print(D, N, M);
#endif

	matrix_delete(C, N, P);
	matrix_delete(A, P, M);
	matrix_delete(D, N, M);

	return 0;
}
