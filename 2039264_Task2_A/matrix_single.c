
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "config.h"
#include "matrix_utils.h"

#define DIVIDE_SPLIT_LEVEL 7

/*************************

Compile with:

	cc ./matrix_single.c matrix_utils.c -lm -o matrix_single
	
**************************/

void single_threaded (float** C, float** A, int n, int p, int m, float** D)
{
	
	int side_length = pow(2, DIVIDE_SPLIT_LEVEL);
	if(n == side_length && p == side_length && m == side_length)
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
	else
	{
		int _n = n / 2, _p = p / 2, _m = m / 2;

		float ** c11 = zero_matrix(_n, _p);
		float ** c12 = zero_matrix(_n, _p);
		float ** c21 = zero_matrix(_n, _p);
		float ** c22 = zero_matrix(_n, _p);

		float ** a11 = zero_matrix(_n, _p);
		float ** a12 = zero_matrix(_n, _p);
		float ** a21 = zero_matrix(_n, _p);
		float ** a22 = zero_matrix(_n, _p);

		float ** d11_1 = zero_matrix(_n, _p);
		float ** d12_1 = zero_matrix(_n, _p);
		float ** d21_1 = zero_matrix(_n, _p);
		float ** d22_1 = zero_matrix(_n, _p);

		float ** d11_2 = zero_matrix(_n, _p);
		float ** d12_2 = zero_matrix(_n, _p);
		float ** d21_2 = zero_matrix(_n, _p);
		float ** d22_2 = zero_matrix(_n, _p);

		for(int i = 0; i < _n; i++)
		{
			for(int j = 0; j < _p; j++)
			{
				c11[i][j] = C[i][j];
				c12[i][j] = C[i][_n + j];
				c21[i][j] = C[_n + i][j];
				c22[i][j] = C[_n + i][_n + j];

				a11[i][j] = A[i][j];
				a12[i][j] = A[i][_n + j];
				a21[i][j] = A[_n + i][j];
				a22[i][j] = A[_n + i][_n + j];
			}
		}

		single_threaded(c11, a11, _n, _p, _m, d11_1);
		single_threaded(c11, a12, _n, _p, _m, d12_1);
		single_threaded(c21, a11, _n, _p, _m, d21_1);
		single_threaded(c21, a12, _n, _p, _m, d22_1);

		single_threaded(c12, a21, _n, _p, _m, d11_2);
		single_threaded(c12, a22, _n, _p, _m, d12_2);
		single_threaded(c22, a21, _n, _p, _m, d21_2);
		single_threaded(c22, a22, _n, _p, _m, d22_2);

		float ** d11 = add(d11_1, d11_2, _n, _m);
		float ** d12 = add(d12_1, d12_2, _n, _m);
		float ** d21 = add(d21_1, d21_2, _n, _m);
		float ** d22 = add(d22_1, d22_2, _n, _m);

		for(int i = 0; i < _n; i++)
		{
			for(int j = 0; j < _p; j++)
			{
				D[i][j] = d11[i][j];
				D[i][_n + j] = d12[i][j];
				D[_n + i][j] = d21[i][j];
				D[_n + i][_n + j] = d22[i][j];
			}
		}

		remove_matrix(c11, _n, _p);
		remove_matrix(c12, _n, _p);
		remove_matrix(c21, _n, _p);
		remove_matrix(c22, _n, _p);

		remove_matrix(a11, _n, _p);
		remove_matrix(a12, _n, _p);
		remove_matrix(a21, _n, _p);
		remove_matrix(a22, _n, _p);

		remove_matrix(d11_1, _n, _p);
		remove_matrix(d12_1, _n, _p);
		remove_matrix(d21_1, _n, _p);
		remove_matrix(d22_1, _n, _p);

		remove_matrix(d11_2, _n, _p);
		remove_matrix(d12_2, _n, _p);
		remove_matrix(d21_2, _n, _p);
		remove_matrix(d22_2, _n, _p);

		remove_matrix(d11, _n, _p);
		remove_matrix(d12, _n, _p);
		remove_matrix(d21, _n, _p);
		remove_matrix(d22, _n, _p);
	}
}

int main(int argc, char** argv)
{
	

	printf("Single threaded of Matrix Multiplication\n");
	printf(" N : %d\n M : %d\n P : %d\n Number Of loops  : %d \n", N, M, P, MATRIX_MULTIPLICATION_TEST_COUNT);
	printf("%10s %10s \n", "loop", "Time(seconds)");

	float** C = random_matrix(N, P);
	float** A = random_matrix(P, M);
	float** D = zero_matrix(N, M);


	struct timespec start, finish;
	double test_time[MATRIX_MULTIPLICATION_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < MATRIX_MULTIPLICATION_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		single_threaded(C, A, N, P, M, D);
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
	print_matrix(C, N, P);
	print_matrix(A, P, M);
	print_matrix(D, N, M);
#endif

	remove_matrix(C, N, P);
	remove_matrix(A, P, M);
	remove_matrix(D, N, M);

	return 0;
}
