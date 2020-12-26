
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "config.h"
#include "matrix_utils.h"



#define DIVIDE_SPLIT_LEVEL 7
/**************************
compile with

	 "cc ./matrix_multi.c matrix_utils.c -lm -lpthread -o matrix_multi"
	 
***************************/	 

typedef struct MatrixMultiplicationArgs {
	float** C;
	float** A;
	float** D;
	uint16_t n;
	uint16_t p;
	uint16_t m;
}MatrixMultiplicationArgs_t;

void* multi_thread(void* args)
{
	MatrixMultiplicationArgs_t* _args = (MatrixMultiplicationArgs_t*) args;
	float** C = _args->C;
	float** A = _args->A;
	float** D = _args->D;
	uint16_t n = _args->n;
	uint16_t p = _args->p;
	uint16_t m = _args->m;

	// Assuming n, p and m are powers of 2
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

		float ** a11 = zero_matrix(_n, _p);
		float ** a12 = zero_matrix(_n, _p);
		float ** a21 = zero_matrix(_n, _p);
		float ** a22 = zero_matrix(_n, _p);

		float ** b11 = zero_matrix(_n, _p);
		float ** b12 = zero_matrix(_n, _p);
		float ** b21 = zero_matrix(_n, _p);
		float ** b22 = zero_matrix(_n, _p);

		float ** c11_1 = zero_matrix(_n, _p);
		float ** c12_1 = zero_matrix(_n, _p);
		float ** c21_1 = zero_matrix(_n, _p);
		float ** c22_1 = zero_matrix(_n, _p);

		float ** c11_2 = zero_matrix(_n, _p);
		float ** c12_2 = zero_matrix(_n, _p);
		float ** c21_2 = zero_matrix(_n, _p);
		float ** c22_2 = zero_matrix(_n, _p);

		for(int i = 0; i < _n; i++)
		{
			for(int j = 0; j < _p; j++)
			{
				a11[i][j] = C[i][j];
				a12[i][j] = C[i][_n + j];
				a21[i][j] = C[_n + i][j];
				a22[i][j] = C[_n + i][_n + j];

				b11[i][j] = A[i][j];
				b12[i][j] = A[i][_n + j];
				b21[i][j] = A[_n + i][j];
				b22[i][j] = A[_n + i][_n + j];
			}
		}

		pthread_t threads[8];
		MatrixMultiplicationArgs_t* _args1 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args1->C = a11;
		_args1->A = b11;
		_args1->D = c11_1;
		_args1->n = _n;
		_args1->p = _p;
		_args1->m = _m;
		pthread_create(&threads[0], NULL, multi_thread, _args1);

		MatrixMultiplicationArgs_t* _args2 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args2->C = a11;
		_args2->A = b12;
		_args2->D = c12_1;
		_args2->n = _n;
		_args2->p = _p;
		_args2->m = _m;
		pthread_create(&threads[1], NULL, multi_thread, _args2);

		MatrixMultiplicationArgs_t* _args3 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args3->C = a21;
		_args3->A = b11;
		_args3->D = c21_1;
		_args3->n = _n;
		_args3->p = _p;
		_args3->m = _m;
		pthread_create(&threads[2], NULL, multi_thread, _args3);

		MatrixMultiplicationArgs_t* _args4 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args4->C = a21;
		_args4->A = b12;
		_args4->D = c22_1;
		_args4->n = _n;
		_args4->p = _p;
		_args4->m = _m;
		pthread_create(&threads[3], NULL, multi_thread, _args4);

		MatrixMultiplicationArgs_t* _args5 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args5->C = a12;
		_args5->A = b21;
		_args5->D = c11_2;
		_args5->n = _n;
		_args5->p = _p;
		_args5->m = _m;
		pthread_create(&threads[4], NULL, multi_thread, _args5);

		MatrixMultiplicationArgs_t* _args6 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args6->C = a12;
		_args6->A = b22;
		_args6->D = c12_2;
		_args6->n = _n;
		_args6->p = _p;
		_args6->m = _m;
		pthread_create(&threads[5], NULL, multi_thread, _args6);

		MatrixMultiplicationArgs_t* _args7 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args7->C = a22;
		_args7->A = b21;
		_args7->D = c21_2;
		_args7->n = _n;
		_args7->p = _p;
		_args7->m = _m;
		pthread_create(&threads[6], NULL, multi_thread, _args7);

		MatrixMultiplicationArgs_t* _args8 = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		_args8->C = a22;
		_args8->A = b22;
		_args8->D = c22_2;
		_args8->n = _n;
		_args8->p = _p;
		_args8->m = _m;
		pthread_create(&threads[7], NULL, multi_thread, _args8);

		for(int i = 0; i < 8; i++)
		{
			pthread_join(threads[i], NULL);
		}

		free(_args1);
		free(_args2);
		free(_args3);
		free(_args4);
		free(_args5);
		free(_args6);
		free(_args7);
		free(_args8);

		float ** c11 = add(c11_1, c11_2, _n, _m);
		float ** c12 = add(c12_1, c12_2, _n, _m);
		float ** c21 = add(c21_1, c21_2, _n, _m);
		float ** c22 = add(c22_1, c22_2, _n, _m);

		for(int i = 0; i < _n; i++)
		{
			for(int j = 0; j < _p; j++)
			{
				D[i][j] = c11[i][j];
				D[i][_n + j] = c12[i][j];
				D[_n + i][j] = c21[i][j];
				D[_n + i][_n + j] = c22[i][j];
			}
		}

		delete_matrix(a11, _n, _p);
		delete_matrix(a12, _n, _p);
		delete_matrix(a21, _n, _p);
		delete_matrix(a22, _n, _p);

		delete_matrix(b11, _n, _p);
		delete_matrix(b12, _n, _p);
		delete_matrix(b21, _n, _p);
		delete_matrix(b22, _n, _p);

		delete_matrix(c11_1, _n, _p);
		delete_matrix(c12_1, _n, _p);
		delete_matrix(c21_1, _n, _p);
		delete_matrix(c22_1, _n, _p);

		delete_matrix(c11_2, _n, _p);
		delete_matrix(c12_2, _n, _p);
		delete_matrix(c21_2, _n, _p);
		delete_matrix(c22_2, _n, _p);

		delete_matrix(c11, _n, _p);
		delete_matrix(c12, _n, _p);
		delete_matrix(c21, _n, _p);
		delete_matrix(c22, _n, _p);
	}
	return NULL;
}

int main(int argc, char** argv)
{
	
	printf("Multi Threaded of Matrix Multiplication \n");
	printf(" N : %d\n M : %d\n P : %d\n Number Of Multiplication : %d\n Number Of threads : %d \n",
			N, M, P, MATRIX_MULTIPLICATION_TEST_COUNT, MATRIX_MULTIPICATION_NO_OF_THREADS);
	printf("%10s %10s \n", "Loop", "Time(seconds)");

	float** C = random_matrix(N, P);
	float** A = random_matrix(P, M);
	float** D = zero_matrix(N, M);


	struct timespec start, finish;
	double test_time[MATRIX_MULTIPLICATION_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < MATRIX_MULTIPLICATION_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		MatrixMultiplicationArgs_t* _args = (MatrixMultiplicationArgs_t*) malloc(sizeof(MatrixMultiplicationArgs_t));
		 _args->C = C;
		 _args->A = A;
		 _args->D = D;
		 _args->n = N;
		 _args->p = P;
		 _args->m = M;
		 multi_thread(_args);
		free(_args);
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

	delete_matrix(C, N, P);
	delete_matrix(A, P, M);
	delete_matrix(D, N, M);

	return 0;
}
