#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>

#include "config.h"
#include "time.h"
#include <math.h>
#include <pthread.h>

/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc ./CrackGP23_multi_threaded.c -lm -lcrypt -lpthread -o CrackGP23

 ******************************************************************************/

int countGP23_mt = 0;     // A counter used to track the number of combinations explored so far

typedef struct CrackPasswordArgs{
	char* salt_and_encrypted;
	char start;
	char stop;
} CrackPasswordArgs_t;

/**
 Required by lack of standard function in C.   
 */

void substrGP23_multi_threaded(char *dest, char *src, int start, int length){
	memcpy(dest, src + start, length);
	*(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the printfs.
 */

static void* crackAZ99_thread_function(void* args){
	CrackPasswordArgs_t* pwd_args = (CrackPasswordArgs_t*)args;
	int x, y, z;     // Loop counters
	char salt[7];    // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
	char plain[7];   // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
	char *enc;       // Pointer to the encrypted password

	substrGP23_multi_threaded(salt, pwd_args->salt_and_encrypted, 0, 6);

	for(x=pwd_args->start; x<=pwd_args->stop; x++){
		for(y='A'; y<='Z'; y++){
			for(z=0; z<=99; z++){
				sprintf(plain, "%c%c%02d", x, y, z);
				enc = (char *) crypt(plain, salt);
				countGP23_mt++;
				if(strcmp(pwd_args->salt_and_encrypted, enc) == 0){
					printf("#%-8d%s %s\n", countGP23_mt, plain, enc);
					// return;	//uncomment this line if you want to speed-up the running time, program will find you the cracked password only without exploring all possibilites
				}
			}
		}
	}

	free(pwd_args->salt_and_encrypted);
	pwd_args->salt_and_encrypted = NULL;
	free(pwd_args);

	return NULL;
}

void crackGP23_multi_threaded(char *salt_and_encrypted){
	pthread_t threads[PWD_CRACK_NO_OF_THREADS];

	char start[PWD_CRACK_NO_OF_THREADS] = {'A', 'N'};
	char stop[PWD_CRACK_NO_OF_THREADS] = {'M', 'Z'};

	for(int i = 0; i < PWD_CRACK_NO_OF_THREADS; i++)
	{
		CrackPasswordArgs_t* pwd_args = (CrackPasswordArgs_t*)calloc(1, sizeof(CrackPasswordArgs_t));
		pwd_args->salt_and_encrypted = strdup(salt_and_encrypted);
		pwd_args->start = start[i];
		pwd_args->stop = stop[i];
		pthread_create(&threads[i], NULL, crackAZ99_thread_function, pwd_args);
	}

	for(int i = 0; i < PWD_CRACK_NO_OF_THREADS; i++)
	{
		pthread_join(threads[i], NULL);
	}
}

int main(int argc, char *argv[]){
	// $6$AS$iG1WMIYWkvE2a0kU7W/DJzDCNOrOzFtSPklYNumsVituTMIOgXGwQyYsQbjEp0pcwdPRavqLV8QxrTgbjXMx/1
	printf("Multi Threaded - Password Cracking of 2 Upper Case Letters And 2 Integer Numbers\n");
	printf("No Of Loops : %d \n", CRYPT_TEST_COUNT);

	struct timespec start, finish;
	double test_time[CRYPT_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < CRYPT_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		crackGP23_multi_threaded("$6$AS$iG1WMIYWkvE2a0kU7W/DJzDCNOrOzFtSPklYNumsVituTMIOgXGwQyYsQbjEp0pcwdPRavqLV8QxrTgbjXMx/1");
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
		printf("%10s %10s \n", "Loop", "Time(seconds)");
		printf("%5d %15.3f \n", (i + 1), time_elapsed);
		fflush(stdout);
	}

	average_time = total_time / CRYPT_TEST_COUNT;

	for(int i = 0; i < CRYPT_TEST_COUNT; i++)
	{
		total_square_time += pow(test_time[i] - average_time, 2);
	}

	variance_time = sqrt(total_square_time / CRYPT_TEST_COUNT);
	printf("\n Average Time %5.3f +/- %5.3f seconds \n", average_time, variance_time);
	return 0;
}

