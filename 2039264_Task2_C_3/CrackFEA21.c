#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>

#include "config.h"
#include "time.h"
#include <math.h>

/******************************************************************************

	Compile file "cc ./CrackFEA21.c -lm -lcrypt -lpthread -o CrackFEA21"

  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 3 uppercase
  letters and a 2 digit integer.

 ******************************************************************************/

int countFEA21 = 0;     // A counter used to track the number of combinations explored so far


void substrFEA21(char *dest, char *src, int start, int length){
	memcpy(dest, src + start, length);
	*(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above.
 */

void crackFEA21(char *salt_and_encrypted){
	int i, j, k, l;     // Loop counters
	char salt[7];    // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
	char plain[7];   // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
	char *enc;       // Pointer to the encrypted password

	substrFEA21(salt, salt_and_encrypted, 0, 6);

	for(i='A'; i<='Z'; i++){
		for(j='A'; j<='Z'; j++){
			for(k='A'; k<='Z'; k++){
				for(l=0; l<=99; l++){
					sprintf(plain, "%c%c%c%02d", i, j, k, l);
					enc = (char *) crypt(plain, salt);
					countFEA21++;
					if(strcmp(salt_and_encrypted, enc) == 0){
						printf("#%-8d%s %s\n", countFEA21, plain, enc);
						// return;	//uncomment this line if you want to speed-up the running time, program will find you the cracked password only without exploring all possibilites
					}
				}
			}
		}
	}
}

int main(int argc, char *argv[]){
	//$6$AS$N50XR7kfyzi2bZ8V938Mk5GTStGvV2JD6wNhdk0soRkQKeYLRPWirPffIDa8QyvqwITGyeBGh4Wz7LPqkNpvH1 FEA21
	
	printf("Single Threaded - Password Cracking Of 3 Upper Case Letters And 2 Integer Numbers\n");
	printf("Number Of Loops : %d \n", CRYPT_TEST_COUNT);

	struct timespec start, finish;
	double test_time[CRYPT_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < CRYPT_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		crackFEA21("$6$AS$N50XR7kfyzi2bZ8V938Mk5GTStGvV2JD6wNhdk0soRkQKeYLRPWirPffIDa8QyvqwITGyeBGh4Wz7LPqkNpvH1");
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

