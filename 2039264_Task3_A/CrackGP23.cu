#include <stdio.h>
#include <stdlib.h>

#define RAW_PASSWORD_SIZE  11
#define DECRYPTED_PASSWORD_SIZE (4 + 1)

#define CRYPT_TEST_COUNT 1


// compile with nvcc CrackGP23.cu -o crackGP23
/**************************
compile with

	 cc nvcc nvcc CrackGP23.cu -o crackGP23
	 
***************************/	
 

// the cuda crypt function takes in a raw password of ONLY 2 letters (ONLY LOWERCASE) and 2 numbers.
// Your task is to take this function and create a "__device__ crypt" function which can be used within the "__global__"
// function. That way, you can encrypt all combinations and check this password with rawPassword to
// determine whether a match has been found.

__device__ void cuda_device(unsigned char* rawPassword, unsigned char* newPassword)
{
	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){ //checking all lower case letter limits
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{ //checking number section
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
}

__global__ void cuda_Crypt(unsigned char* raw_password, unsigned char* decrypted_password)
{
	int block_idx = blockIdx.x;
	int thread_idx = threadIdx.x;

	unsigned char current_password[DECRYPTED_PASSWORD_SIZE];
	unsigned char encrypted_password[RAW_PASSWORD_SIZE];

	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
		{
			char is_correct = 0;
			current_password[0] = 97 + block_idx;
			current_password[1] = 97 + thread_idx;
			current_password[2] = 48 + i;
			current_password[3] = 48 + j;
			cuda_device(current_password, encrypted_password);

			for(int k = 0; k < RAW_PASSWORD_SIZE; k++)
			{
				if(encrypted_password[k] == raw_password[k])
				{
					++is_correct;
				}
			}

			if(is_correct == RAW_PASSWORD_SIZE)
			{
				for(int k = 0; k < DECRYPTED_PASSWORD_SIZE; k++)
				{
					decrypted_password[k] = current_password[k];
				}
			}
		}
	}
}

void crypt()
{

	unsigned char rawPassword[RAW_PASSWORD_SIZE] = "y}zous4071";
	unsigned char decryptedPassword[DECRYPTED_PASSWORD_SIZE] = {0};

	const int blocks = 26;
	const int threads = 26;

	// declare GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocate GPU memory
	cudaMalloc((void**) &d_in, RAW_PASSWORD_SIZE);
	cudaMalloc((void**) &d_out, DECRYPTED_PASSWORD_SIZE);

	cudaMemcpy(d_in, rawPassword, RAW_PASSWORD_SIZE, cudaMemcpyHostToDevice);

	// launch the kernel
	cuda_Crypt<<<blocks, threads>>>(d_in, d_out);

	// copy back the result array to the CPU
	cudaMemcpy(decryptedPassword, d_out, DECRYPTED_PASSWORD_SIZE, cudaMemcpyDeviceToHost);

	//free(image);
	//free(host_imageInput);
	cudaFree(d_in);
	cudaFree(d_out);

	decryptedPassword[DECRYPTED_PASSWORD_SIZE - 1] = '\0';
}

int main(int argc, char *argv[]){
// y}zous4071 GP23
	printf("CUDA - Password Cracking of 2 Upper Case Letter And 2 Integer Numbers  \n");
	printf("No Of Loops : %d \n", CRYPT_TEST_COUNT);

	struct timespec start, finish;
	double test_time[CRYPT_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < CRYPT_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		crypt();
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
	printf("\n AVERAGE TIME %5.3f +/- %5.3f seconds \n", average_time, variance_time);
	return 0;
}
