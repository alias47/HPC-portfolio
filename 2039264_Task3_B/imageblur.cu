#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"

#define CRYPT_TEST_COUNT 10

/**************************
compile with

	 cc nvcc imageblur.cu lodepng.cu -o imageblur 	 
***************************/	



__global__ void blur_gaussian_GPUFunction(unsigned char * gpu_imageOuput, unsigned char * gpu_imageInput,
										  unsigned int height, unsigned int width, unsigned int filter_size){

	int r = 0;
	int g = 0;
	int b = 0;
	int t = 0;
	int sumR = 0;
	int sumG = 0;
	int sumB = 0;
	int count = 0;

	unsigned int half_filter_size = filter_size / 2;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int pixel = idx*4;

	t = gpu_imageInput[3 + pixel];

	for(int filter_y = blockIdx.x - half_filter_size; filter_y < blockIdx.x + half_filter_size; filter_y++)
	{
		/* Check boundary condition along y - direction ( If we have gone out of bounds) */
		if(filter_y >= 0 && filter_y < height)
		{
			for(int filter_x = threadIdx.x - half_filter_size; filter_x < threadIdx.x + half_filter_size; filter_x++)
			{
				/* Check boundary condition along x - direction ( If we have gone out of bounds) */
				if(filter_x >= 0 && filter_x < width)
				{
					r = gpu_imageInput[4 * blockDim.x * filter_y + 4 * filter_x + 0];
					g = gpu_imageInput[4 * blockDim.x * filter_y + 4 * filter_x + 1];
					b = gpu_imageInput[4 * blockDim.x * filter_y + 4 * filter_x + 2];
					sumR += r; /* Compute sum of R values to calculate average */
					sumG += g; /* Compute sum of G values to calculate average */
					sumB += b; /* Compute sum of B values to calculate average */
					++count; /* Compute no. of values to calculate average */
				}
			}
		}
	}

	gpu_imageOuput[pixel] = (sumR / count);
	gpu_imageOuput[1 + pixel] = (sumG / count);
	gpu_imageOuput[2 + pixel] = (sumB / count);
	gpu_imageOuput[3 + pixel] = t;
}

void blur_gaussian(){

	unsigned int error;
	unsigned int encError;
	unsigned char* image;
	unsigned int width;
	unsigned int height;
	unsigned int filter_size = 7;

	const char* filename = "man.png";
	const char* newFileName = "generatedman.png";

	error = lodepng_decode32_file(&image, &width, &height, filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	const int ARRAY_SIZE = width*height*4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char host_imageInput[ARRAY_SIZE * 4];
	unsigned char host_imageOutput[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_imageInput[i] = image[i];
	}

	// GPU memory pointers
	unsigned char * d_in;
	unsigned char * d_out;
	unsigned int* gpu_height;
	unsigned int* gpu_width;
	unsigned int* gpu_filter_size;

	// allocatation of GPU memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);
	cudaMalloc((void**) &gpu_height, sizeof(int));
	cudaMalloc((void**) &gpu_width, sizeof(int));
	cudaMalloc((void**) &gpu_filter_size, sizeof(int));

	cudaMemcpy(d_in, host_imageInput, ARRAY_BYTES, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_height, &height, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_width, &width, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_filter_size, &filter_size, sizeof(int), cudaMemcpyHostToDevice);

	// Launch of Kernal
	blur_gaussian_GPUFunction<<<height, width>>>(d_out, d_in, height, width, filter_size);

	
	cudaMemcpy(host_imageOutput, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	encError = lodepng_encode32_file(newFileName, host_imageOutput, width, height);
	if(encError){
		printf("Error %u: %s\n", error, lodepng_error_text(encError));
	}

	//free(image);
	//free(host_imageInput);
	cudaFree(d_in);
	cudaFree(d_out);
}

int main(int argc, char *argv[]){

	printf("Blur Image\n");
	printf("No Of Loops : %d \n", CRYPT_TEST_COUNT);

	struct timespec start, finish;
	double test_time[CRYPT_TEST_COUNT];
	double total_time = 0, total_square_time = 0, average_time = 0, variance_time = 0;

	for(int i = 0; i < CRYPT_TEST_COUNT; i++)
	{
		clock_gettime(CLOCK_REALTIME, &start);
		blur_gaussian();
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
		printf("%10s %10s \n", "Loops", "Time(seconds)");
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
