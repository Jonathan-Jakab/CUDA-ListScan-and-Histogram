#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define NUM_BINS 4096
#define PRIVATE_NUM_BINS 4096
#define BLOCK_SIZE 512 
#define MAX_BIN_VALUE 127
#define BIN_COUNTER 32

#define CUDA_CHECK(ans)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
            file, line);
    if (abort)
      exit(code);
  }
}

__global__ void histogram(unsigned int *input, unsigned int *bins,
						  unsigned int num_elements,
						  unsigned int num_bins) {
	//@@ Write the kernel that computes the histogram
	//@@ Make sure to use the privitization technique
	//(hint: since NUM_BINS=4096 is larger than maximum allowed number of threads per block, 
	//be aware that threads would need to initialize more than one shared memory bin 
	//and update more than one global memory bin)

	__shared__ unsigned int private_histogram[PRIVATE_NUM_BINS];

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int stride = blockDim.x * gridDim.x;


	if (threadIdx.x < num_bins) {
			private_histogram[threadIdx.x] = 0.0;
	}
	__syncthreads();
	
	while (i < num_elements) {
		atomicAdd(&(bins[input[i]]), 1);
		i += stride;
	}

	__syncthreads();

	if (threadIdx.x < num_bins) {
		atomicAdd(&(bins[threadIdx.x]), private_histogram[threadIdx.x]);
	}


	//if (num_bins > BLOCK_SIZE) {
	//	for (int j = threadIdx.x; j < num_bins; j += BLOCK_SIZE) {
	//		if (j < num_bins) {
	//			private_histogram[j] = 0.0;
	//		}
	//	}
	//} else {
	//	if (threadIdx.x < num_bins) {
	//		private_histogram[threadIdx.x] = 0.0;
	//	}
	//}

	//__syncthreads();

	//if (i < num_elements) {
	//	atomicAdd(&(private_histogram[input[i]]), 1);
	//	//atomicAdd(&(bins[input[i]]), 1);
	//}
	//
	//__syncthreads();

	//if (num_bins > BLOCK_SIZE) {
	//	for (int j = threadIdx.x; j < num_bins; j += BLOCK_SIZE) {
	//		if (j < num_bins) {
	//			atomicAdd(&(bins[j]), private_histogram[j]);
	//		}
	//	}
	//} else {
	//	if (threadIdx.x < num_bins) {
	//		atomicAdd(&(bins[threadIdx.x]), private_histogram[threadIdx.x]);
	//	}
	//}
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ Write the kernel that applies saturtion to counters (i.e., if the bin value is more than 127, make it equal to 127)
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < num_bins) {
		
		if (bins[i] > MAX_BIN_VALUE) {
			bins[i] = MAX_BIN_VALUE;
		}
	}
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0),
                                       &inputLength, "Integer");
  hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The number of bins is ", NUM_BINS);

  wbTime_start(GPU, "Allocating device memory");
  //@@ Allocate device memory here
  cudaMalloc((void**) &deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc((void**) &deviceBins, NUM_BINS * sizeof(unsigned int));

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Allocating device memory");

  wbTime_start(GPU, "Copying input host memory to device");
  //@@ Copy input host memory to device
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(GPU, "Copying input host memory to device");

  wbTime_start(GPU, "Clearing the bins on device");
  //@@ zero out the deviceBins using cudaMemset()
  cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

  wbTime_stop(GPU, "Clearing the bins on device");

  //@@ Initialize the grid and block dimensions here
  //dim3 dimGrid = (ceil(inputLength / (float)BLOCK_SIZE));										//Both of the dim3 versions work but why does BLOCK_SIZE need to be cast to a float?
  //dim3 dimBlock = (BLOCK_SIZE);

  dim3 dimGridHist = (ceil(inputLength / (float)BLOCK_SIZE));
  dim3 dimBlockHist = (BLOCK_SIZE);

  dim3 dimGridSat = (ceil(NUM_BINS / (float)BLOCK_SIZE));
  dim3 dimBlockSat = (BLOCK_SIZE);

  wbLog(TRACE, "Launching kernel");
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Invoke kernels: first call histogram kernel and then call saturate kernel  
  //histogram << <dimGrid, dimBlock >> > (deviceInput, deviceBins, inputLength, NUM_BINS);		//Both of the kernal seemed to work on the examples. See dim3 for reason why there is 2.
  //saturate << <dimGrid, dimBlock >> > (deviceBins, NUM_BINS);

  histogram<<<dimGridHist, dimBlockHist>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  saturate<<<dimGridSat, dimBlockSat>>>(deviceBins, NUM_BINS);

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output device memory to host");
  //@@ Copy output device memory to host
  //cudaMemcpy(hostInput, deviceInput, inputLength * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  CUDA_CHECK(cudaDeviceSynchronize());
  wbTime_stop(Copy, "Copying output device memory to host");

  wbTime_start(GPU, "Freeing device memory");
  //@@ Free the device memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  wbTime_stop(GPU, "Freeing device memory");

  wbSolution(args, hostBins, NUM_BINS);

  free(hostBins);
  free(hostInput);
  return 0;
}
