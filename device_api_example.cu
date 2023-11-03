#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <omp.h>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#define CUDA_CALL(x) do { if((x) != cudaSuccess) { printf("Error at %s:%d\n", __FILE__,__LINE__); return EXIT_FAILURE; }} while(0)

__global__ void setup_kernel(curandState *state)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  /* Each thread gets device index seed, a different sequence number, no offset */
  curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_uniform_kernel(curandState *state, double_t *result, int iteration)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState localState = state[id];
  for (int i = 0; i < iteration; ++i) {
    result[id] = curand_uniform_double(&localState);
  }
  //printf(id);
}

int main(int argc, char *argv[1])
{
  const unsigned int threadPerBlock = 1024;
  const unsigned int blockCount = 1024;
  const unsigned int totalThreads = threadPerBlock * blockCount;
  int iteration = 1;
  curandState *devStates;
  double_t *devResults;
  cudaMalloc((void **)&devResults, totalThreads * sizeof(double_t));
  cudaMemset(devResults, 0, totalThreads * sizeof(double_t));
  cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
  setup_kernel<<<blockCount, threadPerBlock>>>(devStates);
  generate_uniform_kernel<<<blockCount, threadPerBlock>>>(devStates, devResults, iteration);
  // how to cast a pointer into a thrust device pointer
  thrust::device_ptr<double_t> device_ptr = thrust::device_pointer_cast(devResults);
  thrust::device_vector<double_t> dev_vec(device_ptr, device_ptr + totalThreads);
  cudaFree(devResults);
  thrust::host_vector<double_t> host_vec = dev_vec;
  for (int i = 0; i < host_vec.size(); ++i) {
    printf("%1.15f ", host_vec[i]);
  }
}
