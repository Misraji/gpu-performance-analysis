#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
__global__ void Scale(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] *= 1.0001f;
  }
}
int main(int argc, char** argv) {
  int mode = argc > 1 ? std::atoi(argv[1]) : 0;
  constexpr int kN = 1 << 24;
  const size_t bytes = kN * sizeof(float);
  float* h = nullptr;
  if (mode == 0) {
    h = new float[kN];
  } else {
    cudaMallocHost(&h, bytes);  // Pageable host memory.
    // Pinned host memory.
  }
  float* d = nullptr;
  cudaMalloc(&d, bytes);
  cudaStream_t s;
  cudaStreamCreate(&s);
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  if (mode < 2) {
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    Scale<<<grid, block>>>(d, kN);
    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
    Scale<<<grid, block, 0, s>>>(d, kN);
    cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
  }
  cudaStreamDestroy(s);
  cudaFree(d);
  if (mode == 0) {
    delete[] h;
  } else {
    cudaFreeHost(h);
  }
}