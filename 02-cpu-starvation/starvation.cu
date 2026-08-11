#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <thread>
__global__ void Tiny(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x[i] * 1.000001f + 1.0f;
  }
}

int main(int argc, char** argv) {
  const bool slow_host = argc < 2 || std::atoi(argv[1]) == 0;
  constexpr int kN = 1 << 20;
  constexpr int kIterations = 200;
  float* d = nullptr;
  cudaMalloc(&d, kN * sizeof(float));
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  for (int iter = 0; iter < kIterations; ++iter) {
    Tiny<<<grid, block>>>(d, kN);
    if (slow_host) {
      std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
  }
  cudaDeviceSynchronize();
  cudaFree(d);
}