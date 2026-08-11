#include <cuda_runtime.h>

#include <iostream>
__global__ void Copy(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i];
  }
}
int main() {
  constexpr int kN = 1 << 27;
  const size_t bytes = kN * sizeof(float);
  float *a = nullptr, *b = nullptr;
  cudaMalloc(&a, bytes);
  cudaMalloc(&b, bytes);
  dim3 block(256), grid((kN + block.x - 1) / block.x);
  for (int i = 0; i < 5; ++i) Copy<<<grid, block>>>(a, b, kN);
  cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  Copy<<<grid, block>>>(a, b, kN);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  double useful_gb = (2.0 * bytes) / 1e9;
  std::cout << "kernel_ms=" << ms << " useful_GB/s=" << useful_gb / (ms / 1e3)
            << "\n";
  cudaFree(a);
  cudaFree(b);
}