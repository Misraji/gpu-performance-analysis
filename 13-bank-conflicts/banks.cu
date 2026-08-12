#include <cuda_runtime.h>

#include <cstdlib>
__global__ void SharedAccess(float* out, const float* in, bool padded) {
  __shared__ float tile_pad[32][33];
  __shared__ float tile_plain[32][32];
  int x = threadIdx.x;
  int y = threadIdx.y;
  int idx = y * 32 + x;
  if (padded) {
    tile_pad[y][x] = in[idx];
    __syncthreads();
    out[idx] = tile_pad[x][y];
  } else {
    tile_plain[y][x] = in[idx];
    __syncthreads();
    out[idx] = tile_plain[x][y];
  }
}
int main(int argc, char** argv) {
  bool padded = argc > 1 && std::atoi(argv[1]);
  float *a = nullptr, *b = nullptr;
  cudaMalloc(&a, 4096);
  cudaMalloc(&b, 4096);
  for (int i = 0; i < 10000; ++i)
    SharedAccess<<<1, dim3(32, 32)>>>(b, a, padded);
  cudaDeviceSynchronize();
  cudaFree(a);
  cudaFree(b);
}