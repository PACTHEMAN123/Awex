#include <shardstream/primitives.cuh>
#include <cassert>
#include <cstdio>
#include <vector>

using namespace shardstream::transport;
#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); std::abort(); } } while (0)

__global__ void materialize(Fragment f, unsigned char* out) {
  quantizeToFifo(f, out, 0, f.nbytes, threadIdx.x, blockDim.x);
}

// Deliberately split record headers and values at unaligned wire boundaries.
// This exercises the actual producer branch used by lowered transport works.
__global__ void copySplit(KernelArgs args, unsigned char* out, std::uint64_t bytes) {
  constexpr std::uint64_t step = 8197;
  const std::uint64_t offset = blockIdx.x * step;
  const std::uint64_t length = bytes - offset < step ? bytes - offset : step;
  Work work{}; work.fragment_count = 1;
  copyFragmentsToContiguous(args, work, out + offset, offset, length, threadIdx.x, blockDim.x);
}

int main() {
  int cases = 0;
  for (int br : {64, 128}) for (int bc : {64, 128}) for (int padding : {0, 128}) {
    const int rows = br * 3, cols = bc * 2, stride = cols + padding;
    const std::size_t bytes = 6 * (br * bc + 16);
    std::vector<__nv_bfloat16> input(rows * stride);
    __nv_bfloat16* device_input;
    unsigned char *staging, *fused, *staged;
    Fragment* descriptor;
    CHECK(cudaMalloc(&device_input, input.size() * sizeof(input[0])));
    CHECK(cudaMalloc(&staging, bytes)); CHECK(cudaMalloc(&fused, bytes)); CHECK(cudaMalloc(&staged, bytes));
    CHECK(cudaMalloc(&descriptor, sizeof(Fragment)));
    Fragment f{};
    f.tensor_ptr = reinterpret_cast<std::uintptr_t>(device_input);
    f.nbytes = bytes; f.tensor_row_bytes = cols * 2; f.tensor_row_stride = stride * 2;
    f.block_rows = br; f.block_cols = bc;
    KernelArgs args{}; args.fragments = descriptor;
    for (int version = 0; version < 3; ++version) {
      for (int r = 0; r < rows; ++r) for (int c = 0; c < stride; ++c)
        input[r * stride + c] = __float2bfloat16(version == 0 ? 0.f : ((r * 17 + c * 3) % 1021 - 510) * (version / 32.f));
      CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice));
      materialize<<<1, 256>>>(f, staging);
      f.staged_ptr = 0;
      CHECK(cudaMemcpy(descriptor, &f, sizeof(f), cudaMemcpyHostToDevice));
      copySplit<<<(bytes + 8196) / 8197, 256>>>(args, fused, bytes);
      f.staged_ptr = reinterpret_cast<std::uintptr_t>(staging);
      CHECK(cudaMemcpy(descriptor, &f, sizeof(f), cudaMemcpyHostToDevice));
      copySplit<<<(bytes + 8196) / 8197, 256>>>(args, staged, bytes);
      CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
      std::vector<unsigned char> a(bytes), b(bytes);
      CHECK(cudaMemcpy(a.data(), fused, bytes, cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(b.data(), staged, bytes, cudaMemcpyDeviceToHost));
      assert(a == b);
      ++cases;
    }
    CHECK(cudaFree(descriptor)); CHECK(cudaFree(device_input));
    CHECK(cudaFree(staging)); CHECK(cudaFree(fused)); CHECK(cudaFree(staged));
  }
  std::printf("{\"passed\":true,\"exact_wire_cases\":%d}\n", cases);
}
