#include "reduce.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace reduction = cuda_learn::reduce;

void check_cuda(cudaError_t status, const char* expression, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " at line " +
                             std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)

struct DeviceBuffer {
  float* data = nullptr;
  explicit DeviceBuffer(std::size_t count) {
    if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(float)));
  }
  ~DeviceBuffer() { if (data) cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
  cudaEvent_t event = nullptr;
  Event() { CUDA_CHECK(cudaEventCreate(&event)); }
  ~Event() { cudaEventDestroy(event); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
};

struct Stream {
  cudaStream_t stream = nullptr;
  Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~Stream() { cudaStreamDestroy(stream); }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
};

double reference_sum(const std::vector<float>& input) {
  double sum = 0.0;
  for (float value : input) sum += static_cast<double>(value);
  return sum;
}

float read_result(const DeviceBuffer& output, cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float result = 0.0f;
  CUDA_CHECK(cudaMemcpy(&result, output.data, sizeof(result), cudaMemcpyDeviceToHost));
  return result;
}

// 常规有限值数据的误差阈值；不承诺任意动态范围/严重抵消下的精度。
void verify(float actual, double expected, double tolerance, const std::string& label) {
  if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected) > tolerance) {
    throw std::runtime_error(label + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual) +
                             ", tolerance " + std::to_string(tolerance));
  }
}

std::vector<float> random_input(std::size_t n) {
  std::mt19937 generator(42);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  std::vector<float> input(n);
  for (float& value : input) value = distribution(generator);
  return input;
}

void test_case(const std::vector<float>& input, bool exact, const std::string& label,
               cudaStream_t stream) {
  const std::size_t n = input.size();
  DeviceBuffer device_input(n), output(1), workspace(reduction::workspace_elements(n));
  if (n) CUDA_CHECK(cudaMemcpyAsync(device_input.data, input.data(), n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
  const double expected = reference_sum(input);
  const double tolerance = exact ? 0.0 : 1e-4 + 5e-5 * std::abs(expected);
  for (auto method : {reduction::Method::Shared, reduction::Method::Shuffle}) {
    // 非零哨兵可以发现空输入未写回，以及复用输出时的意外累加。
    const float sentinel = 123.0f;
    CUDA_CHECK(cudaMemcpyAsync(output.data, &sentinel, sizeof(sentinel), cudaMemcpyHostToDevice, stream));
    for (int repeat = 0; repeat < 2; ++repeat) {
      CUDA_CHECK(reduction::reduce_sum(device_input.data, output.data, n,
                                       workspace.data, method, stream));
      verify(read_result(output, stream), expected, tolerance, label);
    }
  }
}

void run_tests() {
  Stream stream;
  // 覆盖 warp、block、tile、两轮/三轮归约，以及 grid-stride 的边界。
  const std::size_t sizes[] = {
      0, 1, 2, 3, 31, 32, 33, 255, 256, 257, 1023, 1024, 1025,
      4097, (1u << 20) - 1, 1u << 20, (1u << 20) + 1,
      (1u << 22) + 17};
  std::size_t cases = 0;
  for (std::size_t n : sizes) {
    // 二进制可精确表示的小数：总和低于 FP32 精确整数范围，要求零误差。
    // 这组数据会直接抓到原 v3 将 float 截断成 int 的错误。
    std::vector<float> fractional(n);
    for (std::size_t i = 0; i < n; ++i) {
      fractional[i] = (static_cast<int>(i % 17) - 8) * 0.125f;
    }
    test_case(fractional, true, "fractional n=" + std::to_string(n), stream.stream);
    test_case(std::vector<float>(n, 1.0f), true, "ones n=" + std::to_string(n), stream.stream);
    ++cases;
  }
  test_case(random_input((1u << 20) + 3), false, "random mixed-sign", stream.stream);
  test_case(std::vector<float>(1025, 0.0f), true, "zeros", stream.stream);
  test_case(std::vector<float>(4099, -0.25f), true, "negative", stream.stream);
  // 只在最后一个元素放非零值，检查尾部有无遗漏。
  std::vector<float> tail((1u << 22) + 17, 0.0f);
  tail.back() = 0.375f;
  test_case(tail, true, "tail impulse", stream.stream);

  DeviceBuffer output(1), input(1025);
  if (reduction::reduce_sum(nullptr, output.data, 1, nullptr) != cudaErrorInvalidValue ||
      reduction::reduce_sum(input.data, output.data, 1025, nullptr) != cudaErrorInvalidValue ||
      reduction::reduce_sum(nullptr, nullptr, 0, nullptr) != cudaErrorInvalidValue) {
    throw std::runtime_error("Invalid argument checks failed");
  }
  std::cout << "PASS: " << cases << " boundary sizes, fractional/ones/random/zeros/negative/tail, "
            << "both methods, repeated calls, non-default stream, invalid arguments\n";
}

float benchmark(reduction::Method method, const DeviceBuffer& input, DeviceBuffer& output,
                 DeviceBuffer& workspace, std::size_t n, int iterations,
                 cudaStream_t stream) {
  for (int warmup = 0; warmup < 10; ++warmup) {
    CUDA_CHECK(reduction::reduce_sum(input.data, output.data, n, workspace.data, method, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  Event start, stop;
  CUDA_CHECK(cudaEventRecord(start.event, stream));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(reduction::reduce_sum(input.data, output.data, n, workspace.data, method, stream));
  }
  CUDA_CHECK(cudaEventRecord(stop.event, stream));
  CUDA_CHECK(cudaEventSynchronize(stop.event));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.event, stop.event));
  return elapsed_ms / iterations;
}

std::size_t parse_size(const char* text, std::size_t limit) {
  const std::string value(text);
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument("Expected a non-negative integer: " + value);
  }
  const unsigned long long parsed = std::stoull(value);
  if (parsed > limit) throw std::out_of_range("Argument is too large: " + value);
  return static_cast<std::size_t>(parsed);
}

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "Usage: reduce [N=16777216] [iterations=100]\n"
                << "       reduce --test\n";
      return EXIT_SUCCESS;
    }
    if (argc == 2 && std::string(argv[1]) == "--test") {
      run_tests();
      return EXIT_SUCCESS;
    }
    if (argc > 3) throw std::invalid_argument("Usage: reduce [N] [iterations] or reduce --test");
    const std::size_t n = argc > 1
        ? parse_size(argv[1], std::numeric_limits<std::size_t>::max() / sizeof(float))
        : (1u << 24);
    const int iterations = argc > 2
        ? static_cast<int>(parse_size(argv[2], std::numeric_limits<int>::max())) : 100;
    if (iterations == 0) throw std::invalid_argument("iterations must be positive");

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::cout << "GPU: " << properties.name << ", N=" << n
              << ", block=" << reduction::kBlockSize
              << ", first-pass blocks=" << reduction::block_count(n)
              << ", iterations=" << iterations << '\n';

    const std::vector<float> host_input = random_input(n);
    const double expected = reference_sum(host_input);
    DeviceBuffer input(n), output(1), workspace(reduction::workspace_elements(n));
    Stream stream;
    if (n) CUDA_CHECK(cudaMemcpyAsync(input.data, host_input.data(), n * sizeof(float),
                                     cudaMemcpyHostToDevice, stream.stream));
    float timings[2] = {};
    int index = 0;
    for (auto method : {reduction::Method::Shared, reduction::Method::Shuffle}) {
      const char* name = method == reduction::Method::Shared ? "shared" : "shuffle";
      const float ms = benchmark(method, input, output, workspace, n, iterations, stream.stream);
      const float result = read_result(output, stream.stream);
      verify(result, expected, 1e-4 + 5e-5 * std::abs(expected), name);
      timings[index++] = ms;
      const double gb_per_s = ms > 0.0f ? double(n) * sizeof(float) / (ms * 1e6) : 0.0;
      std::cout << std::setprecision(9) << name << ": sum=" << result
                << ", abs_error=" << std::abs(double(result) - expected)
                << ", " << ms << " ms, input_effective_bandwidth=" << gb_per_s << " GB/s\n";
    }
    if (timings[1] > 0.0f) std::cout << "shared/shuffle time ratio: " << timings[0] / timings[1] << "x\n";
    std::cout << "CPU double reference: " << std::setprecision(17) << expected << '\n';
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
