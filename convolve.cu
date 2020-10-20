#include <cmath>

#include <boost/gil.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <flash/core.hpp>

namespace shino {
std::size_t ceil(std::size_t nominator, std::size_t denominator) {
  return static_cast<std::size_t>(std::ceil(static_cast<double>(nominator) /
                                            static_cast<double>(denominator)));
}

__device__ std::size_t flat_index(std::size_t x, std::size_t y,
                                  std::size_t width) {
  return width * y + x;
}

__global__ void convolve(float *input, float *kernel, float *output,
                         std::size_t width, std::size_t height,
                         std::size_t kernel_width, std::size_t kernel_height) {
  std::size_t origin_x = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t origin_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (origin_y >= height || origin_x >= width) {
    return;
  }

  auto half_k_w = kernel_width / 2;
  auto half_k_h = kernel_height / 2;
  float sum = 0;

  for (std::size_t y = 0; y < kernel_height; ++y) {
    for (std::size_t x = 0; x < kernel_width; ++x) {
      sum += input[flat_index(origin_x + x - half_k_w, origin_y + y - half_k_h,
                              width)] *
             kernel[flat_index(x, y, kernel_width)];
    }
  }

  output[flat_index(origin_x, origin_y, width)] = sum;
}
} // namespace shino

namespace gil = boost::gil;

int main() {
  gil::gray8_image_t molecule_image;
  gil::read_image("gray-molecule.png", molecule_image, gil::png_tag{});
  blaze::DynamicMatrix<float> input(molecule_image.height(),
                                    molecule_image.width());
  flash::to_matrix(gil::view(molecule_image), input);
  /*  gil::gray8_image_t molecule_image;
    gil::read_image("gray-molecule.png", molecule_image, gil::png_tag{});
    gil::gray32f_image_t input_image(molecule_image.dimensions());
    auto input = gil::view(input_image);
    gil::copy_pixels(gil::view(molecule_image), input);

    float *dev_input = nullptr;
    cudaMalloc(&dev_input, sizeof(float) * input.size());
    cudaMemcpy(dev_input, &(input(0, 0)[0]), sizeof(float) * input.size(),
               cudaMemcpyHostToDevice);

    float kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float *dev_kernel = nullptr;
    cudaMalloc(&dev_kernel, sizeof(kernel));
    cudaMemcpy(dev_kernel, kernel, sizeof(float) * input.size(),
               cudaMemcpyHostToDevice);

    float *dev_output = nullptr;
    cudaMalloc(&dev_output, sizeof(float) * input.size());

    dim3 threadsPerBlock(32, 32);
    dim3 blocks(30, 40);
    shino::convolve<<<blocks, threadsPerBlock>>>(
        dev_input, dev_kernel, dev_output, input.width(), input.height(), 3, 3);

    gil::gray32f_image_t output_image(input.dimensions());
    auto output = gil::view(output_image);
    cudaMemcpy(&(output(0, 0)[0]), dev_output, sizeof(float) * input.size(),
               cudaMemcpyDeviceToHost);
    gil::write_view("convolved.png",
                    gil::color_converted_view<gil::gray8_pixel_t>(output),
                    gil::png_tag{});
  */
}
