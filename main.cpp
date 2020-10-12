#include <boost/gil.hpp>
#include <boost/gil/extension/dynamic_image/image_view_factory.hpp>
#include <boost/gil/extension/io/png.hpp>
#include <boost/gil/extension/io/png/tags.hpp>
#include <boost/gil/extension/numeric/convolve.hpp>
#include <boost/gil/image_processing/numeric.hpp>
#include <boost/gil/image_view_factory.hpp>
#include <boost/gil/io/write_view.hpp>
#include <boost/gil/typedefs.hpp>
#include <fstream>
#include <taskflow/core/taskflow.hpp>
#include <taskflow/taskflow.hpp>

namespace gil = boost::gil;

int main()
{
    tf::Executor executor;
    tf::Taskflow taskflow;
    taskflow.name("Gradient computation");

    gil::gray8_image_t molecule_image;
    gil::gray16s_image_t dx;
    gil::gray16s_image_t dy;
    gil::gray16s_image_t gradient;
    auto [read_image, compute_dx, compute_dy, save_dx, save_dy, compute_gradient, save_gradient] =
        taskflow.emplace(
            [&molecule_image, &dx, &dy, &gradient]()
            {
                gil::read_image("gray-molecule.png", molecule_image, gil::png_tag{});
                dx = gil::gray16s_image_t(molecule_image.dimensions());
                dy = gil::gray16s_image_t(molecule_image.dimensions());
                gradient = gil::gray16s_image_t(molecule_image.dimensions());
            },
            [&molecule_image, &dx]()
            {
                auto sobel_x = gil::generate_dx_sobel();
                auto input =
                    gil::color_converted_view<gil::gray16s_pixel_t>(gil::view(molecule_image));
                auto dest = gil::view(dx);

                gil::detail::convolve_2d(input, sobel_x, dest);
            },
            [&molecule_image, &dy]()
            {
                auto sobel_y = gil::generate_dy_sobel();
                auto input =
                    gil::color_converted_view<gil::gray16s_pixel_t>(gil::view(molecule_image));
                auto dest = gil::view(dy);

                gil::detail::convolve_2d(input, sobel_y, dest);
            },
            [&dx]()
            {
                auto dx16_view = gil::view(dx);
                auto dx_view = gil::color_converted_view<gil::gray8_pixel_t>(dx16_view);
                gil::write_view("gray-molecule-dx.png", dx_view, gil::png_tag{});
            },
            [&dy]()
            {
                auto dy16_view = gil::view(dy);
                auto dy_view = gil::color_converted_view<gil::gray8_pixel_t>(dy16_view);
                gil::write_view("gray-molecule-dy.png", dy_view, gil::png_tag{});
            },
            [&dx, &dy, &gradient]()
            {
                auto dx_view = gil::view(dx);
                auto dy_view = gil::view(dy);
                auto gradient_view = gil::view(gradient);
                using pixel_type = gil::gray16s_pixel_t;
                gil::transform_pixels(dx_view, dy_view, gradient_view,
                                      [](const pixel_type& lhs, const pixel_type& rhs)
                                      {
                                          return pixel_type(
                                              std::sqrt(lhs[0] * lhs[0] + rhs[0] * rhs[0]));
                                      });
            },
            [&gradient]()
            {
                auto gradient_view =
                    gil::color_converted_view<gil::gray8_pixel_t>(gil::view(gradient));
                gil::write_view("gray-molecule-gradient.png", gradient_view, gil::png_tag{});
            });

    read_image.name("read_image");
    compute_dx.name("compute_dx");
    compute_dy.name("compute_dy");
    save_dx.name("save_dx");
    save_dy.name("save_dy");
    compute_gradient.name("compute_gradient");
    save_gradient.name("save_gradient");

    read_image.precede(compute_dx, compute_dy);
    compute_dx.precede(save_dx);
    compute_dy.precede(save_dy);
    compute_gradient.succeed(compute_dx, compute_dy);
    save_gradient.succeed(compute_gradient);

    executor.run(taskflow).wait();
    std::ofstream graphviz_state("taskflow-graph.dot");
    taskflow.dump(graphviz_state);
}