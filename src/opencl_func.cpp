#include <CL/cl.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

std::string loadKernel(const std::string& filename) {
    std::ifstream file(filename);
    return std::string(std::istreambuf_iterator<char>(file), {});
}

void runRansacKernel(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                     float A, float B, float C, float D,
                     float distanceThreshold)
{
    // 1. Flatten cloud into float4
    std::vector<cl_float4> h_points(cloud->points.size());
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        h_points[i].x = cloud->points[i].x;
        h_points[i].y = cloud->points[i].y;
        h_points[i].z = cloud->points[i].z;
        h_points[i].w = cloud->points[i].intensity;
    }

    // 2. Setup OpenCL
    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = platform.getDevices(CL_DEVICE_TYPE_GPU)[0];
    cl::Context context(device);
    cl::Program program(context, loadKernel("kernel.cl"));
    program.build({device});
    cl::CommandQueue queue(context, device);

    // 3. Create Buffers
    cl::Buffer d_points(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(cl_float4) * h_points.size(), h_points.data());
    cl::Buffer d_inliers(context, CL_MEM_WRITE_ONLY,
                         sizeof(cl_int) * h_points.size());

    // 4. Set up kernel
    cl::Kernel kernel(program, "compute_inliers");
    kernel.setArg(0, d_points);
    kernel.setArg(1, (int)h_points.size());
    kernel.setArg(2, A);
    kernel.setArg(3, B);
    kernel.setArg(4, C);
    kernel.setArg(5, D);
    kernel.setArg(6, distanceThreshold);
    kernel.setArg(7, d_inliers);

    // 5. Run kernel
    cl::NDRange global(h_points.size());
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    // 6. Read back inlier flags
    std::vector<int> inliers(h_points.size());
    queue.enqueueReadBuffer(d_inliers, CL_TRUE, 0, sizeof(int) * inliers.size(), inliers.data());

    // 7. Print inliers
    std::cout << "Inliers:\n";
    for (size_t i = 0; i < inliers.size(); ++i) {
        if (inliers[i]) {
            std::cout << "Point " << i << " is an inlier\n";
        }
    }
}
