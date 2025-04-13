template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> 
ProcessPointClouds<PointT>::SegmentPlaneGPU(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Memory for storing best inlier indices (two versions, one could be used if you need an additional copy)
    pcl::PointIndices::Ptr bestInliers(new pcl::PointIndices());
    pcl::PointIndices::Ptr bestInliers_check(new pcl::PointIndices());
    
    // Flatten the point cloud into a continuous float vector (x,y,z)
    std::vector<float>* h_points = new std::vector<float>();
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        h_points->push_back(cloud->points[i].x);
        h_points->push_back(cloud->points[i].y);
        h_points->push_back(cloud->points[i].z);
    }
    const int numPoints = cloud->points.size();

    // Setup OpenCL platform, device and context.
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Load and build kernel program
    const char* kernel_filename = "../src/ransac_kernel.cl";
    std::string kernel_source = loadKernel(kernel_filename);
    const char* kernel_source_cstr = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, NULL);
    if(clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        // Retrieve and print build log in case of errors.
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Error building OpenCL program:" << std::endl << build_log.data() << std::endl;
        exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "ransac_plane_segmentation", NULL);

    // Create device buffers.
    size_t pointsBufferSize = numPoints * 3 * sizeof(float);
    cl_mem d_points_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, pointsBufferSize, NULL, NULL);
    size_t planeBufferSize = 5 * sizeof(float);
    cl_mem d_plane_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, planeBufferSize, NULL, NULL);
    size_t inliersBufferSize = numPoints * sizeof(int);
    cl_mem d_inliers_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inliersBufferSize, NULL, NULL);

    // Write the flattened point cloud to the device (it does not change in the loop).
    clEnqueueWriteBuffer(queue, d_points_buffer, CL_TRUE, 0, pointsBufferSize, h_points->data(), 0, NULL, NULL);

    // Variables to track the best inlier count.
    size_t bestInlierCount = 0;

    auto startTime = std::chrono::steady_clock::now();

    // Main RANSAC iterations
    while(maxIterations-- > 0)
    {
        // Randomly sample 3 points from the host flattened cloud.
        int p1_id = (rand() % numPoints) * 3;
        int p2_id = (rand() % numPoints) * 3;
        int p3_id = (rand() % numPoints) * 3;
        float p1_x = h_points->at(p1_id);
        float p1_y = h_points->at(p1_id+1);
        float p1_z = h_points->at(p1_id+2);
        float p2_x = h_points->at(p2_id);
        float p2_y = h_points->at(p2_id+1);
        float p2_z = h_points->at(p2_id+2);
        float p3_x = h_points->at(p3_id);
        float p3_y = h_points->at(p3_id+1);
        float p3_z = h_points->at(p3_id+2);

        // Fit the plane: Ax + By + Cz + D = 0.
        float A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y);
        float B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z);
        float C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x);
        float D = -1 * (A * p1_x + B * p1_y + C * p1_z);
        float normFactor = sqrt(A * A + B * B + C * C);

        // Prepare plane parameters vector: [A, B, C, D, normFactor]
        std::vector<float> planeParams = {A, B, C, D, normFactor};

        // Write plane parameters to device.
        clEnqueueWriteBuffer(queue, d_plane_buffer, CL_TRUE, 0, planeBufferSize, planeParams.data(), 0, NULL, NULL);

        // Set kernel arguments.
        int argIdx = 0;
        clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_points_buffer);
        clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_plane_buffer);
        clSetKernelArg(kernel, argIdx++, sizeof(int), &numPoints);
        clSetKernelArg(kernel, argIdx++, sizeof(float), &distanceThreshold);
        clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_inliers_buffer);

        // Launch kernel with one work-item per point.
        size_t global_size = numPoints;
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clFinish(queue);

        // Read back inliers.
        std::vector<int> inlierFlags(numPoints, 0);
        clEnqueueReadBuffer(queue, d_inliers_buffer, CL_TRUE, 0, inliersBufferSize, inlierFlags.data(), 0, NULL, NULL);

        // Count and store indices for the current iteration.
        size_t inlierCount = 0;
        pcl::PointIndices::Ptr currentInliers(new pcl::PointIndices());
        for (int i = 0; i < numPoints; i++)
        {
            if (inlierFlags[i] == 1)
            {
                currentInliers->indices.push_back(i);
                inlierCount++;
            }
        }

        // Update best inliers if the current set is larger.
        if (inlierCount > bestInlierCount)
        {
            bestInlierCount = inlierCount;
            bestInliers = currentInliers;
            bestInliers_check = currentInliers;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    // Separate cloud using best inliers.
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(bestInliers_check, cloud);

    // Cleanup: release OpenCL resources.
    clReleaseMemObject(d_points_buffer);
    clReleaseMemObject(d_plane_buffer);
    clReleaseMemObject(d_inliers_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete h_points;
    
    return segResult;
}
