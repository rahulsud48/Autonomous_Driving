// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in the function to do voxel grid point reduction and region based filtering

    pcl::VoxelGrid<PointT> vg;
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered(new pcl::PointCloud<PointT>);

    vg.setInputCloud(cloud);
    vg.setLeafSize(filterRes, filterRes, filterRes);
    vg.filter(*cloudFiltered);

    typename pcl::PointCloud<PointT>::Ptr cloudRegion(new pcl::PointCloud<PointT>);

    pcl::CropBox<PointT> region(true);
    region.setMin(minPoint);
    region.setMax(maxPoint);
    region.setInputCloud(cloudFiltered);
    region.filter(*cloudRegion);

    std::vector<int> indices;

    pcl::CropBox<PointT> roof(true);
    roof.setMin(Eigen::Vector4f (-1.5,-1.7,-1,1));
    roof.setMax(Eigen::Vector4f (2.6,1.7,-.4,1));
    roof.setInputCloud(cloudRegion);
    roof.filter(indices);

    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    for(int point : indices){
        inliers->indices.push_back(point);
    }

    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloudRegion);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter (*cloudRegion);


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "filtering took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;

}



template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
  // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr obstCloud {new pcl::PointCloud<PointT>};
    typename pcl::PointCloud<PointT>::Ptr planeCloud {new pcl::PointCloud<PointT>};

    for (int index : inliers->indices)
    {
        planeCloud->points.push_back(cloud->points[index]);
    }
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud (cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    extract.filter(*obstCloud);

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstCloud, planeCloud);
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlanePCL(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();
	// pcl::PointIndices::Ptr inliers;

    // TODO:: Fill in this function to find inliers for the cloud.
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers {new pcl::PointIndices};
    pcl::ModelCoefficients::Ptr coefficients {new pcl::ModelCoefficients};

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    // Segment the largest planar object: which is the road
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if(inliers->indices.size() == 0)
    {
        std::cout<<"model couldn't get the planar road surface"<<std::endl;
    }


    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers,cloud);
    return segResult;
}


template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlaneCPU(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)

{
    pcl::PointIndices::Ptr inliersResult{new pcl::PointIndices()};
    pcl::PointIndices::Ptr inliersResult_check{new pcl::PointIndices()};
    typename pcl::PointCloud<PointT>::Ptr inlierPoints(new pcl::PointCloud<PointT>());

    

    srand(time(NULL));
    auto startTime = std::chrono::steady_clock::now();

    //------------------------------------------------------
    // 2. Initialize data on the HOST
    //------------------------------------------------------
    // 1. Flatten cloud into float4
    /*
    Flatten the cloud is important because in memory, the point cloud is not continuous, which does not help us to optimize the memory.
    Also, in production, it is important to make sure that we read bin directly to STL Vector.
    Thus, I will not be profiling this operation
    */
    std::vector<float>* h_points = new std::vector<float>();  // detele h_points: free nmemory
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        h_points->push_back(cloud->points[i].x);
        h_points->push_back(cloud->points[i].y);
        h_points->push_back(cloud->points[i].z);
    }

    // //------------------------------------------------------
    // // 3. Platform and device setup
    // //------------------------------------------------------
    // cl_platform_id platform;
    // clGetPlatformIDs(1, &platform, NULL);
    // cl_device_id device;
    // clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    // // Max Work GroupSize of the device
    // size_t max_work_group_size;
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

    // //------------------------------------------------------
    // // 4. Create a context and command queue
    // //------------------------------------------------------
    // // Create context and command queue with profiling enabled
    // cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    // cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // //------------------------------------------------------
    // // 5. Build the program and create the kernel
    // //------------------------------------------------------
    // const char* kernel_filename = "../src/ransac_kernel.cl";
    // std::string kernel_source = loadKernel(kernel_filename);

    // //------------------------------------------------------
    // // // 6. Create memory buffers on the DEVICE
    // // //------------------------------------------------------
    // // // Create buffers for input and output
    // cl_mem h_point_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 20000 * sizeof(float), NULL, NULL);
    // cl_mem d_point_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 5 * sizeof(float), NULL, NULL);
    // cl_mem inlier_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , 20000 * sizeof(int), NULL, NULL);
    


    // Segment the planar component from the cloud, represented by indicies of inliers from fitted plane with most inliers
    while (maxIterations-- > 0)
    {
        // Randomly sample subset
        // PointT point1 = cloud->points.at(rand() % (cloud->points.size()));
        // PointT point2 = cloud->points.at(rand() % (cloud->points.size()));
        // PointT point3 = cloud->points.at(rand() % (cloud->points.size()));
        int p1_id = (rand() % (h_points->size() / 3)) * 3;
        float p1_x = h_points->at(p1_id);
        float p1_y = h_points->at(p1_id+1);
        float p1_z = h_points->at(p1_id+2);

        int p2_id = (rand() % (h_points->size() / 3)) * 3;
        float p2_x = h_points->at(p2_id);
        float p2_y = h_points->at(p2_id+1);
        float p2_z = h_points->at(p2_id+2);

        int p3_id = (rand() % (h_points->size() / 3)) * 3;
        float p3_x = h_points->at(p3_id);
        float p3_y = h_points->at(p3_id+1);
        float p3_z = h_points->at(p3_id+2);




        // Fit a plane, Ax+By+Cz+D=0
        std::vector<float> d_points(5);
        float A, B, C, D;
        // A = (point2.y - point1.y) * (point3.z - point1.z) - (point2.z - point1.z) * (point3.y - point1.y); // (y2 - y1)(z3 - z1) - (z2 - z1)(y3 - y1)
        // B = (point2.z - point1.z) * (point3.x - point1.x) - (point2.x - point1.x) * (point3.z - point1.z); // (z2 - z1)(x3 - x1) - (x2 - x1)(z3 - z1)
        // C = (point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x); // (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)
        // D = -1 * (A * point1.x + B * point1.y + C * point1.z); // -(A * x1 + B * y1 + C * z1)

        A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y); // (y2 - y1)(z3 - z1) - (z2 - z1)(y3 - y1)
        B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z); // (z2 - z1)(x3 - x1) - (x2 - x1)(z3 - z1)
        C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x); // (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)
        D = -1 * (A * p1_x + B * p1_y + C * p1_z); // -(A * x1 + B * y1 + C * z1)

        float sqrt_denom = sqrt(A * A + B * B + C * C);
        d_points[0] = A;
        d_points[1] = B;
        d_points[2] = C;
        d_points[3] = D;
        d_points[4] = sqrt_denom;



        // Measure distance between every point and fitted plane
        // this needs to be sent on GPU
        std::vector<int> inlier;
        pcl::PointIndices::Ptr inliersTemp{new pcl::PointIndices()};
        pcl::PointIndices::Ptr inliersTemp_check{new pcl::PointIndices()};
        int idx = 0;
        for (auto it = cloud->points.begin(); it != cloud->points.end(); ++it)
        {
            float d = fabs(A * (*it).x + B * (*it).y + C * (*it).z + D) / sqrt_denom; // |A*x+B*y+C*z+D|/(A^2+B^2+C^2)
            // If distance is smaller than threshold count it as inlier
            if (d <= distanceThreshold)
            {
                inliersTemp->indices.push_back(it - cloud->begin());
                
            }
            idx++;
        }
        for (int it = 0; it < h_points->size() ; it=it+3)
        {
            float d = fabs(A * h_points->at(it) + B * h_points->at(it+1) + C * h_points->at(it+2) + D) / sqrt_denom; // |A*x+B*y+C*z+D|/(A^2+B^2+C^2)
            // If distance is smaller than threshold count it as inlier
            if (d <= distanceThreshold)
            {
                inlier.push_back(1);
            }
            else
            {
                inlier.push_back(0);
            }
        }

        for (int it = 0; it < inlier.size() ; it++)
        {
            if (inlier[it] == 1)
            {
                // PointT inlier_point;
                // inlier_point.x = h_points->at(it);
                // inlier_point.y = h_points->at(it+1);
                // inlier_point.z = h_points->at(it+2);

                // inlierPoints->points.push_back(inlier_point);

                inliersTemp_check->indices.push_back(it);

            }
            
        }
        // inliersResult_check->indices = inlier;
        // inliersResult_check = inlierPoints;


        if (inliersTemp->indices.size() > inliersResult->indices.size())
        {
            inliersResult = inliersTemp;
            inliersResult_check = inliersTemp_check;
        }
    }

    if (inliersResult->indices.size() == 0)
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    // std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliersResult, cloud);
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliersResult_check, cloud);

    delete h_points;

    return segResult;
}

template<typename PointT>
std::string ProcessPointClouds<PointT>::loadKernel(const char* filename) 
{
    FILE* file = fopen(filename, "r");
    if (!file) 
    {
        fprintf(stderr, "Error: Could not open kernel file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(length + 1);
    if (!source) 
    {
        fprintf(stderr, "Error: Could not allocate memory for kernel source\n");
        fclose(file);
        return NULL;
    }

    fread(source, 1, length, file);
    source[length] = '\0'; // Null-terminate the string

    fclose(file);
    return source;
}


// template<typename PointT>
// std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlaneGPU(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
// {
//     pcl::PointIndices::Ptr inliersResult{new pcl::PointIndices()};
//     pcl::PointIndices::Ptr inliersResult_check{new pcl::PointIndices()};
//     typename pcl::PointCloud<PointT>::Ptr inlierPoints(new pcl::PointCloud<PointT>());

    

//     srand(time(NULL));
//     auto startTime = std::chrono::steady_clock::now();

//     //------------------------------------------------------
//     // 2. Initialize data on the HOST
//     //------------------------------------------------------
//     // 1. Flatten cloud into float4
//     /*
//     Flatten the cloud is important because in memory, the point cloud is not continuous, which does not help us to optimize the memory.
//     Also, in production, it is important to make sure that we read bin directly to STL Vector.
//     Thus, I will not be profiling this operation
//     */
//     std::vector<float>* h_points = new std::vector<float>();  // detele h_points: free nmemory
//     for (size_t i = 0; i < cloud->points.size(); ++i) {
//         h_points->push_back(cloud->points[i].x);
//         h_points->push_back(cloud->points[i].y);
//         h_points->push_back(cloud->points[i].z);
//     }

//     //------------------------------------------------------
//     // 3. Platform and device setup
//     //------------------------------------------------------
//     cl_platform_id platform;
//     clGetPlatformIDs(1, &platform, NULL);
//     cl_device_id device;
//     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//     // Max Work GroupSize of the device
//     size_t max_work_group_size;
//     clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

//     //------------------------------------------------------
//     // 4. Create a context and command queue
//     //------------------------------------------------------
//     // Create context and command queue with profiling enabled
//     cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
//     cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

//     //------------------------------------------------------
//     // 5. Build the program and create the kernel
//     //------------------------------------------------------
//     const char* kernel_filename = "../src/ransac_kernel.cl";
//     std::string kernel_source = loadKernel(kernel_filename);

//     //------------------------------------------------------
//     // // 6. Create memory buffers on the DEVICE
//     // //------------------------------------------------------
//     // // Create buffers for input and output
//     cl_mem h_point_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 20000 * sizeof(float), NULL, NULL);
//     cl_mem d_point_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, 5 * sizeof(float), NULL, NULL);
//     cl_mem inlier_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY , 20000 * sizeof(int), NULL, NULL);
    


//     // Segment the planar component from the cloud, represented by indicies of inliers from fitted plane with most inliers
//     while (maxIterations-- > 0)
//     {
//         // Randomly sample subset
//         // PointT point1 = cloud->points.at(rand() % (cloud->points.size()));
//         // PointT point2 = cloud->points.at(rand() % (cloud->points.size()));
//         // PointT point3 = cloud->points.at(rand() % (cloud->points.size()));
//         int p1_id = (rand() % (h_points->size() / 3)) * 3;
//         float p1_x = h_points->at(p1_id);
//         float p1_y = h_points->at(p1_id+1);
//         float p1_z = h_points->at(p1_id+2);

//         int p2_id = (rand() % (h_points->size() / 3)) * 3;
//         float p2_x = h_points->at(p2_id);
//         float p2_y = h_points->at(p2_id+1);
//         float p2_z = h_points->at(p2_id+2);

//         int p3_id = (rand() % (h_points->size() / 3)) * 3;
//         float p3_x = h_points->at(p3_id);
//         float p3_y = h_points->at(p3_id+1);
//         float p3_z = h_points->at(p3_id+2);




//         // Fit a plane, Ax+By+Cz+D=0
//         std::vector<float> d_points(5);
//         float A, B, C, D;
//         // A = (point2.y - point1.y) * (point3.z - point1.z) - (point2.z - point1.z) * (point3.y - point1.y); // (y2 - y1)(z3 - z1) - (z2 - z1)(y3 - y1)
//         // B = (point2.z - point1.z) * (point3.x - point1.x) - (point2.x - point1.x) * (point3.z - point1.z); // (z2 - z1)(x3 - x1) - (x2 - x1)(z3 - z1)
//         // C = (point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x); // (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)
//         // D = -1 * (A * point1.x + B * point1.y + C * point1.z); // -(A * x1 + B * y1 + C * z1)

//         A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y); // (y2 - y1)(z3 - z1) - (z2 - z1)(y3 - y1)
//         B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z); // (z2 - z1)(x3 - x1) - (x2 - x1)(z3 - z1)
//         C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x); // (x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)
//         D = -1 * (A * p1_x + B * p1_y + C * p1_z); // -(A * x1 + B * y1 + C * z1)

//         float sqrt_denom = sqrt(A * A + B * B + C * C);
//         d_points[0] = A;
//         d_points[1] = B;
//         d_points[2] = C;
//         d_points[3] = D;
//         d_points[4] = sqrt_denom;



//         // Measure distance between every point and fitted plane
//         // this needs to be sent on GPU
//         std::vector<int> inlier;
//         pcl::PointIndices::Ptr inliersTemp{new pcl::PointIndices()};
//         pcl::PointIndices::Ptr inliersTemp_check{new pcl::PointIndices()};
//         int idx = 0;
//         for (auto it = cloud->points.begin(); it != cloud->points.end(); ++it)
//         {
//             float d = fabs(A * (*it).x + B * (*it).y + C * (*it).z + D) / sqrt_denom; // |A*x+B*y+C*z+D|/(A^2+B^2+C^2)
//             // If distance is smaller than threshold count it as inlier
//             if (d <= distanceThreshold)
//             {
//                 inliersTemp->indices.push_back(it - cloud->begin());
                
//             }
//             idx++;
//         }
//         for (int it = 0; it < h_points->size() ; it=it+3)
//         {
//             float d = fabs(A * h_points->at(it) + B * h_points->at(it+1) + C * h_points->at(it+2) + D) / sqrt_denom; // |A*x+B*y+C*z+D|/(A^2+B^2+C^2)
//             // If distance is smaller than threshold count it as inlier
//             if (d <= distanceThreshold)
//             {
//                 inlier.push_back(1);
//             }
//             else
//             {
//                 inlier.push_back(0);
//             }
//         }

//         for (int it = 0; it < inlier.size() ; it++)
//         {
//             if (inlier[it] == 1)
//             {
//                 // PointT inlier_point;
//                 // inlier_point.x = h_points->at(it);
//                 // inlier_point.y = h_points->at(it+1);
//                 // inlier_point.z = h_points->at(it+2);

//                 // inlierPoints->points.push_back(inlier_point);

//                 inliersTemp_check->indices.push_back(it);

//             }
            
//         }
//         // inliersResult_check->indices = inlier;
//         // inliersResult_check = inlierPoints;


//         if (inliersTemp->indices.size() > inliersResult->indices.size())
//         {
//             inliersResult = inliersTemp;
//             inliersResult_check = inliersTemp_check;
//         }
//     }

//     if (inliersResult->indices.size() == 0)
//     {
//         std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
//     }
//     auto endTime = std::chrono::steady_clock::now();
//     auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//     std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

//     // std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliersResult, cloud);
//     std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliersResult_check, cloud);

//     delete h_points;

//     return segResult;
// }

// template<typename PointT>
// std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> 
// ProcessPointClouds<PointT>::SegmentPlaneGPU(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
// {
//     // Memory for storing best inlier indices (two versions, one could be used if you need an additional copy)
//     pcl::PointIndices::Ptr bestInliers(new pcl::PointIndices());
//     pcl::PointIndices::Ptr bestInliers_check(new pcl::PointIndices());
    
//     // Flatten the point cloud into a continuous float vector (x,y,z)
//     std::vector<float>* h_points = new std::vector<float>();
//     for (size_t i = 0; i < cloud->points.size(); ++i)
//     {
//         h_points->push_back(cloud->points[i].x);
//         h_points->push_back(cloud->points[i].y);
//         h_points->push_back(cloud->points[i].z);
//     }
//     const int numPoints = cloud->points.size();

//     // Setup OpenCL platform, device and context.
//     cl_platform_id platform;
//     clGetPlatformIDs(1, &platform, NULL);
//     cl_device_id device;
//     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//     size_t max_work_group_size;
//     clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
//     cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
//     cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

//     // profiler
//     cl_event write_event, kernel_event, read_event;



//     // Load and build kernel program
//     const char* kernel_filename = "../src/ransac_kernel.cl";
//     std::string kernel_source = loadKernel(kernel_filename);
//     const char* kernel_source_cstr = kernel_source.c_str();
//     cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, NULL);
//     if(clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
//         // Retrieve and print build log in case of errors.
//         size_t log_size;
//         clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//         std::vector<char> build_log(log_size);
//         clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
//         std::cerr << "Error building OpenCL program:" << std::endl << build_log.data() << std::endl;
//         exit(1);
//     }
//     cl_kernel kernel = clCreateKernel(program, "ransac_plane_segmentation", NULL);

//     // Create device buffers.
//     size_t pointsBufferSize = numPoints * 3 * sizeof(float);
//     cl_mem d_points_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, pointsBufferSize, NULL, NULL);
//     size_t planeBufferSize = 5 * sizeof(float);
//     cl_mem d_plane_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, planeBufferSize, NULL, NULL);
//     size_t inliersBufferSize = numPoints * sizeof(int);
//     cl_mem d_inliers_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inliersBufferSize, NULL, NULL);

//     // Write the flattened point cloud to the device (it does not change in the loop).
//     clEnqueueWriteBuffer(queue, d_points_buffer, CL_TRUE, 0, pointsBufferSize, h_points->data(), 0, NULL, &write_event);

//     // Variables to track the best inlier count.
//     size_t bestInlierCount = 0;

//     auto startTime = std::chrono::steady_clock::now();

//     // Main RANSAC iterations
//     while(maxIterations-- > 0)
//     {
//         // Randomly sample 3 points from the host flattened cloud.
//         int p1_id = (rand() % numPoints) * 3;
//         int p2_id = (rand() % numPoints) * 3;
//         int p3_id = (rand() % numPoints) * 3;
//         float p1_x = h_points->at(p1_id);
//         float p1_y = h_points->at(p1_id+1);
//         float p1_z = h_points->at(p1_id+2);
//         float p2_x = h_points->at(p2_id);
//         float p2_y = h_points->at(p2_id+1);
//         float p2_z = h_points->at(p2_id+2);
//         float p3_x = h_points->at(p3_id);
//         float p3_y = h_points->at(p3_id+1);
//         float p3_z = h_points->at(p3_id+2);

//         // Fit the plane: Ax + By + Cz + D = 0.
//         float A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y);
//         float B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z);
//         float C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x);
//         float D = -1 * (A * p1_x + B * p1_y + C * p1_z);
//         float normFactor = sqrt(A * A + B * B + C * C);

//         // Prepare plane parameters vector: [A, B, C, D, normFactor]
//         std::vector<float> planeParams = {A, B, C, D, normFactor};

//         // Write plane parameters to device.
//         clEnqueueWriteBuffer(queue, d_plane_buffer, CL_TRUE, 0, planeBufferSize, planeParams.data(), 0, NULL, NULL);

//         // Set kernel arguments.
//         int argIdx = 0;
//         clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_points_buffer);
//         clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_plane_buffer);
//         clSetKernelArg(kernel, argIdx++, sizeof(int), &numPoints);
//         clSetKernelArg(kernel, argIdx++, sizeof(float), &distanceThreshold);
//         clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_inliers_buffer);

//         // Launch kernel with one work-item per point.
//         size_t global_size = numPoints;
//         clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
//         clFinish(queue);

//         // Read back inliers.
//         std::vector<int> inlierFlags(numPoints, 0);
//         clEnqueueReadBuffer(queue, d_inliers_buffer, CL_TRUE, 0, inliersBufferSize, inlierFlags.data(), 0, NULL, &read_event);

//         // Count and store indices for the current iteration.
//         size_t inlierCount = 0;
//         pcl::PointIndices::Ptr currentInliers(new pcl::PointIndices());
//         for (int i = 0; i < numPoints; i++)
//         {
//             if (inlierFlags[i] == 1)
//             {
//                 currentInliers->indices.push_back(i);
//                 inlierCount++;
//             }
//         }

//         // Update best inliers if the current set is larger.
//         if (inlierCount > bestInlierCount)
//         {
//             bestInlierCount = inlierCount;
//             bestInliers = currentInliers;
//             bestInliers_check = currentInliers;
//         }

//         auto print_event_duration = [](cl_event e, const std::string& label) {
//             cl_ulong start, end;
//             clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
//             clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
//             double time_ms = (end - start) * 1e-6; // ns to ms
//             std::cout << label << ": " << time_ms << " ms\n";
//         };
//         // print_event_duration(write_event, "WriteBuffer Time");
//         // print_event_duration(kernel_event, "Kernel Execution Time");
//         // print_event_duration(read_event, "ReadBuffer Time");


//     }

//     auto endTime = std::chrono::steady_clock::now();
//     auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//     std::cout << "Plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

//     // Separate cloud using best inliers.
//     std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(bestInliers_check, cloud);

//     // Cleanup: release OpenCL resources.
//     clReleaseMemObject(d_points_buffer);
//     clReleaseMemObject(d_plane_buffer);
//     clReleaseMemObject(d_inliers_buffer);
//     clReleaseKernel(kernel);
//     clReleaseProgram(program);
//     clReleaseCommandQueue(queue);
//     clReleaseContext(context);
//     delete h_points;
    
//     return segResult;
// }

// template<typename PointT>
// std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> 
// ProcessPointClouds<PointT>::SegmentPlaneGPU(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
// {
//     // Pre-flatten the point cloud into a continuous vector [x, y, z, ...].
//     const int numPoints = cloud->points.size();
//     std::vector<float> h_points;
//     h_points.reserve(numPoints * 3);
//     for (size_t i = 0; i < numPoints; ++i)
//     {
//         h_points.push_back(cloud->points[i].x);
//         h_points.push_back(cloud->points[i].y);
//         h_points.push_back(cloud->points[i].z);
//     }
    
//     // Generate 3 random indices for each iteration (total maxIterations * 3).
//     std::vector<int> randomSamples(maxIterations * 3);
//     for (int i = 0; i < maxIterations; ++i)
//     {
//         randomSamples[3 * i + 0] = rand() % numPoints;
//         randomSamples[3 * i + 1] = rand() % numPoints;
//         randomSamples[3 * i + 2] = rand() % numPoints;
//     }
    
//     // Setup OpenCL: platform, device, context and queue.
//     cl_platform_id platform;
//     clGetPlatformIDs(1, &platform, NULL);
//     cl_device_id device;
//     clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
//     cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
//     cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
//     // profiler
//     cl_event write_event, kernel_event, read_event;
    
//     // Load and build kernel program.
//     const char* kernel_filename = "../src/ransac_kernel.cl";
//     std::string kernel_source = loadKernel(kernel_filename);
//     const char* kernel_source_cstr = kernel_source.c_str();
//     cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, NULL);
//     if(clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
//         size_t log_size;
//         clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//         std::vector<char> build_log(log_size);
//         clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
//         std::cerr << "Error building OpenCL program:" << std::endl << build_log.data() << std::endl;
//         exit(1);
//     }
//     cl_kernel kernel = clCreateKernel(program, "ransac_plane_parallel", NULL);
    
//     // Create device buffers.
//     // Points: same data for every iteration.
//     size_t pointsBufferSize = numPoints * 3 * sizeof(float);
//     cl_mem d_points_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, pointsBufferSize, NULL, NULL);
//     // Random samples: one triplet per iteration.
//     size_t samplesBufferSize = maxIterations * 3 * sizeof(int);
//     cl_mem d_samples_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, samplesBufferSize, NULL, NULL);
//     // Output buffer for plane parameters for each iteration (each with 5 floats).
//     size_t planeBufferSize = maxIterations * 5 * sizeof(float);
//     cl_mem d_plane_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, planeBufferSize, NULL, NULL);
//     // Output buffer for inlier flags for each iteration (maxIterations * numPoints ints).
//     size_t inliersBufferSize = maxIterations * numPoints * sizeof(int);
//     cl_mem d_inliers_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inliersBufferSize, NULL, NULL);
    
//         auto startTime = std::chrono::steady_clock::now();
//     // Write data to device.
//     clEnqueueWriteBuffer(queue, d_points_buffer, CL_TRUE, 0, pointsBufferSize, h_points.data(), 0, NULL,  &write_event);
//     clEnqueueWriteBuffer(queue, d_samples_buffer, CL_TRUE, 0, samplesBufferSize, randomSamples.data(), 0, NULL, NULL);
    
//     // Set kernel arguments.
//     int argIdx = 0;
//     clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_points_buffer);
//     clSetKernelArg(kernel, argIdx++, sizeof(int), &numPoints);
//     clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_samples_buffer);
//     clSetKernelArg(kernel, argIdx++, sizeof(int), &maxIterations);
//     clSetKernelArg(kernel, argIdx++, sizeof(float), &distanceThreshold);
//     clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_plane_buffer);
//     clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_inliers_buffer);
    
//     // Launch the kernel with one work-item per iteration.
//     size_t global_size = maxIterations;
//     clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, &kernel_event);
//     clFinish(queue);
    
//     // Retrieve results.
//     std::vector<float> h_planeParams(maxIterations * 5, 0);
//     std::vector<int> h_inlierFlags(maxIterations * numPoints, 0);
//     clEnqueueReadBuffer(queue, d_plane_buffer, CL_TRUE, 0, planeBufferSize, h_planeParams.data(), 0, NULL, NULL);
//     clEnqueueReadBuffer(queue, d_inliers_buffer, CL_TRUE, 0, inliersBufferSize, h_inlierFlags.data(), 0, NULL, &read_event);

//     auto print_event_duration = [](cl_event e, const std::string& label) {
//         cl_ulong start, end;
//         clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
//         clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
//         double time_ms = (end - start) * 1e-6; // ns to ms
//         std::cout << label << ": " << time_ms << " ms\n";
//     };
//     print_event_duration(write_event, "WriteBuffer Time");
//     print_event_duration(kernel_event, "Kernel Execution Time");
//     print_event_duration(read_event, "ReadBuffer Time");
    
//     // Determine best iteration by counting inliers (this could also be done on GPU if desired).
//     size_t bestIteration = 0;
//     size_t bestInlierCount = 0;
//     for (int iter = 0; iter < maxIterations; ++iter) {
//         size_t count = 0;
//         int offset = iter * numPoints;
//         for (int i = 0; i < numPoints; ++i)
//         {
//             if (h_inlierFlags[offset + i] == 1)
//                 count++;
//         }
//         if (count > bestInlierCount) {
//             bestInlierCount = count;
//             bestIteration = iter;
//         }
//     }
    
//     // Now, using the best iteration, you could separate your cloud.
//     // You might want to store the best plane parameters and/or best inlier mask.
//     pcl::PointIndices::Ptr bestInliers(new pcl::PointIndices());
//     {
//         int offset = bestIteration * numPoints;
//         for (int i = 0; i < numPoints; ++i)
//         {
//             if (h_inlierFlags[offset + i] == 1)
//                 bestInliers->indices.push_back(i);
//         }
//     }
    
//     // Cleanup: release OpenCL objects.
//     clReleaseMemObject(d_points_buffer);
//     clReleaseMemObject(d_samples_buffer);
//     clReleaseMemObject(d_plane_buffer);
//     clReleaseMemObject(d_inliers_buffer);
//     clReleaseKernel(kernel);
//     clReleaseProgram(program);
//     clReleaseCommandQueue(queue);
//     clReleaseContext(context);

//     auto endTime = std::chrono::steady_clock::now();
//     auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
//     std::cout << "Plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;
    
//     // Separate cloud using best inliers.
//     std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(bestInliers, cloud);
//     return segResult;
// }

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> 
ProcessPointClouds<PointT>::SegmentPlaneGPU(typename pcl::PointCloud<PointT>::Ptr cloud, 
                                                     int maxIterations, float distanceThreshold)
{
    const int numPoints = cloud->points.size();

    // Flatten the input point cloud into a continuous vector [x, y, z, ...]
    std::vector<float> h_points;
    h_points.reserve(numPoints * 3);
    for (size_t i = 0; i < numPoints; ++i)
    {
        h_points.push_back(cloud->points[i].x);
        h_points.push_back(cloud->points[i].y);
        h_points.push_back(cloud->points[i].z);
    }

    // Precompute candidate plane parameters on the host.
    // Each candidate will store [A, B, C, D, normFactor] for the plane: Ax+By+Cz+D = 0
    std::vector<float> h_planeParams(maxIterations * 5, 0.0f);
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Sample 3 random indices from the cloud.
        int idx1 = rand() % numPoints;
        int idx2 = rand() % numPoints;
        int idx3 = rand() % numPoints;
        int base1 = idx1 * 3;
        int base2 = idx2 * 3;
        int base3 = idx3 * 3;
        
        // Get the points.
        float p1_x = h_points[base1 + 0];
        float p1_y = h_points[base1 + 1];
        float p1_z = h_points[base1 + 2];
        
        float p2_x = h_points[base2 + 0];
        float p2_y = h_points[base2 + 1];
        float p2_z = h_points[base2 + 2];
        
        float p3_x = h_points[base3 + 0];
        float p3_y = h_points[base3 + 1];
        float p3_z = h_points[base3 + 2];
        
        // Compute the plane coefficients using the cross product.
        float A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y);
        float B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z);
        float C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x);
        float D = -(A * p1_x + B * p1_y + C * p1_z);
        float normFactor = sqrt(A * A + B * B + C * C);

        // Store into candidate array.
        int offset = iter * 5;
        h_planeParams[offset + 0] = A;
        h_planeParams[offset + 1] = B;
        h_planeParams[offset + 2] = C;
        h_planeParams[offset + 3] = D;
        h_planeParams[offset + 4] = normFactor;
    }
    
    // === Setup OpenCL ===
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Load and build the kernel program
    const char* kernel_filename = "../src/ransac_test_kernel.cl"; // update filename as needed
    std::string kernel_source = loadKernel(kernel_filename);
    const char* kernel_source_cstr = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, NULL);
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Error building OpenCL program:\n" << build_log.data() << std::endl;
        exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, "ransac_inlier_test", NULL);

    // Create buffers:
    size_t pointsBufferSize = numPoints * 3 * sizeof(float);
    cl_mem d_points = clCreateBuffer(context, CL_MEM_READ_ONLY, pointsBufferSize, NULL, NULL);
    
    size_t planeParamsBufferSize = maxIterations * 5 * sizeof(float);
    cl_mem d_planeParams = clCreateBuffer(context, CL_MEM_READ_ONLY, planeParamsBufferSize, NULL, NULL);
    
    // Allocate an output buffer to record inlier flags.
    // The size is maxIterations * numPoints (each work-item writes numPoints ints)
    size_t inliersBufferSize = maxIterations * numPoints * sizeof(int);
    cl_mem d_inliers = clCreateBuffer(context, CL_MEM_WRITE_ONLY, inliersBufferSize, NULL, NULL);

    // Write data to device.
    clEnqueueWriteBuffer(queue, d_points, CL_TRUE, 0, pointsBufferSize, h_points.data(), 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_planeParams, CL_TRUE, 0, planeParamsBufferSize, h_planeParams.data(), 0, NULL, NULL);

    // Set kernel arguments.
    int argIdx = 0;
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_points);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &numPoints);
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_planeParams);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &maxIterations);
    clSetKernelArg(kernel, argIdx++, sizeof(float), &distanceThreshold);
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &d_inliers);

    // Launch one work-item per candidate plane (i.e. per iteration).
    size_t global_size = maxIterations;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    // Retrieve the inlier flags.
    std::vector<int> h_inlierFlags(maxIterations * numPoints, 0);
    clEnqueueReadBuffer(queue, d_inliers, CL_TRUE, 0, inliersBufferSize, h_inlierFlags.data(), 0, NULL, NULL);

    // Evaluate the best candidate on the host.
    size_t bestIteration = 0;
    size_t bestInlierCount = 0;
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        size_t count = 0;
        int offset = iter * numPoints;
        for (int i = 0; i < numPoints; ++i)
        {
            if (h_inlierFlags[offset + i] == 1)
                count++;
        }
        if (count > bestInlierCount)
        {
            bestInlierCount = count;
            bestIteration = iter;
        }
    }

    // Build a pcl::PointIndices using the best candidate's inlier flags.
    pcl::PointIndices::Ptr bestInliers(new pcl::PointIndices());
    {
        int offset = bestIteration * numPoints;
        for (int i = 0; i < numPoints; ++i)
        {
            if (h_inlierFlags[offset + i] == 1)
                bestInliers->indices.push_back(i);
        }
    }

    // Cleanup OpenCL resources.
    clReleaseMemObject(d_points);
    clReleaseMemObject(d_planeParams);
    clReleaseMemObject(d_inliers);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Return the segmented clouds based on bestInliers.
    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(bestInliers, cloud);
    return segResult;
}





template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{

    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    typename pcl::search::KdTree<PointT>::Ptr tree{new pcl::search::KdTree<PointT>};
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance);
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusterIndices);
    for(pcl::PointIndices getIndices: clusterIndices){
        typename pcl::PointCloud<PointT>::Ptr cloudCluster (new pcl::PointCloud<PointT>);

        for(int index : getIndices.indices){
            cloudCluster->points.push_back(cloud->points[index]);
        }

        cloudCluster->width = cloudCluster->points.size();
        cloudCluster->height = 1;
        cloudCluster->is_dense = true;
        clusters.push_back(cloudCluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadBIN(std::string infile)
{
	// load point cloud
	fstream input(infile.c_str(), ios::in | ios::binary);
	if(!input.good()){
		cerr << "Could not read file: " << infile << endl;
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);

	typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

	int i;
	for (i=0; input.good() && !input.eof(); i++) {
		PointT point;
		input.read((char *) &(point.x), 3*sizeof(float));
		input.read((char *) &(point.intensity), sizeof(float));
		cloud->push_back(point);
	}
	input.close();
    return cloud;

	// cout << "Read KTTI point cloud with " << i << " points, writing to " << outfile << std::endl;

    // pcl::PCDWriter writer;

    // // Save DoN features
    // writer.write<PointT> (outfile, *cloud, false);
}