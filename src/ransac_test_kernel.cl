__kernel void ransac_inlier_test(__global const float* points,  // flattened point cloud data
                                 const int numPoints,
                                 __global const float* planeParams, // array of maxIterations x 5
                                 const int maxIterations,
                                 const float distanceThreshold,
                                 __global int* inlierFlags)         // output: maxIterations x numPoints ints
{
    // Each work-item corresponds to one candidate plane (one iteration).
    int iter = get_global_id(0);
    if (iter >= maxIterations)
        return;
    
    // Read the candidate plane parameters provided by the host.
    int planeOffset = iter * 5;
    float A = planeParams[planeOffset + 0];
    float B = planeParams[planeOffset + 1];
    float C = planeParams[planeOffset + 2];
    float D = planeParams[planeOffset + 3];
    float normFactor = planeParams[planeOffset + 4];
    
    // For every point in the cloud, compute the distance to the candidate plane.
    int inlierOffset = iter * numPoints;
    for (int i = 0; i < numPoints; ++i)
    {
        int base = i * 3;
        float x = points[base + 0];
        float y = points[base + 1];
        float z = points[base + 2];
        
        // Compute the distance from the point to the plane.
        float dist = fabs(A * x + B * y + C * z + D) / normFactor;
        inlierFlags[inlierOffset + i] = (dist <= distanceThreshold) ? 1 : 0;
    }
}
