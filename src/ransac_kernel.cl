// ransac_kernel.cl
__kernel void ransac_plane_segmentation(__global const float* points,
                                        __global const float* planeParams,
                                        const int numPoints,
                                        const float distanceThreshold,
                                        __global int* restrict inliers)
{
    // Get the global index for the point.
    int idx = get_global_id(0);
    if (idx >= numPoints)
        return;
    
    // Each point consists of 3 floats (x, y, z).
    int base = idx * 3;
    float x = points[base];
    float y = points[base + 1];
    float z = points[base + 2];

    // Plane parameters: planeParams[0] = A, [1] = B, [2] = C, [3] = D, [4] = normalization factor.
    float A = planeParams[0];
    float B = planeParams[1];
    float C = planeParams[2];
    float D = planeParams[3];
    float normFactor = planeParams[4];

    // Calculate the distance from point to the plane.
    float dist = fabs(A * x + B * y + C * z + D) / normFactor;
    inliers[idx] = (dist <= distanceThreshold);

}
