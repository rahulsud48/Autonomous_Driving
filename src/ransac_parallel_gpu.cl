__kernel void ransac_plane_parallel(__global const float* points,
                                    const int numPoints,
                                    __global const int* randomSamples,  // length: maxIterations * 3
                                    const int maxIterations,            // e.g., 25
                                    const float distanceThreshold,
                                    __global float* planeParams,        // output: maxIterations * 5 floats
                                    __global int* inlierFlags)          // output: maxIterations * numPoints ints
{
    // Each work-item handles one RANSAC iteration.
    int iter = get_global_id(0);
    if(iter >= maxIterations) return;
    
    // For this iteration, load the random samples.
    int sample_offset = iter * 3;
    int idx1 = randomSamples[sample_offset + 0];
    int idx2 = randomSamples[sample_offset + 1];
    int idx3 = randomSamples[sample_offset + 2];
    
    // Load the three points; each point has 3 coordinates.
    int base1 = idx1 * 3;
    int base2 = idx2 * 3;
    int base3 = idx3 * 3;
    
    float p1_x = points[base1 + 0];
    float p1_y = points[base1 + 1];
    float p1_z = points[base1 + 2];
    
    float p2_x = points[base2 + 0];
    float p2_y = points[base2 + 1];
    float p2_z = points[base2 + 2];
    
    float p3_x = points[base3 + 0];
    float p3_y = points[base3 + 1];
    float p3_z = points[base3 + 2];
    
    // Compute the plane coefficients (Ax + By + Cz + D = 0).
    float A = (p2_y - p1_y) * (p3_z - p1_z) - (p2_z - p1_z) * (p3_y - p1_y);
    float B = (p2_z - p1_z) * (p3_x - p1_x) - (p2_x - p1_x) * (p3_z - p1_z);
    float C = (p2_x - p1_x) * (p3_y - p1_y) - (p2_y - p1_y) * (p3_x - p1_x);
    float D = - (A * p1_x + B * p1_y + C * p1_z);
    float normFactor = sqrt(A * A + B * B + C * C);
    
    // Store the calculated plane parameters in the output array.
    int plane_offset = iter * 5;
    planeParams[plane_offset + 0] = A;
    planeParams[plane_offset + 1] = B;
    planeParams[plane_offset + 2] = C;
    planeParams[plane_offset + 3] = D;
    planeParams[plane_offset + 4] = normFactor;
    
    // For every point in the cloud, calculate its distance to this plane.
    int inlier_offset = iter * numPoints;
    for (int i = 0; i < numPoints; ++i)
    {
        int base = i * 3;
        float x = points[base + 0];
        float y = points[base + 1];
        float z = points[base + 2];
        float dist = fabs(A * x + B * y + C * z + D) / normFactor;
        inlierFlags[inlier_offset + i] = (dist <= distanceThreshold) ? 1 : 0;
    }
}