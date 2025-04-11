__kernel void compute_inliers(__global const float4* points,
                              int numPoints,
                              float A, float B, float C, float D,
                              float threshold,
                              __global int* inliers)
{
    int i = get_global_id(0);
    if (i >= numPoints) return;

    float x = points[i].x;
    float y = points[i].y;
    float z = points[i].z;

    float dist = fabs(A * x + B * y + C * z + D) / sqrt(A*A + B*B + C*C);
    inliers[i] = (dist <= threshold) ? 1 : 0;
}
