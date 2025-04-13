// ransac_kernel.cl
// This kernel processes batched candidate planes for RANSAC segmentation.
__kernel void batched_ransac_plane_segmentation(__global const float* points,          // [numPoints * 3]
                                                __global const float* candidatePlanes, // [numCandidates * 5]
                                                const int numPoints,
                                                const float distanceThreshold,
                                                __global int* candidateCounts,
                                                __local int* localSum)
{
    // Get candidate index (first dimension) and point index (second dimension).
    int cand_idx = get_global_id(0);   // Candidate plane index.
    int point_idx = get_global_id(1);    // Point index within the candidate.
    
    // Local index within the point dimension.
    int local_idx = get_local_id(1);
    int local_size = get_local_size(1);
    
    // Each work-group processes one candidate in the first dimension.
    int candOffset = cand_idx * 5;
    float A = candidatePlanes[candOffset + 0];
    float B = candidatePlanes[candOffset + 1];
    float C = candidatePlanes[candOffset + 2];
    float D = candidatePlanes[candOffset + 3];
    float normFactor = candidatePlanes[candOffset + 4];

    // Compute the global inlier flag for this candidate and point.
    int flag = 0;
    int pointOffset = point_idx * 3;
    float x = points[pointOffset + 0];
    float y = points[pointOffset + 1];
    float z = points[pointOffset + 2];
    float dist = fabs(A * x + B * y + C * z + D) / normFactor;
    if (dist <= distanceThreshold) {
        flag = 1;
    }
    
    // Each work-item writes its flag into the local sum array.
    localSum[local_idx] = flag;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory: sum the inlier flags.
    // (Assumes local_size is a power of two for simplicity.)
    for (int stride = local_size >> 1; stride > 0; stride >>= 1) {
        if (local_idx < stride && (local_idx + stride) < numPoints) {
            localSum[local_idx] += localSum[local_idx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write the reduction result: only local index 0 writes the candidate's total inlier count.
    if (local_idx == 0) {
        candidateCounts[cand_idx] = localSum[0];
    }
}
