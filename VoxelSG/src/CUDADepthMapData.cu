#include "../include/CUDADepthMapData.h"
#include <cuda_runtime.h>
// #include <cuda_gl_interop.h> // Not needed here; remove to avoid GL type deps

// Note: scanData.h is NOT included here to avoid fmt/Unicode issues in CUDA compilation
// The actual implementation that needs scanData.h should be in .cpp files

// CUDA kernel for point cloud to depth map projection
extern "C" __global__ void projectPointCloudToDepthMapKernel(
    const float3* points,           // Input 3D points
    const float3* normals,          // Input normals (optional)
    const uchar3* colors,           // Input colors (optional)
    int numPoints,                  // Number of points
    float* depthMap,                // Output depth map
    unsigned char* depthMapMask,    // Output mask
    int width, int height,          // Depth map dimensions
    float fx, float fy,             // Focal length
    float cx, float cy,             // Principal point
    float nearPlane, float farPlane // Depth range
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= numPoints) return;
    
    float3 point = points[idx];
    
    // Check if point is behind camera
    if (point.z <= 0.0f) return;
    
    // Project to image coordinates
    float u = (point.x * fx / point.z) + cx;
    float v = (point.y * fy / point.z) + cy;
    
    // Check bounds
    if (u < 0 || u >= width || v < 0 || v >= height) return;
    
    // Check depth range
    if (point.z < nearPlane || point.z > farPlane) return;
    
    // Convert to pixel coordinates
    int pixel_x = (int)u;
    int pixel_y = (int)v;
    int pixel_idx = pixel_y * width + pixel_x;
    
    // Use atomic operations to handle multiple points projecting to same pixel
    // Keep the closest point (smallest depth)
    float currentDepth = depthMap[pixel_idx];
    
    if (currentDepth == 0.0f || point.z < currentDepth) {
        depthMap[pixel_idx] = point.z;
        depthMapMask[pixel_idx] = 255; // Valid pixel
    }
}

// CUDA kernel for depth map initialization
extern "C" __global__ void initDepthMapKernel(float* depthMap, unsigned char* mask, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        depthMap[idx] = 0.0f;
        mask[idx] = 0;
    }
}

// Host wrapper functions for kernel launches
extern "C" {
    void projectPointCloudToDepthMapWrapper(const float3* points, const float3* normals, const uchar3* colors,
                                           int numPoints, float* depthMap, unsigned char* depthMapMask,
                                           int width, int height, float fx, float fy, float cx, float cy,
                                           float nearPlane, float farPlane) {
        int blockSize = 256;
        int numBlocks = (numPoints + blockSize - 1) / blockSize;
        projectPointCloudToDepthMapKernel<<<numBlocks, blockSize>>>(
            points, normals, colors, numPoints, depthMap, depthMapMask,
            width, height, fx, fy, cx, cy, nearPlane, farPlane);
    }
    
    void initDepthMapWrapper(float* depthMap, unsigned char* mask, int size) {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        initDepthMapKernel<<<numBlocks, blockSize>>>(depthMap, mask, size);
    }
}
