#pragma once

#include <cuda_runtime.h>

// 4x4 matrix type for CUDA
struct Matrix4x4 {
    float m[16];  // Row-major 4x4 matrix
};

/**
 * DepthCameraParams: Camera intrinsic and extrinsic parameters
 */
struct DepthCameraParams {
    // Intrinsic parameters
    float fx, fy;           // Focal lengths
    float cx, cy;           // Principal point
    int width, height;      // Image dimensions
    
    // Extrinsic parameters (camera pose)
    Matrix4x4 extrinsic;    // 4x4 transformation matrix (world to camera)
    
    // Depth range
    float depthMin;
    float depthMax;
};

/**
 * DepthCameraData: Depth and color image data
 */
struct DepthCameraData {
    // Device pointers
    float* d_depth;         // Depth map (GPU)
    uchar3* d_color;        // Color map (RGB, GPU)
    
    // Image dimensions
    int width;
    int height;
};

