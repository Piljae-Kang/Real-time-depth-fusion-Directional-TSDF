#pragma once
#include "DepthCameraData.h"
#include <opencv2/opencv.hpp>

// Forward declaration to avoid including scanData.h (which includes open3d/fmt)
// This prevents fmt Unicode assertion errors when CUDA files include this header
struct PointCloudFormat;

class CUDADepthMapData {
private:
    // GPU memory for depth map
    float* d_depthMap;           // Depth values (GPU)
    unsigned char* d_depthMapMask; // Valid pixel mask (GPU)
    
    // Camera parameters
    float fx, fy;                // Focal length
    float cx, cy;                // Principal point
    float nearPlane, farPlane;   // Depth range
    
    // Depth map dimensions
    int depthMapWidth, depthMapHeight;
    
    // Point cloud data (GPU)
    float3* d_points;            // 3D points
    float3* d_normals;           // Point normals
    uchar3* d_colors;            // Point colors
    int numPoints;               // Number of points
    
    // Camera pose (world to camera transform)
    Matrix4x4 cameraPose;        // 4x4 transformation matrix
    
    bool isAllocated;
    
    // Helper functions
    void allocateGPU();
    void deallocateGPU();
    void copyPointCloudToGPU(const PointCloudFormat& pointCloud);
    
public:
    CUDADepthMapData();
    ~CUDADepthMapData();
    
    // Initialize depth map from point cloud
    void buildDepthMapFromPointCloud(const PointCloudFormat& pointCloud, 
                                   const cv::Mat& cameraPose,
                                   int width, int height);
    
    // Set camera intrinsics
    void setCameraParameters(float fx, float fy, float cx, float cy, 
                           float nearPlane = 0.1f, float farPlane = 10.0f);
    
    // Get depth map data (for integration)
    float* getDepthMapGPU() const { return d_depthMap; }
    unsigned char* getDepthMapMaskGPU() const { return d_depthMapMask; }
    
    // Get dimensions
    int getWidth() const { return depthMapWidth; }
    int getHeight() const { return depthMapHeight; }
    
    // Debug: copy depth map to host for visualization
    cv::Mat getDepthMapHost() const;
    cv::Mat getDepthMapMaskHost() const;
};