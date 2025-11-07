//#include "../include/CUDADepthMapData.h"
//#include "../include/scanData.h"  // Include here for implementation
//#include <cuda_runtime.h>
//
//// Declare CUDA wrapper functions (defined in .cu file)
//extern "C" {
//    void projectPointCloudToDepthMapWrapper(const float3* points, const float3* normals, const uchar3* colors,
//                                           int numPoints, float* depthMap, unsigned char* depthMapMask,
//                                           int width, int height, float fx, float fy, float cx, float cy,
//                                           float nearPlane, float farPlane);
//    void initDepthMapWrapper(float* depthMap, unsigned char* mask, int size);
//}
//
//// Constructor
//CUDADepthMapData::CUDADepthMapData() 
//    : d_depthMap(nullptr), d_depthMapMask(nullptr), d_points(nullptr), 
//      d_normals(nullptr), d_colors(nullptr), numPoints(0),
//      depthMapWidth(275), depthMapHeight(225),
//      fx(275.0f), fy(275.0f), cx(137.5f), cy(112.5f),
//      nearPlane(0.1f), farPlane(10.0f), isAllocated(false) {
//}
//
//// Destructor
//CUDADepthMapData::~CUDADepthMapData() {
//    deallocateGPU();
//}
//
//// Set camera parameters
//void CUDADepthMapData::setCameraParameters(float fx, float fy, float cx, float cy, 
//                                         float nearPlane, float farPlane) {
//    this->fx = fx;
//    this->fy = fy;
//    this->cx = cx;
//    this->cy = cy;
//    this->nearPlane = nearPlane;
//    this->farPlane = farPlane;
//}
//
//// Build depth map from point cloud
//void CUDADepthMapData::buildDepthMapFromPointCloud(const PointCloudFormat& pointCloud,
//                                                   const cv::Mat& cameraPose,
//                                                   int width, int height) {
//    depthMapWidth = width;
//    depthMapHeight = height;
//    
//    // Copy camera pose matrix
//    // Note: cameraPose should be world-to-camera transformation
//    // Implementation depends on your coordinate system convention
//    
//    // Allocate GPU memory if needed
//    if (!isAllocated) {
//        allocateGPU();
//    }
//    
//    // Copy point cloud to GPU
//    copyPointCloudToGPU(pointCloud);
//    
//    // Initialize depth map
//    int mapSize = width * height;
//    int blockSize = 256;
//    int numBlocks = (mapSize + blockSize - 1) / blockSize;
//    
//    initDepthMapWrapper(d_depthMap, d_depthMapMask, mapSize);
//    
//    // Project points to depth map
//    projectPointCloudToDepthMapWrapper(
//        d_points, d_normals, d_colors, numPoints,
//        d_depthMap, d_depthMapMask,
//        width, height,
//        fx, fy, cx, cy, nearPlane, farPlane
//    );
//    
//    cudaDeviceSynchronize();
//}
//
//// Allocate GPU memory
//void CUDADepthMapData::allocateGPU() {
//    if (isAllocated) return;
//    
//    // Allocate depth map
//    size_t mapSize = depthMapWidth * depthMapHeight * sizeof(float);
//    cudaMalloc(&d_depthMap, mapSize);
//    
//    size_t maskSize = depthMapWidth * depthMapHeight * sizeof(unsigned char);
//    cudaMalloc(&d_depthMapMask, maskSize);
//    
//    isAllocated = true;
//}
//
//// Deallocate GPU memory
//void CUDADepthMapData::deallocateGPU() {
//    if (d_depthMap) {
//        cudaFree(d_depthMap);
//        d_depthMap = nullptr;
//    }
//    if (d_depthMapMask) {
//        cudaFree(d_depthMapMask);
//        d_depthMapMask = nullptr;
//    }
//    if (d_points) {
//        cudaFree(d_points);
//        d_points = nullptr;
//    }
//    if (d_normals) {
//        cudaFree(d_normals);
//        d_normals = nullptr;
//    }
//    if (d_colors) {
//        cudaFree(d_colors);
//        d_colors = nullptr;
//    }
//    
//    isAllocated = false;
//}
//
//// Copy point cloud to GPU
//void CUDADepthMapData::copyPointCloudToGPU(const PointCloudFormat& pointCloud) {
//    numPoints = pointCloud.points.size();
//    
//    if (numPoints == 0) return;
//    
//    // Allocate GPU memory for points
//    if (d_points) cudaFree(d_points);
//    cudaMalloc(&d_points, numPoints * sizeof(float3));
//    
//    if (d_normals) cudaFree(d_normals);
//    cudaMalloc(&d_normals, numPoints * sizeof(float3));
//    
//    if (d_colors) cudaFree(d_colors);
//    cudaMalloc(&d_colors, numPoints * sizeof(uchar3));
//    
//    // Copy points
//    std::vector<float3> hostPoints(numPoints);
//    for (int i = 0; i < numPoints; i++) {
//        hostPoints[i] = make_float3(pointCloud.points[i].x, 
//                                   pointCloud.points[i].y, 
//                                   pointCloud.points[i].z);
//    }
//    cudaMemcpy(d_points, hostPoints.data(), numPoints * sizeof(float3), cudaMemcpyHostToDevice);
//    
//    // Copy normals
//    std::vector<float3> hostNormals(numPoints);
//    for (int i = 0; i < numPoints; i++) {
//        hostNormals[i] = make_float3(pointCloud.normals[i].x, 
//                                    pointCloud.normals[i].y, 
//                                    pointCloud.normals[i].z);
//    }
//    cudaMemcpy(d_normals, hostNormals.data(), numPoints * sizeof(float3), cudaMemcpyHostToDevice);
//    
//    // Copy colors
//    std::vector<uchar3> hostColors(numPoints);
//    for (int i = 0; i < numPoints; i++) {
//        hostColors[i] = make_uchar3(pointCloud.colors[i][0], 
//                                   pointCloud.colors[i][1], 
//                                   pointCloud.colors[i][2]);
//    }
//    cudaMemcpy(d_colors, hostColors.data(), numPoints * sizeof(uchar3), cudaMemcpyHostToDevice);
//}
//
//// Get depth map on host (for debugging)
//cv::Mat CUDADepthMapData::getDepthMapHost() const {
//    if (!d_depthMap) return cv::Mat();
//    
//    cv::Mat depthMap(depthMapHeight, depthMapWidth, CV_32F);
//    cudaMemcpy(depthMap.data, d_depthMap, 
//               depthMapWidth * depthMapHeight * sizeof(float), cudaMemcpyDeviceToHost);
//    
//    return depthMap;
//}
//
//// Get depth map mask on host (for debugging)
//cv::Mat CUDADepthMapData::getDepthMapMaskHost() const {
//    if (!d_depthMapMask) return cv::Mat();
//    
//    cv::Mat mask(depthMapHeight, depthMapWidth, CV_8UC1);
//    cudaMemcpy(mask.data, d_depthMapMask, 
//               depthMapWidth * depthMapHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
//    
//    return mask;
//}
//
