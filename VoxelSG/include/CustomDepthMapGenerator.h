#pragma once
#include "../include/scanData.h"
#include "../include/scanData2.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

/**
 * Custom Depth Map Generator
 * 
 * This class provides functionality to generate depth maps from point clouds
 * using various projection methods. It can create depth maps with different
 * resolutions, camera parameters, and projection techniques.
 */
class CustomDepthMapGenerator {
public:
    // Camera parameters structure
    struct CameraParams {
        float fx, fy;           // Focal length
        float cx, cy;           // Principal point
        float nearPlane;        // Near clipping plane
        float farPlane;         // Far clipping plane
        int width, height;      // Image dimensions
        
        CameraParams() : fx(400.0f), fy(400.0f), cx(200.0f), cy(240.0f),
                        nearPlane(0.1f), farPlane(10.0f), width(400), height(480) {}
    };
    
    // Projection method types
    enum class ProjectionMethod {
        PERSPECTIVE,        // Standard perspective projection
        ORTHOGRAPHIC,      // Orthographic projection
        CUSTOM_CAMERA      // Custom camera parameters
    };
    
    // Generated depth map data structure
    struct GeneratedDepthMap {
        std::vector<float> depthmap;        // Depth values only (Z component)
        std::vector<cv::Vec3f> pointmap;    // 3D points (X, Y, Z in camera coordinates)
        std::vector<cv::Vec3f> normalmap;   // Surface normals
        std::vector<cv::Vec3b> colormap;    // Colors
        int width, height;                   // Dimensions
        
        GeneratedDepthMap() : width(0), height(0) {}
    };
    
    // Container for multiple frames of generated depth maps (similar to PointCloudParams structure)
    struct GeneratedDepthMapFrames {
        std::vector<GeneratedDepthMap> src_0;   // Depth maps for src_0 (0-degree view)
        std::vector<GeneratedDepthMap> src_45;   // Depth maps for src_45 (45-degree view)
        std::vector<GeneratedDepthMap> total;    // Depth maps for total (src_0 + src_45 combined)
        
        GeneratedDepthMapFrames() {}
        
        // Get total number of frames
        size_t getFrameCount() const {
            return std::max({src_0.size(), src_45.size(), total.size()});
        }
        
        // Check if empty
        bool empty() const {
            return src_0.empty() && src_45.empty() && total.empty();
        }
    };

public:
    CustomDepthMapGenerator();
    ~CustomDepthMapGenerator();
    
    // Set camera parameters
    void setCameraParams(const CameraParams& params);
    void setCameraParams(float fx, float fy, float cx, float cy, 
                        float nearPlane, float farPlane, int width, int height);
    
    // Set camera parameters from ScanDataLoader (convenience method)
    void setCameraParamsFromLoader(const ScanDataLoader& loader);
    
    // Set camera parameters from ScanDataLoader2 (convenience method)
    void setCameraParamsFromLoader(const ScanDataLoader2& loader);
    
    // Set projection method
    void setProjectionMethod(ProjectionMethod method);
    
    // Generate depth map from point cloud
    GeneratedDepthMap generateFromPointCloud(
        const std::vector<cv::Point3f>& points,
        const std::vector<cv::Point3f>& normals = std::vector<cv::Point3f>(),
        const std::vector<cv::Vec3b>& colors = std::vector<cv::Vec3b>()
    );
    
    // Generate depth map from combined point clouds (src_0 + src_45)
    // Combines two point cloud formats and generates a single depth map
    GeneratedDepthMap generateFromCombinedPointClouds(
        const PointCloudFormat& src0,
        const PointCloudFormat& src45
    );
    
    // Generate depth map from existing depth map (reprojection)
    GeneratedDepthMap generateFromExistingDepthMap(
        const std::vector<cv::Vec3f>& sourceDepthmap,
        int sourceWidth, int sourceHeight,
        const CameraParams& sourceCamera,
        const CameraParams& targetCamera
    );
    
    // Generate depth map with custom transform
    GeneratedDepthMap generateWithTransform(
        const std::vector<cv::Point3f>& points,
        const cv::Mat& transform,
        const std::vector<cv::Point3f>& normals = std::vector<cv::Point3f>(),
        const std::vector<cv::Vec3b>& colors = std::vector<cv::Vec3b>()
    );
    
    // Save generated depth map
    bool saveDepthMap(const GeneratedDepthMap& depthmap, 
                     const std::string& outputPath, 
                     const std::string& prefix = "custom",
                     bool saveBinary = true,
                     bool savePng = false) const;
    
    // Save point cloud from depth map as PLY (world coordinates)
    // Transforms camera coordinates to world coordinates using cameraToWorld transform
    bool savePointCloudPLY(const GeneratedDepthMap& depthmap,
                            const cv::Mat& cameraToWorld,
                            const std::string& filePath) const;
    
    // Get current camera parameters
    const CameraParams& getCameraParams() const { return cameraParams_; }
    
    // Get current projection method
    ProjectionMethod getProjectionMethod() const { return projectionMethod_; }

private:
    CameraParams cameraParams_;
    ProjectionMethod projectionMethod_;
    
    // Internal projection functions
    cv::Point2f projectPoint(const cv::Point3f& point) const;
    bool isPointInBounds(const cv::Point2f& projectedPoint) const;
    bool isPointInDepthRange(float depth) const;
    
    // Normal calculation
    cv::Point3f calculateNormal(const std::vector<cv::Point3f>& points, 
                               const std::vector<cv::Point2f>& projectedPoints,
                               int centerIdx, int width, int height) const;
    
    // Color interpolation
    cv::Vec3b interpolateColor(const std::vector<cv::Vec3b>& colors,
                              const std::vector<cv::Point2f>& projectedPoints,
                              int centerIdx) const;
    
    // CUDA-based projection (if available)
    bool useCUDA_;
    void* cudaDepthMap_;      // CUDA memory for depth map
    void* cudaNormalMap_;     // CUDA memory for normal map
    void* cudaColorMap_;       // CUDA memory for color map
    
    bool initializeCUDA();
    void cleanupCUDA();
    GeneratedDepthMap generateFromPointCloudCUDA(
        const std::vector<cv::Point3f>& points,
        const std::vector<cv::Point3f>& normals,
        const std::vector<cv::Vec3b>& colors
    );
};
