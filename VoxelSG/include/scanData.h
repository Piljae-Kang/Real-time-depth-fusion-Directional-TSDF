#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Note: Avoid including <filesystem> in headers to keep CUDA compilation clean


struct BaseParams {
    int* patch_info;
    int patchID = 0;
    int Tolerance = 0;

    int depthmapInterpolateBaseLevel = 0;
	int depthmapInterpolateLevel = 0;

	float RobustVoxelRatio = 0.f;
	int RobustVoxelCount = 0;

	float xUnit = 1.f;
	float yUnit = 1.f;
	float pointMag = 1.f;

    bool bUseAASF = false;
    

    bool bDeepLearningEnable = false;

};

struct CameraParams {
    float fx = 0.0f;  // Focal length x
    float fy = 0.0f;  // Focal length y
    float cx = 0.0f;  // Principal point x
    float cy = 0.0f;  // Principal point y
    float nearPlane = 0.1f;  // Near clipping plane
    float farPlane = 10.0f;  // Far clipping plane
};

struct ImageParams {
    

    std::vector<short> deeplearning_inference; // depth map data

    std::vector<unsigned char> current_img_0; // RGBA image data (height x width x 4)
    std::vector<unsigned char> current_img_45; // RGBA image data (height x width x 4)

    std::vector<float> confidence_map_0; 
    std::vector<float> confidence_map_45; 
    std::vector<short> depthmap_mask;

};

struct MatrixParams {
    std::vector<cv::Mat> cameraPoses;
    std::vector<cv::Mat> transformationMatrices;
    std::vector<cv::Mat> transform_0;      // Transform_0 matrices (Local -> World, 0 degrees)
    std::vector<cv::Mat> transform_45;   // Transform_45 matrices (Local -> World, 45 degrees)
    std::vector<cv::Mat> localToCamera;   // CameraRT matrices (Local -> Camera)
    std::vector<cv::Mat> cameraToWorld0;  // Camera -> World (for 0 degrees view)
    std::vector<cv::Mat> cameraToWorld45; // Camera -> World (for 45 degrees view)
};

struct DepthMapParams {
    struct DepthMapFrame {
        std::vector<cv::Vec3b> colormap;   // RGB color data (uint8 x3 per pixel)
        std::vector<cv::Vec3f> depthmap;   // 3D positions (float3 per pixel)
        std::vector<cv::Vec3f> normalmap;  // Normals (float3 per pixel)
    };
    
    std::vector<DepthMapFrame> high_resolution;
    std::vector<DepthMapFrame> low_resolution;
};

struct PointCloudFormat {
    
    std::vector<cv::Point3f> points;    // OpenCV type
    std::vector<cv::Point3f> normals;   // OpenCV type
    std::vector<cv::Vec3b> colors;      // OpenCV type
};

struct PointCloudParams {

    PointCloudFormat src_0;
    PointCloudFormat src_45;
};

class ScanDataLoader {
private:
    
    std::string rootPath;

    BaseParams baseParams;
    ImageParams imageParams;
    MatrixParams matrixParams;
    DepthMapParams depthMapParams;
    std::vector<PointCloudParams> pointCloudParams;  // Changed to vector for multiple frames
    CameraParams cameraParams;


    // Load base parameters
    void loadBaseParams();
    
    // Load image parameters
    void loadImageParams();
    
    // Load matrix parameters
    void loadMatrixParams();
    
    // Load depth map parameters (frame_idx only)
    void loadDepthMapParams();
    
    // Load point cloud parameters
    void loadPointCloudParams();
    
    // Load individual point cloud format (with frame index)
    void loadPointCloudFormat(const std::string& formatPath, PointCloudFormat& format, int frameIndex);
    
    template<typename T>
    bool loadBinaryImage(const std::string& folderName, std::vector<T>& targetVector);
    
    // Dynamically check file count
    int countAvailableFrames(const std::string& directoryPath, const std::string& filePattern);

    // Load camera parameters
    void loadCameraParams();
    
public:

    ScanDataLoader(const std::string& path);
    ~ScanDataLoader(); // Destructor added
    

    bool load();
    
    const BaseParams& getBaseParams() const { return baseParams; }
    const ImageParams& getImageParams() const { return imageParams; }
    const MatrixParams& getMatrixParams() const { return matrixParams; }
    const DepthMapParams& getDepthMapParams() const { return depthMapParams; }
    const PointCloudParams& getPointCloudParams() const { return pointCloudParams.empty() ? throw std::runtime_error("No point cloud data loaded") : pointCloudParams[0]; }  // Backward compatibility: returns first frame
    const PointCloudParams& getPointCloudParams(int frameIndex) const { return pointCloudParams.at(frameIndex); }  // Get specific frame
    const std::vector<PointCloudParams>& getAllPointCloudParams() const { return pointCloudParams; }  // Get all frames
    const CameraParams& getCameraParams() const { return cameraParams; }
    
    void printSummary() const;

    // Load individual frame (memory efficient)
    bool loadSingleDepthMapFrame(int frameIndex, const std::string& resolution, DepthMapParams::DepthMapFrame& frame);

    // Save a single depth map frame to disk (writes bin + preview PNG)
    bool saveDepthMapFrame(const std::string& outputDir, const std::string& resolution, int frameIndex) const;

    int frame_idx = 2;

    int width = 400;  int height = 480; // image resolution (img_params)
    int h_depthWidth = 450; int h_depthHeight = 550; // depthmap resolution (different from image)
    int l_depthWidth = 450; int l_depthHeight = 550; // depthmap resolution (different from image)
};


