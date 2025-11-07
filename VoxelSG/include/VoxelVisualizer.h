#pragma once

#include "CustomDepthMapGenerator.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <open3d/Open3D.h>
#include <Eigen/Dense>


// Open3D test
void TestOpen3D();

// Point cloud visualize function
void VisualizePointCloudWithOpen3D(const std::vector<cv::Point3f>& points,
                                  const std::vector<cv::Point3f>& normals = {},
                                  const std::vector<cv::Vec3b>& colors = {},
                                  const std::string& windowName = "Point Cloud");

void VisualizeScanDataPointCloud(const std::string& scanDataPath, int frameIdx = 0);

// Depth map visualization with OpenCV
void VisualizeDepthMapWithOpenCV(const class ScanDataLoader& loader, const std::string& resolution, int frameIndex);

// Depth map to 3D point cloud conversion and visualization
void VisualizeDepthMapAsPointCloud(const class ScanDataLoader& loader, const std::string& resolution, int frameIndex);

// Custom depth map visualization
void VisualizeCustomDepthMap(const class CustomDepthMapGenerator::GeneratedDepthMap& depthmap);

// Transform and visualize PCD in World coordinates
void TransformAndVisualizePCDInWorld(const class ScanDataLoader& loader, int frameIdx = 0);

// Transform and visualize PCD in World coordinates (for ScanDataLoader2)
void TransformAndVisualizePCDInWorld(const class ScanDataLoader2& loader, int frameIndex = 0);

// Show rendered depth (from RayCastRender::downloadResults) using OpenCV
struct float4; // forward declaration for CUDA float4
void VisualizeRenderedDepthImShow(const float4* depthOut, int width, int height, const std::string& windowName = "Rendered Depth");

// Show 1-channel float depth directly (depth pointer points to width*height floats)
void VisualizeRenderedDepthFloat(const float* depthOut, int width, int height, const std::string& windowName = "Rendered Depth (1ch)");

//// Voxel grid visualization
//cv::Mat VisualizeVoxelGrid(const std::vector<cv::Point3f>& voxels, 
//                           int gridSize = 512, 
//                           const std::string& windowName = "Voxel Grid");
//
//// Point cloud visualization  
//cv::Mat VisualizePointCloud(const std::vector<cv::Point3f>& points,
//                           const std::vector<cv::Vec3b>& colors = {},
//                           const std::string& windowName = "Point Cloud");
//
 //2D map visualization  
//cv::mat visualize2dmap(const std::vector<cv::point2f>& points,
//                       const std::string& windowname = "2d map");

//// Mesh rendering
//cv::Mat VisualizeMesh(const std::vector<cv::Point3f>& vertices,
//                      const std::vector<cv::Vec3b>& faces,
//                      const std::string& windowName = "Mesh");