#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "scanData.h"  // Need full type definitions for member variables

namespace fs = std::filesystem;

class ScanDataLoader2 {


public:

	ScanDataLoader2(std::string dataPath);

	void loadCameraFromFile();

	// Load image parameters
	void loadImageFromFile();

	// Load matrix parameters
	void loadTransformFile();

	// Load point cloud parameters
	void loadPointCloudfromFile();

	// Helper to load point cloud format from PLY file
	void loadPointCloudFormat(const std::string& formatPath, PointCloudFormat& format, int frame_idx, int type);

	int width = 400;  int height = 480; // image resolution (img_params)

	int h_depthWidth = 400; int h_depthHeight = 480;

	// Getter methods (compatible with ScanDataLoader interface)
	const BaseParams& getBaseParams() const { return baseParams_; }
	const ImageParams& getImageParams() const { return imageParams_; }
	const MatrixParams& getMatrixParams() const { return matrixParams; }
	const DepthMapParams& getDepthMapParams() const { return depthMapParams_; }
	const PointCloudParams& getPointCloudParams() const { return pointCloudParams_; }  // Backward compatibility: returns first frame
	const PointCloudParams& getPointCloudParams(int frameIndex) const { return PCDs.at(frameIndex); }  // Get specific frame
	const std::vector<PointCloudParams>& getAllPointCloudParams() const { return PCDs; }  // Get all frames
	const CameraParams& getCameraParams() const { return cameraParams; }
	
	void printSummary() const;


private:

	int frameCount_; // current frame 
	int frameN_; // the total number of images
	int startIdx_;

	std::string dataPath_;

	std::vector<cv::Mat> color_imgs;
	
	CameraParams cameraParams;

	std::vector<PointCloudParams> PCDs;
	MatrixParams matrixParams;
	cv::Mat localToCamera;

	// Empty/default params for compatibility (ScanDataLoader2 only has matrix and point cloud)
	BaseParams baseParams_;
	ImageParams imageParams_;
	DepthMapParams depthMapParams_;
	PointCloudParams pointCloudParams_; // Will be populated from PCDs


};