#include "../include/scanData2.h"  // Already includes scanData.h
#include <open3d/Open3D.h>  // For PLY file loading

ScanDataLoader2::ScanDataLoader2(std::string filepath) {

	dataPath_ = filepath;
	startIdx_ = 0;
	frameN_ = 50;
	//frameN_ = 50; // Must be deleted later

	// Initialize default/empty params for compatibility
	baseParams_.patch_info = nullptr;
	baseParams_.patchID = 0;
	baseParams_.Tolerance = 0;
	baseParams_.depthmapInterpolateBaseLevel = 0;
	baseParams_.depthmapInterpolateLevel = 0;
	baseParams_.RobustVoxelRatio = 0.0f;
	baseParams_.RobustVoxelCount = 0;
	baseParams_.xUnit = 1.0f;
	baseParams_.yUnit = 1.0f;
	baseParams_.pointMag = 1.0f;
	baseParams_.bUseAASF = false;
	baseParams_.bDeepLearningEnable = false;

	// ImageParams and DepthMapParams remain empty (ScanDataLoader2 doesn't load them)

}


void ScanDataLoader2::loadCameraFromFile() {

	localToCamera = cv::Mat::eye(4, 4, CV_32F);

	std::string cameraFileName = dataPath_ + "/../camera.txt";

	FILE* in = fopen(cameraFileName.c_str(), "r");

	int imageWidth, imageHeight;
	float fx, fy, cx, cy;
	bool columnMajor = false;

	fscanf(in, "%d %d\n", &imageWidth, &imageHeight);
	fscanf(in, "%f %f %f  %f", &fx, &fy, &cx, &cy);

	for (int i = 0; i < 16; i++) {

		float value;

		if (columnMajor) {

			fscanf(in, "%f", &value);

			int row = i % 4;
			int col = i / 4;

			localToCamera.at<float>(row, col) = value;
		}
		else {

			fscanf(in, "%f", &value);

			int row = i / 4;
			int col = i % 4;

			localToCamera.at<float>(row, col) = value;

		}
	}

	width = imageWidth; height = imageHeight;
	cameraParams.fx = fx;
	cameraParams.fy = fy;
	cameraParams.cx = cx;
	cameraParams.cy = cy;


	std::cout << "----------camera parameter ----------\n";
	std::cout << cameraParams.fx << " " << cameraParams.fy << std::endl;
	std::cout << cameraParams.cx << " " << cameraParams.cy << std::endl;
	std::cout << width << " x " << height << std::endl;
	std::cout << "--------------------\n";

	std::cout << "---------- localToCamera ----------\n";
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {

			std::cout << localToCamera.at<float>(i, j) << " ";
		}
		std::cout << "\n";
	}
	std::cout << "--------------------\n";

}

inline void readMatFromString (char* buf, cv::Mat& transform) {

	float a[16];
	sscanf(buf, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f",
		&a[0], &a[4], &a[8], &a[12],
		&a[1], &a[5], &a[9], &a[13],
		&a[2], &a[6], &a[10], &a[14],
		&a[3], &a[7], &a[11], &a[15]);


	//printf(" buf : %s\n", buf);
	std::memcpy(transform.data, a, transform.total() * transform.elemSize());
	//transform = cv::Mat(4, 4, CV_32F, a);


}


void ScanDataLoader2::loadTransformFile() {


	std::string transformFileName = dataPath_ + "/transform_0.txt";

	// Check if LocalToCamera is actually loaded
	bool isLocalToCameraLoaded = false;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			if (i == j && abs(localToCamera.at<float>(i, j) - 1.0f) > 1e-6) {
				isLocalToCameraLoaded = true;
				break;
			}
			else if (i != j && abs(localToCamera.at<float>(i, j)) > 1e-6) {
				isLocalToCameraLoaded = true;
				break;
			}
		}
		if (isLocalToCameraLoaded) break;
	}

	std::cout << "LocalToCamera loaded: " << (isLocalToCameraLoaded ? "YES" : "NO") << std::endl;


	FILE* in = fopen(transformFileName.c_str(), "r");

	char* buf1 = new char[1000];
	char* buf2 = new char[100];
	int frameIdx, rawIdx;

	if (in == NULL)
		printf("file at %s is not loaded correctly. ", buf1);

	cv::Mat transform_0 = cv::Mat(4, 4, CV_32F);
	cv::Mat transform_45 = cv::Mat(4, 4, CV_32F);

	while (!feof(in)) {

		// READ index mapping line
		fgets(buf1, 1000, in);
		sscanf(buf1, "%d %s", &frameIdx, buf2);
		char* tok = strtok(buf2, ":");
		if (tok != NULL) tok = strtok(NULL, ":");
		rawIdx = atoi(tok);

		//DEBUG
		printf("%d %d\n", frameIdx, rawIdx);

		// READ matrix for 0 degree
		fgets(buf1, 1000, in);
		fgets(buf1, 1000, in);
		readMatFromString(buf1, transform_0);

		// READ matrix for 45 degree
		fgets(buf1, 1000, in);
		fgets(buf1, 1000, in);
		readMatFromString(buf1, transform_45);

		matrixParams.transform_0.push_back(transform_0.clone());
		matrixParams.transform_45.push_back(transform_45.clone());
		matrixParams.localToCamera.push_back(localToCamera.clone());


		if (isLocalToCameraLoaded) {
			cv::Mat localToCameraInv = localToCamera.inv(cv::DECOMP_SVD);
			std::cout << "LocalToCamera^(-1):" << std::endl;
			std::cout << localToCameraInv << std::endl;

			std::cout << "transform_0 :" << std::endl;
			std::cout << transform_0 << std::endl;

			cv::Mat cameraToWorld0 = transform_0 * localToCameraInv;
			std::cout << "CameraToWorld0 = Transform_0 * CameraRT^(-1):" << std::endl;
			std::cout << cameraToWorld0 << std::endl;

			cv::Mat cameraToWorld45 = transform_45 * localToCameraInv;

			matrixParams.cameraToWorld0.push_back(cameraToWorld0);
			matrixParams.cameraToWorld45.push_back(cameraToWorld45);
		}
		else {
			std::cout << "WARNING: LocalToCamera is identity, using Transform_0/45 directly" << std::endl;
			matrixParams.cameraToWorld0.push_back(transform_0);
			matrixParams.cameraToWorld45.push_back(transform_45);
		}
		std::cout << "========================\n" << std::endl;

	}


	delete[] buf1;
	delete[] buf2;

}


void ScanDataLoader2::loadImageFromFile() {

	// ScanDataLoader2 doesn't load image params, so this is a placeholder
	std::cout << "loadImageFromFile: ScanDataLoader2 doesn't load image parameters" << std::endl;

}


void ScanDataLoader2::loadPointCloudfromFile() {

	std::string pointCloudParamsPath = dataPath_;

	if (!fs::exists(pointCloudParamsPath)) {
		std::cout << "Warning: pointColud_params directory not found" << std::endl;
		return;
	}

	std::cout << "Loading point cloud parameters..." << std::endl;

	for (int i = 0; i < frameN_; i++) {

		PointCloudParams pointCloudFrame;

		// Load src_0_mesh point cloud (type 0)
		loadPointCloudFormat(pointCloudParamsPath, pointCloudFrame.src_0, i, 0);

		// Load src_45_mesh point cloud (type 1)
		loadPointCloudFormat(pointCloudParamsPath, pointCloudFrame.src_45, i, 1);

		std::cout << "Loaded frame " << i << " - src_0: " << pointCloudFrame.src_0.points.size() 
		          << " points, src_45: " << pointCloudFrame.src_45.points.size() << " points" << std::endl;

		PCDs.push_back(pointCloudFrame);
	}

	// Set pointCloudParams_ from first frame if available (for compatibility)
	if (!PCDs.empty()) {
		pointCloudParams_ = PCDs[0];
	}

	std::cout << "PCDs is loaded !! size : " << PCDs.size() << "\n";

}

void ScanDataLoader2::loadPointCloudFormat(const std::string& formatPath, PointCloudFormat& format, int frame_idx, int type) {
	char filename[32];
	std::string plyPath = "";

	if (type == 0) {
		sprintf_s(filename, sizeof(filename), "%04dpatch0.ply", frame_idx);
		plyPath = formatPath + "/" + filename;
	}
	else {
		sprintf_s(filename, sizeof(filename), "%04dpatch45.ply", frame_idx);
		plyPath = formatPath + "/" + filename;
	}



	if (!fs::exists(plyPath)) {
		std::cout << "Warning: File not found: " << plyPath << std::endl;
		return;
	}

	auto pcd = open3d::io::CreatePointCloudFromFile(plyPath);
	if (!pcd || pcd->points_.empty()) {
		std::cout << "Warning: Failed to read or empty PLY file: " << plyPath << std::endl;
		return;
	}

	size_t pointCount = pcd->points_.size();
	format.points.resize(pointCount);
	format.normals.resize(pointCount);
	format.colors.resize(pointCount);

	// Transform points from local to camera coordinates using localToCamera matrix
	for (size_t i = 0; i < pointCount; ++i) {
		auto& p = pcd->points_[i];
		
		// Original point in local coordinates
		cv::Point3f localPoint(p(0), p(1), p(2));
		
		// Transform point from local to camera coordinates
		cv::Mat pointHomogeneous = (cv::Mat_<float>(4, 1) << 
			localPoint.x,
			localPoint.y,
			localPoint.z,
			1.0f);
		
		cv::Mat cameraPoint = localToCamera * pointHomogeneous;
		format.points[i] = cv::Point3f(
			cameraPoint.at<float>(0),
			cameraPoint.at<float>(1),
			cameraPoint.at<float>(2)
		);

		// Transform normal (only rotation, no translation)
		if (!pcd->normals_.empty()) {
			auto& n = pcd->normals_[i];
			cv::Point3f localNormal(n(0), n(1), n(2));
			
			// Normal transformation (only rotation part)
			cv::Mat normalHomogeneous = (cv::Mat_<float>(4, 1) << 
				localNormal.x,
				localNormal.y,
				localNormal.z,
				0.0f);  // w=0 for vectors
			
			cv::Mat cameraNormal = localToCamera * normalHomogeneous;
			cv::Point3f transformedNormal(
				cameraNormal.at<float>(0),
				cameraNormal.at<float>(1),
				cameraNormal.at<float>(2)
			);
			
			// Normalize the transformed normal
			float len = sqrtf(transformedNormal.x * transformedNormal.x + 
			                  transformedNormal.y * transformedNormal.y + 
			                  transformedNormal.z * transformedNormal.z);
			if (len > 1e-6f) {
				format.normals[i] = cv::Point3f(
					transformedNormal.x / len,
					transformedNormal.y / len,
					transformedNormal.z / len
				);
			} else {
				format.normals[i] = cv::Point3f(0, 0, 0);
			}
		}
		else {
			format.normals[i] = cv::Point3f(0, 0, 0);
		}

		// Colors don't need transformation
		if (!pcd->colors_.empty()) {
			auto& c = pcd->colors_[i];
			format.colors[i] = cv::Vec3b(
				static_cast<unsigned char>(c(2) * 255),
				static_cast<unsigned char>(c(1) * 255),
				static_cast<unsigned char>(c(0) * 255));
		}
		else {
			format.colors[i] = cv::Vec3b(255, 255, 255);
		}
	}

	std::cout << "Loaded PLY point cloud: " << pointCount << " points from " << plyPath << std::endl;
	std::cout << "  Transformed from local to camera coordinates using localToCamera matrix" << std::endl;
}

void ScanDataLoader2::printSummary() const {
	std::cout << "\n=== Scan Data Summary (ScanDataLoader2) ===" << std::endl;
	std::cout << "Note: ScanDataLoader2 only loads Matrix and Point Cloud data" << std::endl;
	
	std::cout << "Matrix Parameters:" << std::endl;
	std::cout << "  - Transform_0 matrices: " << matrixParams.transform_0.size() << std::endl;
	std::cout << "  - Transform_45 matrices: " << matrixParams.transform_45.size() << std::endl;
	std::cout << "  - LocalToCamera matrices: " << matrixParams.localToCamera.size() << std::endl;
	std::cout << "  - CameraToWorld0 matrices: " << matrixParams.cameraToWorld0.size() << std::endl;
	std::cout << "  - CameraToWorld45 matrices: " << matrixParams.cameraToWorld45.size() << std::endl;
	
	std::cout << "Point Cloud Parameters:" << std::endl;
	std::cout << "  - PCDs frames loaded: " << PCDs.size() << std::endl;
	if (!pointCloudParams_.src_0.points.empty()) {
		std::cout << "  - src_0 Points: " << pointCloudParams_.src_0.points.size() << std::endl;
		std::cout << "  - src_0 Normals: " << pointCloudParams_.src_0.normals.size() << std::endl;
		std::cout << "  - src_0 Colors: " << pointCloudParams_.src_0.colors.size() << std::endl;
	}
	if (!pointCloudParams_.src_45.points.empty()) {
		std::cout << "  - src_45 Points: " << pointCloudParams_.src_45.points.size() << std::endl;
		std::cout << "  - src_45 Normals: " << pointCloudParams_.src_45.normals.size() << std::endl;
		std::cout << "  - src_45 Colors: " << pointCloudParams_.src_45.colors.size() << std::endl;
	}
	
	std::cout << "Camera Parameters:" << std::endl;
	std::cout << "  - fx: " << cameraParams.fx << ", fy: " << cameraParams.fy << std::endl;
	std::cout << "  - cx: " << cameraParams.cx << ", cy: " << cameraParams.cy << std::endl;
	std::cout << "  - Image Size: " << width << "x" << height << std::endl;
}
