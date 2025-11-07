#include "../include/VoxelVisualizer.h"
#include "../include/scanData.h"
#include "../include/scanData2.h"
#include "../include/CustomDepthMapGenerator.h"
#include <limits>
#include <algorithm>
#include <cmath>
#include <cfloat>  // For FLT_MAX
#include <cuda_runtime.h> // for float4

void TestOpen3D() {
    std::cout << "Open3D test started..." << std::endl;
    
    // Create a simple test point cloud
    std::vector<cv::Point3f> testPoints;
    std::vector<cv::Vec3b> testColors;
    
    // Create a cube pattern for testing
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
                testPoints.push_back(cv::Point3f(i * 0.1f, j * 0.1f, k * 0.1f));
                testColors.push_back(cv::Vec3b(i * 25, j * 25, k * 25));
            }
        }
    }
    
    std::cout << "Test point cloud created: " << testPoints.size() << " points" << std::endl;
    
    // Use the same visualization function
    VisualizePointCloudWithOpen3D(testPoints, {}, testColors, "Test_PointCloud");
}

void VisualizePointCloudWithOpen3D(const std::vector<cv::Point3f>& points,
                                  const std::vector<cv::Point3f>& normals,
                                  const std::vector<cv::Vec3b>& colors,
                                  const std::string& windowName) {
    
    if (points.empty()) {
        std::cout << "Warning: Point cloud is empty!" << std::endl;
        return;
    }
    
    std::cout << "Starting Open3D point cloud visualization..." << std::endl;
    std::cout << "Point count: " << points.size() << std::endl;
    
    // Create Open3D PointCloud object
    auto pointcloud = std::make_shared<open3d::geometry::PointCloud>();
    
    // Add points and check for valid data
    bool hasValidPoints = false;
    int invalidCount = 0;
    std::vector<int> validIndices;
    
    for (size_t i = 0; i < points.size(); i++) {
        const auto& point = points[i];
        // Check for NaN, infinite values, or extremely large values
        if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
            std::abs(point.x) < 1e6 && std::abs(point.y) < 1e6 && std::abs(point.z) < 1e6) {

            //printf("%f %f %f\n", point.x, point.y, point.z);

            pointcloud->points_.push_back(Eigen::Vector3d(point.x, point.y, point.z));
            validIndices.push_back(static_cast<int>(i));
            hasValidPoints = true;
        } else {
            invalidCount++;
        }
    }
    
    if (invalidCount > 0) {
        std::cout << "WARNING: " << invalidCount << " invalid points filtered out!" << std::endl;
    }
    
    if (!hasValidPoints) {
        std::cout << "ERROR: No valid points found in the point cloud!" << std::endl;
        return;
    }
    
    std::cout << "Valid points added: " << pointcloud->points_.size() << std::endl;
    
    // Add normals (if available) - only for valid points
    if (!normals.empty() && normals.size() == points.size()) {
        std::cout << "Adding normal vectors..." << std::endl;
        for (int idx : validIndices) {
            const auto& normal = normals[idx];
            if (std::isfinite(normal.x) && std::isfinite(normal.y) && std::isfinite(normal.z)) {
                pointcloud->normals_.push_back(Eigen::Vector3d(normal.x, normal.y, normal.z));
            } else {
                pointcloud->normals_.push_back(Eigen::Vector3d(0, 0, 1)); // Default normal
            }
        }
    }
    
    // Add colors (if available) - only for valid points
    if (!colors.empty() && colors.size() == points.size()) {
        std::cout << "Adding color information..." << std::endl;
        for (int idx : validIndices) {
            const auto& color = colors[idx];
            pointcloud->colors_.push_back(Eigen::Vector3d(color[0] / 255.0, 
                                                         color[1] / 255.0, 
                                                         color[2] / 255.0));
        }
    } else {
        // Apply height-based coloring if no colors
        std::cout << "Applying height-based coloring..." << std::endl;
        double minZ = 1e10;
        double maxZ = -1e10;
        
        // Find min/max Z from valid points only
        for (const auto& point : pointcloud->points_) {
            if (point.z() < minZ) minZ = point.z();
            if (point.z() > maxZ) maxZ = point.z();
        }
        
        // Apply coloring to valid points only
        for (const auto& point : pointcloud->points_) {
            double normalizedZ = (point.z() - minZ) / (maxZ - minZ);
            // Blue to red gradient
            pointcloud->colors_.push_back(Eigen::Vector3d(normalizedZ, 0.5, 1.0 - normalizedZ));
        }
    }
    
    // Save as PLY file
    std::string filename = windowName + "_visualization.ply";
    if (open3d::io::WritePointCloud(filename, *pointcloud)) {
        std::cout << "PLY file saved: " << filename << std::endl;
    } else {
        std::cout << "PLY file save failed!" << std::endl;
    }
    
    // Debug: Check point cloud bounds
    if (!pointcloud->points_.empty()) {
        Eigen::Vector3d minBound = pointcloud->GetMinBound();
        Eigen::Vector3d maxBound = pointcloud->GetMaxBound();
        std::cout << "Point cloud bounds:" << std::endl;
        std::cout << "  Min: (" << minBound.x() << ", " << minBound.y() << ", " << minBound.z() << ")" << std::endl;
        std::cout << "  Max: (" << maxBound.x() << ", " << maxBound.y() << ", " << maxBound.z() << ")" << std::endl;
        
        // Check if points are too small or too far
        double range = (maxBound - minBound).norm();
        std::cout << "  Range: " << range << std::endl;
        
        if (range < 1e-6) {
            std::cout << "WARNING: Point cloud range is very small, might not be visible!" << std::endl;
        } else if (range > 1000.0) {
            std::cout << "WARNING: Point cloud range is very large, might be hard to see!" << std::endl;
        }
        
        // Estimate point size for better visibility
        double estimatedRadius = range * 0.01; // 1% of range
        std::cout << "Estimated point radius: " << estimatedRadius << std::endl;
        
        // Normalize point cloud if range is too extreme
        if (range < 1e-6 || range > 1000.0) {
            std::cout << "Normalizing point cloud for better visualization..." << std::endl;
            Eigen::Vector3d center = pointcloud->GetCenter();
            pointcloud->Translate(-center);
            pointcloud->Scale(1.0 / range, center);
        }
    }
    
    // Open3D visualization with better settings
    std::cout << "Opening Open3D visualization window..." << std::endl;
    
    // Create visualization parameters
    open3d::visualization::Visualizer visualizer;
    visualizer.CreateVisualizerWindow(windowName, 1200, 800);
    visualizer.AddGeometry(pointcloud);
    
    // Set rendering options for better visibility
    auto render_option = visualizer.GetRenderOption();
    render_option.point_size_ = 3.0;  // Increase point size
    render_option.background_color_ = Eigen::Vector3d(0.1, 0.1, 0.1);  // Dark background
    
    visualizer.Run();
    visualizer.DestroyVisualizerWindow();
    
    std::cout << "Point cloud visualization completed!" << std::endl;
}

void VisualizeScanDataPointCloud(const std::string& scanDataPath, int frameIdx) {
    std::cout << "=== VoxelSG Point Cloud Visualization ===" << std::endl;
    std::cout << "Scan data path: " << scanDataPath << std::endl;
    std::cout << "Frame index: " << frameIdx << std::endl;
    
    // Load data with ScanDataLoader
    ScanDataLoader loader(scanDataPath);
    loader.frame_idx = frameIdx;
    
    if (!loader.load()) {
        std::cerr << "Failed to load scan data!" << std::endl;
        return;
    }
    
    // Print loaded data summary
    loader.printSummary();
    
    const auto& pointCloudParams = loader.getPointCloudParams();
    
    // Visualize src_0 point cloud
    if (!pointCloudParams.src_0.points.empty()) {
        std::cout << "\n=== src_0 Point Cloud Visualization ===" << std::endl;
        VisualizePointCloudWithOpen3D(pointCloudParams.src_0.points,
                                     pointCloudParams.src_0.normals,
                                     pointCloudParams.src_0.colors,
                                     "VoxelSG_src_0");
    } else {
        std::cout << "src_0 point cloud is empty." << std::endl;
    }
    
    // Visualize src_45 point cloud
    if (!pointCloudParams.src_45.points.empty()) {
        std::cout << "\n=== src_45 Point Cloud Visualization ===" << std::endl;
        VisualizePointCloudWithOpen3D(pointCloudParams.src_45.points,
                                     pointCloudParams.src_45.normals,
                                     pointCloudParams.src_45.colors,
                                     "VoxelSG_src_45");
    } else {
        std::cout << "src_45 point cloud is empty." << std::endl;
    }
    
    std::cout << "\n=== All Point Cloud Visualization Completed ===" << std::endl;
}

void VisualizeDepthMapWithOpenCV(const ScanDataLoader& loader, const std::string& resolution, int frameIndex) {
    std::cout << "=== Depth Map OpenCV Visualization ===" << std::endl;
    std::cout << "Resolution: " << resolution << ", Frame: " << frameIndex << std::endl;
    
    const auto& depthMapParams = loader.getDepthMapParams();
    const std::vector<DepthMapParams::DepthMapFrame>* frames = nullptr;
    
    if (resolution == "high_resolution") {
        frames = &depthMapParams.high_resolution;
    } else if (resolution == "low_resolution") {
        frames = &depthMapParams.low_resolution;
    } else {
        std::cout << "Error: Invalid resolution. Use 'high_resolution' or 'low_resolution'" << std::endl;
        return;
    }
    
    if (frames->empty()) {
        std::cout << "Error: No frames loaded for " << resolution << std::endl;
        return;
    }
    
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames->size())) {
        std::cout << "Error: Frame index " << frameIndex << " out of range [0, " << frames->size() - 1 << "]" << std::endl;
        return;
    }
    
    const auto& frame = (*frames)[frameIndex];
    
    // Get actual depth map dimensions based on resolution
    int actualWidth, actualHeight;
    if (resolution == "high_resolution") {
        actualWidth = loader.h_depthWidth;
        actualHeight = loader.h_depthHeight;
    } else if (resolution == "low_resolution") {
        actualWidth = loader.l_depthWidth;
        actualHeight = loader.l_depthHeight;
    } else {
        std::cout << "Error: Unknown resolution type!" << std::endl;
        return;
    }

    std::cout << "Actual Resolution : " << actualHeight << " x " << actualWidth << "\n";
    
    // Visualize depth map (Z channel)
    if (!frame.depthmap.empty()) {
        std::cout << "Visualizing depth map (Z channel)..." << std::endl;
        
        // Create depth image from Z channel
        cv::Mat depthImage(actualHeight, actualWidth, CV_32FC1);
        for (int y = 0; y < actualHeight; y++) {
            for (int x = 0; x < actualWidth; x++) {
                int idx = y * actualWidth + x;
                if (idx < static_cast<int>(frame.depthmap.size())) {
                    depthImage.at<float>(y, x) = frame.depthmap[idx][2]; // Z channel (index 2)
                } else {
                    depthImage.at<float>(y, x) = 0.0f;
                }
            }
        }
        
        // Normalize for visualization
        double minVal, maxVal;
        cv::minMaxLoc(depthImage, &minVal, &maxVal);
        
        cv::Mat depthNormalized;
        if (maxVal > minVal) {
            depthImage.convertTo(depthNormalized, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        } else {
            depthNormalized = cv::Mat::zeros(actualHeight, actualWidth, CV_8U);
        }
        
        // Apply colormap for better visualization
        cv::Mat depthColored;
        cv::applyColorMap(depthNormalized, depthColored, cv::COLORMAP_JET);
        
        // Show depth map
        std::string depthWindowName = "Depth Map - " + resolution + " Frame " + std::to_string(frameIndex);
        cv::imshow(depthWindowName, depthColored);
        cv::waitKey(1); // Non-blocking wait
        
        std::cout << "Depth range: [" << minVal << ", " << maxVal << "]" << std::endl;
        std::cout << "Press any key in the depth map window to continue..." << std::endl;
    }
    
    // Visualize color map
    if (!frame.colormap.empty()) {
        std::cout << "Visualizing color map..." << std::endl;
        
        // Create color image
        cv::Mat colorImage(actualHeight, actualWidth, CV_8UC3);
        for (int y = 0; y < actualHeight; y++) {
            for (int x = 0; x < actualWidth; x++) {
                int idx = y * actualWidth + x;
                if (idx < static_cast<int>(frame.colormap.size())) {
                    const auto& color = frame.colormap[idx];
                    colorImage.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]); // BGR for OpenCV
                } else {
                    colorImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }
        
        // Show color map
        std::string colorWindowName = "Color Map - " + resolution + " Frame " + std::to_string(frameIndex);
        cv::imshow(colorWindowName, colorImage);
        cv::waitKey(1); // Non-blocking wait
        
        std::cout << "Press any key in the color map window to continue..." << std::endl;
    }
    
    // Visualize normal map (as RGB)
    if (!frame.normalmap.empty()) {
        std::cout << "Visualizing normal map..." << std::endl;
        
        // Create normal image (normalize to [0, 255])
        cv::Mat normalImage(actualHeight, actualWidth, CV_8UC3);
        for (int y = 0; y < actualHeight; y++) {
            for (int x = 0; x < actualWidth; x++) {
                int idx = y * actualWidth + x;
                if (idx < static_cast<int>(frame.normalmap.size())) {
                    const auto& normal = frame.normalmap[idx];
                    // Normalize normal vector to [0, 255] range
                    cv::Vec3b normalColor(
                        static_cast<unsigned char>((normal[0] + 1.0f) * 127.5f),
                        static_cast<unsigned char>((normal[1] + 1.0f) * 127.5f),
                        static_cast<unsigned char>((normal[2] + 1.0f) * 127.5f)
                    );
                    normalImage.at<cv::Vec3b>(y, x) = normalColor;
                } else {
                    normalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(127, 127, 127); // Neutral gray
                }
            }
        }
        
        // Show normal map
        std::string normalWindowName = "Normal Map - " + resolution + " Frame " + std::to_string(frameIndex);
        cv::imshow(normalWindowName, normalImage);
        cv::waitKey(1); // Non-blocking wait
        
        std::cout << "Press any key in the normal map window to continue..." << std::endl;
    }
    
    std::cout << "All depth map visualizations are ready. Press any key in any window to close all windows..." << std::endl;
    cv::waitKey(0); // Wait for key press
    cv::destroyAllWindows(); // Close all windows
    
    std::cout << "Depth map visualization completed!" << std::endl;
}

void VisualizeDepthMapAsPointCloud(const ScanDataLoader& loader, const std::string& resolution, int frameIndex) {
    std::cout << "=== Depth Map to 3D Point Cloud Visualization ===" << std::endl;
    std::cout << "Resolution: " << resolution << ", Frame: " << frameIndex << std::endl;
    
    const auto& depthMapParams = loader.getDepthMapParams();
    const auto& cameraParams = loader.getCameraParams();
    const std::vector<DepthMapParams::DepthMapFrame>* frames = nullptr;
    
    if (resolution == "high_resolution") {
        frames = &depthMapParams.high_resolution;
    } else if (resolution == "low_resolution") {
        frames = &depthMapParams.low_resolution;
    } else {
        std::cout << "Error: Invalid resolution. Use 'high_resolution' or 'low_resolution'" << std::endl;
        return;
    }
    
    if (frames->empty()) {
        std::cout << "Error: No frames loaded for " << resolution << std::endl;
        return;
    }
    
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames->size())) {
        std::cout << "Error: Frame index " << frameIndex << " out of range [0, " << frames->size() - 1 << "]" << std::endl;
        return;
    }
    
    const auto& frame = (*frames)[frameIndex];
    
    if (frame.depthmap.empty()) {
        std::cout << "Error: Depth map is empty!" << std::endl;
        return;
    }
    
    // Get actual depth map dimensions based on resolution
    int actualWidth, actualHeight;
    if (resolution == "high_resolution") {
        actualWidth = loader.h_depthWidth;
        actualHeight = loader.h_depthHeight;
    } else if (resolution == "low_resolution") {
        actualWidth = loader.l_depthWidth;
        actualHeight = loader.l_depthHeight;
    } else {
        std::cout << "Error: Unknown resolution type!" << std::endl;
        return;
    }
    
    std::cout << "Converting depth map to 3D point cloud..." << std::endl;
    std::cout << "Depth map already contains 3D world coordinates - no projection needed!" << std::endl;
    std::cout << "Current depth map size: " << actualWidth << "x" << actualHeight << std::endl;
    
    // Convert depth map to 3D points
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> colors3D;
    std::vector<cv::Point3f> normals3D;
    
    int validPointCount = 0;
    int invalidPointCount = 0;

    cv::Mat mask = cv::Mat::zeros(actualHeight, actualWidth, CV_8U);
    
    for (int y = 0; y < actualHeight; y++) {
        for (int x = 0; x < actualWidth; x++) {
            int idx = y * actualWidth + x;
            
            if (idx >= static_cast<int>(frame.depthmap.size())) {
                continue;
            }
            
            const auto& depthPoint = frame.depthmap[idx];
            
            // Check if depth point is valid (not zero, invalid, or FLT_MAX)
            //float depth = depthPoint[2];
            //if (depth <= 0.0f || 
            //    !std::isfinite(depth) || 
            //    depth >= (FLT_MAX * 0.9f) ||  // FLT_MAX or near-FLT_MAX indicates invalid/no data
            //    depth > 1000.0f) {  // Reasonable depth limit
            //    invalidPointCount++;
            //    continue;
            //}

            float depth = depthPoint[2];
            if (depth >= (FLT_MAX * 0.9f) || depth == 0) {  // Reasonable depth limit
                invalidPointCount++;
                continue;
            }
            
            // Depth map already contains 3D world coordinates (x,y,z)
            // No need for reverse projection - use directly
            float X = depthPoint[0];  // World X coordinate
            float Y = depthPoint[1];  // World Y coordinate
            float Z = depthPoint[2];  // World Z coordinate
            
            // Check if the 3D coordinates are valid
            if (std::isfinite(X) && std::isfinite(Y) && std::isfinite(Z) &&
                std::abs(X) < 1000.0f && std::abs(Y) < 1000.0f && std::abs(Z) < 1000.0f) {
                
                points3D.push_back(cv::Point3f(X, Y, Z));
                validPointCount++;

                mask.at<uchar>(y, x) = 255;  // Valid pixel marked as white
                
                // Add color if available
                if (idx < static_cast<int>(frame.colormap.size())) {
                    const auto& color = frame.colormap[idx];
                    colors3D.push_back(cv::Vec3b(color[0], color[1], color[2]));
                } else {
                    // Default color based on depth
                    float normalizedDepth = (Z - 0.1f) / (10.0f - 0.1f);
                    normalizedDepth = fmaxf(0.0f, fminf(1.0f, normalizedDepth));
                    colors3D.push_back(cv::Vec3b(
                        static_cast<unsigned char>(normalizedDepth * 255),
                        static_cast<unsigned char>((1.0f - normalizedDepth) * 255),
                        128
                    ));
                }
                
                // Add normal if available
                if (idx < static_cast<int>(frame.normalmap.size())) {
                    const auto& normal = frame.normalmap[idx];
                    normals3D.push_back(cv::Point3f(normal[0], normal[1], normal[2]));
                } else {
                    // Default normal (pointing towards camera)
                    normals3D.push_back(cv::Point3f(0.0f, 0.0f, -1.0f));
                }
            } else {
                invalidPointCount++;
            }
        }
    }

    // Display mask image showing valid depth pixels
    std::cout << "Displaying mask image (white = valid depth pixels)..." << std::endl;
    cv::imshow("Depth Mask - Valid Pixels", mask);
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    std::cout << "Point cloud conversion completed!" << std::endl;
    std::cout << "  Valid points: " << validPointCount << std::endl;
    std::cout << "  Invalid points: " << invalidPointCount << std::endl;
    std::cout << "  Total pixels: " << actualWidth * actualHeight << std::endl;
    
    if (points3D.empty()) {
        std::cout << "Error: No valid 3D points generated!" << std::endl;
        return;
    }
    
    // Calculate point cloud statistics
    float minX = 1e10f, maxX = -1e10f;
    float minY = 1e10f, maxY = -1e10f;
    float minZ = 1e10f, maxZ = -1e10f;
    
    for (const auto& point : points3D) {
        minX = fminf(minX, point.x);
        maxX = fmaxf(maxX, point.x);
        minY = fminf(minY, point.y);
        maxY = fmaxf(maxY, point.y);
        minZ = fminf(minZ, point.z);
        maxZ = fmaxf(maxZ, point.z);
    }
    
    std::cout << "Point cloud bounds:" << std::endl;
    std::cout << "  X: [" << minX << ", " << maxX << "] (range: " << (maxX - minX) << ")" << std::endl;
    std::cout << "  Y: [" << minY << ", " << maxY << "] (range: " << (maxY - minY) << ")" << std::endl;
    std::cout << "  Z: [" << minZ << ", " << maxZ << "] (range: " << (maxZ - minZ) << ")" << std::endl;
    
    // Visualize with Open3D
    std::string windowName = "DepthMap_PointCloud_" + resolution + "_Frame" + std::to_string(frameIndex);
    VisualizePointCloudWithOpen3D(points3D, normals3D, colors3D, windowName);
    
    std::cout << "Depth map to point cloud visualization completed!" << std::endl;
}

void VisualizeCustomDepthMap(const CustomDepthMapGenerator::GeneratedDepthMap& depthmap) {
    std::cout << "Visualizing custom generated depth map..." << std::endl;
    std::cout << "Resolution: " << depthmap.width << "x" << depthmap.height << std::endl;
    
    if (depthmap.depthmap.empty()) {
        std::cout << "Error: Empty depth map!" << std::endl;
        return;
    }
    
    // Create depth image (Z channel)
    cv::Mat depthImage(depthmap.height, depthmap.width, CV_32FC1);
    cv::Mat colorImage(depthmap.height, depthmap.width, CV_8UC3);
    cv::Mat normalImage(depthmap.height, depthmap.width, CV_8UC3);
    
    // Find depth range for normalization
    float minDepth = FLT_MAX;
    float maxDepth = -FLT_MAX;
    int validPixels = 0;
    
    for (int y = 0; y < depthmap.height; y++) {
        for (int x = 0; x < depthmap.width; x++) {
            int idx = y * depthmap.width + x;
            float depthValue = depthmap.depthmap[idx];
            
            if (depthValue > 0.0f && depthValue < FLT_MAX) {
                depthImage.at<float>(y, x) = depthValue;
                minDepth = fminf(minDepth, depthValue);
                maxDepth = fmaxf(maxDepth, depthValue);
                validPixels++;
                
                // Set color
                if (idx < static_cast<int>(depthmap.colormap.size())) {
                    const cv::Vec3b& color = depthmap.colormap[idx];
                    colorImage.at<cv::Vec3b>(y, x) = color;
                }
                
                // Set normal (convert to RGB visualization)
                if (idx < static_cast<int>(depthmap.normalmap.size())) {
                    const cv::Vec3f& normal = depthmap.normalmap[idx];
                    cv::Vec3b normalColor(
                        static_cast<unsigned char>((normal[0] + 1.0f) * 127.5f),
                        static_cast<unsigned char>((normal[1] + 1.0f) * 127.5f),
                        static_cast<unsigned char>((normal[2] + 1.0f) * 127.5f)
                    );
                    normalImage.at<cv::Vec3b>(y, x) = normalColor;
                }
            } else {
                depthImage.at<float>(y, x) = 0.0f;
                colorImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                normalImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    std::cout << "Valid pixels: " << validPixels << "/" << (depthmap.width * depthmap.height) 
              << " (" << (100.0f * validPixels / (depthmap.width * depthmap.height)) << "%)" << std::endl;
    std::cout << "Depth range: " << minDepth << " - " << maxDepth << std::endl;
    
    // Normalize depth image for visualization
    cv::Mat depthNormalized;
    if (maxDepth > minDepth) {
        depthImage.convertTo(depthNormalized, CV_8U, 255.0f / (maxDepth - minDepth), -minDepth * 255.0f / (maxDepth - minDepth));
    } else {
        depthNormalized = cv::Mat::zeros(depthmap.height, depthmap.width, CV_8U);
    }
    
    // Apply color map to depth
    cv::Mat depthColored;
    cv::applyColorMap(depthNormalized, depthColored, cv::COLORMAP_JET);
    
    // Display images
    cv::imshow("Custom Depth Map (Z)", depthColored);
    cv::imshow("Custom Color Map", colorImage);
    cv::imshow("Custom Normal Map", normalImage);
    
    std::cout << "Press any key to continue..." << std::endl;
    cv::waitKey(0);
    
    //// Also create 3D point cloud visualization
    //std::vector<cv::Point3f> points;
    //std::vector<cv::Point3f> normals;
    //std::vector<cv::Vec3b> colors;

    //for (int y = 0; y < depthmap.height; y++) {
    //    for (int x = 0; x < depthmap.width; x++) {
    //        int idx = y * depthmap.width + x;
    //        float depthValue = depthmap.depthmap[idx];
    //        
    //        if (depthValue > 0.0f && depthValue < FLT_MAX) {
    //            // Back-project to 3D point using camera intrinsics
    //            // Note: This assumes depth map was created with standard perspective projection
    //            // You may need to pass camera parameters if needed
    //            float fx = 3171.533820f;  // Default focal length (should be passed as parameter)
    //            float fy = 3172.366484f;
    //            float cx = 308.147818f;  // Default principal point (should be passed as parameter)
    //            float cy = 207.796650f;
    //            
    //            float pointX = (x - cx) * depthValue / fx;
    //            float pointY = (y - cy) * depthValue / fy;
    //            points.push_back(cv::Point3f(pointX, pointY, depthValue));
    //            
    //            if (idx < static_cast<int>(depthmap.normalmap.size())) {
    //                const cv::Vec3f& normal = depthmap.normalmap[idx];
    //                normals.push_back(cv::Point3f(normal[0], normal[1], normal[2]));
    //            }
    //            
    //            if (idx < static_cast<int>(depthmap.colormap.size())) {
    //                colors.push_back(depthmap.colormap[idx]);
    //            }
    //        }
    //    }
    //}
    //
    //std::cout << "Generated " << points.size() << " 3D points from custom depth map" << std::endl;
    
    // Visualize with Open3D
    //VisualizePointCloudWithOpen3D(points, normals, colors, "Custom Generated Depth Map");
    //
    //std::cout << "Custom depth map visualization completed!" << std::endl;


}

void TransformAndVisualizePCDInWorld(const ScanDataLoader& loader, int frameIdx) {
    std::cout << "\n=== Transforming PCD to World Coordinates ===" << std::endl;
    
    const auto& pointCloudParams = loader.getPointCloudParams(frameIdx);
    const auto& matrixParams = loader.getMatrixParams();
    
    if (!matrixParams.cameraToWorld0.empty()) {
        cv::Mat cameraToWorld = matrixParams.cameraToWorld0[0];
        std::cout << "Using CameraToWorld0 matrix:" << std::endl;
        std::cout << cameraToWorld << std::endl;
        
        // Transform src_0 to world coordinates
        if (!pointCloudParams.src_0.points.empty()) {
            std::cout << "Transforming src_0 point cloud to world coordinates..." << std::endl;
            std::vector<cv::Point3f> transformedPoints;
            std::vector<cv::Point3f> transformedNormals;
            
            for (size_t i = 0; i < pointCloudParams.src_0.points.size(); ++i) {
                // Transform point from camera to world
                cv::Mat pointHomogeneous = (cv::Mat_<float>(4, 1) << 
                    pointCloudParams.src_0.points[i].x,
                    pointCloudParams.src_0.points[i].y,
                    pointCloudParams.src_0.points[i].z,
                    1.0f);
                
                cv::Mat worldPoint = cameraToWorld * pointHomogeneous;
                transformedPoints.push_back(cv::Point3f(
                    worldPoint.at<float>(0),
                    worldPoint.at<float>(1),
                    worldPoint.at<float>(2)
                ));
                
                // Transform normal (only rotation)
                if (i < pointCloudParams.src_0.normals.size()) {
                    cv::Mat normalHomogeneous = (cv::Mat_<float>(4, 1) << 
                        pointCloudParams.src_0.normals[i].x,
                        pointCloudParams.src_0.normals[i].y,
                        pointCloudParams.src_0.normals[i].z,
                        0.0f);
                    
                    cv::Mat worldNormal = cameraToWorld * normalHomogeneous;
                    transformedNormals.push_back(cv::Point3f(
                        worldNormal.at<float>(0),
                        worldNormal.at<float>(1),
                        worldNormal.at<float>(2)
                    ));
                }
            }
            
            std::cout << "Transformed " << transformedPoints.size() << " points to world coordinates" << std::endl;
            VisualizePointCloudWithOpen3D(transformedPoints, transformedNormals, pointCloudParams.src_0.colors, "PCD_World_src_0");
        } else {
            std::cout << "src_0 point cloud is empty!" << std::endl;
        }
    } else {
        std::cout << "CameraToWorld0 matrix not available!" << std::endl;
    }
}

void TransformAndVisualizePCDInWorld(const ScanDataLoader2& loader, int frameIndex) {
    std::cout << "\n=== Transforming PCD to World Coordinates (ScanDataLoader2) ===" << std::endl;
    
    const auto& pointCloudParams = loader.getPointCloudParams(frameIndex);
    const auto& matrixParams = loader.getMatrixParams();
    
    if (!matrixParams.cameraToWorld0.empty()) {
        cv::Mat cameraToWorld = matrixParams.cameraToWorld0[0];
        std::cout << "Using CameraToWorld0 matrix:" << std::endl;
        std::cout << cameraToWorld << std::endl;
        
        // Transform src_0 to world coordinates
        if (!pointCloudParams.src_0.points.empty()) {
            std::cout << "Transforming src_0 point cloud to world coordinates..." << std::endl;
            std::vector<cv::Point3f> transformedPoints;
            std::vector<cv::Point3f> transformedNormals;
            
            for (size_t i = 0; i < pointCloudParams.src_0.points.size(); ++i) {
                // Transform point from camera to world
                cv::Mat pointHomogeneous = (cv::Mat_<float>(4, 1) << 
                    pointCloudParams.src_0.points[i].x,
                    pointCloudParams.src_0.points[i].y,
                    pointCloudParams.src_0.points[i].z,
                    1.0f);
                
                cv::Mat worldPoint = cameraToWorld * pointHomogeneous;
                transformedPoints.push_back(cv::Point3f(
                    worldPoint.at<float>(0),
                    worldPoint.at<float>(1),
                    worldPoint.at<float>(2)
                ));
                
                // Transform normal (only rotation)
                if (i < pointCloudParams.src_0.normals.size()) {
                    cv::Mat normalHomogeneous = (cv::Mat_<float>(4, 1) << 
                        pointCloudParams.src_0.normals[i].x,
                        pointCloudParams.src_0.normals[i].y,
                        pointCloudParams.src_0.normals[i].z,
                        0.0f);
                    
                    cv::Mat worldNormal = cameraToWorld * normalHomogeneous;
                    transformedNormals.push_back(cv::Point3f(
                        worldNormal.at<float>(0),
                        worldNormal.at<float>(1),
                        worldNormal.at<float>(2)
                    ));
                }
            }
            
            std::cout << "Transformed " << transformedPoints.size() << " points to world coordinates" << std::endl;
            VisualizePointCloudWithOpen3D(transformedPoints, transformedNormals, pointCloudParams.src_0.colors, "PCD_World_src_0");
        } else {
            std::cout << "src_0 point cloud is empty!" << std::endl;
        }
    } else {
        std::cout << "CameraToWorld0 matrix not available!" << std::endl;
    }
}


void VisualizeRenderedDepthImShow(const float4* depthOut, int width, int height, const std::string& windowName) {
    if (depthOut == nullptr || width <= 0 || height <= 0) {
        std::cout << "VisualizeRenderedDepthImShow: invalid input" << std::endl;
        return;
    }

    const float MINF = 1e10f;
    cv::Mat depth(height, width, CV_32FC1);

    for (int y = 0; y < height; ++y) {
        float* row = depth.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            const float4 v = depthOut[y * width + x];
            row[x] = (fabsf(v.x) >= MINF * 0.5f) ? 0.0f : v.x;
        }
    }

    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    if (!(maxVal > minVal)) { minVal = 0.0; maxVal = 1.0; }

    cv::Mat norm8u, colored;
    depth.convertTo(norm8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::applyColorMap(norm8u, colored, cv::COLORMAP_JET);

    cv::imshow(windowName, colored);
    cv::waitKey(1);
}

void VisualizeRenderedDepthFloat(const float* depthOut, int width, int height, const std::string& windowName) {
    if (depthOut == nullptr || width <= 0 || height <= 0) {
        std::cout << "VisualizeRenderedDepthFloat: invalid input" << std::endl;
        return;
    }

    const float MINF = 1e10f;
    cv::Mat depth(height, width, CV_32FC1);

    for (int y = 0; y < height; ++y) {
        float* row = depth.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            float v = depthOut[y * width + x];
            row[x] = (fabsf(v) >= MINF * 0.5f) ? 0.0f : v;
        }
    }

    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    if (!(maxVal > minVal)) { minVal = 0.0; maxVal = 1.0; }

    cv::Mat norm8u, colored;
    depth.convertTo(norm8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    cv::applyColorMap(norm8u, colored, cv::COLORMAP_JET);

    cv::imshow(windowName, colored);
    cv::waitKey(1);
}
