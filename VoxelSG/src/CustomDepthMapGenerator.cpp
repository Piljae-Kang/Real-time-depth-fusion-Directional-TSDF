#include "../include/CustomDepthMapGenerator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

CustomDepthMapGenerator::CustomDepthMapGenerator() 
    : projectionMethod_(ProjectionMethod::PERSPECTIVE), useCUDA_(false),
      cudaDepthMap_(nullptr), cudaNormalMap_(nullptr), cudaColorMap_(nullptr) {
    
    // Initialize with default camera parameters
    cameraParams_ = CameraParams();
    
    // Try to initialize CUDA
    useCUDA_ = initializeCUDA();
    if (useCUDA_) {
        std::cout << "CustomDepthMapGenerator: CUDA support enabled" << std::endl;
    } else {
        std::cout << "CustomDepthMapGenerator: Using CPU implementation" << std::endl;
    }
}

CustomDepthMapGenerator::~CustomDepthMapGenerator() {
    cleanupCUDA();
}

void CustomDepthMapGenerator::setCameraParams(const CameraParams& params) {
    cameraParams_ = params;
    std::cout << "Camera parameters updated: " << params.width << "x" << params.height 
              << ", fx=" << params.fx << ", fy=" << params.fy << std::endl;
}

void CustomDepthMapGenerator::setCameraParams(float fx, float fy, float cx, float cy,
                                            float nearPlane, float farPlane, int width, int height) {
    cameraParams_.fx = fx;
    cameraParams_.fy = fy;
    cameraParams_.cx = cx;
    cameraParams_.cy = cy;
    cameraParams_.nearPlane = nearPlane;
    cameraParams_.farPlane = farPlane;
    cameraParams_.width = width;
    cameraParams_.height = height;
    
    std::cout << "Camera parameters updated: " << width << "x" << height 
              << ", fx=" << fx << ", fy=" << fy << std::endl;
}

void CustomDepthMapGenerator::setCameraParamsFromLoader(const ScanDataLoader& loader) {
    const auto& cameraParams = loader.getCameraParams();
    setCameraParams(
        cameraParams.fx, cameraParams.fy,
        cameraParams.cx, cameraParams.cy,
        cameraParams.nearPlane, cameraParams.farPlane,
        cameraParams_.width, cameraParams_.height
    );
    std::cout << "Camera parameters set from ScanDataLoader: " 
              << loader.h_depthWidth << "x" << loader.h_depthHeight << std::endl;
}

void CustomDepthMapGenerator::setCameraParamsFromLoader(const ScanDataLoader2& loader) {
    const auto& cameraParams = loader.getCameraParams();
    setCameraParams(
        cameraParams.fx, cameraParams.fy,
        cameraParams.cx, cameraParams.cy,
        cameraParams.nearPlane, cameraParams.farPlane,
        loader.width, loader.height
    );
    std::cout << "Camera parameters set from ScanDataLoader2: " 
              << loader.width << "x" << loader.height << std::endl;
}

void CustomDepthMapGenerator::setProjectionMethod(ProjectionMethod method) {
    projectionMethod_ = method;
    std::cout << "Projection method set to: " << (int)method << std::endl;
}

CustomDepthMapGenerator::GeneratedDepthMap CustomDepthMapGenerator::generateFromPointCloud(
    const std::vector<cv::Point3f>& points,
    const std::vector<cv::Point3f>& normals,
    const std::vector<cv::Vec3b>& colors) {
    
    std::cout << "Generating depth map from " << points.size() << " points..." << std::endl;
    
    if (useCUDA_) {
        return generateFromPointCloudCUDA(points, normals, colors);
    }
    
    // CPU implementation
    GeneratedDepthMap result;
    result.width = cameraParams_.width;
    result.height = cameraParams_.height;
    int totalPixels = result.width * result.height;

    std::cout << "resolution  : " << result.height << " x " << result.width << "\n";
    std::cout << "Camera params: fx=" << cameraParams_.fx << ", fy=" << cameraParams_.fy 
              << ", cx=" << cameraParams_.cx << ", cy=" << cameraParams_.cy 
              << ", width=" << cameraParams_.width << ", height=" << cameraParams_.height << std::endl;
    
    // Initialize output arrays with invalid values (0 = no depth data)
    result.depthmap.resize(totalPixels, 0.0f);  // Depth values, initialize to 0.0 (invalid)
    result.normalmap.resize(totalPixels, cv::Vec3f(0, 0, 0));
    result.colormap.resize(totalPixels, cv::Vec3b(0, 0, 0));
    result.pointmap.resize(totalPixels, cv::Vec3f(0, 0, 0));  // 3D points (camera coordinates)
    
    // Project each point
    for (size_t i = 0; i < points.size(); ++i) {
        const cv::Point3f& point = points[i];
        
        // Check if point is in front of camera
        if (point.z <= 0.0f) continue;
        
        // Project point to image coordinates
        cv::Point2f projected = projectPoint(point);

        //std::cout << "3d points : " << point.x << ", " << point.y << ", " << point.z << "\n";
        //std::cout << "pixel : " << projected.x << ", " << projected.y << "\n";


        
        // Check bounds and depth range
        if (!isPointInBounds(projected)) {

            //std::cout << "3d points : " << point.x << ", " << point.y << ", " << point.z << "\n";
            //std::cout << "pixel : " << projected.x << ", " << projected.y << "\n";


            continue;
        }

        //std::cout << "3d points : " << point.x << ", " << point.y << ", " << point.z << "\n";
        //std::cout << "pixel : " << projected.x << ", " << projected.y << "\n";
        
        // Convert to pixel coordinates
        int pixelX = static_cast<int>(std::floor(projected.x));
        int pixelY = static_cast<int>(std::floor(projected.y));
        // Clamp to valid range to avoid overflow due to floating rounding near borders

        if (pixelX < 0 || pixelY < 0 || pixelX >= result.width || pixelY >= result.height) continue;

        int pixelIdx = pixelY * result.width + pixelX;
        
        // Store 3D point (camera coordinates)
        result.pointmap[pixelIdx] = cv::Vec3f(point.x, point.y, point.z);
        
        // Store depth value
        result.depthmap[pixelIdx] = point.z;
        
        // Store normal if available
        if (i < normals.size()) {
            result.normalmap[pixelIdx] = cv::Vec3f(normals[i].x, normals[i].y, normals[i].z);
        }
        
        // Store color if available
        if (i < colors.size()) {
            result.colormap[pixelIdx] = colors[i];
        }
    }
    
    // Calculate normals for points without normals
    if (normals.empty()) {
        std::cout << "Calculating normals from depth map..." << std::endl;
        for (int y = 1; y < result.height - 1; ++y) {
            for (int x = 1; x < result.width - 1; ++x) {
                int idx = y * result.width + x;
                
                if (result.depthmap[idx] > 0.0f) {  // Valid depth data
                    // Back-project to 3D points using camera intrinsics
                    float fx = cameraParams_.fx;
                    float fy = cameraParams_.fy;
                    float cx = cameraParams_.cx;
                    float cy = cameraParams_.cy;
                    
                    float centerDepth = result.depthmap[idx];
                    cv::Point3f center = cv::Point3f(
                        (x - cx) * centerDepth / fx,
                        (y - cy) * centerDepth / fy,
                        centerDepth
                    );
                    
                    float rightDepth = result.depthmap[idx + 1];
                    cv::Point3f right = cv::Point3f(
                        (x + 1 - cx) * rightDepth / fx,
                        (y - cy) * rightDepth / fy,
                        rightDepth
                    );
                    
                    float downDepth = result.depthmap[(y + 1) * result.width + x];
                    cv::Point3f down = cv::Point3f(
                        (x - cx) * downDepth / fx,
                        (y + 1 - cy) * downDepth / fy,
                        downDepth
                    );
                    
                    cv::Point3f normal = (right - center).cross(down - center);
                    float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                    
                    if (length > 0.0f) {
                        normal = normal * (1.0f / length);
                        result.normalmap[idx] = cv::Vec3f(normal.x, normal.y, normal.z);
                    }
                }
            }
        }
    }
    
    // Count valid pixels
    int validPixels = 0;
    for (int i = 0; i < totalPixels; ++i) {
        if (result.depthmap[i] > 0.0f) {  // depth > 0 means valid
            validPixels++;
        }
    }
    
    std::cout << "Generated depth map: " << validPixels << "/" << totalPixels 
              << " valid pixels (" << (100.0f * validPixels / totalPixels) << "%)" << std::endl;
    
    return result;
}

CustomDepthMapGenerator::GeneratedDepthMap CustomDepthMapGenerator::generateFromExistingDepthMap(
    const std::vector<cv::Vec3f>& sourceDepthmap,
    int sourceWidth, int sourceHeight,
    const CameraParams& sourceCamera,
    const CameraParams& targetCamera) {
    
    std::cout << "Reprojecting depth map from " << sourceWidth << "x" << sourceHeight 
              << " to " << targetCamera.width << "x" << targetCamera.height << std::endl;
    
    GeneratedDepthMap result;
    result.width = targetCamera.width;
    result.height = targetCamera.height;
    int totalPixels = result.width * result.height;
    
    // Initialize output arrays
    result.depthmap.resize(totalPixels, 0.0f);  // 0 = no depth data
    result.normalmap.resize(totalPixels, cv::Vec3f(0, 0, 0));
    result.colormap.resize(totalPixels, cv::Vec3b(0, 0, 0));
    
    // Reproject each valid pixel
    for (int sy = 0; sy < sourceHeight; ++sy) {
        for (int sx = 0; sx < sourceWidth; ++sx) {
            int sourceIdx = sy * sourceWidth + sx;
            const cv::Vec3f& sourcePoint = sourceDepthmap[sourceIdx];
            
            // Skip invalid points
            if (sourcePoint[2] <= 0.0f || sourcePoint[2] >= sourceCamera.farPlane) {
                continue;
            }
            
            // Convert to 3D point in camera space
            cv::Point3f point3D(sourcePoint[0], sourcePoint[1], sourcePoint[2]);
            
            // Project to target camera
            cv::Point2f projected = projectPoint(point3D);
            
            // Check bounds
            if (!isPointInBounds(projected)) continue;
            
            // Convert to pixel coordinates
            int pixelX = static_cast<int>(projected.x + 0.5f);
            int pixelY = static_cast<int>(projected.y + 0.5f);
            int pixelIdx = pixelY * result.width + pixelX;
            
            // Z-buffering
            if (point3D.z < result.depthmap[pixelIdx]) {
                result.depthmap[pixelIdx] = point3D.z;
            }
        }
    }
    
    // Calculate normals (using depth values and camera intrinsics for 3D reconstruction)
    // Note: We need camera intrinsics to back-project depth to 3D
    for (int y = 1; y < result.height - 1; ++y) {
        for (int x = 1; x < result.width - 1; ++x) {
            int idx = y * result.width + x;
            
            if (result.depthmap[idx] > 0.0f) {  // Valid depth data
                // Back-project to 3D points using camera intrinsics
                float fx = targetCamera.fx;
                float fy = targetCamera.fy;
                float cx = targetCamera.cx;
                float cy = targetCamera.cy;
                
                float centerDepth = result.depthmap[idx];
                cv::Point3f center = cv::Point3f(
                    (x - cx) * centerDepth / fx,
                    (y - cy) * centerDepth / fy,
                    centerDepth
                );
                
                float rightDepth = result.depthmap[idx + 1];
                cv::Point3f right = cv::Point3f(
                    (x + 1 - cx) * rightDepth / fx,
                    (y - cy) * rightDepth / fy,
                    rightDepth
                );
                
                float downDepth = result.depthmap[(y + 1) * result.width + x];
                cv::Point3f down = cv::Point3f(
                    (x - cx) * downDepth / fx,
                    (y + 1 - cy) * downDepth / fy,
                    downDepth
                );
                
                cv::Point3f normal = (right - center).cross(down - center);
                float length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
                
                if (length > 0.0f) {
                    normal = normal * (1.0f / length);
                    result.normalmap[idx] = cv::Vec3f(normal.x, normal.y, normal.z);
                }
            }
        }
    }
    
    int validPixels = 0;
    for (int i = 0; i < totalPixels; ++i) {
        if (result.depthmap[i] > 0.0f) {  // depth > 0 means valid
            validPixels++;
        }
    }
    
    std::cout << "Reprojected depth map: " << validPixels << "/" << totalPixels 
              << " valid pixels (" << (100.0f * validPixels / totalPixels) << "%)" << std::endl;
    
    return result;
}

CustomDepthMapGenerator::GeneratedDepthMap CustomDepthMapGenerator::generateWithTransform(
    const std::vector<cv::Point3f>& points,
    const cv::Mat& transform,
    const std::vector<cv::Point3f>& normals,
    const std::vector<cv::Vec3b>& colors) {
    
    std::cout << "Generating depth map with custom transform from " << points.size() << " points..." << std::endl;
    
    // Transform points
    std::vector<cv::Point3f> transformedPoints;
    std::vector<cv::Point3f> transformedNormals;
    
    for (size_t i = 0; i < points.size(); ++i) {
        const cv::Point3f& point = points[i];
        
        // Apply transformation
        cv::Mat pointMat = (cv::Mat_<float>(4, 1) << point.x, point.y, point.z, 1.0f);
        cv::Mat transformedMat = transform * pointMat;
        
        cv::Point3f transformedPoint(transformedMat.at<float>(0), 
                                   transformedMat.at<float>(1), 
                                   transformedMat.at<float>(2));
        
        transformedPoints.push_back(transformedPoint);
        
        // Transform normal if available
        if (i < normals.size()) {
            cv::Mat normalMat = (cv::Mat_<float>(3, 1) << normals[i].x, normals[i].y, normals[i].z);
            cv::Mat transformedNormalMat = transform(cv::Rect(0, 0, 3, 3)).inv().t() * normalMat;
            
            cv::Point3f transformedNormal(transformedNormalMat.at<float>(0),
                                         transformedNormalMat.at<float>(1),
                                         transformedNormalMat.at<float>(2));
            
            // Normalize
            float length = sqrt(transformedNormal.x * transformedNormal.x + 
                              transformedNormal.y * transformedNormal.y + 
                              transformedNormal.z * transformedNormal.z);
            if (length > 0.0f) {
                transformedNormal = transformedNormal * (1.0f / length);
            }
            
            transformedNormals.push_back(transformedNormal);
        }
    }
    
    // Generate depth map with transformed points
    return generateFromPointCloud(transformedPoints, transformedNormals, colors);
}

bool CustomDepthMapGenerator::saveDepthMap(const GeneratedDepthMap& depthmap,
                                         const std::string& outputPath,
                                         const std::string& prefix,
                                         bool saveBinary,
                                         bool savePng) const {
    
    if (!saveBinary && !savePng) {
        std::cerr << "CustomDepthMapGenerator::saveDepthMap: No output format selected (binary/png)." << std::endl;
        return false;
    }

    if (depthmap.width <= 0 || depthmap.height <= 0) {
        std::cerr << "CustomDepthMapGenerator::saveDepthMap: Invalid depth map dimensions." << std::endl;
        return false;
    }

    try {
        // Create output directory if it doesn't exist
        if (!fs::exists(outputPath)) {
            fs::create_directories(outputPath);
        }
        
        const size_t expectedPixelCount = static_cast<size_t>(depthmap.width) * static_cast<size_t>(depthmap.height);

        if (saveBinary) {
            // Save depth map
            std::string depthPath = outputPath + "/" + prefix + "_depthmap.bin";
            std::ofstream depthFile(depthPath, std::ios::binary);
            if (depthFile.is_open()) {
                depthFile.write(reinterpret_cast<const char*>(&depthmap.width), sizeof(depthmap.width));
                depthFile.write(reinterpret_cast<const char*>(&depthmap.height), sizeof(depthmap.height));
                if (!depthmap.depthmap.empty()) {
                    depthFile.write(reinterpret_cast<const char*>(depthmap.depthmap.data()),
                                    depthmap.depthmap.size() * sizeof(float));
                }
                depthFile.close();
                std::cout << "Saved depth map: " << depthPath << std::endl;
            } else {
                std::cerr << "Error: Cannot open file for writing: " << depthPath << std::endl;
            }
            
            // Save normal map
            std::string normalPath = outputPath + "/" + prefix + "_normalmap.bin";
            std::ofstream normalFile(normalPath, std::ios::binary);
            if (normalFile.is_open()) {
                normalFile.write(reinterpret_cast<const char*>(&depthmap.width), sizeof(depthmap.width));
                normalFile.write(reinterpret_cast<const char*>(&depthmap.height), sizeof(depthmap.height));
                if (!depthmap.normalmap.empty()) {
                    normalFile.write(reinterpret_cast<const char*>(depthmap.normalmap.data()),
                                     depthmap.normalmap.size() * sizeof(cv::Vec3f));
                }
                normalFile.close();
                std::cout << "Saved normal map: " << normalPath << std::endl;
            }
            
            // Save color map
            std::string colorPath = outputPath + "/" + prefix + "_colormap.bin";
            std::ofstream colorFile(colorPath, std::ios::binary);
            if (colorFile.is_open()) {
                colorFile.write(reinterpret_cast<const char*>(&depthmap.width), sizeof(depthmap.width));
                colorFile.write(reinterpret_cast<const char*>(&depthmap.height), sizeof(depthmap.height));
                if (!depthmap.colormap.empty()) {
                    colorFile.write(reinterpret_cast<const char*>(depthmap.colormap.data()),
                                    depthmap.colormap.size() * sizeof(cv::Vec3b));
                }
                colorFile.close();
                std::cout << "Saved color map: " << colorPath << std::endl;
            }
        }

        if (savePng) {
            // Depth map PNG (8-bit normalized to match imshow, plus colorized preview)
            if (!depthmap.depthmap.empty() && depthmap.depthmap.size() >= expectedPixelCount) {
                const float* depthPtr = depthmap.depthmap.data();
                double minDepth = std::numeric_limits<double>::max();
                double maxDepth = std::numeric_limits<double>::lowest();

                for (size_t idx = 0; idx < expectedPixelCount; ++idx) {
                    float v = depthPtr[idx];
                    if (v > 0.0f) {
                        if (v < minDepth) minDepth = v;
                        if (v > maxDepth) maxDepth = v;
                    }
                }

                cv::Mat depthPng(depthmap.height, depthmap.width, CV_8UC1, cv::Scalar(0));
                if (maxDepth > minDepth && minDepth < std::numeric_limits<double>::max()) {
                    const double invRange = 255.0 / (maxDepth - minDepth);
                    for (int y = 0; y < depthmap.height; ++y) {
                        unsigned char* rowPtr = depthPng.ptr<unsigned char>(y);
                        for (int x = 0; x < depthmap.width; ++x) {
                            size_t idx = static_cast<size_t>(y) * depthmap.width + x;
                            float v = depthPtr[idx];
                            if (v > 0.0f) {
                                rowPtr[x] = static_cast<unsigned char>(std::clamp(
                                    (v - minDepth) * invRange, 0.0, 255.0) + 0.5);
                            }
                        }
                    }
                }

                std::string depthPngPath = outputPath + "/" + prefix + "_depthmap.png";
                if (cv::imwrite(depthPngPath, depthPng)) {
                    std::cout << "Saved depth map PNG: " << depthPngPath << std::endl;
                } else {
                    std::cerr << "Error: Failed to write depth map PNG: " << depthPngPath << std::endl;
                }

                // Optional: colorized depth map for easier inspection
                cv::Mat colorized;
                cv::applyColorMap(depthPng, colorized, cv::COLORMAP_JET);
                std::string depthColorPath = outputPath + "/" + prefix + "_depthmap_color.png";
                if (cv::imwrite(depthColorPath, colorized)) {
                    std::cout << "Saved colorized depth map PNG: " << depthColorPath << std::endl;
                } else {
                    std::cerr << "Error: Failed to write colorized depth map PNG: " << depthColorPath << std::endl;
                }
            }

            // Normal map PNG (RGB 8-bit)
            if (!depthmap.normalmap.empty() && depthmap.normalmap.size() >= expectedPixelCount) {
                cv::Mat normalPng(depthmap.height, depthmap.width, CV_8UC3, cv::Scalar(0, 0, 0));
                auto toByte = [](float value) -> unsigned char {
                    float mapped = (value * 0.5f + 0.5f) * 255.0f;
                    mapped = std::clamp(mapped, 0.0f, 255.0f);
                    return static_cast<unsigned char>(mapped + 0.5f);
                };

                for (int y = 0; y < depthmap.height; ++y) {
                    cv::Vec3b* rowPtr = normalPng.ptr<cv::Vec3b>(y);
                    for (int x = 0; x < depthmap.width; ++x) {
                        size_t idx = static_cast<size_t>(y) * depthmap.width + x;
                        const cv::Vec3f& normal = depthmap.normalmap[idx];
                        rowPtr[x][2] = toByte(normal[0]); // R <- X
                        rowPtr[x][1] = toByte(normal[1]); // G <- Y
                        rowPtr[x][0] = toByte(normal[2]); // B <- Z
                    }
                }

                std::string normalPngPath = outputPath + "/" + prefix + "_normalmap.png";
                if (cv::imwrite(normalPngPath, normalPng)) {
                    std::cout << "Saved normal map PNG: " << normalPngPath << std::endl;
                } else {
                    std::cerr << "Error: Failed to write normal map PNG: " << normalPngPath << std::endl;
                }
            }

            // Color map PNG (BGR 8-bit)
            if (!depthmap.colormap.empty() && depthmap.colormap.size() >= expectedPixelCount) {
                cv::Mat colorPng(depthmap.height, depthmap.width, CV_8UC3,
                                 const_cast<cv::Vec3b*>(depthmap.colormap.data()));
                std::string colorPngPath = outputPath + "/" + prefix + "_colormap.png";
                if (cv::imwrite(colorPngPath, colorPng)) {
                    std::cout << "Saved color map PNG: " << colorPngPath << std::endl;
                } else {
                    std::cerr << "Error: Failed to write color map PNG: " << colorPngPath << std::endl;
                }
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving depth map: " << e.what() << std::endl;
        return false;
    }
}

cv::Point2f CustomDepthMapGenerator::projectPoint(const cv::Point3f& point) const {
    switch (projectionMethod_) {
        case ProjectionMethod::PERSPECTIVE:
            return cv::Point2f(
                (point.x * cameraParams_.fx / point.z) + cameraParams_.cx,
                (point.y * cameraParams_.fy / point.z) + cameraParams_.cy
            );
            
        case ProjectionMethod::ORTHOGRAPHIC:
            return cv::Point2f(
                point.x + cameraParams_.cx,
                point.y + cameraParams_.cy
            );
            
        case ProjectionMethod::CUSTOM_CAMERA:

            std::cout << "fx : " << cameraParams_.fx << ", " << "fy : " << cameraParams_.fy << "\n";
            std::cout << "Cx : " << cameraParams_.cx << ", " << "Cy : " << cameraParams_.cy << "\n";

            // Same as perspective for now
            return cv::Point2f(
                (point.x * cameraParams_.fx / point.z) + cameraParams_.cx,
                (point.y * cameraParams_.fy / point.z) + cameraParams_.cy
            );
            
        default:
            return cv::Point2f(0, 0);
    }
}

bool CustomDepthMapGenerator::isPointInBounds(const cv::Point2f& projectedPoint) const {


    //std::cout << "camera width : " << cameraParams_.width << "camera height : " << cameraParams_.height << "\n";

    return projectedPoint.x >= 0 && projectedPoint.x < cameraParams_.width &&
           projectedPoint.y >= 0 && projectedPoint.y < cameraParams_.height;
}

bool CustomDepthMapGenerator::isPointInDepthRange(float depth) const {
    return depth >= cameraParams_.nearPlane && depth <= cameraParams_.farPlane;
}

bool CustomDepthMapGenerator::initializeCUDA() {
    // TODO: Implement CUDA initialization
    // For now, return false to use CPU implementation
    return false;
}

void CustomDepthMapGenerator::cleanupCUDA() {
    // TODO: Implement CUDA cleanup
}

CustomDepthMapGenerator::GeneratedDepthMap CustomDepthMapGenerator::generateFromPointCloudCUDA(
    const std::vector<cv::Point3f>& points,
    const std::vector<cv::Point3f>& normals,
    const std::vector<cv::Vec3b>& colors) {
    
    // TODO: Implement CUDA-based generation
    // For now, fall back to CPU implementation
    return generateFromPointCloud(points, normals, colors);
}
