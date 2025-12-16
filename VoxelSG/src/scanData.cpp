#include "../include/scanData.h"
#include <filesystem>
#include <algorithm>  // For std::max

namespace fs = std::filesystem;

ScanDataLoader::ScanDataLoader(const std::string& path) : rootPath(path) {
    
    // frame_idx = 200;
    // frame_idx = 400;
    frame_idx = 1000;
    //width = 400;
    //height = 480;

    // frame_idx = 2;
    // width = 480;
    // height = 400;
    
    baseParams.patch_info = nullptr;
    baseParams.patchID = 0;
    baseParams.Tolerance = 0;
    baseParams.depthmapInterpolateBaseLevel = 0;
    baseParams.depthmapInterpolateLevel = 0;
    baseParams.RobustVoxelRatio = 0.0f;
    baseParams.RobustVoxelCount = 0;
    baseParams.xUnit = 1.0f;
    baseParams.yUnit = 1.0f;
    baseParams.pointMag = 1.0f;
    baseParams.bUseAASF = false;
    baseParams.bDeepLearningEnable = false;
    
    // Initialize camera parameters with default values
    cameraParams.fx = width * 0.5f;  // Assume 90 degree FOV
    cameraParams.fy = height * 0.5f;
    cameraParams.cx = width * 0.5f;
    cameraParams.cy = height * 0.5f;
    cameraParams.nearPlane = 0.1f;
    cameraParams.farPlane = 10.0f;
}

ScanDataLoader::~ScanDataLoader() {
   
    if (baseParams.patch_info != nullptr) {
        delete[] baseParams.patch_info;
        baseParams.patch_info = nullptr;
    }
}


void ScanDataLoader::loadBaseParams() {


    std::string baseParamsPath = rootPath + "/base_params";
    if (!fs::exists(baseParamsPath)) {
        std::cout << "Warning: base_params directory not found" << std::endl;
        return;
    }
    
    std::cout << "Loading base parameters..." << std::endl;


    char filename[32];
    sprintf_s(filename, sizeof(filename), "%06d.txt", frame_idx);
    std::string filePath = baseParamsPath + "/" + filename;
    
    if (!fs::exists(filePath)) {
        std::cout << "Warning: File not found: " << filePath << std::endl;
        return;
    }
    
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Warning: Cannot open file: " << filePath << std::endl;
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Parse key-value pairs separated by colon (:)
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            continue;
        }
        
        std::string key = line.substr(0, colonPos);
        std::string value = line.substr(colonPos + 1);
        
 
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        

        if (key == "Patch Info") {

            std::istringstream iss(value);
            std::vector<int> patchInfo;
            int num;
            while (iss >> num) {
                patchInfo.push_back(num);
            }
            if (!patchInfo.empty()) {
                baseParams.patch_info = new int[patchInfo.size()];
                std::copy(patchInfo.begin(), patchInfo.end(), baseParams.patch_info);
            }
        }
        else if (key == "Patch ID") {
            baseParams.patchID = std::stoi(value);
        }
        else if (key == "Tolerance") {
            baseParams.Tolerance = std::stoi(value);
        }
        else if (key == "Depthmap Interpolate Base Level") {
            baseParams.depthmapInterpolateBaseLevel = std::stoi(value);
        }
        else if (key == "Depthmap Interpolate Level") {
            baseParams.depthmapInterpolateLevel = std::stoi(value);
        }
        else if (key == "Robust Voxel Ratio") {
            baseParams.RobustVoxelRatio = std::stof(value);
        }
        else if (key == "Robust Voxel Count") {
            baseParams.RobustVoxelCount = std::stoi(value);
        }
        else if (key == "X Unit") {
            baseParams.xUnit = std::stof(value);
        }
        else if (key == "Y Unit") {
            baseParams.yUnit = std::stof(value);
        }
        else if (key == "Point Mag") {
            baseParams.pointMag = std::stof(value);
        }
        else if (key == "bUseAASF") {
            baseParams.bUseAASF = (std::stoi(value) != 0);
        }
        else if (key == "bDeepLearningEnable") {
            baseParams.bDeepLearningEnable = (std::stoi(value) != 0);
        }
    }
    
    file.close();
    
    std::cout << "Loaded from: " << filePath << std::endl;
    std::cout << "Patch ID: " << baseParams.patchID << std::endl;
    std::cout << "Tolerance: " << baseParams.Tolerance << std::endl;
    std::cout << "Robust Voxel Ratio: " << baseParams.RobustVoxelRatio << std::endl;
    std::cout << "Use AASF: " << (baseParams.bUseAASF ? "true" : "false") << std::endl;
    std::cout << "Deep Learning Enable: " << (baseParams.bDeepLearningEnable ? "true" : "false") << std::endl;
}


// Template function implementation
template<typename T>
bool ScanDataLoader::loadBinaryImage(const std::string& folderName, std::vector<T>& targetVector) {
    std::string imgParamsPath = rootPath + "/img_params";
    std::string folderPath = imgParamsPath + "/" + folderName;
    
    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        return false;
    }
    
    char filename[32];
    sprintf_s(filename, sizeof(filename), "%06d.bin", frame_idx);
    std::string filePath = folderPath + "/" + filename;
    
    if (!fs::exists(filePath)) {
        std::cout << "Warning: " << folderName << " file not found: " << filePath << std::endl;
        return false;
    }
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Warning: Cannot open " << folderName << " file: " << filePath << std::endl;
        return false;
    }
    

    int fileWidth, fileHeight;
    file.read(reinterpret_cast<char*>(&fileWidth), sizeof(fileWidth));
    file.read(reinterpret_cast<char*>(&fileHeight), sizeof(fileHeight));
    
    int totalPixels = fileWidth * fileHeight;
    targetVector.resize(totalPixels);
    file.read(reinterpret_cast<char*>(targetVector.data()), 
             totalPixels * sizeof(T));
    
    file.close();
    std::cout << "Loaded " << folderName << ": " << filePath << " (" << fileWidth << "x" << fileHeight << ")" << std::endl;
    return true;
}

void ScanDataLoader::loadImageParams() {
    std::string imgParamsPath = rootPath + "/img_params";
    if (!fs::exists(imgParamsPath)) {
        std::cout << "Warning: img_params directory not found" << std::endl;
        return;
    }
    
    std::cout << "Loading image parameters..." << std::endl;
    
    loadBinaryImage("deeplearning_inference", imageParams.deeplearning_inference);
    loadBinaryImage("current_img_0", imageParams.current_img_0);
    loadBinaryImage("current_img_45", imageParams.current_img_45);
    loadBinaryImage("confidence_map_0", imageParams.confidence_map_0);
    loadBinaryImage("confidence_map_45", imageParams.confidence_map_45);
    loadBinaryImage("depthmap_mask", imageParams.depthmap_mask);
    
    std::cout << "Image loading completed!" << std::endl;
    std::cout << "Deeplearning inference size: " << imageParams.deeplearning_inference.size() << " pixels" << std::endl;
    std::cout << "Current img 0 size: " << imageParams.current_img_0.size() << " pixels" << std::endl;
    std::cout << "Current img 45 size: " << imageParams.current_img_45.size() << " pixels" << std::endl;
    std::cout << "Confidence map 0 size: " << imageParams.confidence_map_0.size() << " pixels" << std::endl;
    std::cout << "Confidence map 45 size: " << imageParams.confidence_map_45.size() << " pixels" << std::endl;
    std::cout << "Depthmap mask size: " << imageParams.depthmap_mask.size() << " pixels" << std::endl;
    std::cout << "Image size: " << width << "x" << height << std::endl;
}


void ScanDataLoader::loadMatrixParams() {
    std::string matrixParamsPath = rootPath + "/matrix_params";
    if (!fs::exists(matrixParamsPath)) {
        std::cout << "Warning: matrix_params directory not found" << std::endl;
        return;
    }
    
    std::cout << "Loading matrix parameters..." << std::endl;
    
    // Dynamically check available matrix file count
    int availableMatrixFiles = countAvailableFrames(matrixParamsPath, ".txt");
    if (availableMatrixFiles == 0) {
        std::cout << "Warning: No matrix files found in " << matrixParamsPath << std::endl;
        return;
    }
    
    int maxFrames = std::min(availableMatrixFiles, frame_idx);
    
    if (maxFrames <= 0) {
        std::cout << "Warning: No matrix files available from frame_idx " << frame_idx << std::endl;
        return;
    }
    
    std::cout << "Found " << availableMatrixFiles << " matrix files" << std::endl;
    std::cout << "Loading matrix files from frame_idx " << frame_idx << " (max " << maxFrames << " frames)..." << std::endl;
    
    // Clear existing matrices
    matrixParams.transform_0.clear();
    matrixParams.transform_45.clear();
    matrixParams.localToCamera.clear();
    matrixParams.cameraToWorld0.clear();
    matrixParams.cameraToWorld45.clear();
    
    matrixParams.transform_0.reserve(maxFrames);
    matrixParams.transform_45.reserve(maxFrames);
    matrixParams.localToCamera.reserve(maxFrames);
    matrixParams.cameraToWorld0.reserve(maxFrames);
    matrixParams.cameraToWorld45.reserve(maxFrames);
    
    // Load multiple frames starting from frame_idx
    for (int currentFrameIdx = 0; currentFrameIdx < maxFrames; currentFrameIdx++) {
        
        char filename[32];
        sprintf_s(filename, sizeof(filename), "%06d.txt", currentFrameIdx);
        std::string filePath = matrixParamsPath + "/" + filename;
        
        if (!fs::exists(filePath)) {
            std::cout << "Warning: Matrix file not found: " << filePath << ", skipping..." << std::endl;
            continue;
        }
        
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cout << "Warning: Cannot open matrix file: " << filePath << ", skipping..." << std::endl;
            continue;
        }
        
        std::string line;
        cv::Mat transform_0 = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat transform_45 = cv::Mat::eye(4, 4, CV_32F);
        cv::Mat localToCamera = cv::Mat::eye(4, 4, CV_32F);
        
        bool reading_transform_0 = false;
        bool reading_transform_45 = false;
        bool reading_localToCamera = false;
        int row = 0;
        
        int lineNum = 0;
        while (std::getline(file, line)) {
            lineNum++;
            
            // Find Transform_0, Transform_45, or CameraRT section
            if (line.find("Transform_0") != std::string::npos) {
                reading_transform_0 = true;
                reading_transform_45 = false;
                reading_localToCamera = false;
                row = 0;
                continue;
            } else if (line.find("Transform_45") != std::string::npos) {
                reading_transform_0 = false;
                reading_transform_45 = true;
                reading_localToCamera = false;
                row = 0;
                continue;
            } else if (line.find("CameraRT") != std::string::npos) {
                reading_transform_0 = false;
                reading_transform_45 = false;
                reading_localToCamera = true;
                row = 0;
                continue;
            }
            
            // Parse matrix data
            if (reading_transform_0 || reading_transform_45 || reading_localToCamera) {
                // Skip if line is empty or doesn't contain numbers
                if (line.empty() || line.find_first_of("0123456789.-") == std::string::npos) {
                    continue;
                }
                
                // Check if this line contains matrix data (should have 4 float numbers)
                // Try to parse the line
                std::string dataPart = line;
                size_t colonPos = line.find(':');
                if (colonPos != std::string::npos) {
                    dataPart = line.substr(colonPos + 1);
                }
                
                std::istringstream dataStream(dataPart);
                std::string token;
                int col = 0;
                bool hasValidData = false;
                
                while (dataStream >> token && col < 4) {
                    try {
                        float value = std::stof(token);
                        if (reading_transform_0) {
                            transform_0.at<float>(row, col) = value;
                        } else if (reading_transform_45) {
                            transform_45.at<float>(row, col) = value;
                        } else if (reading_localToCamera) {
                            localToCamera.at<float>(row, col) = value;
                        }
                        col++;
                        hasValidData = true;
                    } catch (const std::exception& e) {
                        // If parsing fails, this line is not matrix data, skip it
                        break;
                    }
                }
                
                // Only increment row if we successfully parsed 4 values
                if (hasValidData && col == 4) {
                    row++;
                }
            }
        }
        
        file.close();
        
        // Validate matrices before storing
        if (transform_0.empty() || transform_0.rows != 4 || transform_0.cols != 4 ||
            transform_45.empty() || transform_45.rows != 4 || transform_45.cols != 4 ||
            localToCamera.empty() || localToCamera.rows != 4 || localToCamera.cols != 4) {
            std::cout << "Warning: Invalid matrix data for frame " << currentFrameIdx << ", skipping..." << std::endl;
            continue;
        }
        
        // Store matrices in vector (use clone() to ensure deep copy)
        // Clone immediately to avoid any reference issues
        cv::Mat transform_0_clone = transform_0.clone();
        cv::Mat transform_45_clone = transform_45.clone();
        cv::Mat localToCamera_clone = localToCamera.clone();
        
        // Validate cloned matrices
        if (transform_0_clone.empty() || transform_45_clone.empty() || localToCamera_clone.empty()) {
            std::cout << "Warning: Failed to clone matrices for frame " << currentFrameIdx << ", skipping..." << std::endl;
            continue;
        }
        
        matrixParams.transform_0.push_back(transform_0_clone);
        matrixParams.transform_45.push_back(transform_45_clone);
        matrixParams.localToCamera.push_back(localToCamera_clone);
        
        // Calculate Camera -> World matrices
        // CameraToWorld = LocalToWorld * CameraToLocal
        // CameraToLocal = (LocalToCamera)^(-1)
        // Therefore: CameraToWorld = LocalToWorld * (LocalToCamera)^(-1)
        // For 0 degrees: CameraToWorld0 = Transform_0 * CameraRT^(-1)
        // For 45 degrees: CameraToWorld45 = Transform_45 * CameraRT^(-1)
        
        // Check if LocalToCamera is actually loaded (use cloned version)
        bool isLocalToCameraLoaded = false;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i == j && abs(localToCamera_clone.at<float>(i, j) - 1.0f) > 1e-6) {
                    isLocalToCameraLoaded = true;
                    break;
                } else if (i != j && abs(localToCamera_clone.at<float>(i, j)) > 1e-6) {
                    isLocalToCameraLoaded = true;
                    break;
                }
            }
            if (isLocalToCameraLoaded) break;
        }
        
        if (isLocalToCameraLoaded) {
            cv::Mat localToCameraInv = localToCamera_clone.inv(cv::DECOMP_SVD);
            
            // Validate inverse matrix
            if (localToCameraInv.empty() || localToCameraInv.rows != 4 || localToCameraInv.cols != 4) {
                std::cout << "Warning: Invalid localToCameraInv for frame " << currentFrameIdx << ", using transform directly" << std::endl;
                matrixParams.cameraToWorld0.push_back(transform_0_clone.clone());
                matrixParams.cameraToWorld45.push_back(transform_45_clone.clone());
            } else {
                cv::Mat cameraToWorld0 = transform_0_clone * localToCameraInv;
                cv::Mat cameraToWorld45 = transform_45_clone * localToCameraInv;
                
                // Validate multiplication results before cloning
                if (cameraToWorld0.empty() || cameraToWorld0.rows != 4 || cameraToWorld0.cols != 4 ||
                    cameraToWorld45.empty() || cameraToWorld45.rows != 4 || cameraToWorld45.cols != 4) {
                    std::cout << "Warning: Invalid cameraToWorld matrices for frame " << currentFrameIdx << ", using transform directly" << std::endl;
                    matrixParams.cameraToWorld0.push_back(transform_0_clone.clone());
                    matrixParams.cameraToWorld45.push_back(transform_45_clone.clone());
                } else {
                    // Clone immediately to avoid any reference issues
                    cv::Mat cameraToWorld0_clone = cameraToWorld0.clone();
                    cv::Mat cameraToWorld45_clone = cameraToWorld45.clone();
                    
                    // Final validation before storing
                    if (cameraToWorld0_clone.empty() || cameraToWorld45_clone.empty()) {
                        std::cout << "Warning: Failed to clone cameraToWorld matrices for frame " << currentFrameIdx << ", using transform directly" << std::endl;
                        matrixParams.cameraToWorld0.push_back(transform_0_clone.clone());
                        matrixParams.cameraToWorld45.push_back(transform_45_clone.clone());
                    } else {
                        matrixParams.cameraToWorld0.push_back(cameraToWorld0_clone);
                        matrixParams.cameraToWorld45.push_back(cameraToWorld45_clone);
                    }
                }
            }
        } else {
            // Use cloned versions
            matrixParams.cameraToWorld0.push_back(transform_0_clone.clone());
            matrixParams.cameraToWorld45.push_back(transform_45_clone.clone());
        }
        
        if ((currentFrameIdx + 1) % 10 == 0 || currentFrameIdx == 0) {
            std::cout << "Loaded matrix for frame " << currentFrameIdx << " (" << (currentFrameIdx + 1) << "/" << maxFrames << ")" << std::endl;
        }
    }
    
    std::cout << "\nMatrix loading completed! Loaded:" << std::endl;
    std::cout << "  - Transform_0 (Local->World, 0°): " << matrixParams.transform_0.size() << " matrices" << std::endl;
    std::cout << "  - Transform_45 (Local->World, 45°): " << matrixParams.transform_45.size() << " matrices" << std::endl;
    std::cout << "  - LocalToCamera: " << matrixParams.localToCamera.size() << " matrices" << std::endl;
    std::cout << "  - CameraToWorld0: " << matrixParams.cameraToWorld0.size() << " matrices" << std::endl;
    std::cout << "  - CameraToWorld45: " << matrixParams.cameraToWorld45.size() << " matrices" << std::endl;
}

int ScanDataLoader::countAvailableFrames(const std::string& directoryPath, const std::string& filePattern) {
    if (!fs::exists(directoryPath)) {
        return 0;
    }
    
    int maxFrame = -1;
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            
            // Check if file pattern matches (e.g., "000000_colormap.bin")
            if (filename.find(filePattern) != std::string::npos) {
                // Extract frame number from filename (e.g., "000000" -> 0)
                std::string frameStr = filename.substr(0, 6); // First 6 characters
                try {
                    int frameNum = std::stoi(frameStr);
                    maxFrame = (std::max)(maxFrame, frameNum);  // Use (std::max) to avoid Windows max macro conflict
                } catch (const std::exception&) {
                    // Ignore if number conversion fails
                }
            }
        }
    }
    
    return maxFrame + 1; // +1 because it's 0-based
}

bool ScanDataLoader::loadSingleDepthMapFrame(int frameIndex, const std::string& resolution, DepthMapParams::DepthMapFrame& frame) {
    std::string depthMapParamsPath = rootPath + "/depthmap_params";
    std::string resolutionPath = depthMapParamsPath + "/" + resolution;
    
    if (!fs::exists(resolutionPath)) {
        std::cout << "Warning: " << resolution << " directory not found" << std::endl;
        return false;
    }
    
    // Load colormap, depthmap, normalmap files
    std::vector<std::string> fileTypes = {"colormap", "depthmap", "normalmap"};
    
    for (const auto& fileType : fileTypes) {
        char filename[64];
        sprintf_s(filename, sizeof(filename), "%06d_%s.bin", frameIndex, fileType.c_str());
        std::string filePath = resolutionPath + "/" + filename;
        
        if (!fs::exists(filePath)) {
            std::cout << "Warning: " << fileType << " file not found: " << filePath << std::endl;
            return false;
        }
        
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Warning: Cannot open " << fileType << " file: " << filePath << std::endl;
            return false;
        }
        
        // Calculate file size
        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (fileType == "colormap") {
            // RGB data (3 bytes per pixel)
            frame.colormap.resize(fileSize);
            file.read(reinterpret_cast<char*>(frame.colormap.data()), fileSize);
        } else if (fileType == "depthmap") {
            // Depth data (4 bytes per pixel, float)
            size_t pixelCount = fileSize / sizeof(float);
            frame.depthmap.resize(pixelCount);
            file.read(reinterpret_cast<char*>(frame.depthmap.data()), fileSize);
        } else if (fileType == "normalmap") {
            // Normal data (4 bytes per component, 3 components per pixel)
            size_t pixelCount = fileSize / (sizeof(float) * 3);
            frame.normalmap.resize(pixelCount * 3);
            file.read(reinterpret_cast<char*>(frame.normalmap.data()), fileSize);
        }
        
        file.close();
    }
    
    std::cout << "Loaded frame " << frameIndex << " from " << resolution << std::endl;
    return true;
}

void ScanDataLoader::loadDepthMapParams() {
    std::string depthMapParamsPath = rootPath + "/depthmap_params";
    if (!fs::exists(depthMapParamsPath)) {
        std::cout << "Warning: depthmap_params directory not found" << std::endl;
        return;
    }
    
    std::cout << "Loading depth map parameters..." << std::endl;
    
    // Load both High resolution and Low resolution
    std::vector<std::string> resolutions = {"high_resolution", "low_resolution"};
    
    for (const auto& resolution : resolutions) {
        std::string resolutionPath = depthMapParamsPath + "/" + resolution;
        if (!fs::exists(resolutionPath)) {
            std::cout << "Warning: " << resolution << " directory not found" << std::endl;
            continue;
        }
        
        std::vector<DepthMapParams::DepthMapFrame>* targetFrames = nullptr;
        if (resolution == "high_resolution") {
            targetFrames = &depthMapParams.high_resolution;
        } else {
            targetFrames = &depthMapParams.low_resolution;
        }
        
        // Always load only frame_idx (default 0) to match other params
        int i = frame_idx;
        std::cout << "Loading depthmap frame " << i << " for " << resolution << std::endl;
        {
            DepthMapParams::DepthMapFrame frame;
            
            // Load colormap, depthmap, normalmap files
            std::vector<std::string> fileTypes = {"colormap", "depthmap", "normalmap"};
            
            for (size_t j = 0; j < fileTypes.size(); ++j) {
                char filename[64];
                sprintf_s(filename, sizeof(filename), "%06d_%s.bin", i, fileTypes[j].c_str());
                std::string filePath = resolutionPath + "/" + filename;
                
                if (!fs::exists(filePath)) {
                    std::cout << "Warning: " << fileTypes[j] << " file not found: " << filePath << std::endl;
                    continue;
                }
                
                std::ifstream file(filePath, std::ios::binary);
                if (!file.is_open()) {
                    std::cout << "Warning: Cannot open " << fileTypes[j] << " file: " << filePath << std::endl;
                    continue;
                }
                
                // Read width/height header (int32)
                int w = 0, h = 0;
                


                file.read(reinterpret_cast<char*>(&w), sizeof(w));
                file.read(reinterpret_cast<char*>(&h), sizeof(h));
                const size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);

                if (resolution == "high_resolution") {
                    h_depthWidth = w; h_depthHeight = h;
                } else if (resolution == "low_resolution") {
                    l_depthWidth = w; l_depthHeight = h;
                }

                std::cout << "-------------------------------\n";
                std::cout << "depth map resolution : " << h << " x " << w << "\n";
                std::cout << "-------------------------------\n";
                            
                if (fileTypes[j] == "colormap") {
                    frame.colormap.resize(pixels);
                    // file payload is 3 * unsigned int per pixel, take low 8-bits per channel
                    std::vector<unsigned int> tmp(pixels * 3);
                    file.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(unsigned int));
                    for (size_t p = 0; p < pixels; ++p) {
                        frame.colormap[p] = cv::Vec3b(
                            static_cast<unsigned char>(tmp[p*3 + 0] & 0xFF),
                            static_cast<unsigned char>(tmp[p*3 + 1] & 0xFF),
                            static_cast<unsigned char>(tmp[p*3 + 2] & 0xFF)
                        );
                    }
                    std::cout << "Loaded colormap (" << resolution << "): "
                              << pixels << " pixels from " << filePath << " ["<<w<<"x"<<h<<"]" << std::endl;
                } else if (fileTypes[j] == "depthmap") {
                    frame.depthmap.resize(pixels);
                    std::vector<cv::Vec3f> tmp(pixels);
                    file.read(reinterpret_cast<char*>(tmp.data()), pixels * sizeof(cv::Vec3f));
                    frame.depthmap.swap(tmp);
                    std::cout << "Loaded depthmap (" << resolution << "): "
                              << pixels << " pixels from " << filePath << " ["<<w<<"x"<<h<<"]" << std::endl;
                } else if (fileTypes[j] == "normalmap") {
                    frame.normalmap.resize(pixels);
                    std::vector<cv::Vec3f> tmp(pixels);
                    file.read(reinterpret_cast<char*>(tmp.data()), pixels * sizeof(cv::Vec3f));
                    frame.normalmap.swap(tmp);
                    std::cout << "Loaded normalmap (" << resolution << "): "
                              << pixels << " pixels from " << filePath << " ["<<w<<"x"<<h<<"]" << std::endl;
                }
                
                file.close();
            }
            
            targetFrames->push_back(frame);
        }
        
        std::cout << "Loaded " << targetFrames->size() << " frames for " << resolution << std::endl;
    }
    
    std::cout << "Depth map loading completed!" << std::endl;
    std::cout << "High resolution frames: " << depthMapParams.high_resolution.size() << std::endl;
    std::cout << "Low resolution frames: " << depthMapParams.low_resolution.size() << std::endl;
}

bool ScanDataLoader::saveDepthMapFrame(const std::string& outputDir, const std::string& resolution, int frameIndex) const {
    const std::vector<DepthMapParams::DepthMapFrame>* sourceFrames = nullptr;
    if (resolution == "high_resolution") sourceFrames = &depthMapParams.high_resolution;
    else if (resolution == "low_resolution") sourceFrames = &depthMapParams.low_resolution;
    else return false;

    if (!sourceFrames || sourceFrames->empty()) return false;
    if (frameIndex < 0 || frameIndex >= static_cast<int>(sourceFrames->size())) return false;

    const auto& frame = (*sourceFrames)[frameIndex];

    // Get actual depth map dimensions based on resolution
    int actualWidth, actualHeight;
    if (resolution == "high_resolution") {
        actualWidth = h_depthWidth;
        actualHeight = h_depthHeight;
    } else if (resolution == "low_resolution") {
        actualWidth = l_depthWidth;
        actualHeight = l_depthHeight;
    } else {
        return false;
    }

    // Ensure output directory exists
    try {
        if (!fs::exists(outputDir)) fs::create_directories(outputDir);
    } catch (...) { return false; }

    // Save raw depth (float) as .bin
    {
        std::string path = outputDir + "/depthmap_" + resolution + "_" + std::to_string(frameIndex) + ".bin";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(frame.depthmap.data()), frame.depthmap.size() * sizeof(float));
        ofs.close();
    }

    // Save normal (float3) as .bin
    {
        std::string path = outputDir + "/normalmap_" + resolution + "_" + std::to_string(frameIndex) + ".bin";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(frame.normalmap.data()), frame.normalmap.size() * sizeof(float));
        ofs.close();
    }

    // Save colormap (uchar3) as .bin
    {
        std::string path = outputDir + "/colormap_" + resolution + "_" + std::to_string(frameIndex) + ".bin";
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(frame.colormap.data()), frame.colormap.size() * sizeof(unsigned char));
        ofs.close();
    }

    // Save preview PNGs (depth normalized, color as is if 3-channel)
    if (!frame.depthmap.empty()) {
        cv::Mat depth(actualHeight, actualWidth, CV_32FC3, const_cast<cv::Vec3f*>(frame.depthmap.data()));
        // visualize Z channel
        std::vector<cv::Mat> ch; cv::split(depth, ch);
        cv::Mat z = ch.size() > 2 ? ch[2] : ch[0];
        double minV, maxV; cv::minMaxLoc(z, &minV, &maxV);
        cv::Mat depth8u; z.convertTo(depth8u, CV_8U, (maxV - minV) > 1e-6 ? 255.0 / (maxV - minV) : 1.0, (maxV - minV) > 1e-6 ? -minV * 255.0 / (maxV - minV) : 0.0);
        cv::imwrite(outputDir + "/depth_preview_" + resolution + "_" + std::to_string(frameIndex) + ".png", depth8u);
    }

    if (!frame.colormap.empty()) {
        cv::Mat color(actualHeight, actualWidth, CV_8UC3, const_cast<cv::Vec3b*>(frame.colormap.data()));
        cv::imwrite(outputDir + "/color_" + resolution + "_" + std::to_string(frameIndex) + ".png", color);
    }

    return true;
}

void ScanDataLoader::loadPointCloudParams() {
    std::string pointCloudParamsPath = rootPath + "/pointColud_params";
    if (!fs::exists(pointCloudParamsPath)) {
        std::cout << "Warning: pointColud_params directory not found" << std::endl;
        return;
    }
    
    std::cout << "Loading point cloud parameters for multiple frames..." << std::endl;
    
    // Check available frames in src_0_mesh directory
    std::string src0Path = pointCloudParamsPath + "/src_0_mesh";
    int availableFrames = countAvailableFrames(src0Path, ".bin");
    
    if (availableFrames == 0) {
        std::cout << "Warning: No point cloud files found in " << src0Path << std::endl;
        return;
    }
    
    // Limit to maximum 10 frames (or available frames if less)
    int maxFrames = std::min(availableFrames, frame_idx);
    std::cout << "Found " << availableFrames << " frames, loading first " << maxFrames << " frames..." << std::endl;
    
    PCDs.clear();
    PCDs.reserve(maxFrames);
    
    // Load multiple frames
    for (int i = 0; i < maxFrames; ++i) {
        PointCloudParams pointCloudFrame;
        
        // Load src_0_mesh point cloud for this frame
        loadPointCloudFormat(src0Path, pointCloudFrame.src_0, i);
        
        // Load src_45_mesh point cloud for this frame
        std::string src45Path = pointCloudParamsPath + "/src_45_mesh";
        loadPointCloudFormat(src45Path, pointCloudFrame.src_45, i);
        
        std::cout << "Loaded frame " << i << " - src_0: " << pointCloudFrame.src_0.points.size() 
                  << " points, src_45: " << pointCloudFrame.src_45.points.size() << " points" << std::endl;
        
        PCDs.push_back(pointCloudFrame);
    }
    
    std::cout << "Loaded " << PCDs.size() << " point cloud frames" << std::endl;
}


void ScanDataLoader::loadPointCloudFormat(const std::string& formatPath, PointCloudFormat& format, int frameIndex) {
    if (!fs::exists(formatPath)) {
        std::cout << "Warning: " << formatPath << " directory not found" << std::endl;
        return;
    }
    
    // Read integrated mesh file for specified frame index
    char filename[32];
    sprintf_s(filename, sizeof(filename), "%06d.bin", frameIndex);
    std::string filePath = formatPath + "/" + filename;
    
    if (!fs::exists(filePath)) {
        std::cout << "Warning: Point cloud file not found: " << filePath << std::endl;
        return;
    }
    
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Warning: Cannot open point cloud file: " << filePath << std::endl;
        return;
    }
    
    size_t pointCount;
    file.read(reinterpret_cast<char*>(&pointCount), sizeof(pointCount));
    
    if (pointCount == 0) {
        std::cout << "Warning: No points in file: " << filePath << std::endl;
        file.close();
        return;
    }
    

    format.points.resize(pointCount);
    format.normals.resize(pointCount);
    format.colors.resize(pointCount);
    
    // Read points + normals + colors for each point in order
    for (size_t i = 0; i < pointCount; ++i) {

        float pointData[3];
        file.read(reinterpret_cast<char*>(pointData), 3 * sizeof(float));
        format.points[i] = cv::Point3f(pointData[0], pointData[1], pointData[2]);
        

        float normalData[3];
        file.read(reinterpret_cast<char*>(normalData), 3 * sizeof(float));
        format.normals[i] = cv::Point3f(normalData[0], normalData[1], normalData[2]);
        

        unsigned char colorData[3];
        file.read(reinterpret_cast<char*>(colorData), 3 * sizeof(unsigned char));
        format.colors[i] = cv::Vec3b(colorData[0], colorData[1], colorData[2]);
    }
    
    file.close();
    std::cout << "Loaded point cloud: " << pointCount << " points from " << filePath << std::endl;
}

void ScanDataLoader::loadCameraParams() {



    std::string cameraParamsPath = rootPath + "/camera_params";
    if (!fs::exists(cameraParamsPath)) {
        std::cout << "Warning: camera_params directory not found" << std::endl;
        return;
    }

    std::cout << "Loading camera parameters..." << std::endl;

    // Read integrated mesh file
    char filename[32];
    sprintf_s(filename, sizeof(filename), "%06d_intrinsic.txt", 0);
    std::string filePath = cameraParamsPath + "/" + filename;


    if (!fs::exists(filePath)) {
        std::cout << "Warning: File not found: " << filePath << std::endl;
        return;
    }

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Warning: Cannot open file: " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        // Parse different parameter types
        if (line.find("Focal Length X (fx):") != std::string::npos) {
            sscanf_s(line.c_str(), "Focal Length X (fx): %f", &cameraParams.fx);
        }
        else if (line.find("Focal Length Y (fy):") != std::string::npos) {
            sscanf_s(line.c_str(), "Focal Length Y (fy): %f", &cameraParams.fy);
        }
        else if (line.find("Principal Point X (cx):") != std::string::npos) {
            sscanf_s(line.c_str(), "Principal Point X (cx): %f", &cameraParams.cx);
        }
        else if (line.find("Principal Point Y (cy):") != std::string::npos) {
            sscanf_s(line.c_str(), "Principal Point Y (cy): %f", &cameraParams.cy);
        }
        else if (line.find("Image Width:") != std::string::npos) {
            int imgWidth;
            sscanf_s(line.c_str(), "Image Width: %d", &imgWidth);
            if (imgWidth != width) {
                width = imgWidth;
                std::cout << "Updated image width to: " << width << std::endl;
            }
        }
        else if (line.find("Image Height:") != std::string::npos) {
            int imgHeight;
            sscanf_s(line.c_str(), "Image Height: %d", &imgHeight);
            if (imgHeight != height) {
                height = imgHeight;
                std::cout << "Updated image height to: " << height << std::endl;
            }
        }
    }

    file.close();

    // Print loaded camera parameters
    std::cout << "=== Loaded Camera Parameters ===" << std::endl;
    std::cout << "Focal Length X (fx): " << cameraParams.fx << std::endl;
    std::cout << "Focal Length Y (fy): " << cameraParams.fy << std::endl;
    std::cout << "Principal Point X (cx): " << cameraParams.cx << std::endl;
    std::cout << "Principal Point Y (cy): " << cameraParams.cy << std::endl;
    std::cout << "Image Size: " << width << "x" << height << std::endl;
    std::cout << "=================================" << std::endl;
}

    
// Main load function   
bool ScanDataLoader::load() {
    std::cout << "Loading scan data from: " << rootPath << std::endl;
    
    if (!fs::exists(rootPath)) {
        std::cerr << "Error: Root path does not exist: " << rootPath << std::endl;
        return false;
    }
    
    try {
        loadBaseParams();
        // loadImageParams();
        loadMatrixParams();
        // loadDepthMapParams();
        loadPointCloudParams();
        loadCameraParams();
        
        std::cout << "Scan data loading completed!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error during loading: " << e.what() << std::endl;
        return false;
    }
}

//bool ScanDataLoader::load(int maxDepthFrames) {
//    std::cout << "Loading scan data from: " << rootPath << std::endl;
//    
//    if (!fs::exists(rootPath)) {
//        std::cerr << "Error: Root path does not exist: " << rootPath << std::endl;
//        return false;
//    }
//    
//    try {
//        loadBaseParams();
//        loadImageParams();
//        loadMatrixParams();
//        loadDepthMapParams();
//        loadPointCloudParams();
//        
//        std::cout << "Scan data loading completed!" << std::endl;
//        return true;
//    } catch (const std::exception& e) {
//        std::cerr << "Error during loading: " << e.what() << std::endl;
//        return false;
//    }
//}


void ScanDataLoader::printSummary() const {
    std::cout << "\n=== Scan Data Summary ===" << std::endl;
    std::cout << "Base Parameters:" << std::endl;
    std::cout << "  - Patch ID: " << baseParams.patchID << std::endl;
    std::cout << "  - Tolerance: " << baseParams.Tolerance << std::endl;
    std::cout << "  - Robust Voxel Ratio: " << baseParams.RobustVoxelRatio << std::endl;
    std::cout << "  - Robust Voxel Count: " << baseParams.RobustVoxelCount << std::endl;
    std::cout << "  - X Unit: " << baseParams.xUnit << std::endl;
    std::cout << "  - Y Unit: " << baseParams.yUnit << std::endl;
    std::cout << "  - Point Magnitude: " << baseParams.pointMag << std::endl;
    std::cout << "  - Use AASF: " << (baseParams.bUseAASF ? "true" : "false") << std::endl;
    std::cout << "  - Deep Learning Enable: " << (baseParams.bDeepLearningEnable ? "true" : "false") << std::endl;
    
    std::cout << "Image Parameters:" << std::endl;
    std::cout << "  - Deeplearning Inference: " << imageParams.deeplearning_inference.size() << " pixels" << std::endl;
    std::cout << "  - Current img 0: " << imageParams.current_img_0.size() << " pixels" << std::endl;
    std::cout << "  - Current img 45: " << imageParams.current_img_45.size() << " pixels" << std::endl;
    std::cout << "  - Confidence map 0: " << imageParams.confidence_map_0.size() << " pixels" << std::endl;
    std::cout << "  - Confidence map 45: " << imageParams.confidence_map_45.size() << " pixels" << std::endl;
    std::cout << "  - Depthmap mask: " << imageParams.depthmap_mask.size() << " pixels" << std::endl;
    std::cout << "  - Image Size: " << width << "x" << height << std::endl;
    
    std::cout << "Matrix Parameters:" << std::endl;
    std::cout << "  - Transform_0 matrices: " << matrixParams.transform_0.size() << std::endl;
    std::cout << "  - Transform_45 matrices: " << matrixParams.transform_45.size() << std::endl;
    
    std::cout << "Depth Map Parameters:" << std::endl;
    std::cout << "  - High resolution frames: " << depthMapParams.high_resolution.size() << std::endl;
    std::cout << "  - Low resolution frames: " << depthMapParams.low_resolution.size() << std::endl;
    if (!depthMapParams.high_resolution.empty()) {
        const auto& firstFrame = depthMapParams.high_resolution[0];
        std::cout << "  - Colormap size: " << firstFrame.colormap.size() << " bytes" << std::endl;
        std::cout << "  - Depthmap size: " << firstFrame.depthmap.size() << " pixels" << std::endl;
        std::cout << "  - Normalmap size: " << firstFrame.normalmap.size() << " components" << std::endl;
    }
    
    std::cout << "Point Cloud Parameters:" << std::endl;
    std::cout << "  - Total frames loaded: " << PCDs.size() << std::endl;
    for (size_t i = 0; i < PCDs.size(); ++i) {
        std::cout << "  - Frame " << i << ": src_0=" << PCDs[i].src_0.points.size()
                  << " points, src_45=" << PCDs[i].src_45.points.size() << " points" << std::endl;
    }
}

