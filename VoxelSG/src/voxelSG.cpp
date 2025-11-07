#include <iostream>
#include "../include/VoxelVisualizer.h"
#include "../include/globalPramasConfig.h"
#include "../include/VoxelScene.h"
#include "../include/CUDAHashData.h"
#include "../include/CustomDepthMapGenerator.h"
#include "../include/RayCastRender.h"
#include "../include/scanData.h"
#include "../include/scanData2.h"

#define ENABLE_RAY_CAST_RENDERING

//#define HUBITZ_DATA_NEW  // Comment out to use scanDataLoader

int main(){


    std::cout << "VoxelSG - Scan Data Loader" << std::endl;

    
    

#ifdef HUBITZ_DATA_NEW
    
    std::string scanDataPath = "C:\\Users\\pjkang\\Desktop\\PLUGIN Release\\PLUGIN Release\\scanData\\scan_20251027_123454.379";
    
    ScanDataLoader loader(scanDataPath);


    
    // Load only first 3 frames for testing (optional)
    // loader.loadDepthMapParams(3); // This will load only 3 frames
    
    if (loader.load()) { // Load maximum 3 frames for depthmap
#else

    std::string scanDataPath = "Z:\\Dataset\\hubitz_project\\Hubitz\\data\\teeth";

    ScanDataLoader2 loader(scanDataPath);
    
    // Load data using ScanDataLoader2 methods
    loader.loadCameraFromFile();
    loader.loadImageFromFile();
    loader.loadTransformFile();
    loader.loadPointCloudfromFile();
    
    // ScanDataLoader2 doesn't have a single load() method, so we continue

#endif
  
    loader.printSummary();
       
    const auto& baseParams = loader.getBaseParams();
    const auto& imageParams = loader.getImageParams();
    const auto& matrixParams = loader.getMatrixParams();
        
    // Matrix data analysis
    std::cout << "\n=== Matrix Data Analysis ===" << std::endl;
    std::cout << "Transform_0 matrices loaded: " << matrixParams.transform_0.size() << std::endl;
    std::cout << "Transform_45 matrices loaded: " << matrixParams.transform_45.size() << std::endl;
        
    if (!matrixParams.transform_0.empty()) {
        cv::Mat firstTransform = matrixParams.transform_0[0];
        std::cout << "First Transform_0 matrix:" << std::endl;
        std::cout << firstTransform << std::endl;
    }
        
#ifdef HUBITZ_DATA_NEW
    // DepthMap data analysis (only available in HUBITZ_DATA_NEW)
    const auto& depthMapParams = loader.getDepthMapParams();
    std::cout << "\n=== Depth Map Data Analysis ===" << std::endl;
    std::cout << "High resolution frames: " << depthMapParams.high_resolution.size() << std::endl;
    std::cout << "Low resolution frames: " << depthMapParams.low_resolution.size() << std::endl;
        
    if (!depthMapParams.high_resolution.empty()) {
        const auto& firstDepthFrame = depthMapParams.high_resolution[0];
        std::cout << "First high-res frame - Colormap: " << firstDepthFrame.colormap.size() 
                    << " bytes, Depthmap: " << firstDepthFrame.depthmap.size() 
                    << " pixels, Normalmap: " << firstDepthFrame.normalmap.size() << " components" << std::endl;
    }
        
    // Individual frame loading example (memory efficient)
    std::cout << "\n=== Individual Frame Loading Example ===" << std::endl;
    DepthMapParams::DepthMapFrame singleFrame;
    if (loader.loadSingleDepthMapFrame(0, "high_resolution", singleFrame)) {
        std::cout << "Successfully loaded single frame 0:" << std::endl;
        std::cout << "  - Colormap: " << singleFrame.colormap.size() << " bytes" << std::endl;
        std::cout << "  - Depthmap: " << singleFrame.depthmap.size() << " pixels" << std::endl;
        std::cout << "  - Normalmap: " << singleFrame.normalmap.size() << " components" << std::endl;
    }
        
    // Depth map visualization with OpenCV
    std::cout << "\n=== Depth Map OpenCV Visualization ===" << std::endl;
    if (!depthMapParams.high_resolution.empty()) {
        VisualizeDepthMapWithOpenCV(loader, "high_resolution", 0);
    }

    // Depth map to 3D point cloud visualization
    std::cout << "\n=== Depth Map to 3D Point Cloud Visualization ===" << std::endl;
    if (!depthMapParams.high_resolution.empty()) {
        VisualizeDepthMapAsPointCloud(loader, "high_resolution", 0);
    }
#endif
        
    // Transform and visualize PCD in World coordinates
    TransformAndVisualizePCDInWorld(loader, 0);
        
    // === Custom Depth Map Generation Example ===
    std::cout << "\n=== Custom Depth Map Generation ===" << std::endl;
        
    // Create custom depth map generator
    CustomDepthMapGenerator customGenerator;
        
    // Set camera parameters from loaded data (much simpler!)
    customGenerator.setCameraParamsFromLoader(loader);
        
    // Get all point cloud frames (now supports multiple frames)
    const auto& allPointCloudFrames = loader.getAllPointCloudParams();

    // Limit test_idx to actual available frames
    int test_idx = (std::min)(20, static_cast<int>(allPointCloudFrames.size()));
    if (test_idx == 0) {
        std::cerr << "No point cloud frames available!" << std::endl;
        return -1;
    }
    
    std::cout << "Processing " << test_idx << " out of " << allPointCloudFrames.size() << " available frames" << std::endl;
    
    // Store generated depth maps for all frames (similar to PCD structure)
    CustomDepthMapGenerator::GeneratedDepthMapFrames allGeneratedDepthMaps;

    for (int i = 0; i < test_idx; i++) {
        
        // Safety check: ensure frame index is valid
        if (i >= static_cast<int>(allPointCloudFrames.size())) {
            std::cerr << "Warning: Frame index " << i << " exceeds available frames. Skipping." << std::endl;
            continue;
        }

        // Use first frame for backward compatibility (same as getPointCloudParams())
        const auto& pointCloudParams = allPointCloudFrames[i];

        std::cout << "Loaded " << allPointCloudFrames.size() << " point cloud frames" << std::endl;

        // Generate depth map from point cloud data (using first frame)
        if (!pointCloudParams.src_0.points.empty()) {
            std::cout << "Generating custom depth map from point cloud (src_0, frame " << i << ")..." << std::endl;

            auto customDepthMap = customGenerator.generateFromPointCloud(
                pointCloudParams.src_0.points,
                pointCloudParams.src_0.normals,
                pointCloudParams.src_0.colors
            );
            
            // Store in vector for later use
            allGeneratedDepthMaps.src_0.push_back(customDepthMap);

            // Visualize custom generated depth map
            VisualizeCustomDepthMap(customDepthMap);

            // Save custom depth map
            std::string frameSuffix = "_frame" + std::to_string(i);
            customGenerator.saveDepthMap(customDepthMap, "output/custom_depthmap", "custom_generated" + frameSuffix);
        }

        // Generate depth map from 45-degree point cloud (also in camera coordinates)
        if (!pointCloudParams.src_45.points.empty()) {
            std::cout << "Generating custom depth map from point cloud (src_45, frame " << i << ")..." << std::endl;

            // Point cloud is already in Camera coordinates, use it directly (no transform needed)
            auto customDepthMap45 = customGenerator.generateFromPointCloud(
                pointCloudParams.src_45.points,
                pointCloudParams.src_45.normals,
                pointCloudParams.src_45.colors
            );
            
            // Store in vector for later use
            allGeneratedDepthMaps.src_45.push_back(customDepthMap45);

            // Visualize 45-degree depth map
            VisualizeCustomDepthMap(customDepthMap45);

            // Save 45-degree depth map
            std::string frameSuffix = "_frame" + std::to_string(i);
            customGenerator.saveDepthMap(customDepthMap45, "output/custom_depthmap", "custom_45_degree" + frameSuffix);
        }

        // Optional: Process additional frames if available
        if (allPointCloudFrames.size() > 1) {
            std::cout << "\n=== Processing Additional Frames ===" << std::endl;
            for (size_t frameIdx = 1; frameIdx < allPointCloudFrames.size() && frameIdx < 5; ++frameIdx) {
                const auto& frame = allPointCloudFrames[frameIdx];
                std::cout << "Frame " << frameIdx << ": src_0=" << frame.src_0.points.size()
                    << " points, src_45=" << frame.src_45.points.size() << " points" << std::endl;
                // You can process additional frames here if needed
            }
        }

        std::cout << "Custom depth map generation completed!" << std::endl;


        std::cout << "\n=== Point Cloud Visualization Started ===" << std::endl;

        // src_0 pcd visualization
        if (!pointCloudParams.src_0.points.empty() && i % 10 == 0) {
            std::cout << "\n=== src_0 Point Cloud Visualization ===" << std::endl;
            VisualizePointCloudWithOpen3D(pointCloudParams.src_0.points,
                pointCloudParams.src_0.normals,
                pointCloudParams.src_0.colors,
                "VoxelSG_src_0");
        }
        else {
            std::cout << "src_0 point cloud is empty." << std::endl;
        }

        // src_45 pcd visualization
        if (!pointCloudParams.src_45.points.empty() && i % 10 == 0) {
            std::cout << "\n=== src_45 Point Cloud Visualization ===" << std::endl;
            VisualizePointCloudWithOpen3D(pointCloudParams.src_45.points,
                pointCloudParams.src_45.normals,
                pointCloudParams.src_45.colors,
                "VoxelSG_src_45");
        }
        else {
            std::cout << "src_45 point cloud is empty." << std::endl;
        }

        std::cout << "\n=== All Point Cloud Visualization Completed ===" << std::endl;
        
    }
    
    // Summary of generated depth maps
    std::cout << "\n=== Generated Depth Maps Summary ===" << std::endl;
    std::cout << "src_0 depth maps: " << allGeneratedDepthMaps.src_0.size() << " frames" << std::endl;
    std::cout << "src_45 depth maps: " << allGeneratedDepthMaps.src_45.size() << " frames" << std::endl;
    
    // Now you can access generated depth maps like:
    // allGeneratedDepthMaps.src_0[frameIndex] for src_0 depth map of specific frame
    // allGeneratedDepthMaps.src_45[frameIndex] for src_45 depth map of specific frame
        
    // =============================================================================
    // Initialize Global Parameters and VoxelScene
    // =============================================================================
    
    std::cout << "\n=== Initializing VoxelScene ===" << std::endl;
    
    // Initialize global parameters
    GlobalParamsConfig::get().initialize();
    GlobalParamsConfig::get().print();
    
    // Create Params from GlobalParamsConfig
    Params params(GlobalParamsConfig::get());
    
    std::cout << "\n=== Creating VoxelScene ===" << std::endl;



    
    try {
        // Create VoxelScene (this will allocate GPU memory and initialize)
        VoxelScene scene(params);
        
        // VoxelScene created successfully
        std::cout << "VoxelScene created successfully" << std::endl;
        
        // TODO: Add integration code here when needed
        
        std::cout << "\n=== VoxelScene Created Successfully ===" << std::endl;
        
        // Print statistics
        scene.printStats();
//        
        // Choose depth map source
        const bool USE_CUSTOM_DEPTHMAP = true;  // true: use custom, false: use loaded
//        
        std::cout << "\n=== Integrating Depth Map ===" << std::endl;
//        
        const auto& cameraParams = loader.getCameraParams();
// 
        

        const auto& depthMapParams = loader.getDepthMapParams();
        std::cout << "Using " << (USE_CUSTOM_DEPTHMAP ? "CUSTOM" : "LOADED") << " depth map" << std::endl;
        
        const DepthMapParams::DepthMapFrame* frame = nullptr;

        int depthWidth, depthHeight;
        
        if (USE_CUSTOM_DEPTHMAP) {


            for (int i = 0; i < test_idx; i++) {
            //for (int i = 0; i < 20; i++) {

                if (loader.getPointCloudParams(i).src_0.points.empty()) {
                    std::cerr << "No point cloud data available!" << std::endl;
                    return -1;
                }

                // Choose which depth map to use
                CustomDepthMapGenerator::GeneratedDepthMap customDepthMap;
                

                customDepthMap = allGeneratedDepthMaps.src_0[i];

                depthWidth = customDepthMap.width;
                depthHeight = customDepthMap.height;


                std::cout << "Using camera parameters:" << std::endl;
                std::cout << "  fx: " << cameraParams.fx << ", fy: " << cameraParams.fy << std::endl;
                std::cout << "  cx: " << cameraParams.cx << ", cy: " << cameraParams.cy << std::endl;
            
                // Convert depthmap data to GPU
                float3* d_depthmap = nullptr;
                uchar3* d_colormap = nullptr;
                float3* d_normalmap = nullptr;

                size_t pixelCount = customDepthMap.pointmap.size();
                std::cout << "-------------------------\n";
                std::cout << "depth map index : " << i << "\n";
                std::cout << "Custom depth map sizes:" << std::endl;
                std::cout << "  pointmap.size(): " << customDepthMap.pointmap.size() << std::endl;
                std::cout << "  depthmap.size(): " << customDepthMap.depthmap.size() << std::endl;
                std::cout << "  colormap.size(): " << customDepthMap.colormap.size() << std::endl;
                std::cout << "  normalmap.size(): " << customDepthMap.normalmap.size() << std::endl;
                std::cout << "  width x height: " << customDepthMap.width << " x " << customDepthMap.height
                    << " = " << (customDepthMap.width * customDepthMap.height) << std::endl;
                std::cout << "Pixel Count (used): " << pixelCount << std::endl;
                std::cout << "-------------------------\n";

                    //Check if pixelCount is valid
                if (pixelCount == 0) {
                    std::cerr << "ERROR: pixelCount is 0! Cannot allocate GPU memory." << std::endl;
                    return -1;
                }
            
                // Allocate GPU memory using cv::Vec3f size (same as float3 but for consistency)
                cudaError_t err1 = cudaMalloc(&d_depthmap, pixelCount * sizeof(cv::Vec3f));
                cudaError_t err2 = cudaMalloc(&d_colormap, pixelCount * sizeof(cv::Vec3b));
                cudaError_t err3 = cudaMalloc(&d_normalmap, pixelCount * sizeof(cv::Vec3f));
            
                // Check for allocation errors
                if (err1 != cudaSuccess) {
                    std::cerr << "CUDA malloc error for depthmap: " << cudaGetErrorString(err1) << std::endl;
                    return -1;
                }
                if (err2 != cudaSuccess) {
                    std::cerr << "CUDA malloc error for colormap: " << cudaGetErrorString(err2) << std::endl;
                    cudaFree(d_depthmap);
                    return -1;
                }
                if (err3 != cudaSuccess) {
                    std::cerr << "CUDA malloc error for normalmap: " << cudaGetErrorString(err3) << std::endl;
                    cudaFree(d_depthmap);
                    cudaFree(d_colormap);
                    return -1;
                }
            
                std::cout << "  GPU memory allocated successfully: d_depthmap=" << (void*)d_depthmap 
                            << ", d_colormap=" << (void*)d_colormap << ", d_normalmap=" << (void*)d_normalmap << std::endl;
                    
                // Copy data to GPU
                cudaError_t err4, err5, err6;

                err4 = cudaMemcpy(d_depthmap, customDepthMap.pointmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);
                err5 = cudaMemcpy(d_colormap, customDepthMap.colormap.data(), pixelCount * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);
                err6 = cudaMemcpy(d_normalmap, customDepthMap.normalmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);

                if (err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
                std::cerr << "CUDA memcpy error: depthmap=" << cudaGetErrorString(err4) 
                          << ", colormap=" << cudaGetErrorString(err5) 
                          << ", normalmap=" << cudaGetErrorString(err6) << std::endl;

                cudaFree(d_depthmap);
                cudaFree(d_colormap);
                cudaFree(d_normalmap);
                return -1;
                }

                 std::cout << "  Data copied to GPU successfully" << std::endl;
                   

                if (!matrixParams.localToCamera.empty() && !matrixParams.cameraToWorld0.empty()) {

                    cv::Mat cameraPoseMatrix = matrixParams.cameraToWorld0[i];

                    // Use global truncation distance
                    float truncationDistance = GlobalParamsConfig::get().g_SDFTruncation;  // Use fixed value for now

                    // Debug: Check if GPU pointers are valid before calling integrateFromScanData
                    std::cout << "  Debug: Before integrateFromScanData call:" << std::endl;
                    std::cout << "    d_depthmap=" << (void*)d_depthmap << ", d_colormap=" << (void*)d_colormap
                        << ", d_normalmap=" << (void*)d_normalmap << std::endl;
                    std::cout << "    depthWidth=" << depthWidth << ", depthHeight=" << depthHeight << std::endl;



                    scene.integrateFromScanData(
                        d_depthmap, d_colormap, d_normalmap,
                        depthWidth, depthHeight,
                        truncationDistance, cameraPoseMatrix,
                        cameraParams.fx, cameraParams.fy, cameraParams.cx, cameraParams.cy);


                    std::cout << "=== Depth Map Integration Complete ===" << std::endl;


#ifdef ENABLE_RAY_CAST_RENDERING
                    // =============================================================================
                    // Ray Cast Rendering
                    // =============================================================================

                    std::cout << "\n=== Starting Ray Cast Rendering ===" << std::endl;

                    // Create ray cast renderer
                    RayCastRender renderer;
                    if (renderer.initialize(depthWidth, depthHeight)) {

                        // Get camera parameters for rendering
                        float3 cameraPos = make_float3(
                            cameraPoseMatrix.at<float>(0, 3),
                            cameraPoseMatrix.at<float>(1, 3),
                            cameraPoseMatrix.at<float>(2, 3)
                        );

                        // Upload camera transform to GPU
                        float* d_cameraTransform = nullptr;
                        cudaMalloc(&d_cameraTransform, 16 * sizeof(float));
                        cudaMemcpy(d_cameraTransform, cameraPoseMatrix.data, 16 * sizeof(float), cudaMemcpyHostToDevice);

                        // Calculate actual depth range from depthmap
                        float minDepth = 1e10f;
                        float maxDepth = -1e10f;


                        for (size_t i = 0; i < customDepthMap.depthmap.size(); i++) {
                            if (customDepthMap.depthmap[i] > 0.0f) {  // Valid depth
                                minDepth = fminf(minDepth, customDepthMap.depthmap[i]);
                                maxDepth = fmaxf(maxDepth, customDepthMap.depthmap[i]);
                            }
                        }

                        // Add some margin (10% on each side)
                        float margin = (maxDepth - minDepth) * 0.1f;
                        minDepth = fmaxf(0.01f, minDepth - margin);  // At least 1cm minimum
                        maxDepth += margin;

                        std::cout << "  Calculated depth range: [" << minDepth << ", " << maxDepth << "] meters" << std::endl;

                        renderer.render(
                            &scene,
                            cameraPos,
                            d_cameraTransform,
                            cameraParams.fx, cameraParams.fy, cameraParams.cx, cameraParams.cy,
                            minDepth, maxDepth
                        );

                        // Download results
                        float4* h_outputDepth = new float4[depthWidth * depthHeight];
                        float4* h_outputColor = new float4[depthWidth * depthHeight];
                        float4* h_outputNormal = new float4[depthWidth * depthHeight];

                        renderer.downloadResults(h_outputDepth, h_outputColor, h_outputNormal);

                        // Also show pure 1-channel depth directly
                        std::vector<float> h_depthFloat(depthWidth * depthHeight);
                        renderer.downloadDepthFloat(h_depthFloat.data());
                        //VisualizeRenderedDepthFloat(h_depthFloat.data(), depthWidth, depthHeight, "RayCast Depth (1ch)");

                        // Also reuse existing VisualizeCustomDepthMap by adapting to its input type
                        {
                            CustomDepthMapGenerator::GeneratedDepthMap rendered;
                            rendered.width = depthWidth;
                            rendered.height = depthHeight;
                            rendered.depthmap.resize((size_t)depthWidth * depthHeight);
                            rendered.colormap.resize(rendered.depthmap.size(), cv::Vec3b(0, 0, 0));
                            rendered.normalmap.resize(rendered.depthmap.size(), cv::Vec3f(0, 0, 0));

                            const float MINF = 1e10f;
                            for (int i = 0; i < depthWidth * depthHeight; ++i) {
                                float v = h_depthFloat[(size_t)i];
                                rendered.depthmap[(size_t)i] = (fabsf(v) >= MINF * 0.5f) ? 0.0f : v;
                            }

                            VisualizeCustomDepthMap(rendered);
                        }

                        std::cout << "  Rendering completed and depth shown." << std::endl;

                        delete[] h_outputDepth;
                        delete[] h_outputColor;
                        delete[] h_outputNormal;
                        cudaFree(d_cameraTransform);
                    }
#endif

                }

                cudaFree(d_depthmap);
                cudaFree(d_colormap);
                cudaFree(d_normalmap);

            }



        } else {
            // Use loaded depth map (only available in HUBITZ_DATA_NEW)
            if (depthMapParams.high_resolution.empty()) {
                std::cout << "No depth map available!" << std::endl;
                return -1;
            }

            for (int i = 0; i < test_idx; i++) {

                frame = &depthMapParams.high_resolution[i];
                depthWidth = loader.h_depthWidth;
                depthHeight = loader.h_depthHeight;

                size_t pixelCount = frame->depthmap.size();
                std::cout << "Loaded depth map sizes:" << std::endl;
                std::cout << "  depthmap.size(): " << frame->depthmap.size() << std::endl;
                std::cout << "  colormap.size(): " << frame->colormap.size() << std::endl;
                std::cout << "  normalmap.size(): " << frame->normalmap.size() << std::endl;


            }
        }
//        
//        if (usingCustom

//            || frame != nullptr

//            ) {
//            
//            std::cout << "Using camera parameters:" << std::endl;
//            std::cout << "  fx: " << cameraParams.fx << ", fy: " << cameraParams.fy << std::endl;
//            std::cout << "  cx: " << cameraParams.cx << ", cy: " << cameraParams.cy << std::endl;
//            std::cout << "  near: " << cameraParams.nearPlane << ", far: " << cameraParams.farPlane << std::endl;
//            
//            // Convert depthmap data to GPU
//            float3* d_depthmap = nullptr;
//            uchar3* d_colormap = nullptr;
//            float3* d_normalmap = nullptr;
//                    
//            // Allocate GPU memory
//            size_t pixelCount;
//            if (usingCustom) {
//                pixelCount = customDepthMap.pointmap.size();
//                std::cout << "-------------------------\n";
//                std::cout << "Custom depth map sizes:" << std::endl;
//                std::cout << "  pointmap.size(): " << customDepthMap.pointmap.size() << std::endl;
//                std::cout << "  depthmap.size(): " << customDepthMap.depthmap.size() << std::endl;
//                std::cout << "  colormap.size(): " << customDepthMap.colormap.size() << std::endl;
//                std::cout << "  normalmap.size(): " << customDepthMap.normalmap.size() << std::endl;
//                std::cout << "  width x height: " << customDepthMap.width << " x " << customDepthMap.height 
//                          << " = " << (customDepthMap.width * customDepthMap.height) << std::endl;
//                std::cout << "Pixel Count (used): " << pixelCount << std::endl;
//                std::cout << "-------------------------\n";

//            } else {
//                pixelCount = frame->depthmap.size();
//                std::cout << "Loaded depth map sizes:" << std::endl;
//                std::cout << "  depthmap.size(): " << frame->depthmap.size() << std::endl;
//                std::cout << "  colormap.size(): " << frame->colormap.size() << std::endl;
//                std::cout << "  normalmap.size(): " << frame->normalmap.size() << std::endl;
//            }

//            
//            // Check if pixelCount is valid
//            if (pixelCount == 0) {
//                std::cerr << "ERROR: pixelCount is 0! Cannot allocate GPU memory." << std::endl;
//                return -1;
//            }
//            
//            // Allocate GPU memory using cv::Vec3f size (same as float3 but for consistency)
//            cudaError_t err1 = cudaMalloc(&d_depthmap, pixelCount * sizeof(cv::Vec3f));
//            cudaError_t err2 = cudaMalloc(&d_colormap, pixelCount * sizeof(cv::Vec3b));
//            cudaError_t err3 = cudaMalloc(&d_normalmap, pixelCount * sizeof(cv::Vec3f));
//            
//            // Check for allocation errors
//            if (err1 != cudaSuccess) {
//                std::cerr << "CUDA malloc error for depthmap: " << cudaGetErrorString(err1) << std::endl;
//                return -1;
//            }
//            if (err2 != cudaSuccess) {
//                std::cerr << "CUDA malloc error for colormap: " << cudaGetErrorString(err2) << std::endl;
//                cudaFree(d_depthmap);
//                return -1;
//            }
//            if (err3 != cudaSuccess) {
//                std::cerr << "CUDA malloc error for normalmap: " << cudaGetErrorString(err3) << std::endl;
//                cudaFree(d_depthmap);
//                cudaFree(d_colormap);
//                return -1;
//            }
//            
//            std::cout << "  GPU memory allocated successfully: d_depthmap=" << (void*)d_depthmap 
//                      << ", d_colormap=" << (void*)d_colormap << ", d_normalmap=" << (void*)d_normalmap << std::endl;
//                    
//            // Copy data to GPU
//            cudaError_t err4, err5, err6;
//            if (usingCustom) {
//                err4 = cudaMemcpy(d_depthmap, customDepthMap.pointmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);
//                err5 = cudaMemcpy(d_colormap, customDepthMap.colormap.data(), pixelCount * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);
//                err6 = cudaMemcpy(d_normalmap, customDepthMap.normalmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);

//            } else {
//                err4 = cudaMemcpy(d_depthmap, frame->depthmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);
//                err5 = cudaMemcpy(d_colormap, frame->colormap.data(), pixelCount * sizeof(cv::Vec3b), cudaMemcpyHostToDevice);
//                err6 = cudaMemcpy(d_normalmap, frame->normalmap.data(), pixelCount * sizeof(cv::Vec3f), cudaMemcpyHostToDevice);
//            }

//            
//            if (err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess) {
//                std::cerr << "CUDA memcpy error: depthmap=" << cudaGetErrorString(err4) 
//                          << ", colormap=" << cudaGetErrorString(err5) 
//                          << ", normalmap=" << cudaGetErrorString(err6) << std::endl;
//                cudaFree(d_depthmap);
//                cudaFree(d_colormap);
//                cudaFree(d_normalmap);
//                return -1;
//            }
//            
//            std::cout << "  Data copied to GPU successfully" << std::endl;
//                    
//            // Get camera transform matrices
//            const auto& matrixParams = loader.getMatrixParams();
//            const auto& localToCameraMatrices = matrixParams.localToCamera;
//            const auto& cameraToWorld0Matrices = matrixParams.cameraToWorld0;
//            
//            if (!localToCameraMatrices.empty() || !cameraToWorld0Matrices.empty()) {
//                cv::Mat cameraPoseMatrix;
//                
//                // Prefer Camera -> World transformation for voxel integration
//                if (!cameraToWorld0Matrices.empty()) {
//                    // Use CameraToWorld (Camera -> World) for 0 degrees view
//                    std::cout << "Using CameraToWorld0 (Camera -> World, 0°)" << std::endl;
//                    cameraPoseMatrix = cameraToWorld0Matrices[0];
//                } else if (!localToCameraMatrices.empty()) {
//                    // Fallback to LocalToCamera (Local -> Camera)
//                    std::cout << "Using LocalToCamera (Local -> Camera)" << std::endl;
//                    cameraPoseMatrix = localToCameraMatrices[0];
//                }
//                        
//                // Get camera parameters
//                const auto& cameraParams = loader.getCameraParams();
//                        
//                // Use global truncation distance
//                float truncationDistance = GlobalParamsConfig::get().g_SDFTruncation;  // Use fixed value for now
//                
//                // Debug: Check if GPU pointers are valid before calling integrateFromScanData
//                std::cout << "  Debug: Before integrateFromScanData call:" << std::endl;
//                std::cout << "    d_depthmap=" << (void*)d_depthmap << ", d_colormap=" << (void*)d_colormap 
//                          << ", d_normalmap=" << (void*)d_normalmap << std::endl;
//                std::cout << "    depthWidth=" << depthWidth << ", depthHeight=" << depthHeight << std::endl;
//                        
//                scene.integrateFromScanData(
//                    d_depthmap, d_colormap, d_normalmap,
//                    depthWidth, depthHeight,
//                    truncationDistance, cameraPoseMatrix,
//                    cameraParams.fx, cameraParams.fy, cameraParams.cx, cameraParams.cy);
//                        
//                std::cout << "=== Depth Map Integration Complete ===" << std::endl;
//            
//#ifdef ENABLE_RAY_CAST_RENDERING
//                // =============================================================================
//                // Ray Cast Rendering
//                // =============================================================================
//                
//                std::cout << "\n=== Starting Ray Cast Rendering ===" << std::endl;
//                
//                // Create ray cast renderer
//                RayCastRender renderer;
//                if (renderer.initialize(depthWidth, depthHeight)) {
//                    
//                    // Get camera parameters for rendering
//                    float3 cameraPos = make_float3(
//                        cameraPoseMatrix.at<float>(0, 3),
//                        cameraPoseMatrix.at<float>(1, 3),
//                        cameraPoseMatrix.at<float>(2, 3)
//                    );
//                    
//                    // Upload camera transform to GPU
//                    float* d_cameraTransform = nullptr;
//                    cudaMalloc(&d_cameraTransform, 16 * sizeof(float));
//                    cudaMemcpy(d_cameraTransform, cameraPoseMatrix.data, 16 * sizeof(float), cudaMemcpyHostToDevice);
//                    
//                    // Calculate actual depth range from depthmap
//                    float minDepth = 1e10f;
//                    float maxDepth = -1e10f;
//                    
//                    if (usingCustom) {
//                        for (size_t i = 0; i < customDepthMap.depthmap.size(); i++) {
//                            if (customDepthMap.depthmap[i] > 0.0f) {  // Valid depth
//                                minDepth = fminf(minDepth, customDepthMap.depthmap[i]);
//                                maxDepth = fmaxf(maxDepth, customDepthMap.depthmap[i]);
//                            }
//                        }
//#ifdef HUBITZ_DATA_NEW
//                    } else {
//                        for (size_t i = 0; i < frame->depthmap.size(); i++) {
//                            float depth = frame->depthmap[i][2];  // Z component
//                            if (depth > 0.0f && depth < 1000.0f) {  // Valid range
//                                minDepth = fminf(minDepth, depth);
//                                maxDepth = fmaxf(maxDepth, depth);
//                            }
//                        }
//                    }
//#endif
//                    
//                    // Add some margin (10% on each side)
//                    float margin = (maxDepth - minDepth) * 0.1f;
//                    minDepth = fmaxf(0.01f, minDepth - margin);  // At least 1cm minimum
//                    maxDepth += margin;
//                    
//                    std::cout << "  Calculated depth range: [" << minDepth << ", " << maxDepth << "] meters" << std::endl;
//                    
//                    renderer.render(
//                        &scene,
//                        cameraPos,
//                        d_cameraTransform,
//                        cameraParams.fx, cameraParams.fy, cameraParams.cx, cameraParams.cy,
//                        minDepth, maxDepth
//                    );
//                    
//                    // Download results
//                    float4* h_outputDepth = new float4[depthWidth * depthHeight];
//                    float4* h_outputColor = new float4[depthWidth * depthHeight];
//                    float4* h_outputNormal = new float4[depthWidth * depthHeight];
//                    
//                    renderer.downloadResults(h_outputDepth, h_outputColor, h_outputNormal);
//
//                    // Also show pure 1-channel depth directly
//                    std::vector<float> h_depthFloat(depthWidth * depthHeight);
//                    renderer.downloadDepthFloat(h_depthFloat.data());
//                    //VisualizeRenderedDepthFloat(h_depthFloat.data(), depthWidth, depthHeight, "RayCast Depth (1ch)");
//
//                    // Also reuse existing VisualizeCustomDepthMap by adapting to its input type
//                    {
//                        CustomDepthMapGenerator::GeneratedDepthMap rendered;
//                        rendered.width = depthWidth;
//                        rendered.height = depthHeight;
//                        rendered.depthmap.resize((size_t)depthWidth * depthHeight);
//                        rendered.colormap.resize(rendered.depthmap.size(), cv::Vec3b(0,0,0));
//                        rendered.normalmap.resize(rendered.depthmap.size(), cv::Vec3f(0,0,0));
//
//                        const float MINF = 1e10f;
//                        for (int i = 0; i < depthWidth * depthHeight; ++i) {
//                            float v = h_depthFloat[(size_t)i];
//                            rendered.depthmap[(size_t)i] = (fabsf(v) >= MINF * 0.5f) ? 0.0f : v;
//                        }
//
//                        VisualizeCustomDepthMap(rendered);
//                    }
//
//                    std::cout << "  Rendering completed and depth shown." << std::endl;
//                    
//                    delete[] h_outputDepth;
//                    delete[] h_outputColor;
//                    delete[] h_outputNormal;
//                    cudaFree(d_cameraTransform);
//                }
//#endif
//            } 
//            else {
//                std::cout << "No camera transform matrix found!" << std::endl;
//                cudaFree(d_depthmap);
//                cudaFree(d_colormap);
//                cudaFree(d_normalmap);
//            }
//        } else {
//            std::cout << "No depth map data available for integration!" << std::endl;
//        }
//        
//        std::cout << "\n=== VoxelScene Test Complete ===" << std::endl;
//        
    } 
    
    catch (const std::exception& e) {
        std::cerr << "Error creating VoxelScene: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n=== Program Completed Successfully ===" << std::endl;




    
    return 0;
}
