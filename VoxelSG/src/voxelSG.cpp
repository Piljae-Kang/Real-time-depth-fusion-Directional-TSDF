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

#define HUBITZ_DATA_NEW  // Comment out to use scanDataLoader

// #define USING_HUBITZ_DEPTHMAP

// Enable/Disable streaming functionality
// Set to 1 to enable multi-threaded streaming (GPU↔CPU block management)
// Set to 0 to disable streaming (all blocks stay in GPU memory)
#define ENABLE_STREAMING 0

#if ENABLE_STREAMING
#include "../include/VoxelStreamingManager.h"
#endif

int main() {


    std::cout << "VoxelSG - Scan Data Loader" << std::endl;




#ifdef HUBITZ_DATA_NEW
    
    // std::string data_name = "scan_20251124_150352.887"; // teeth 1
     
    // std::string data_name = "scan_20251125_110032.152"; // teeth 2

    // std::string data_name = "scan_20251126_151102.503"; // teeth 3

    std::string data_name = "scan_20251126_195334.652"; // teeth 400 framesssssssssssssssssssssssssssssssssssssssss

    // std::string data_name = "scan_20251127_103912.743"; // front teeth 400 frame

    // std::string data_name = "scan_20251127_152736.862"; // whole teeth 1000 frame

    // std::string data_name = "scan_20251125_144629.196"; // Thin object

    // std::string data_name = "scan_20251125_153835.792"; // Thin object2

    std::string scanDataPath = "C:\\Users\\pjkang\\Desktop\\PLUGIN Release\\PLUGIN Release\\scanData\\" + data_name;

    ScanDataLoader loader(scanDataPath);



    // Load only first 3 frames for testing (optional)
    // loader.loadDepthMapParams(3); // This will load only 3 frames

    if (loader.load()) {
        std::cout << "Data loaded successfully\n";


#ifdef USING_HUBITZ_DEPTHMAP

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


#endif // DEBUG


    }
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
      
        
    // Transform and visualize PCD in World coordinates
    TransformAndVisualizePCDInWorld(loader, 0);
        
    // =============================================================================
    // Initialize Global Parameters and VoxelScene (before any file outputs)
    // =============================================================================
    std::cout << "\n=== Initializing VoxelScene ===" << std::endl;

    GlobalParamsConfig::get().initialize();
    GlobalParamsConfig::get().print();

    GlobalParamsConfig::get().g_data_name = data_name;

    Params params(GlobalParamsConfig::get());

    std::cout << "\n=== Creating VoxelScene ===" << std::endl;

    VoxelScene scene(params);
    std::cout << "VoxelScene created successfully" << std::endl;
    scene.printStats();

#if ENABLE_STREAMING
    // Initialize VoxelStreamingManager
    std::cout << "\n=== Initializing VoxelStreamingManager ===" << std::endl;
    const auto& gpc = GlobalParamsConfig::get();
    vec3f voxelExtents = vec3f(
        gpc.g_streamingVoxelExtents.x,
        gpc.g_streamingVoxelExtents.y,
        gpc.g_streamingVoxelExtents.z
    );
    vec3i gridDimensions = gpc.g_streamingGridDimensions;
    vec3i minGridPos = gpc.g_streamingMinGridPos;
    unsigned int initialChunkListSize = 1000;  // Initial size for chunk vectors
    bool streamingEnabled = true;  // Enable multi-threaded streaming
    
    VoxelStreamingManager streamingManager(
        scene, params, voxelExtents, gridDimensions, minGridPos,
        initialChunkListSize, streamingEnabled
    );
    std::cout << "VoxelStreamingManager initialized successfully" << std::endl;
#else
    std::cout << "\n=== Streaming DISABLED (all blocks stay in GPU memory) ===" << std::endl;
#endif

    const std::string runOutputDir = scene.getOutputDirectory();
        
    // === Custom Depth Map Generation Example ===
    std::cout << "\n=== Custom Depth Map Generation ===" << std::endl;
        
    // Create custom depth map generator
    CustomDepthMapGenerator customGenerator;
        
    // Set camera parameters from loaded data (much simpler!)
    customGenerator.setCameraParamsFromLoader(loader);
        
    // Get all point cloud frames (now supports multiple frames)
    const auto& allPointCloudFrames = loader.getAllPointCloudParams();

    // Limit test_idx to actual available frames : loader frame = test_idx
    int test_idx = (std::min)(loader.frame_idx, static_cast<int>(allPointCloudFrames.size()));
    //int test_idx = (std::min)(1, static_cast<int>(allPointCloudFrames.size()));
    if (test_idx == 0) {
        std::cerr << "No point cloud frames available!" << std::endl;
        return -1;
    }
    
    std::cout << "Processing " << test_idx << " out of " << allPointCloudFrames.size() << " available frames" << std::endl;
    
    // Store generated depth maps for all frames (similar to PCD structure)
    CustomDepthMapGenerator::GeneratedDepthMapFrames allGeneratedDepthMaps;

    bool Debugging_pcd = false;

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
            VisualizeCustomDepthMap(customDepthMap, runOutputDir, "custom0");

            // Save custom depth map
            //std::string frameSuffix = "_frame" + std::to_string(i);
            //customGenerator.saveDepthMap(
            //    customDepthMap,
            //    runOutputDir + "/custom_depthmap",
            //    "custom_generated" + frameSuffix,
            //    false,
            //    true);
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
            VisualizeCustomDepthMap(customDepthMap45, runOutputDir, "custom45");

            // Save 45-degree depth map
            //std::string frameSuffix = "_frame" + std::to_string(i);
            //customGenerator.saveDepthMap(
            //    customDepthMap45,
            //    runOutputDir + "/custom_depthmap",
            //    "custom_45_degree" + frameSuffix,
            //    false,
            //    true);
        }

        // Generate depth map from total (src_0 + src_45 combined) point cloud
        {
            std::cout << "Generating custom depth map from total point cloud (src_0 + src_45, frame " << i << ")..." << std::endl;
            
            auto customDepthMapTotal = customGenerator.generateFromCombinedPointClouds(
                pointCloudParams.src_0,
                pointCloudParams.src_45
            );
            
            if (!customDepthMapTotal.depthmap.empty()) {
                // Store in vector for later use
                allGeneratedDepthMaps.total.push_back(customDepthMapTotal);

                // Visualize total depth map
                VisualizeCustomDepthMap(customDepthMapTotal, runOutputDir, "custom_total");

                // Save total point cloud as PLY (world coordinates)
                if (!matrixParams.cameraToWorld0.empty() && i < static_cast<int>(matrixParams.cameraToWorld0.size())) {
                    std::string plyPath = runOutputDir + "/totoal_pcd" + "/total_pointcloud_frame_" + std::to_string(i) + ".ply";
                    customGenerator.savePointCloudPLY(customDepthMapTotal, matrixParams.cameraToWorld0[i], plyPath);
                }
            }
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

            if (Debugging_pcd) {

                VisualizePointCloudWithOpen3D(pointCloudParams.src_0.points,
                    pointCloudParams.src_0.normals,
                    pointCloudParams.src_0.colors,
                    "VoxelSG_src_0");
            }


        }
        else {
            std::cout << "src_0 point cloud is empty." << std::endl;
        }

        // src_45 pcd visualization
        if (!pointCloudParams.src_45.points.empty() && i % 10 == 0) {
            std::cout << "\n=== src_45 Point Cloud Visualization ===" << std::endl;


            if (Debugging_pcd) {

                VisualizePointCloudWithOpen3D(pointCloudParams.src_45.points,
                    pointCloudParams.src_45.normals,
                    pointCloudParams.src_45.colors,
                    "VoxelSG_src_45");


            }

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
    std::cout << "total depth maps: " << allGeneratedDepthMaps.total.size() << " frames" << std::endl;
    
    // Now you can access generated depth maps like:
    // allGeneratedDepthMaps.src_0[frameIndex] for src_0 depth map of specific frame
    // allGeneratedDepthMaps.src_45[frameIndex] for src_45 depth map of specific frame
        
    try {
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

                // Choose which depth map to use (use total instead of src_0)
                CustomDepthMapGenerator::GeneratedDepthMap customDepthMap;

                customDepthMap = allGeneratedDepthMaps.total[i];
                //customDepthMap = allGeneratedDepthMaps.src_0[i];
                
                //// Use total depth map (src_0 + src_45 combined) if available, otherwise fall back to src_0
                //if (i < static_cast<int>(allGeneratedDepthMaps.total.size()) && !allGeneratedDepthMaps.total[i].depthmap.empty()) {
                //    customDepthMap = allGeneratedDepthMaps.total[i];
                //    std::cout << "Using total depth map (src_0 + src_45 combined) for frame " << i << std::endl;
                //} else if (i < static_cast<int>(allGeneratedDepthMaps.src_0.size()) && !allGeneratedDepthMaps.src_0[i].depthmap.empty()) {
                //    customDepthMap = allGeneratedDepthMaps.src_0[i];
                //    std::cout << "Using src_0 depth map (fallback) for frame " << i << std::endl;
                //} else {
                //    std::cerr << "No depth map available for frame " << i << std::endl;
                //    continue;
                //}

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

#if ENABLE_STREAMING
                    // Extract camera position from transform matrix
                    vec3f cameraPos = vec3f(
                        cameraPoseMatrix.at<float>(0, 3),
                        cameraPoseMatrix.at<float>(1, 3),
                        cameraPoseMatrix.at<float>(2, 3)
                    );
                    
                    // Extract camera view direction (pose matrix's Z axis = camera forward direction)
                    // Note: If cameraPoseMatrix is camera-to-world, Z axis is forward. If world-to-camera, -Z is forward.
                    vec3f viewDir = vec3f(
                        cameraPoseMatrix.at<float>(0, 2),  // Z direction (camera-to-world transform)
                        cameraPoseMatrix.at<float>(1, 2),
                        cameraPoseMatrix.at<float>(2, 2)
                    );
                    
                    // Calculate sphere center: 100m ahead of camera in view direction
                    vec3f sphereCenter = cameraPos + viewDir * 110.0f;
                    
                    // Streaming radius (same for both stream out and stream in)
                    // Both use sphere center (100m ahead of camera) with the same radius
                    float streamingRadius = 10.0f;  // 10m radius from sphere center
                    
                    bool useParts = true;  // Use parts for streaming out (divide work across frames)
                    
                    // Update sphere center and radius for auxiliary thread (stream in)
                    // (Auxiliary thread will call streamInCopyToGPUBuffer using these values)
                    streamingManager.posCamera_ = sphereCenter;  // Use sphere center (100m ahead)
                    streamingManager.radius_ = streamingRadius;   // Same radius for both stream out and stream in
                    
                    // ===================================================================
                    // STREAMING: Stream out blocks from GPU to CPU (before integration)
                    // Main thread: Find blocks to stream out (auxiliary thread will copy to CPU)
                    // (Like expert code: streamOutToCPUPass0GPU is called before integration)
                    // Stream out uses sphere center (100m ahead) with streamingRadius
                    // ===================================================================
                    std::cout << "\n=== Streaming Out (GPU->CPU) - Find Blocks ===\n";
                    streamingManager.m_frameNumber = i;  // Set frame number for .xyz file naming
                    streamingManager.m_outputDirectory = runOutputDir;  // Set output directory for .xyz files
                    streamingManager.streamOutFindBlocksOnGPU(sphereCenter, streamingRadius, useParts, true);
                    
                    // ===================================================================
                    // STREAMING: Stream in blocks from CPU to GPU (before integration)
                    // Main thread: Only insert to hash table (auxiliary thread does the CPU→GPU copy)
                    // (Like expert code: streamInToGPUPass1GPU is called before integration)
                    // Stream in uses sphereCenter (100m ahead) with streamInRadius (10m)
                    // ===================================================================
                    std::cout << "\n=== Streaming In (CPU->GPU) - Insert to Hash ===\n";
                    streamingManager.streamInInsertToHashTable(true);
#endif
                    
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

                            VisualizeCustomDepthMap(rendered, runOutputDir, "rendered");
                        }

                        std::string plyOutputPath = runOutputDir + "/raycast_outputs/frame_" + std::to_string(i) + ".ply";
                        if (!renderer.savePointCloudPLY(plyOutputPath)) {
                            std::cerr << "Failed to save raycast point cloud: " << plyOutputPath << std::endl;
                        }

                        // Extract surface points at frame 30
                        if (i % 10 ==0) {
                            std::cout << "\n=== Extracting Surface Points at Frame 30 ===" << std::endl;
                            
                            // Use wider SDF range to capture more surface points
                            // -0.05 to 0.05 covers more voxels near the surface
                            int numPoints = renderer.extractSurfacePoints(&scene, -0.1f, 0.1f, 5);
                            
                            if (numPoints > 0) {
                                std::string surfacePlyPath = runOutputDir + "/extract_points_from_voxel/surface_points_frame_" + std::to_string(i) + ".ply";
                                if (renderer.saveExtractedSurfacePointsPLY(surfacePlyPath)) {
                                    std::cout << "Successfully saved " << numPoints << " surface points to " << surfacePlyPath << std::endl;
                                } else {
                                    std::cerr << "Failed to save surface points to " << surfacePlyPath << std::endl;
                                }
                            } else {
                                std::cerr << "No surface points extracted!" << std::endl;
                            }
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

    } 
    
    catch (const std::exception& e) {
        std::cerr << "Error during VoxelScene processing: " << e.what() << std::endl;
        return -1;
    }
    
#if ENABLE_STREAMING
    // Stop streaming thread before cleanup
    std::cout << "\n=== Stopping Streaming Thread ===" << std::endl;
    streamingManager.stopMultiThreading();
#endif
    
    std::cout << "\n=== Program Completed Successfully ===" << std::endl;




    
    return 0;
}
