#include "../include/globalPramasConfig.h"

GlobalParamsConfig::GlobalParamsConfig() {
    setDefault();
    m_bIsInitialized = false;
}

void GlobalParamsConfig::setDefault() {

    // =========================================================================================
    // Sensor and Scene Configuration Parameters
    // =========================================================================================

    g_depth_map_H = 400;
    g_depth_map_W = 480;

    // =========================================================================================
    // SDF Voxel Processing Parameters
    // =========================================================================================
    
    // SDF voxel size: Size of each voxel in the Signed Distance Function (meters)
    // g_SDFVoxelSize = 0.004f;
    //g_SDFVoxelSize = 0.05f;
    g_SDFVoxelSize = 0.01f;
    
    // SDF marching cubes threshold factor: Threshold factor for triangle generation in Marching Cubes algorithm
    g_SDFMarchingCubeThreshFactor = 10.0f;
    
    // SDF truncation distance
    //g_SDFTruncation = 0.08f;
    //g_SDFTruncation = 0.5f;
    g_SDFTruncation = 0.2f;

    
    // SDF truncation scale: Weighted scale factor for truncation (meters)
    g_SDFTruncationScale = 0.01f;
    
    // SDF maximum integration distance: Maximum distance for integrating observations (meters)
    g_SDFMaxIntegrationDistance = 3.0f;
    
    // SDF integration weight sampling count: Number of weight mappings used in observation integration
    g_SDFIntegrationWeightSample = 3;
    
    // SDF integration maximum weight: Maximum weight value used in observation integration
    g_SDFIntegrationWeightMax = 255;

    // Hash table bucket count: Number of buckets in the hash table storing voxel data
    // g_hashNumSlots = 500000;
    g_hashNumSlots = 1000000;
    
    // Hash table SDF block count: Maximum number of SDF blocks that can be stored in the hash table
    g_hashNumSDFBlocks = 1000000;
    
    // Hash collision linked list maximum size: Maximum length of linked list for hash collision resolution
    g_hashMaxCollisionLinkedListSize = 5;

    // Entries per hash slot
    g_hashSlotSize = 10;
    
    // Number of hash buckets
    g_hashNumBuckets = g_hashNumSlots;


    // =========================================================================================
    // Ray Casting Configuration Parameters
    // =========================================================================================
    
    // SDF ray increment factor: Incremental advance parameter in ray casting
    g_SDFRayIncrementFactor = 0.8f;
    
    // SDF ray threshold sample distance factor: Threshold factor for sampling distance during ray casting
    g_SDFRayThresSampleDistFactor = 50.5f;
    
    // SDF ray threshold distance factor: Distance threshold factor where ray terminates
    g_SDFRayThresDistFactor = 50.0f;
    
    // SDF use gradients: Whether to use numerical gradients in ray casting
    g_SDFUseGradients = false;
    
    // Binary dump sensor use trajectory: Whether to use sensor trajectory information
    g_binaryDumpSensorUseTrajectory = false;
    
    // Binary dump sensor use trajectory only init: Whether to use trajectory only during initialization
    g_binaryDumpSensorUseTrajectoryOnlyInit = false;


    // =========================================================================================
    // Directional TSDF Processing Parameters
    // =========================================================================================
    
    // Maximum directional slots: Maximum number of direction slots supported in Directional TSDF
    //g_maxDirectionalSlots = 20;
    //
    // Direction threshold cosine value: Cosine threshold for determining directional consistency in TSDF calculation
    //g_directionThresholdCosine = 0.8f;
    //
    // Weight maximum value: Maximum weight value to use in directional TSDF
    //g_weightMax = 100.0f;


    // =========================================================================================
    // Streaming Processing Parameters
    // =========================================================================================
    
    // Streaming enabled: Whether to process large data in chunk-based streaming
    g_streamingEnabled = true;
    
    // Streaming voxel extents: Spatial extent of each streaming chunk (meters)
    g_streamingVoxelExtents = vec3f(1.1f, 1.0f, 1.0f);
    
    // Streaming grid dimensions: X, Y, Z dimension size of entire streaming grid
    g_streamingGridDimensions = vec3i(513, 513, 513);
    
    // Streaming minimum grid position: Starting coordinates of streaming grid (integer pixel units)
    g_streamingMinGridPos = vec3i(-256, -256, -256);
    
    // Streaming initial chunk list size: Number of chunks to load at streaming start
    g_streamingInitialChunkListSize = 2000;
    
    // Streaming radius: Distance radius for maintaining cache
    g_streamingRadius = 4.0f;
    
    // Streaming position: Center point coordinates for streaming processing
    g_streamingPos = vec3f(0.0f, 0.0f, 3.0f);
    
    // Streaming output parts: Number of parts to divide during streaming output
    g_streamingOutParts = 80;


    // =========================================================================================
    // Camera Configuration Parameters
    // =========================================================================================
    
    // Gain value: Camera sensor sensitivity setting (brightness adjustment)
    g_gain = 1;
    
    // Exposure time: Camera shutter speed (milliseconds, lower = faster shutter)
    g_exposure = 200;
    
    // Auto exposure enabled: Whether camera automatically adjusts optimal exposure time
    g_autoExposureEnabled = false;
    
    // Auto white balance enabled: Whether camera automatically adjusts color temperature
    g_autoWhiteBalanceEnabled = false;

    // =========================================================================================
    // Rendering System Configuration Parameters
    // =========================================================================================
    
    // Render to file: Whether to save completed frames to files
    g_renderToFile = false;
    
    // Voxel rendering enabled: Whether to use volumetric rendering mode
    g_voxelRenderingEnabled = false;
    
    // Window width: Horizontal pixel size of rendering window
    g_windowWidth = 640;
    
    // Window height: Vertical pixel size of rendering window
    g_windowHeight = 480;
    
    // Adapter width: Horizontal size of buffer output by graphics adapter
    g_adapterWidth = 640;
    
    // Adapter height: Vertical size of buffer output by graphics adapter
    g_adapterHeight = 480;

    // =========================================================================================
    // Texture Processing Configuration Parameters
    // =========================================================================================
    
    // Texture key frames: Number of keyframes to store texture information
    g_texKeyFrames = 4;
    
    // Video total frame count: Total number of video frames to process
    g_nVideoFrame = 900;
    
    // Start frame: Starting frame index for data processing
    g_startFrame = 0;
    
    // Texture pool patch width: Horizontal size of each patch in texture memory
    g_texPoolPatchWidth = 4;
    
    // Texture pool patch count: Maximum number of patches to allocate in texture memory
    g_texPoolNumPatches = 1000000;
    
    // Texture tile width: Horizontal size of tiles used in texture mapping
    g_numTextureTileWidth = 1024;
    
    // Texture integration maximum weight: Maximum weight value when mixing textures
    g_texIntegrationWeightMax = 255;
    
    // Texture integration weight sampling count: Sampling mapping to consider when mixing textures
    g_texIntegrationWeightSample = 30;

    // =========================================================================================
    // Erosion Processing Configuration Parameters
    // =========================================================================================
    
    // Erosion sigma stretch: Stretch factor to apply in morphological operations
    g_erodeSigmaStretch = 2.0f;
    
    // Erosion iteration stretch box filter: Number of iterations to stretch with box filter
    g_erodeIterStretchBox = 0;
    
    // Erosion iteration stretch gaussian filter: Number of iterations to stretch with gaussian filter
    g_erodeIterStretchGauss = 0;
    
    // Erosion iteration occlusion depth: Number of erosion iterations to apply to depth information
    g_erodeIterOccDepth = 6;
    
    // Texture angle threshold (depth): Texture angle discrimination threshold when calculating depth
    g_texAngleThreshold_depth = 0.05f;
    
    // Texture angle threshold (update): Texture angle discrimination threshold when updating texture
    g_texAngleThreshold_update = 0.5f;
    
    // Screen boundary width: Pixel width of screen edge processing range
    g_screenBoundaryWidth = 40.0f;
    
    // Sigma angle: Standard deviation parameter used in angle calculation
    g_sigmaAngle = 0.5f;
    
    // Sigma depth: Standard deviation parameter used in depth calculation
    g_sigmaDepth = 0.5f;
    
    // Sigma area: Standard deviation parameter used in area-based filtering
    g_sigmaArea = 1.0f;

    // =========================================================================================
    // Sensor Depth Configuration Parameters
    // =========================================================================================
    
    // Sensor depth maximum: Maximum depth the sensor can measure (meters)
    g_sensorDepthMax = 3.0f;
    
    // Sensor depth minimum: Minimum depth the sensor can measure (meters)
    g_sensorDepthMin = 0.4f;

    // =========================================================================================
    // Color Cropping Configuration Parameters
    // =========================================================================================
    
    // Color cropping enabled: Whether to use cropping feature to process only specific regions
    g_enableColorCropping = false;
    
    // Crop region X coordinate: Starting X coordinate of region to crop (pixel units)
    g_colorCropX = 320;
    
    // Crop region Y coordinate: Starting Y coordinate of region to crop (pixel units)
    g_colorCropY = 272;
    
    // Crop region width: Horizontal size of region to crop (pixel units)
    g_colorCropWidth = 640;
    
    // Crop region height: Vertical size of region to crop (pixel units)
    g_colorCropHeight = 480;


    // =========================================================================================
    // Filtering Configuration Parameters
    // =========================================================================================
    
    // Depth sigma D: Standard deviation parameter for depth data smoothing (spatial domain)
    g_depthSigmaD = 2.0f;
    
    // Depth sigma R: Standard deviation parameter for depth data smoothing (range domain)
    g_depthSigmaR = 0.1f;
    
    // Depth filter enabled: Whether to apply bilateral filter to depth data
    g_depthFilter = true;
    
    // Color sigma D: Standard deviation parameter for color data smoothing (spatial domain)
    g_colorSigmaD = 1.0f;
    
    // Color sigma R: Standard deviation parameter for color data smoothing (range domain)
    g_colorSigmaR = 0.08f;
    
    // Color filter enabled: Whether to apply bilateral filter to color data
    g_colorFilter = false;

    // =========================================================================================
    // Processing Mode Configuration Parameters
    // =========================================================================================
    
    // Integration processing enabled: Whether to integrate new observations into existing model
    g_integrationEnabled = true;
    
    // Tracking processing enabled: Whether to track camera movement
    g_trackingEnabled = true;
    
    // Timings detailed enabled: Whether to measure time consumed by each processing step
    g_timingsDetailledEnabled = false;
    
    // Total timing enabled: Whether to measure time consumed by entire processing pipeline
    g_timingsTotalEnabled = false;
    
    // Garbage collection enabled: Whether to automatically perform memory deallocation and cleanup
    g_garbageCollectionEnabled = false;
    
    // Garbage collection starve: Number of frames to maintain before performing garbage collection
    g_garbageCollectionStarve = 15;

    // =========================================================================================
    // Material Rendering Configuration Parameters
    // =========================================================================================
    
    // Material shininess: Glossiness property of material (adjusts color reflection intensity)
    g_materialShininess = 16.0f;
    
    // Material ambient: Color reflected by material in ambient light (RGBA format)
    g_materialAmbient = vec4f(0.75f, 0.65f, 0.5f, 1.0f);
    
    // Material diffuse: Color reflected diffusely by material (RGBA format)
    g_materialDiffuse = vec4f(1.0f, 0.9f, 0.7f, 1.0f);
    
    // Material specular: Color reflected directly by material (RGBA format)
    g_materialSpecular = vec4f(1.0f, 1.0f, 1.0f, 1.0f);

    // =========================================================================================
    // Lighting Rendering Configuration Parameters
    // =========================================================================================
    
    // Light ambient: Color and intensity of overall ambient lighting (RGBA format)
    g_lightAmbient = vec4f(0.4f, 0.4f, 0.4f, 1.0f);
    
    // Light diffuse: Color and intensity of diffuse lighting from main light source (RGBA format)
    g_lightDiffuse = vec4f(0.6f, 0.52944f, 0.4566f, 0.6f);
    
    // Light specular: Color and intensity of specular lighting from main light source (RGBA format)
    g_lightSpecular = vec4f(0.3f, 0.3f, 0.3f, 1.0f);
    
    // Light direction: Direction vector of main light source (normalized 3D vector)
    g_lightDirection = vec3f(0.0f, -1.0f, 2.0f);

    // =========================================================================================
    // Rendering Options Configuration Parameters
    // =========================================================================================
    
    g_RnderMode = 1;
    
    // Use color for rendering: Whether to use texture color information when rendering
    g_useColorForRendering = false;
    
    // Play data enabled: Whether to play back sensor data sequentially
    g_playData = true;
    
    // Rendering depth discontinuity threshold offset: Offset value for determining depth discontinuity during rendering
    g_renderingDepthDiscontinuityThresOffset = 0.012f;
    
    // Rendering depth discontinuity threshold linear: Linear coefficient for determining depth discontinuity during rendering
    g_renderingDepthDiscontinuityThresLin = 0.001f;
    
    //// Remapping depth discontinuity threshold offset: Offset value for determining discontinuity during depth remapping
    //g_rmappingDepthDiscontinuityThresOffset = 0.012f;
    //
    //// Remapping depth discontinuity threshold linear: Linear coefficient for determining discontinuity during depth remapping
    //g_remappingDepthDiscontinuityThresLin = 0.016f;
    
    // Use camera calibration: Whether to use internal camera calibration parameters
    g_bUseCameraCalibration = false;
    
    // Marching cubes maximum triangle count: Maximum number of triangles to generate in Marching Cubes algorithm
    g_marchingCubesMaxNumTriangles = 250000;

    // =========================================================================================
    // Data Processing Configuration Parameters
    // =========================================================================================
    
    // Record data enabled: Whether to record processed data to file
    g_recordData = false;
    
    // Record compression enabled: Whether to compress data when saving
    g_recordCompression = true;
    
    // Record data file: Path and name of file where data will be saved
    g_recordDataFile = "Dump/test.sens";
    
    // 3D reconstruction enabled: Whether to reconstruct 3D model from input data
    g_reconstructionEnabled = true;
    
    // Offline processing enabled: Whether to process in batch mode rather than real-time
    g_offlineProcessing = false;
    
    // Warping mode: Warping mode number to control texture mapping method
    g_warpingMode = 0;
}

void GlobalParamsConfig::initialize() {
    validateParameters();
    m_bIsInitialized = true;
}

bool GlobalParamsConfig::isInitialized() const {
    return m_bIsInitialized;
}

void GlobalParamsConfig::validateParameters() const {
    if (g_hashNumBuckets == 0) throw std::invalid_argument("Invalid hash buckets");
    if (g_SDFVoxelSize <= 0.0f) throw std::invalid_argument("Invalid voxel size");
    //if (g_maxDirectionalSlots != 2) throw std::invalid_argument("Only 2 directional slots supported");
}

GlobalParamsConfig& GlobalParamsConfig::getInstance() {
    static GlobalParamsConfig s;
    return s;
}

GlobalParamsConfig& GlobalParamsConfig::get() {
    return getInstance();
}

void GlobalParamsConfig::print() const {
    std::cout << "GlobalParamsConfig parameters:" << std::endl;
    std::cout << "g_depth_map_H = " << g_depth_map_H << std::endl;
    std::cout << "g_depth_map_W = " << g_depth_map_W << std::endl;
    std::cout << "g_SDFVoxelSize = " << g_SDFVoxelSize << std::endl;
    std::cout << "g_hashNumSlots = " << g_hashNumSlots << std::endl;
    std::cout << "g_hashSlotSize = " << g_hashSlotSize << std::endl;
    // Add other parameters as needed
}
