// GlobalParamsConfig.h
#pragma once
#include <string>
#include <iostream>

// #define SDF_BLOCK_SIZE 8
#define SDF_BLOCK_SIZE 8
#define HASH_BUCKET_SIZE 10

struct vec3f {
    float x, y, z;
    vec3f() : x(0), y(0), z(0) {}
    vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct vec3i {
    int x, y, z;
    vec3i() : x(0), y(0), z(0) {}
    vec3i(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
};

struct vec4f {
    float x, y, z, w;
    vec4f() : x(0), y(0), z(0), w(0) {}
    vec4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
};

class GlobalParamsConfig {
public:
    // Global variable declaration
    unsigned int g_sensorIdx;
    std::string g_sceneName;
    std::string g_binaryDumpSensorFile0;
    unsigned int g_optimizationIdx;
    int g_gain;
    int g_exposure;
    bool g_autoExposureEnabled;
    bool g_autoWhiteBalanceEnabled;
    bool g_renderToFile;
    bool g_voxelRenderingEnabled;
    unsigned int g_texKeyFrames;
    int g_nVideoFrame;
    int g_startFrame;
    unsigned int g_texPoolPatchWidth;
    unsigned int g_texPoolNumPatches;
    unsigned int g_numTextureTileWidth;
    unsigned int g_texIntegrationWeightMax;
    unsigned int g_texIntegrationWeightSample;
    float g_erodeSigmaStretch;
    unsigned int g_erodeIterStretchBox;
    unsigned int g_erodeIterStretchGauss;
    unsigned int g_erodeIterOccDepth;
    float g_texAngleThreshold_depth;
    float g_texAngleThreshold_update;
    float g_screenBoundaryWidth;
    float g_sigmaAngle;
    float g_sigmaDepth;
    float g_sigmaArea;
    unsigned int g_windowWidth;
    unsigned int g_windowHeight;
    unsigned int g_adapterWidth;
    unsigned int g_adapterHeight;
    float g_sensorDepthMax;
    float g_sensorDepthMin;
    bool g_enableColorCropping;
    unsigned int g_colorCropX;
    unsigned int g_colorCropY;
    unsigned int g_colorCropWidth;
    unsigned int g_colorCropHeight;
    float g_SDFVoxelSize;
    float g_SDFMarchingCubeThreshFactor;
    float g_SDFTruncation;
    float g_SDFTruncationScale;
    float g_SDFMaxIntegrationDistance;
    unsigned int g_SDFIntegrationWeightSample;
    unsigned int g_SDFIntegrationWeightMax;
    unsigned int g_hashNumSlots;
    unsigned int g_hashNumSDFBlocks;
    unsigned int g_hashMaxCollisionLinkedListSize;
    unsigned int g_hashSlotSize;
    unsigned int g_hashNumBuckets;
    float g_SDFRayIncrementFactor;
    float g_SDFRayThresSampleDistFactor;
    float g_SDFRayThresDistFactor;
    bool g_SDFUseGradients;
    bool g_binaryDumpSensorUseTrajectory;
    bool g_binaryDumpSensorUseTrajectoryOnlyInit;
    float g_depthSigmaD;
    float g_depthSigmaR;
    bool g_depthFilter;
    float g_colorSigmaD;
    float g_colorSigmaR;
    bool g_colorFilter;
    vec4f g_materialAmbient;
    vec4f g_materialSpecular;
    vec4f g_materialDiffuse;
    float g_materialShininess;
    vec4f g_lightAmbient;
    vec4f g_lightDiffuse;
    vec4f g_lightSpecular;
    vec3f g_lightDirection;
    unsigned int g_RnderMode;
    bool g_useColorForRendering;
    bool g_playData;
    float g_renderingDepthDiscontinuityThresLin;
    float g_rmappingDepthDiscontinuityThresLin;
    float g_renderingDepthDiscontinuityThresOffset;
    float g_remappingDepthDiscontinuityThresOffset;
    bool g_bUseCameraCalibration;
    unsigned int g_marchingCubesMaxNumTriangles;
    bool g_streamingEnabled;
    vec3f g_streamingVoxelExtents;
    vec3i g_streamingGridDimensions;
    vec3i g_streamingMinGridPos;
    signed int g_streamingInitialChunkListSize;
    float g_streamingRadius;
    vec3f g_streamingPos;
    unsigned int g_streamingOutParts;
    bool g_recordData;
    bool g_recordCompression;
    std::string g_recordDataFile;
    bool g_reconstructionEnabled;
    bool g_offlineProcessing;
    bool g_integrationEnabled;
    bool g_trackingEnabled;
    bool g_timingsDetailledEnabled;
    bool g_timingsDetailledEnabledOurs;
    bool g_timingsTotalEnabled;
    bool g_garbageCollectionEnabled;
    unsigned int g_garbageCollectionStarve;
    int g_warpingMode;
    int g_depth_map_H;
    int g_depth_map_W;

    // Directional TSDF
    int g_maxDirectionalSlots;
    float g_directionThresholdCosine;
    float g_weightMax;

    // Constructor
    GlobalParamsConfig();

    // Singleton
    static GlobalParamsConfig& getInstance();
    static GlobalParamsConfig& get();

    // Methods
    void initialize();
    bool isInitialized() const;
    void setDefault();
    void print() const;
    void validateParameters() const;

private:
    bool m_bIsInitialized;
};
