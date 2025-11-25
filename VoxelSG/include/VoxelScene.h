#pragma once

#include "CUDAHashData.h"
#include "globalPramasConfig.h"
#include "CUDAHashRef.h"
#include "DepthCameraData.h"
#include "CUDADepthMapData.h"

#include <cuda_runtime.h>
#include <string>

// External CUDA functions (defined in .cu files)
extern "C" void resetHashDataCUDA(CUDAHashRef & hashData, const Params & params);
extern "C" void allocBlocksCUDA(CUDAHashRef & hashData, const Params & params,
    const DepthCameraData & depthCameraData,
    const DepthCameraParams & depthCameraParams);
extern "C" void integrateDepthMapCUDA(CUDAHashRef & hashData, const Params & params,
    const DepthCameraData & depthCameraData,
    const DepthCameraParams & depthCameraParams);

// New depthmap-based voxel allocation kernels (3-step process)
// Method 1: Camera direction allocation
extern "C" void allocBlocksFromDepthMapMethod1CUDA(CUDAHashRef & hashData, const Params & params,
    const float3* depthmap, int width, int height, float truncationDistance,
    float3 cameraPos, float* cameraTransform);

// Method 2: Normal direction allocation (for inside surface)
extern "C" void allocBlocksFromDepthMapMethod2CUDA(CUDAHashRef & hashData, const Params & params,
    const float3* depthmap, const float3* normalmap, int width, int height, float truncationDistance,
    float3 cameraPos, float* cameraTransform);

// Legacy function (calls Method 1 for backward compatibility)
extern "C" void allocBlocksFromDepthMapCUDA(CUDAHashRef & hashData, const Params & params,
    const float3* depthmap, const float3* normalmap, int width, int height, float truncationDistance,
    float3 cameraPos, float* cameraTransform);
extern "C" void integrateDepthMapIntoBlocksCUDA(CUDAHashRef & hashData, const Params & params,
    const float3* depthmap, const uchar3* colormap, const float3* normalmap,
    int width, int height, float truncationDistance, float3 cameraPos, float* cameraTransform,
    float fx, float fy, float cx, float cy);
extern "C" void compactifyHashCUDA(CUDAHashRef & hashData, const Params & params);

/**
 * VoxelScene: Main class for GPU-based Voxel Hashing and 3D Reconstruction
 *
 * Key Features:
 * - HashData memory allocation and initialization
 * - Depth data integration into SDF
 * - Active block management
 */
class VoxelScene {
public:
    /**
     * Constructor: Initialize VoxelScene with parameters
     * @param params Hash parameters (block count, bucket count, etc.)
     */
    VoxelScene(const Params& params);

    /**
     * Destructor: Free GPU memory
     */
    ~VoxelScene();

    /**
     * Initialize HashData (heap, hash table, all values to 0)
     */
    void reset();

    /**
     * Integrate depth data into SDF
     * @param depthCameraData depth/color data
     * @param depthCameraParams camera parameters
     * @param transform camera transformation matrix (4x4)
     */
    void integrate(const DepthCameraData& depthCameraData,
        const DepthCameraParams& depthCameraParams,
        const float* transform = nullptr);

    /**
     * Integrate depth map from ScanData into SDF (3-step process)
     * @param depthmap 3D positions (float3 per pixel)
     * @param colormap RGB colors (uchar3 per pixel)
     * @param normalmap surface normals (float3 per pixel)
     * @param width depth map width
     * @param height depth map height
     * @param truncationDistance truncation distance for SDF
     */
    void integrateFromScanData(const float3* depthmap, const uchar3* colormap, const float3* normalmap,
        int width, int height, float truncationDistance, const cv::Mat& cameraTransform,
        float fx, float fy, float cx, float cy);


    /**
     * Compact active hash entries (for memory/processing efficiency)
     */
    void compactifyHashEntries();

    /**
     * Get number of free slots in heap (debug)
     * @return Number of free heap slots
     */
    unsigned int getHeapFreeCount() const;

    /**
     * Get number of integrated frames
     * @return Number of integrated frames
     */
    unsigned int getNumIntegratedFrames() const {
        return m_numIntegratedFrames;
    }

    /**
     * Check if memory is allocated
     * @return True if allocated
     */
    bool isAllocated() const {
        return m_isAllocated;
    }

    /**
     * Get CUDAHashRef reference (for debug/mesh extraction)
     * @return CUDAHashRef struct reference
     */
    CUDAHashRef& getHashData() {
        return m_hashData;
    }

    /**
     * Get const CUDAHashRef reference
     * @return Const CUDAHashRef struct reference
     */
    const CUDAHashRef& getHashData() const {
        return m_hashData;
    }

    /**
     * Get Params reference
     * @return Params struct reference
     */
    Params& getParams() {
        return m_params;
    }

    /**
     * Get per-run output directory path (lazy-initialized).
     */
    std::string getOutputDirectory() const;

    /**
     * Get const Params reference
     * @return Const Params struct reference
     */
    const Params& getParams() const {
        return m_params;
    }

    /**
     * Print HashData statistics (debug)
     */
    void printStats() const;

    /**
     * Save to mesh file (after Marching Cubes)
     * @param filename Output file path (.ply)
     * @return Success status
     */
    bool saveToMesh(const std::string& filename);

private:
    /**
     * Allocate HashData memory
     */
    void allocate();

    /**
     * Prepare timestamped output directory for logs and dumps.
     */
    void initializeOutputDirectory();

    /**
     * Write current global parameters into the active output directory.
     */
    void writeGlobalParametersLog() const;

    /**
     * Free HashData memory
     */
    void deallocate();

    /**
     * Allocate SDF blocks needed for depth map
     */
    void allocBlocks(const DepthCameraData& depthCameraData,
        const DepthCameraParams& depthCameraParams);

    /**
     * Integrate depth map into SDF
     */
    void integrateDepthMap(const DepthCameraData& depthCameraData,
        const DepthCameraParams& depthCameraParams);

private:
    Params m_params;                    // Hash parameters
    CUDAHashRef m_hashData;             // GPU memory (hash table, heap, etc.)
    unsigned int m_numIntegratedFrames; // Number of integrated frames
    bool m_isAllocated;                 // Memory allocation status
    std::string m_outputDirectory; // Per-run output directory
};
