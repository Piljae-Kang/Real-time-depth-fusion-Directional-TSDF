#pragma once

#include "CUDAHashData.h"
#include "CUDAHashRef.h"
#include <cuda_runtime.h>
#include <string>

// Forward declaration
class VoxelScene;

/**
 * RayCastRender: Ray casting renderer for voxel-based SDF
 *
 * Key Features:
 * - Ray casting through SDF voxel grid
 * - Zero-crossing detection using bisection
 * - Trilinear interpolation for smooth SDF sampling
 * - Output depth, color, and normal maps
 */
class RayCastRender {
private:
    bool m_isAllocated;
    int m_width;
    int m_height;
    
    // GPU output buffers
    float4* m_d_outputDepth;
    float4* m_d_outputColor;
    float4* m_d_outputNormal;
    float4* m_d_outputPosition;
    
    // Extracted surface points buffers
    float4* m_d_extractedPositions;  // xyz + padding
    float4* m_d_extractedColors;     // rgb + padding
    float4* m_d_extractedNormals;    // normal xyz + padding
    int* m_d_extractedCount;         // Atomic counter for extracted points
    int m_maxExtractedPoints;        // Maximum number of points to extract
    int m_numExtractedPoints;         // Actual number of extracted points
    
public:
    /**
     * Constructor
     */
    RayCastRender();
    
    /**
     * Destructor
     */
    ~RayCastRender();
    
    /**
     * Initialize renderer with output size
     * @param width output image width
     * @param height output image height
     */
    bool initialize(int width, int height);
    
    /**
     * Render SDF using ray casting
     * @param voxelScene pointer to VoxelScene containing SDF data
     * @param cameraPos camera position in world space
     * @param cameraTransform camera transformation matrix (4x4)
     * @param fx, fy, cx, cy camera intrinsics
     * @param minDepth, maxDepth depth range
     */
    void render(const VoxelScene* voxelScene,
                float3 cameraPos,
                float* cameraTransform,
                float fx, float fy, float cx, float cy,
                float minDepth, float maxDepth);
    
    /**
     * Download rendered results from GPU
     * @param h_outputDepth host buffer for depth
     * @param h_outputColor host buffer for color
     * @param h_outputNormal host buffer for normals
     */
    void downloadResults(float4* h_outputDepth,
                        float4* h_outputColor,
                        float4* h_outputNormal);

    // Download only depth (1-channel float) for direct imshow
    void downloadDepthFloat(float* h_outputDepthFloat);

    /**
     * Save current raycast point cloud (world positions + colors) as PLY
     */
    bool savePointCloudPLY(const std::string& filePath);
    
    /**
     * Extract surface points from voxel grid (voxelwise extraction)
     * Similar to Marching Cubes point extraction but simpler
     * @param voxelScene pointer to VoxelScene containing SDF data
     * @param minSDF minimum SDF threshold for surface extraction
     * @param maxSDF maximum SDF threshold for surface extraction
     * @param minWeight minimum weight threshold for point extraction (default: 1)
     * @return number of extracted points
     */
    int extractSurfacePoints(const VoxelScene* voxelScene,
                             float minSDF = -0.01f,
                             float maxSDF = 0.01f,
                             unsigned int minWeight = 1);
    
    /**
     * Save extracted surface points as PLY
     * @param filePath output PLY file path
     * @return success status
     */
    bool saveExtractedSurfacePointsPLY(const std::string& filePath);
    
    /**
     * Get GPU buffer pointers (for direct GPU access)
     */
    float4* getOutputDepth() const { return m_d_outputDepth; }
    float4* getOutputColor() const { return m_d_outputColor; }
    float4* getOutputNormal() const { return m_d_outputNormal; }
    float4* getOutputPositions() const { return m_d_outputPosition; }
    
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    
    bool isAllocated() const { return m_isAllocated; }
};

