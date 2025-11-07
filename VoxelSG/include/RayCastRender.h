#pragma once

#include "CUDAHashData.h"
#include "CUDAHashRef.h"
#include <cuda_runtime.h>

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
     * Get GPU buffer pointers (for direct GPU access)
     */
    float4* getOutputDepth() const { return m_d_outputDepth; }
    float4* getOutputColor() const { return m_d_outputColor; }
    float4* getOutputNormal() const { return m_d_outputNormal; }
    
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    
    bool isAllocated() const { return m_isAllocated; }
};

