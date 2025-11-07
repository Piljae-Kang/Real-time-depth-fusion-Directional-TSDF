#include "../include/RayCastRender.h"
#include "../include/VoxelScene.h"
#include "../include/CUDAHashRef.h"
#include "../include/globalPramasConfig.h"
#include "../include/CUDAHashData.h"
#include <cuda_runtime.h>
#include <iostream>

// Forward declaration of CUDA types
struct float3;
struct float4;
struct uchar3;
struct int3;

// CUDA kernel declaration (external)
extern "C" void rayCastRenderCUDA(CUDAHashRef& hashData, int width, int height,
                                   float* d_depth, float4* d_depth4, float4* d_colors, float4* d_normals,
                                   float3 cameraPos, float* cameraTransform, float fx, float fy, float cx, float cy,
                                   float minDepth, float maxDepth, float voxelSize,
                                   int numBuckets, int bucketSize, int totalHashSize);

/**
 * Constructor
 */
RayCastRender::RayCastRender()
    : m_isAllocated(false)
    , m_width(0)
    , m_height(0)
    , m_d_outputDepth(nullptr)
    , m_d_outputColor(nullptr)
    , m_d_outputNormal(nullptr)
{
}

/**
 * Destructor
 */
RayCastRender::~RayCastRender() {
    if (m_isAllocated) {
        cudaFree(m_d_outputDepth);
        cudaFree(m_d_outputColor);
        cudaFree(m_d_outputNormal);
    }
}

/**
 * Initialize renderer with output size
 */
bool RayCastRender::initialize(int width, int height) {
    if (m_isAllocated) {
        std::cerr << "RayCastRender: Already initialized!" << std::endl;
        return false;
    }
    
    m_width = width;
    m_height = height;
    
    size_t bufferSize = width * height * sizeof(float4);
    
    // Allocate GPU buffers
    cudaError_t err1 = cudaMalloc(&m_d_outputDepth, width * height * sizeof(float));
    cudaError_t err2 = cudaMalloc(&m_d_outputColor, bufferSize);
    cudaError_t err3 = cudaMalloc(&m_d_outputNormal, bufferSize);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << "RayCastRender: Failed to allocate GPU memory!" << std::endl;
        if (m_d_outputDepth) cudaFree(m_d_outputDepth);
        if (m_d_outputColor) cudaFree(m_d_outputColor);
        if (m_d_outputNormal) cudaFree(m_d_outputNormal);
        return false;
    }
    
    m_isAllocated = true;
    std::cout << "RayCastRender: Initialized with size " << width << "x" << height << std::endl;
    
    return true;
}

/**
 * Render SDF using ray casting
 */
void RayCastRender::render(const VoxelScene* voxelScene,
                           float3 cameraPos,
                           float* cameraTransform,
                           float fx, float fy, float cx, float cy,
                           float minDepth, float maxDepth) {
    if (!m_isAllocated) {
        std::cerr << "RayCastRender: Not initialized!" << std::endl;
        return;
    }
    
    if (!voxelScene || !voxelScene->isAllocated()) {
        std::cerr << "RayCastRender: Invalid voxel scene!" << std::endl;
        return;
    }
    
    std::cout << "RayCastRender: Starting ray cast rendering..." << std::endl;
    std::cout << "  Output size: " << m_width << "x" << m_height << std::endl;
    std::cout << "  Depth range: [" << minDepth << ", " << maxDepth << "]" << std::endl;
    
    // Get hash data and params from VoxelScene
    CUDAHashRef hashData = voxelScene->getHashData();
    const Params& params = voxelScene->getParams();
    
    // Call CUDA kernel
    float voxelSize = GlobalParamsConfig::get().g_SDFVoxelSize;
    
    // Allocate temporary buffers for d_depth4
    float4* d_depth4 = nullptr;
    cudaMalloc(&d_depth4, m_width * m_height * sizeof(float4));
    
    rayCastRenderCUDA(
        hashData,
        m_width,
        m_height,
        (float*)m_d_outputDepth,
        d_depth4,
        m_d_outputColor,
        m_d_outputNormal,
        cameraPos,
        cameraTransform,
        fx, fy, cx, cy,
        minDepth,
        maxDepth,
        voxelSize,
        params.hashSlotNum, params.slotSize, params.totalHashSize
    );
    
    cudaFree(d_depth4);
    
    std::cout << "RayCastRender: Ray cast rendering completed!" << std::endl;
}

/**
 * Download rendered results from GPU
 */
void RayCastRender::downloadResults(float4* h_outputDepth,
                                    float4* h_outputColor,
                                    float4* h_outputNormal) {
    if (!m_isAllocated) {
        std::cerr << "RayCastRender: Not initialized!" << std::endl;
        return;
    }
    
    size_t bufferSize = m_width * m_height * sizeof(float4);
    
    // Download depth as float first, then convert to float4
    float* h_depthFloat = new float[m_width * m_height];
    cudaMemcpy(h_depthFloat, m_d_outputDepth, m_width * m_height * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Convert to float4
    for (int i = 0; i < m_width * m_height; i++) {
        h_outputDepth[i] = make_float4(h_depthFloat[i], h_depthFloat[i], h_depthFloat[i], 1.0f);
    }
    delete[] h_depthFloat;
    
    cudaError_t err2 = cudaMemcpy(h_outputColor, m_d_outputColor, bufferSize, cudaMemcpyDeviceToHost);
    cudaError_t err3 = cudaMemcpy(h_outputNormal, m_d_outputNormal, bufferSize, cudaMemcpyDeviceToHost);
    
    if (err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << "RayCastRender: Failed to download results!" << std::endl;
    }
}

void RayCastRender::downloadDepthFloat(float* h_outputDepthFloat) {
    if (!m_isAllocated || h_outputDepthFloat == nullptr) return;
    cudaMemcpy(h_outputDepthFloat, m_d_outputDepth, m_width * m_height * sizeof(float), cudaMemcpyDeviceToHost);
}

