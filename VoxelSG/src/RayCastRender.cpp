#include "../include/RayCastRender.h"
#include "../include/VoxelScene.h"
#include "../include/CUDAHashRef.h"
#include "../include/globalPramasConfig.h"
#include "../include/CUDAHashData.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <limits>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

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
    , m_d_outputPosition(nullptr)
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
        cudaFree(m_d_outputPosition);
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
    cudaError_t err4 = cudaMalloc(&m_d_outputPosition, bufferSize);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) {
        std::cerr << "RayCastRender: Failed to allocate GPU memory!" << std::endl;
        if (m_d_outputDepth) cudaFree(m_d_outputDepth);
        if (m_d_outputColor) cudaFree(m_d_outputColor);
        if (m_d_outputNormal) cudaFree(m_d_outputNormal);
        if (m_d_outputPosition) cudaFree(m_d_outputPosition);
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
    
    rayCastRenderCUDA(
        hashData,
        m_width,
        m_height,
        (float*)m_d_outputDepth,
        m_d_outputPosition,
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

bool RayCastRender::savePointCloudPLY(const std::string& filePath) {
    if (!m_isAllocated) {
        std::cerr << "RayCastRender: Not initialized!" << std::endl;
        return false;
    }

    size_t pixelCount = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
    std::vector<float4> h_positions(pixelCount);
    std::vector<float4> h_colors(pixelCount);
    std::vector<float> h_depth(pixelCount);

    cudaError_t errPos = cudaMemcpy(h_positions.data(), m_d_outputPosition, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaError_t errCol = cudaMemcpy(h_colors.data(), m_d_outputColor, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);
    cudaError_t errDepth = cudaMemcpy(h_depth.data(), m_d_outputDepth, pixelCount * sizeof(float), cudaMemcpyDeviceToHost);

    if (errPos != cudaSuccess || errCol != cudaSuccess || errDepth != cudaSuccess) {
        std::cerr << "RayCastRender: Failed to download point cloud buffers!" << std::endl;
        return false;
    }

    auto isValidPoint = [](const float4& p, float depthValue) {
        constexpr float kInvalidThreshold = 1e9f;
        return std::isfinite(depthValue) &&
               depthValue > 0.0f &&
               std::fabs(depthValue) < kInvalidThreshold &&
               std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
    };

    size_t validCount = 0;
    for (size_t idx = 0; idx < pixelCount; ++idx) {
        if (isValidPoint(h_positions[idx], h_depth[idx])) {
            ++validCount;
        }
    }

    if (validCount == 0) {
        std::cerr << "RayCastRender: No valid raycast points to export." << std::endl;
        return false;
    }

    fs::path outPath(filePath);
    if (!outPath.has_parent_path()) {
        // nothing
    } else {
        fs::create_directories(outPath.parent_path());
    }

    std::ofstream ofs(outPath);
    if (!ofs) {
        std::cerr << "RayCastRender: Failed to open " << outPath << " for writing." << std::endl;
        return false;
    }

    ofs << "ply\nformat ascii 1.0\n";
    ofs << "comment Generated by RayCastRender\n";
    ofs << "element vertex " << validCount << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\n";
    ofs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    ofs << "end_header\n";

    auto toColorChannel = [](float v) -> int {
        if (!std::isfinite(v)) return 255;
        v = std::min(std::max(v, 0.0f), 1.0f);
        return static_cast<int>(v * 255.0f + 0.5f);
    };

    for (size_t idx = 0; idx < pixelCount; ++idx) {
        const auto& pos = h_positions[idx];
        if (!isValidPoint(pos, h_depth[idx])) continue;

        const auto& col = h_colors[idx];
        int r = toColorChannel(col.x);
        int g = toColorChannel(col.y);
        int b = toColorChannel(col.z);

        ofs << pos.x << " " << pos.y << " " << pos.z << " "
            << r << " " << g << " " << b << "\n";
    }

    std::cout << "RayCastRender: Saved point cloud to " << outPath << " (" << validCount << " points)" << std::endl;
    return true;
}

