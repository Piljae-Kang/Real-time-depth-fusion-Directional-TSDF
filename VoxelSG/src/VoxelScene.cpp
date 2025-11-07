#include "../include/VoxelScene.h"
//#include "../include/DepthCameraData.h"
#include <iostream>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

/**
 * Constructor: Initialize VoxelScene
 */
VoxelScene::VoxelScene(const Params& params)
    : m_params(params)
    , m_numIntegratedFrames(0)
    , m_isAllocated(false)
{
    std::cout << "VoxelScene: Initializing..." << std::endl;

    // Allocate HashData memory
    allocate();

    // Initialize HashData (set all values to 0)
    reset();

    std::cout << "VoxelScene: Initialization complete" << std::endl;
    printStats();
}

/**
 * Destructor: Free GPU memory
 */
VoxelScene::~VoxelScene() {
    std::cout << "VoxelScene: Destroying..." << std::endl;
    deallocate();
}

/**
 * Allocate HashData memory
 */
void VoxelScene::allocate() {
    if (m_isAllocated) {
        std::cerr << "VoxelScene: Already allocated!" << std::endl;
        return;
    }

    std::cout << "VoxelScene: Allocating GPU memory..." << std::endl;

    // Let CUDAHashRef allocate its own memory
    m_hashData.HashDataAllocation(m_params);

    m_isAllocated = true;

    // Calculate total memory usage
    size_t voxelCount = (size_t)m_params.SDFBlockNum *
        m_params.SDFBlockSize *
        m_params.SDFBlockSize *
        m_params.SDFBlockSize;
    
    size_t totalBytes =
        sizeof(unsigned int) * m_params.SDFBlockNum +  // heap
        sizeof(unsigned int) +                          // heapCounter
        sizeof(HashSlot) * m_params.totalHashSize * 2 +  // hash + compactified
        sizeof(int) +                                   // compactifiedCounter
        sizeof(VoxelData) * voxelCount +                   // SDF blocks
        sizeof(int) * m_params.hashSlotNum +           // mutex
        sizeof(int) * m_params.totalHashSize * 2;      // decision arrays

    std::cout << "VoxelScene: Allocated " << (totalBytes / (1024.0 * 1024.0))
        << " MB on GPU" << std::endl;
}

/**
 * Free HashData memory
 */
void VoxelScene::deallocate() {
    if (!m_isAllocated) {
        return;
    }

    std::cout << "VoxelScene: Deallocating GPU memory..." << std::endl;

    // Let CUDAHashRef free its own memory
    if (m_hashData.d_heap) CUDA_CHECK(cudaFree(m_hashData.d_heap));
    if (m_hashData.d_heapCounter) CUDA_CHECK(cudaFree(m_hashData.d_heapCounter));
    if (m_hashData.d_hashTable) CUDA_CHECK(cudaFree(m_hashData.d_hashTable));
    if (m_hashData.d_CompactifiedHashTable) CUDA_CHECK(cudaFree(m_hashData.d_CompactifiedHashTable));
    if (m_hashData.d_hashCompactifiedCounter) CUDA_CHECK(cudaFree(m_hashData.d_hashCompactifiedCounter));
    if (m_hashData.d_SDFBlocks) CUDA_CHECK(cudaFree(m_hashData.d_SDFBlocks));
    if (m_hashData.d_hashBucketMutex) CUDA_CHECK(cudaFree(m_hashData.d_hashBucketMutex));
    if (m_hashData.d_hashDecision) CUDA_CHECK(cudaFree(m_hashData.d_hashDecision));
    if (m_hashData.d_hashDecisionPrefix) CUDA_CHECK(cudaFree(m_hashData.d_hashDecisionPrefix));
    if (m_hashData.d_voxelZeroCross) CUDA_CHECK(cudaFree(m_hashData.d_voxelZeroCross));
    if (m_hashData.d_voxelZeroCrossCounter) CUDA_CHECK(cudaFree(m_hashData.d_voxelZeroCrossCounter));

    m_hashData.d_heap = nullptr;
    m_hashData.d_heapCounter = nullptr;
    m_hashData.d_hashTable = nullptr;
    m_hashData.d_CompactifiedHashTable = nullptr;
    m_hashData.d_hashCompactifiedCounter = nullptr;
    m_hashData.d_SDFBlocks = nullptr;
    m_hashData.d_hashBucketMutex = nullptr;
    m_hashData.d_hashDecision = nullptr;
    m_hashData.d_hashDecisionPrefix = nullptr;
    m_hashData.d_voxelZeroCross = nullptr;
    m_hashData.d_voxelZeroCrossCounter = nullptr;

    m_isAllocated = false;
}

/**
 * Initialize HashData (heap, hash table, set all values to 0)
 */
void VoxelScene::reset() {
    if (!m_isAllocated) {
        std::cerr << "VoxelScene: Not allocated yet!" << std::endl;
        return;
    }

    std::cout << "VoxelScene: Resetting HashData..." << std::endl;

    // Call CUDA kernel (defined in CUDAHashRef.cu)
    resetHashDataCUDA(m_hashData, m_params);

    // GPU synchronization
    CUDA_CHECK(cudaDeviceSynchronize());

    m_numIntegratedFrames = 0;

    std::cout << "VoxelScene: Reset complete" << std::endl;
}

/**
 * Integrate depth data into SDF
 */
void VoxelScene::integrate(const DepthCameraData& depthCameraData,
    const DepthCameraParams& depthCameraParams,
    const float* transform) {
    if (!m_isAllocated) {
        std::cerr << "VoxelScene: Not allocated yet!" << std::endl;
        return;
    }

    std::cout << "VoxelScene: Integrating frame " << m_numIntegratedFrames << "..." << std::endl;

    // 1. Allocate needed SDF blocks
    allocBlocks(depthCameraData, depthCameraParams);

    // 2. Compact active blocks
    compactifyHashEntries();

    // 3. Integrate depth into SDF
    integrateDepthMap(depthCameraData, depthCameraParams);

    // GPU synchronization
    CUDA_CHECK(cudaDeviceSynchronize());

    m_numIntegratedFrames++;

    std::cout << "VoxelScene: Integration complete" << std::endl;
}

/**
 * Allocate SDF blocks needed for depth map
 */
void VoxelScene::allocBlocks(const DepthCameraData& depthCameraData,
    const DepthCameraParams& depthCameraParams) {
    // Call CUDA kernel
    allocBlocksCUDA(m_hashData, m_params, depthCameraData, depthCameraParams);
}

/**
 * Compact active hash entries
 */
void VoxelScene::compactifyHashEntries() {
    // Call CUDA kernel
    compactifyHashCUDA(m_hashData, m_params);
}

/**
 * Integrate depth map into SDF
 */
void VoxelScene::integrateDepthMap(const DepthCameraData& depthCameraData,
    const DepthCameraParams& depthCameraParams) {
    // Call CUDA kernel
    integrateDepthMapCUDA(m_hashData, m_params, depthCameraData, depthCameraParams);
}

/**
 * Get number of free slots in heap
 */
unsigned int VoxelScene::getHeapFreeCount() const {
    unsigned int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, m_hashData.d_heapCounter,
        sizeof(unsigned int), cudaMemcpyDeviceToHost));
    return count + 1; // heap counter starts from 0, so +1
}

/**
 * Print HashData statistics
 */
void VoxelScene::printStats() const {
    std::cout << "\n=== VoxelScene Stats ===" << std::endl;
    std::cout << "Hash Slots: " << m_params.hashSlotNum << std::endl;
    std::cout << "Bucket Size: " << m_params.slotSize << std::endl;
    std::cout << "Total Hash Entries: " << m_params.totalHashSize << std::endl;
    std::cout << "SDF Blocks: " << m_params.SDFBlockNum << std::endl;
    std::cout << "Block Size: " << m_params.SDFBlockSize << "x"
        << m_params.SDFBlockSize << "x" << m_params.SDFBlockSize << std::endl;
    std::cout << "Total Voxels: " << m_params.totalBlockSize << std::endl;
    std::cout << "Integrated Frames: " << m_numIntegratedFrames << std::endl;

    if (m_isAllocated) {
        unsigned int freeCount = getHeapFreeCount();
        std::cout << "Free Blocks: " << freeCount << " / " << m_params.SDFBlockNum
            << " (" << (freeCount * 100.0 / m_params.SDFBlockNum) << "%)" << std::endl;
    }

    std::cout << "========================\n" << std::endl;
}


/**
 * Integrate depth map from ScanData into SDF (3-step process)
 */
void VoxelScene::integrateFromScanData(const float3* depthmap, const uchar3* colormap, const float3* normalmap,
    int width, int height, float truncationDistance, const cv::Mat& cameraTransform,
    float fx, float fy, float cx, float cy) {
    
    std::cout << "VoxelScene: Starting 3-step depth map integration..." << std::endl;
    std::cout << "  Depth map size: " << width << "x" << height << std::endl;
    std::cout << "  Truncation distance: " << truncationDistance << std::endl;
    
    // Debug: Check if pointers are valid
    std::cout << "  Debug: depthmap=" << (void*)depthmap << ", colormap=" << (void*)colormap 
              << ", normalmap=" << (void*)normalmap << std::endl;
    
    if (depthmap == nullptr || colormap == nullptr || normalmap == nullptr) {
        std::cerr << "VoxelScene: Invalid pointers (nullptr)!" << std::endl;
        return;
    }
    
    if (!m_isAllocated) {
        std::cerr << "VoxelScene: Not allocated!" << std::endl;
        return;
    }
    
    // Convert OpenCV Mat to float array and upload to GPU
    float* d_transform = nullptr;
    cudaError_t err = cudaMalloc(&d_transform, 16 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc error for transform: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    std::cout << "print matrix : \n";

    float transformArray[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            transformArray[i * 4 + j] = cameraTransform.at<float>(i, j);
            std::cout << cameraTransform.at<float>(i, j) << " ";
        }
        std::cout << "\n";
    }
    err = cudaMemcpy(d_transform, transformArray, 16 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error for transform: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_transform);
        return;
    }
    
    // Extract camera position from transform matrix
    // cameraTransform is camera->world, so camera in world = R * camera_in_camera + T
    // camera_in_camera = (0, 0, 0), so camera in world = T = (transformArray[3], transformArray[7], transformArray[11])


    
    // For camera-to-world matrix: [R | T], camera position in world = T
    float3 cameraPos = make_float3(
        transformArray[3],   // Translation X
        transformArray[7],   // Translation Y
        transformArray[11]   // Translation Z
    );

    std::cout << "cam pos : " << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << "\n";
    
    // Debug: Check pointers before calling kernel
    std::cout << "Debug: Checking pointers..." << std::endl;
    std::cout << "  depthmap: " << (void*)depthmap << std::endl;
    std::cout << "  m_hashData.d_hashTable: " << (void*)m_hashData.d_hashTable << std::endl;
    std::cout << "  m_hashData.d_heapCounter: " << (void*)m_hashData.d_heapCounter << std::endl;
    std::cout << "  d_transform: " << (void*)d_transform << std::endl;
    
    // Step 1: Allocate blocks based on depthmap
    std::cout << "  Step 1: Allocating blocks..." << std::endl;
    allocBlocksFromDepthMapCUDA(
        m_hashData,
        m_params,
        depthmap,
        width,
        height,
        truncationDistance,
        cameraPos,
        d_transform
    );
    
    // Step 2: Compactify hash entries (optional, for efficiency)
    std::cout << "  Step 2: Compactifying hash entries..." << std::endl;
    compactifyHashCUDA(m_hashData, m_params);
    
    // Step 3: Integrate depth map into allocated blocks
    std::cout << "  Step 3: Integrating depth map..." << std::endl;
    integrateDepthMapIntoBlocksCUDA(
        m_hashData,
        m_params,
        depthmap,
        colormap,
        normalmap,
        width,
        height,
        truncationDistance,
        cameraPos,
        d_transform,
        fx, fy, cx, cy
    );
    
    // Get active block count from compactified counter
    int numActiveBlocks = 0;
    cudaMemcpy(&numActiveBlocks, m_hashData.d_hashCompactifiedCounter, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  Active blocks: " << numActiveBlocks << std::endl;
    
#ifdef SAVE_VOXEL_TO_XYZ
    // Download compactified hash table from GPU
    HashSlot* h_compactified = new HashSlot[numActiveBlocks];
    cudaError_t err = cudaMemcpy(h_compactified, m_hashData.d_CompactifiedHashTable, 
                                  numActiveBlocks * sizeof(HashSlot), 
                                  cudaMemcpyDeviceToHost);

    if (err == cudaSuccess) {
        std::cout << "  Saving updated voxel positions to updated_voxel_positions.xyz..." << std::endl;
        
        // Download voxel data from GPU
        VoxelData* h_SDFBlocks = new VoxelData[numActiveBlocks * 512];
        cudaError_t err2 = cudaMemcpy(h_SDFBlocks, m_hashData.d_SDFBlocks,
                                     numActiveBlocks * 512 * sizeof(VoxelData),
                                     cudaMemcpyDeviceToHost);
        
        if (err2 == cudaSuccess) {
            FILE* fp_xyz = fopen("updated_voxel_positions.xyz", "w");
            FILE* fp_full = fopen("updated_voxel_full.txt", "w");
            
            if (fp_xyz && fp_full) {
                fprintf(fp_full, "# Voxel data: x, y, z, sdf, weight, r, g, b\n");
                
                float voxelSize = GlobalParamsConfig::get().g_SDFVoxelSize;
                int totalVoxels = 0;
                int updatedVoxels = 0;
                
                for (int i = 0; i < numActiveBlocks; i++) {
                    int3 blockCoord = h_compactified[i].pos;
                    
                    // Save block center/corner instead of all 512 voxels
                    float3 blockCenter = make_float3(
                        blockCoord.x * 8 * voxelSize,
                        blockCoord.y * 8 * voxelSize,
                        blockCoord.z * 8 * voxelSize
                    );
                    
                    // Standard .xyz format: block center
                    fprintf(fp_xyz, "%.6f %.6f %.6f\n",
                        blockCenter.x, blockCenter.y, blockCenter.z);
                    
                    // Full data: just block coords
                    fprintf(fp_full, "%.6f %.6f %.6f %d %d %d\n",
                        blockCenter.x, blockCenter.y, blockCenter.z,
                        blockCoord.x, blockCoord.y, blockCoord.z);
                    
                    updatedVoxels++;
                    totalVoxels++;
                }
                
                if (fp_xyz) fclose(fp_xyz);
                if (fp_full) fclose(fp_full);
                
                std::cout << "  Saved " << updatedVoxels << " updated voxels (out of " << totalVoxels << " total)" << std::endl;
                std::cout << "  - updated_voxel_positions.xyz: x y z (standard format)" << std::endl;
                std::cout << "  - updated_voxel_full.txt: x y z sdf weight r g b" << std::endl;
            }
        }
        delete[] h_SDFBlocks;
    }
    delete[] h_compactified;
#endif
    
    // Clean up GPU memory
    cudaFree(d_transform);
    
    // Increment frame counter
    m_numIntegratedFrames++;
    
    std::cout << "VoxelScene: 3-step integration completed. Total frames: " << m_numIntegratedFrames << std::endl;
}

/**
 * Save to mesh file (after Marching Cubes implementation)
 */
bool VoxelScene::saveToMesh(const std::string& filename) {
    std::cout << "VoxelScene: Saving mesh to " << filename << "..." << std::endl;

    // TODO: Implement Marching Cubes
    std::cout << "VoxelScene: Mesh export not implemented yet!" << std::endl;

    return false;
}