#include "../include/VoxelStreamingManager.h"

#include <cmath>
#include <cuda_runtime.h>

#include "../include/CUDAHashRef.h"
#include "../include/CUDAHashData.h"

// Forward declaration of CUDA functions
extern "C" void streamOutFindBlocksCUDA_OPTIMIZED(
    const CUDAHashRef& hashData,
    const Params& params,
    int numActiveBlocks,
    float radius,
    float3 cameraPosition,
    unsigned int* d_outputCounter,
    SDFBlockInfo* d_output
);

extern "C" float compactifyHashCUDA(CUDAHashRef& hashData, const Params& params);
extern "C" float repackCompactifiedCUDA(CUDAHashRef& hashData, int numActiveBlocks);

extern "C" void streamOutCopyBlocksCUDA(
    const CUDAHashRef& hashData,
    const Params& params,
    const SDFBlockInfo* d_blockInfos,
    SDFBlock* d_outputBlocks,
    unsigned int nSDFBlocks
);




DWORD WINAPI AuxiliaryStreamingFunc(LPVOID lParam) {

    VoxelStreamingManager* chunkStreamingManager = (VoxelStreamingManager*)lParam;
    

    // Call Auxiliary Thread function



    return 0;

}



VoxelStreamingManager::VoxelStreamingManager(VoxelScene& sceneHashRef, Params& params, const vec3f& voxelExtends, 
    const vec3i& gridDimensions, const vec3i& minGridPos,
    unsigned int initialChunkListSize, bool streamingEnabled, unsigned int streamOutParts) {

    sceneHashRef_ = &sceneHashRef;
    params_ = params;

    m_currentPart = 0;
    m_streamOutParts = streamOutParts;

    m_maxNumberOfSDFBlocksIntegrateFromGlobalHash = 100000; // initial value from voxel hashing

    h_SDFBlockInfoOutput = NULL;
    h_SDFBlockOutput = NULL;
    d_SDFBlockInfoOutput = NULL;
    d_SDFBlockInfoInput = NULL;
    d_SDFBlockOutput = NULL;
    d_SDFBlockInput = NULL;
    d_SDFBlockCounter = NULL;

    d_bitMask = NULL;

    s_terminateThread = true;	//by default the thread is disabled

    initialize(voxelExtends, gridDimensions, minGridPos, initialChunkListSize, streamingEnabled);

}


VoxelStreamingManager::~VoxelStreamingManager() = default;

void VoxelStreamingManager::initialize(const vec3f& voxelExtends, const vec3i& gridDimensions,
    const vec3i& minGridPos, unsigned int initialChunkListSize, bool streamingEnabled) {

    m_voxelExtents = voxelExtends;
    m_gridDimensions = gridDimensions;
    m_initialChunkListSize = initialChunkListSize;

    m_minGridPos = minGridPos;
    m_maxGridPos = minGridPos + gridDimensions;

    m_grid.resize(m_gridDimensions.x * m_gridDimensions.y * m_gridDimensions.z, NULL);

    m_bitMask = BitArray<unsigned int>(m_gridDimensions.x * m_gridDimensions.y * m_gridDimensions.z);


    cudaHostAlloc(&h_SDFBlockInfoOutput, sizeof(SDFBlockInfo) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash, cudaHostAllocDefault);
    cudaHostAlloc(&h_SDFBlockOutput, sizeof(SDFBlock) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash, cudaHostAllocDefault);

    cudaMalloc(&d_SDFBlockInfoOutput, sizeof(SDFBlockInfo) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash);
    cudaMalloc(&d_SDFBlockInfoInput, sizeof(SDFBlockInfo) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash);
    cudaMalloc(&d_SDFBlockOutput, sizeof(SDFBlock) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash);
    cudaMalloc(&d_SDFBlockInput, sizeof(SDFBlock) * m_maxNumberOfSDFBlocksIntegrateFromGlobalHash);
    cudaMalloc(&d_SDFBlockCounter, sizeof(unsigned int));

    cudaMalloc(&d_bitMask, m_bitMask.getByteWidth());

    if (streamingEnabled) startAuxiliaryThread();


}

void VoxelStreamingManager::streamOut(VoxelScene& scene,
                                      const float3& cameraPos,
                                      float radius) {
    // TODO: Scan GPU hash, copy blocks outside radius to CPU storage,
    //       and free GPU entries.





}

void VoxelStreamingManager::streamIn(VoxelScene& scene,
                                     const float3& cameraPos,
                                     float radius) {
    // TODO: Find CPU blocks within radius, allocate them back on the GPU,
    //       and remove them from CPU storage.





}



bool VoxelStreamingManager::isInsideRadius(const float3& blockWorldCenter,
                                           const float3& cameraPos,
                                           float radius) const {
    float dx = blockWorldCenter.x - cameraPos.x;
    float dy = blockWorldCenter.y - cameraPos.y;
    float dz = blockWorldCenter.z - cameraPos.z;
    return (dx * dx + dy * dy + dz * dz) <= radius * radius;


}

float3 VoxelStreamingManager::blockCoordToWorld(const int3& blockCoord) const {
    float voxelSize = params_.voxelSize;
    float blockSize = static_cast<float>(SDF_BLOCK_SIZE);

    return make_float3(blockCoord.x * blockSize * voxelSize,
                       blockCoord.y * blockSize * voxelSize,
                       blockCoord.z * blockSize * voxelSize);
}


void VoxelStreamingManager::startAuxiliaryThread() {


    hStreamingThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)AuxiliaryStreamingFunc, (LPVOID)this, 0, &dwStreamingThreadID);

}

  
void VoxelStreamingManager::streamOutFindBlocksOnGPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded) { // Main Thread 1 : Find Block for Stream Out


    if (multiThreaded) {
        WaitForSingleObject(hEventOutProduce, INFINITE);
        WaitForSingleObject(hMutexOut, INFINITE);
    }


    // Task 1 : Reset Hash Data Mutex

    resetHasMutexCUDA(sceneHashRef_->getHashData(), sceneHashRef_->getParams());
    clearSDFBlockCounter();

    // Task 2 : Find all SDFBlocks that have to be transfered
    // OPTIMIZED APPROACH: Scan compactified hash (active blocks only), then regenerate it!
    // This is MORE efficient than scanning entire hash table.
    
    // Get number of active blocks from compactified counter
    int numActiveBlocks = 0;
    if (sceneHashRef_->getHashData().d_hashCompactifiedCounter) {
        cudaMemcpy(&numActiveBlocks, 
                   sceneHashRef_->getHashData().d_hashCompactifiedCounter, 
                   sizeof(int), 
                   cudaMemcpyDeviceToHost);
    }
    
    if (numActiveBlocks > 0) {
        // Convert vec3f to float3
        float3 cameraPos = make_float3(posCamera.x, posCamera.y, posCamera.z);
        
        // Launch OPTIMIZED kernel (scans only active blocks, not entire hash!)
        streamOutFindBlocksCUDA_OPTIMIZED(
            sceneHashRef_->getHashData(),
            sceneHashRef_->getParams(),
            numActiveBlocks,
            radius,
            cameraPos,
            d_SDFBlockCounter,
            d_SDFBlockInfoOutput
        );
        
        // Check for errors
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] streamOutFindBlocksCUDA_OPTIMIZED failed: %s\n", 
                   cudaGetErrorString(err));
        }
        
        // Get number of blocks found for streaming out
        unsigned int nSDFBlockDescs = getSDFBlockCounter();
        if (nSDFBlockDescs > 0) {
            printf("[STREAM OUT] Found %u blocks to stream out (scanned %d active blocks, not %u total slots)\n", 
                   nSDFBlockDescs, numActiveBlocks, sceneHashRef_->getParams().totalHashSize);
        }
    } else {
        printf("[STREAM OUT] No active blocks to scan\n");
    }

    // Task 3 :  Copy SDFBlocks to output buffer
    unsigned int nSDFBlockDescs = getSDFBlockCounter();
    
    if (nSDFBlockDescs > 0) {
        // Pass 2: Copy voxel data from GPU to output buffer and delete from hash
        streamOutCopyBlocksCUDA(
            sceneHashRef_->getHashData(),
            sceneHashRef_->getParams(),
            d_SDFBlockInfoOutput,
            d_SDFBlockOutput,
            nSDFBlockDescs
        );
        
        // Check for errors
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] streamOutCopyBlocksCUDA failed: %s\n", 
                   cudaGetErrorString(err));
        }
        
        // Copy block info and data from GPU to CPU
        cudaMemcpy(h_SDFBlockInfoOutput, 
                   d_SDFBlockInfoOutput, 
                   sizeof(SDFBlockInfo) * nSDFBlockDescs, 
                   cudaMemcpyDeviceToHost);
        
        cudaMemcpy(h_SDFBlockOutput, 
                   d_SDFBlockOutput, 
                   sizeof(SDFBlock) * nSDFBlockDescs, 
                   cudaMemcpyDeviceToHost);
        
        // Check for copy errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] GPU→CPU copy failed: %s\n", 
                   cudaGetErrorString(err));
        } else {
            printf("[STREAM OUT] Copied %u blocks to CPU (%.2f MB)\n", 
                   nSDFBlockDescs,
                   (sizeof(SDFBlockInfo) * nSDFBlockDescs + sizeof(SDFBlock) * nSDFBlockDescs) / (1024.0f * 1024.0f));
        }
        
        // Store number of streamed blocks for later use
        nStreamdOutBlocks_ = nSDFBlockDescs;
        
        // IMPORTANT: Repack compactified array to remove gaps!
        // After streaming out, some blocks were removed, creating gaps in compactified array.
        // OPTIMIZED: Only repack compactified array (don't scan entire hash table!)
        printf("[STREAM OUT] Repacking compactified array to remove gaps (scanned %d blocks, not %u total slots)...\n", 
               numActiveBlocks, sceneHashRef_->getParams().totalHashSize);
        float repackTime = repackCompactifiedCUDA(sceneHashRef_->getHashData(), numActiveBlocks);
        printf("[STREAM OUT] Compactified array repacked in %.3f ms\n", repackTime);
    } else {
        nStreamdOutBlocks_ = 0;
    }



    if (multiThreaded) {
        SetEvent(hEventOutConsume);
        ReleaseMutex(hMutexOut);
    }

}