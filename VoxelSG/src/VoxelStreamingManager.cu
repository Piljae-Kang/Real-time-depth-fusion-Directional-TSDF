#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/VoxelStreamingManager.h"
#include "../include/CUDAHashRef.h"
#include "../include/CUDAHashData.h"

#define T_PER_BLOCK 8

// Shared device functions (same as VoxelScene.cu, but defined here to avoid linking issues)
// These are inline functions, so multiple definitions are OK
__device__ __forceinline__ unsigned int hashBlockCoordinate(int3 blockCoord, int numBuckets) {
    unsigned int hash = (blockCoord.x * 73856093) ^
                       (blockCoord.y * 19349663) ^
                       (blockCoord.z * 83492791);
    return hash % numBuckets;
}

__device__ __forceinline__ float3 SDFBlockToWorld(const int3& sdfBlock, float voxelSize) {
    return make_float3(
        sdfBlock.x * SDF_BLOCK_SIZE * voxelSize,
        sdfBlock.y * SDF_BLOCK_SIZE * voxelSize,
        sdfBlock.z * SDF_BLOCK_SIZE * voxelSize
    );
}

/**
 * OPTIMIZED Streaming Out Kernel: Scan compactified hash (active blocks only)
 * After streaming out, we will regenerate compactified to remove gaps.
 * This is MORE efficient than scanning entire hash table!
 */
__global__ void streamOutFindBlocksKernel_OPTIMIZED(
    const HashSlot* d_hashCompactified,    // Active blocks only (much faster!)
    int numActiveBlocks,                    // Number of active blocks
    float voxelSize,
    float radius,
    float3 cameraPosition,
    unsigned int* d_outputCounter,
    SDFBlockInfo* d_output
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only scan active blocks (much smaller than full hash table!)
    if (idx < numActiveBlocks) {
        const HashSlot& slot = d_hashCompactified[idx];
        
        // Skip free slots (shouldn't happen in compactified, but safety check)
        if (slot.ptr == -1) return;
        
        // Convert block coordinates to world position
        float3 blockWorldPos = SDFBlockToWorld(slot.pos, voxelSize);
        
        // Calculate distance from camera
        float dx = blockWorldPos.x - cameraPosition.x;
        float dy = blockWorldPos.y - cameraPosition.y;
        float dz = blockWorldPos.z - cameraPosition.z;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        
        // If block is outside radius, add to stream-out list
        if (distance >= radius) {
            // Initialize SDFBlockInfo directly without constructor (device code)
            // Use direct member assignment - don't call constructor
            SDFBlockInfo info;
            // Manually initialize members (avoid constructor call)
            info.pos.x = slot.pos.x;
            info.pos.y = slot.pos.y;
            info.pos.z = slot.pos.z;
            info.ptr = slot.ptr;
            
            // Atomically add to output list
            unsigned int outputIdx = atomicAdd(d_outputCounter, 1);
            if (outputIdx < 100000) {  // Safety check against buffer overflow
                d_output[outputIdx] = info;
            }
        }
    }
}

// Note: hashBlockCoordinate and SDFBlockToWorld are defined in VoxelScene.cu
// They are shared device functions, so we don't redefine them here.

/**
 * OPTIMIZED Kernel: Repack compactified array to remove gaps
 * Only scans compactified array (not entire hash table) - MUCH faster!
 * This removes gaps created by streaming out without scanning entire hash.
 */
__global__ void repackCompactifiedKernel(
    const HashSlot* d_compactifiedInput,   // Input: compactified with gaps
    HashSlot* d_compactifiedOutput,        // Output: dense compactified
    int* d_outputCounter,                  // Counter for new compactified size
    int numActiveBlocks                    // Current size (may have gaps)
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numActiveBlocks) {
        const HashSlot& slot = d_compactifiedInput[idx];
        
        // Only copy valid blocks (skip gaps where ptr == -1)
        if (slot.ptr != -1) {
            unsigned int outputIdx = atomicAdd((unsigned int*)d_outputCounter, 1);
            d_compactifiedOutput[outputIdx] = slot;
        }
    }
}

/**
 * Host function: Repack compactified array to remove gaps (OPTIMIZED)
 * Only scans compactified array, not entire hash table!
 */
extern "C" float repackCompactifiedCUDA(
    CUDAHashRef& hashData,
    int numActiveBlocks
) {
    if (numActiveBlocks <= 0) return 0.0f;
    
    // Time measurement
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);
    
    // Reset counter
    cudaMemset(hashData.d_hashCompactifiedCounter, 0, sizeof(int));
    
    // Set up grid and block dimensions
    const unsigned int threadsPerBlock = 256;
    const unsigned int numBlocks = (numActiveBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate temporary buffer (same size as compactified)
    HashSlot* d_tempCompactified = nullptr;
    cudaMalloc(&d_tempCompactified, sizeof(HashSlot) * numActiveBlocks);
    
    // Repack: copy valid blocks to temp buffer
    repackCompactifiedKernel<<<numBlocks, threadsPerBlock>>>(
        hashData.d_CompactifiedHashTable,
        d_tempCompactified,
        hashData.d_hashCompactifiedCounter,
        numActiveBlocks
    );
    
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        printf("[ERROR] repackCompactifiedKernel launch failed: %s\n", 
               cudaGetErrorString(kernelErr));
        cudaFree(d_tempCompactified);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }
    
    // Copy back from temp to compactified (DeviceToDevice!)
    int newCount = 0;
    cudaMemcpy(&newCount, hashData.d_hashCompactifiedCounter, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (newCount > 0) {
        cudaMemcpy(hashData.d_CompactifiedHashTable, 
                   d_tempCompactified, 
                   sizeof(HashSlot) * newCount, 
                   cudaMemcpyDeviceToDevice);
    }
    
    cudaFree(d_tempCompactified);
    
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);
    
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] repackCompactifiedCUDA failed: %s\n", 
               cudaGetErrorString(err));
        return 0.0f;
    }
    
    return elapsedMs;
}

/**
 * Host function: Launch OPTIMIZED kernel to find blocks for streaming out
 * Scans compactified hash (active blocks only) - MUCH faster!
 * After streaming out, compactified will be repacked to remove gaps.
 */
extern "C" void streamOutFindBlocksCUDA_OPTIMIZED(
    const CUDAHashRef& hashData,
    const Params& params,
    int numActiveBlocks,                    // From d_hashCompactifiedCounter
    float radius,
    float3 cameraPosition,
    unsigned int* d_outputCounter,
    SDFBlockInfo* d_output
) {
    if (numActiveBlocks <= 0) return;
    
    const unsigned int threadsPerBlock = T_PER_BLOCK * T_PER_BLOCK;  // 64 threads
    const unsigned int numBlocks = (numActiveBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    streamOutFindBlocksKernel_OPTIMIZED<<<numBlocks, threadsPerBlock>>>(
        hashData.d_CompactifiedHashTable,  // Active blocks only!
        numActiveBlocks,
        params.voxelSize,
        radius,
        cameraPosition,
        d_outputCounter,
        d_output
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] streamOutFindBlocksCUDA_OPTIMIZED kernel launch failed: %s\n", 
               cudaGetErrorString(err));
    }
}

/**
 * Pass 2 Kernel: Copy SDF block voxel data to output buffer and delete from hash
 * Each CUDA block processes one SDF block, each thread processes one voxel
 */
__global__ void streamOutCopyBlocksKernel(
    const SDFBlockInfo* d_blockInfos,      // Block info from Pass 1
    VoxelData* d_SDFBlocks,                // Source: GPU voxel data
    SDFBlock* d_outputBlocks,               // Output: Block data to copy to CPU
    HashSlot* d_hashTable,                  // Hash table to delete entries from
    unsigned int* d_heap,                   // Heap to return blocks to
    unsigned int* d_heapCounter,            // Heap counter
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    unsigned int nSDFBlocks
) {
    const unsigned int idxBlock = blockIdx.x;  // Which block to process (renamed to avoid conflict with CUDA built-in)
    const unsigned int voxelIdx = threadIdx.x; // Which voxel within the block
    
    if (idxBlock >= nSDFBlocks) return;
    
    const unsigned int linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    if (voxelIdx >= linBlockSize) return;
    
    const SDFBlockInfo& info = d_blockInfos[idxBlock];
    
    // Copy voxel data from GPU to output buffer
    unsigned int srcVoxelIdx = info.ptr * linBlockSize + voxelIdx;
    d_outputBlocks[idxBlock].data[voxelIdx] = d_SDFBlocks[srcVoxelIdx];
    
    // Clear voxel data (optional, for safety)
    if (voxelIdx == 0) {
        // Only first thread per block handles hash deletion and heap return
        
        // Find hash entry and delete it
        int3 blockCoord = make_int3(info.pos.x, info.pos.y, info.pos.z);
        
        // Hash function (same as allocation)
        unsigned int bucketId = hashBlockCoordinate(blockCoord, numBuckets);
        unsigned int bucketStart = bucketId * bucketSize;
        
        // Search in bucket
        bool found = false;
        for (int j = 0; j < bucketSize; j++) {
            unsigned int slotIdx = bucketStart + j;
            HashSlot* slot = &d_hashTable[slotIdx];
            
            if (slot->ptr != -1 &&
                slot->pos.x == blockCoord.x &&
                slot->pos.y == blockCoord.y &&
                slot->pos.z == blockCoord.z) {
                // Found! Delete hash entry
                slot->ptr = -1;
                slot->pos = make_int3(-1, -1, -1);
                slot->offset = 0;
                found = true;
                break;
            }
        }
        
        // If not found in bucket, search entire table (open addressing)
        if (!found) {
            for (int i = bucketSize; i < totalHashSize; i++) {
                unsigned int slotIdx = (bucketStart + i) % totalHashSize;
                HashSlot* slot = &d_hashTable[slotIdx];
                
                if (slot->ptr != -1 &&
                    slot->pos.x == blockCoord.x &&
                    slot->pos.y == blockCoord.y &&
                    slot->pos.z == blockCoord.z) {
                    // Found! Delete hash entry
                    slot->ptr = -1;
                    slot->pos = make_int3(-1, -1, -1);
                    slot->offset = 0;
                    found = true;
                    break;
                }
            }
        }
        
        // Return block to heap (block index = ptr / linBlockSize)
        if (found && info.ptr >= 0) {
            unsigned int blockIndex = info.ptr / linBlockSize;
            unsigned int heapIdx = atomicAdd((unsigned int*)d_heapCounter, 1);
            if (heapIdx < (unsigned int)(totalHashSize / linBlockSize)) {  // Safety check
                d_heap[heapIdx] = blockIndex;
            }
        }
    }
}

/**
 * Host function: Launch Pass 2 kernel to copy blocks and delete from hash
 */
extern "C" void streamOutCopyBlocksCUDA(
    const CUDAHashRef& hashData,
    const Params& params,
    const SDFBlockInfo* d_blockInfos,
    SDFBlock* d_outputBlocks,
    unsigned int nSDFBlocks
) {
    if (nSDFBlocks <= 0) return;
    
    const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;  // 512 threads
    const unsigned int numBlocks = nSDFBlocks;
    
    streamOutCopyBlocksKernel<<<numBlocks, threadsPerBlock>>>(
        d_blockInfos,
        hashData.d_SDFBlocks,
        d_outputBlocks,
        hashData.d_hashTable,
        hashData.d_heap,
        hashData.d_heapCounter,
        params.hashSlotNum,
        params.slotSize,
        params.totalHashSize,
        nSDFBlocks
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] streamOutCopyBlocksCUDA kernel launch failed: %s\n", 
               cudaGetErrorString(err));
    }
}

