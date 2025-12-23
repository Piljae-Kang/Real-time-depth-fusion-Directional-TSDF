#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/VoxelStreamingManager.h"
#include "../include/CUDAHashRef.h"
#include "../include/CUDAHashData.h"

#define T_PER_BLOCK 8

// Hash table constants (same as VoxelScene.cu)
#define HASH_SLOT_FREE      (-1)
#define HASH_BUCKET_UNLOCKED 0
#define HASH_BUCKET_LOCKED    1

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
    float3 sphereCenter,                    // Sphere center position (typically 100m ahead of camera, for stream out: blocks outside radius from sphere center are streamed out)
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

   
        
        // Calculate distance from sphere center (typically 100m ahead of camera)
        float dx = blockWorldPos.x - sphereCenter.x;
        float dy = blockWorldPos.y - sphereCenter.y;
        float dz = blockWorldPos.z - sphereCenter.z;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        
        // If block is outside radius from sphere center, add to stream-out list
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
    // OPTIMIZED: Use shared memory local counter to reduce atomicAdd contention
    // (Similar to expert code's compactifyHashAllInOneKernel)
    __shared__ int localCounter;
    __shared__ int addrGlobal;
    
    if (threadIdx.x == 0) {
        localCounter = 0;
    }
    __syncthreads();
    
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int addrLocal = -1;
    
    if (idx < numActiveBlocks) {
        const HashSlot& slot = d_compactifiedInput[idx];
        
        // Only copy valid blocks (skip gaps where ptr == -1)
        if (slot.ptr != -1) {
            addrLocal = atomicAdd(&localCounter, 1);  // Local atomic (much faster!)
        }
    }
    
    __syncthreads();
    
    // One thread per block updates global counter
    if (threadIdx.x == 0 && localCounter > 0) {
        addrGlobal = atomicAdd((unsigned int*)d_outputCounter, localCounter);
    }
    
    __syncthreads();
    
    // Copy to output using global base + local offset
    if (addrLocal != -1) {
        const unsigned int outputIdx = addrGlobal + addrLocal;
        const HashSlot& slot = d_compactifiedInput[idx];
        d_compactifiedOutput[outputIdx] = slot;
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
    float radius,                           // Stream out radius (blocks outside this radius from sphere center are streamed out)
    float3 sphereCenter,                    // Sphere center position (typically 100m ahead of camera, for stream out: blocks outside radius from sphere center are streamed out)
    unsigned int* d_outputCounter,
    SDFBlockInfo* d_output
) {
    if (numActiveBlocks <= 0) return;
    
    const unsigned int threadsPerBlock = T_PER_BLOCK * T_PER_BLOCK;  // 64 threads
    const unsigned int numBlocks = (numActiveBlocks + threadsPerBlock - 1) / threadsPerBlock;

    //printf("sphere center : %f %f %f", sphereCenter.x, sphereCenter.y, sphereCenter.z);
    
    streamOutFindBlocksKernel_OPTIMIZED<<<numBlocks, threadsPerBlock>>>(
        hashData.d_CompactifiedHashTable,  // Active blocks only!
        numActiveBlocks,
        params.voxelSize,
        radius,
        sphereCenter,
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


    VoxelData* voxel = &d_SDFBlocks[srcVoxelIdx];
    voxel->color = make_uchar3(0, 0, 0);
    voxel->weight = 0;
    voxel->sdf = 0.0f;
    voxel->isUpdated = 0;
    voxel->isZeroCrossing = 0;
    
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
            
            if (slot->ptr != HASH_SLOT_FREE &&
                slot->pos.x == blockCoord.x &&
                slot->pos.y == blockCoord.y &&
                slot->pos.z == blockCoord.z) {
                // Found! Delete hash entry
                // Expert code uses: pos = (0,0,0), ptr = FREE_ENTRY (-2)
                // We use: pos = (0,0,0), ptr = HASH_SLOT_FREE (-1) for consistency with our codebase
                slot->ptr = HASH_SLOT_FREE;
                slot->pos = make_int3(0, 0, 0);  // Expert code uses (0,0,0), not (-1,-1,-1)
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
                
                if (slot->ptr != HASH_SLOT_FREE &&
                    slot->pos.x == blockCoord.x &&
                    slot->pos.y == blockCoord.y &&
                    slot->pos.z == blockCoord.z) {
                    // Found! Delete hash entry
                    slot->ptr = HASH_SLOT_FREE;
                    slot->pos = make_int3(0, 0, 0);
                    slot->offset = 0;
                    found = true;
                    break;
                }
            }
        }
        
        // Return block to heap ONLY if we found and deleted the hash entry
        // CRITICAL: If found == false, the block was already streamed out or doesn't exist
        // In that case, we should NOT increment heap counter (would cause mismatch between TotalAllocated and ActiveBlocks)
        if (found && info.ptr >= 0) {
            unsigned int heapIdx = atomicAdd((unsigned int*)d_heapCounter, 1);
            unsigned int blockIndex = info.ptr;  // Already block index, no division needed!
            d_heap[heapIdx + 1] = blockIndex;  // Match expert code: store at heapIdx+1
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
    
    // DEBUG: Check if d_heapCounter is valid
    if (hashData.d_heapCounter == nullptr) {
        printf("[ERROR] streamOutCopyBlocksCUDA: d_heapCounter is nullptr!\n");
        return;
    }
    
    if (hashData.d_heap == nullptr) {
        printf("[ERROR] streamOutCopyBlocksCUDA: d_heap is nullptr!\n");
        return;
    }
    
    // DEBUG: Read current heap counter value
    unsigned int currentValue = 0;
    cudaError_t err = cudaMemcpy(&currentValue, hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] streamOutCopyBlocksCUDA: Failed to read d_heapCounter: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Calculate allocated blocks from heap counter
    unsigned int initialHeapCounter = params.SDFBlockNum - 1;
    unsigned int totalAllocatedBlocks = initialHeapCounter - currentValue;
    
    // Console output (minimal)
    printf("[STREAM_OUT] HeapCounter: %u, Allocated: %u, StreamOut: %u", 
           currentValue, totalAllocatedBlocks, nSDFBlocks);
    if (totalAllocatedBlocks < nSDFBlocks) {
        printf(" [WARNING: More blocks to stream out than allocated!]");
    }
    printf("\n");
    
    // Log to file will be written by host function (VoxelStreamingManager.cpp)
    
    const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;  // 512 threads
    const unsigned int numBlocks = nSDFBlocks;


    printf("--------------------------------------\n");
    printf("threadsPerBlock : %d, numBlocks : %d\n", threadsPerBlock, numBlocks);
    printf("--------------------------------------\n");
        
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
    
    // Reuse err variable (already declared above)
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] streamOutCopyBlocksCUDA kernel launch failed: %s\n", 
               cudaGetErrorString(err));
    }
}

// ============================================================================
// Stream In: CPU to GPU (Insert blocks into hash table)
// ============================================================================

//-------------------------------------------------------
// Pass 1: Allocate memory in hash table
//-------------------------------------------------------

/**
 * Stream In Pass 1 Kernel: Allocate blocks in hash table
 * Each thread handles one SDF block descriptor
 */
__global__ void streamInHashAllocKernel(
    CUDAHashRef hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    int* d_hashBucketMutex,
    int SDFBlockNum,  // Add heap array size for bounds checking
    StreamInDebugInfo* d_debugInfo  // Debug info array for logging
) {
    const unsigned int blockID = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    
    if (blockID >= numSDFBlockDescs) return;
    
    // Get block index from heap (mass allocation: use blocks from heapCountPrev backwards)
    // heapCountPrev is the number of free blocks, so we use indices from heapCountPrev-1 down to heapCountPrev-numSDFBlockDescs
    if (blockID >= heapCountPrev) {
        printf("[ERROR] streamInHashAllocKernel: blockID %u >= heapCountPrev %u\n", blockID, heapCountPrev);
        return;  // Not enough free blocks
    }

    
    
    unsigned int heapIdx = heapCountPrev - blockID - 1;  // -1 because heapCountPrev is count, not index
    
    printf("Block ID : %d, heap index : %d\n", blockID, heapIdx);

    // Bounds check: heapIdx must be within [0, SDFBlockNum)
    if (heapIdx >= (unsigned int)SDFBlockNum) {
        printf("[ERROR] streamInHashAllocKernel: heapIdx %u >= SDFBlockNum %d\n", heapIdx, SDFBlockNum);
        return;
    }
    
    unsigned int blockIndex = hashData.d_heap[heapIdx];
    
    // Bounds check: blockIndex must be within [0, SDFBlockNum)
    if (blockIndex >= (unsigned int)SDFBlockNum) {
        printf("[ERROR] streamInHashAllocKernel: blockIndex %u >= SDFBlockNum %d\n", blockIndex, SDFBlockNum);
        return;
    }
    
    // IMPORTANT: HashSlot.ptr stores BLOCK INDEX, not voxel index!
    // In integrateDepthMapIntoBlocksKernel, we use: d_SDFBlocks + (slot.ptr * voxelsPerBlock)
    // So ptr should be blockIndex, not blockIndex * linBlockSize
    unsigned int ptr = blockIndex;  // Block index, not voxel index!
    
    // Get block coordinates from descriptor
    const SDFBlockInfo& desc = d_SDFBlockDescs[blockID];
    int3 blockCoord = make_int3(desc.pos.x, desc.pos.y, desc.pos.z);
    
    // Allocate block in hash table (similar to allocBlock, but we already have the block index)
    unsigned int baseBucketId = hashBlockCoordinate(blockCoord, numBuckets);
    unsigned int baseBucketStart = baseBucketId * bucketSize;
    
    bool inserted = false;
    
    // Try to insert in the hashed bucket first
    for (int j = 0; j < bucketSize; j++) {
        unsigned int slotIdx = baseBucketStart + j;
        HashSlot* slot = &hashData.d_hashTable[slotIdx];
        
        if (slot->ptr == -1) {  // Free slot found
            // Acquire mutex
            int expected = 0;  // UNLOCKED
            if (atomicCAS(&d_hashBucketMutex[baseBucketId], expected, 1) == expected) {
                // Double-check slot is still free
                if (slot->ptr == -1) {
                    slot->pos = blockCoord;
                    slot->ptr = ptr;  // Block index
                    slot->offset = 0;
                    inserted = true;
                }
                atomicExch(&d_hashBucketMutex[baseBucketId], 0);  // Release mutex
                if (inserted) {
                    // Store debug info
                    if (d_debugInfo != nullptr) {
                        d_debugInfo[blockID].blockCoord = blockCoord;
                        d_debugInfo[blockID].result = INSERT_RESULT_NEW_SLOT;
                        d_debugInfo[blockID].slotIdx = slotIdx;
                        d_debugInfo[blockID].bucketId = baseBucketId;
                    }
                    return;  // Successfully inserted, exit function
                }
            }
        } else if (slot->ptr == ptr &&
                   slot->pos.x == blockCoord.x &&
                   slot->pos.y == blockCoord.y &&
                   slot->pos.z == blockCoord.z) {
            // Already exists, skip
            inserted = true;
            // Store debug info
            if (d_debugInfo != nullptr) {
                d_debugInfo[blockID].blockCoord = blockCoord;
                d_debugInfo[blockID].result = INSERT_RESULT_EXISTS_PRIMARY;
                d_debugInfo[blockID].slotIdx = slotIdx;
                d_debugInfo[blockID].bucketId = baseBucketId;
            }
            return;  // Block already exists, exit function
        }
    }
    
    // If not inserted in bucket, try open addressing
    if (!inserted) {
        const int MAX_ITERATIONS = totalHashSize;  // Prevent infinite loop
        for (int i = bucketSize; i < MAX_ITERATIONS; i++) {
            unsigned int slotIdx = (baseBucketStart + i) % totalHashSize;
            
            // Calculate the bucket ID for this slot (for mutex)
            unsigned int targetBucketId = slotIdx / bucketSize;
            
            HashSlot* slot = &hashData.d_hashTable[slotIdx];
            
            if (slot->ptr == -1) {
                // Try to acquire mutex for the target bucket
                int expected = 0;
                if (atomicCAS(&d_hashBucketMutex[targetBucketId], expected, 1) == expected) {
                    if (slot->ptr == -1) {
                        slot->pos = blockCoord;
                        slot->ptr = ptr;  // Block index
                        slot->offset = 0;
                        inserted = true;
                    }
                    atomicExch(&d_hashBucketMutex[targetBucketId], 0);
                    if (inserted) {
                        // Store debug info
                        if (d_debugInfo != nullptr) {
                            d_debugInfo[blockID].blockCoord = blockCoord;
                            d_debugInfo[blockID].result = INSERT_RESULT_NEW_OPEN;
                            d_debugInfo[blockID].slotIdx = slotIdx;
                            d_debugInfo[blockID].bucketId = targetBucketId;
                        }
                        return;  // Successfully inserted, exit function
                    }
                }
            } else if (slot->ptr == ptr &&
                       slot->pos.x == blockCoord.x &&
                       slot->pos.y == blockCoord.y &&
                       slot->pos.z == blockCoord.z) {
                // Already exists in open addressing slot
                inserted = true;
                // Store debug info
                if (d_debugInfo != nullptr) {
                    d_debugInfo[blockID].blockCoord = blockCoord;
                    d_debugInfo[blockID].result = INSERT_RESULT_EXISTS_OPEN;
                    d_debugInfo[blockID].slotIdx = slotIdx;
                    d_debugInfo[blockID].bucketId = targetBucketId;
                }
                return;  // Block already exists, exit function
            }
        }
    }
    
    // If we reach here, insertion failed (hash table might be full)
    if (d_debugInfo != nullptr) {
        d_debugInfo[blockID].blockCoord = blockCoord;
        d_debugInfo[blockID].result = INSERT_RESULT_FAIL;
        d_debugInfo[blockID].slotIdx = 0xFFFFFFFF;  // Invalid slot index
        d_debugInfo[blockID].bucketId = baseBucketId;
    }
}

extern "C" void streamInHashAllocCUDA(
    const Params& params,
    const CUDAHashRef& hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks,
    StreamInDebugInfo* d_debugInfo  // Debug info array (can be nullptr)
) {
    if (numSDFBlockDescs == 0) return;
    
    // Additional safety check
    if (heapCountPrev < numSDFBlockDescs) {
        printf("[ERROR] streamInHashAllocCUDA: heapCountPrev %u < numSDFBlockDescs %u\n", 
               heapCountPrev, numSDFBlockDescs);
        return;
    }
    
    if (heapCountPrev > (unsigned int)params.SDFBlockNum) {
        printf("[ERROR] streamInHashAllocCUDA: heapCountPrev %u > SDFBlockNum %d\n", 
               heapCountPrev, params.SDFBlockNum);
        return;
    }
    
    const dim3 gridSize((numSDFBlockDescs + T_PER_BLOCK * T_PER_BLOCK - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
    const dim3 blockSize(T_PER_BLOCK * T_PER_BLOCK, 1);
    
    streamInHashAllocKernel<<<gridSize, blockSize>>>(
        hashData,
        numSDFBlockDescs,
        heapCountPrev,
        d_SDFBlockDescs,
        d_SDFBlocks,
        params.hashSlotNum,
        params.slotSize,
        params.totalHashSize,
        hashData.d_hashBucketMutex,
        params.SDFBlockNum,  // Pass heap array size
        d_debugInfo  // Pass debug info array
    );
    
    // Check for kernel execution errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] streamInHashAllocCUDA kernel execution failed: %s\n", 
               cudaGetErrorString(err));
    }
}

//-------------------------------------------------------
// Pass 2: Copy SDF block data
//-------------------------------------------------------

/**
 * Stream In Pass 2 Kernel: Copy voxel data to allocated blocks
 * Each block gets one thread block, each voxel gets one thread
 */
__global__ void streamInHashInitKernel(
    CUDAHashRef hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks,
    int SDFBlockNum  // Add heap array size for bounds checking
) {
    const unsigned int blockID = blockIdx.x;  // Which SDF block to process
    const unsigned int voxelIdx = threadIdx.x;  // Which voxel within the block
    const unsigned int linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    
    if (voxelIdx >= linBlockSize) return;
    if (blockID >= numSDFBlockDescs) return;
    
    // Get block index from heap
    if (blockID >= heapCountPrev) {
        printf("[ERROR] streamInHashInitKernel: blockID %u >= heapCountPrev %u\n", blockID, heapCountPrev);
        return;
    }
    
    unsigned int heapIdx = heapCountPrev - blockID - 1;
    
    // Bounds check: heapIdx must be within [0, SDFBlockNum)
    if (heapIdx >= (unsigned int)SDFBlockNum) {
        printf("[ERROR] streamInHashInitKernel: heapIdx %u >= SDFBlockNum %d\n", heapIdx, SDFBlockNum);
        return;
    }
    
    unsigned int blockIndex = hashData.d_heap[heapIdx];
    
    // Bounds check: blockIndex must be within [0, SDFBlockNum)
    if (blockIndex >= (unsigned int)SDFBlockNum) {
        printf("[ERROR] streamInHashInitKernel: blockIndex %u >= SDFBlockNum %d\n", blockIndex, SDFBlockNum);
        return;
    }
    
    unsigned int ptr = blockIndex * linBlockSize;
    
    // Bounds check: ptr + voxelIdx must be within d_SDFBlocks array
    unsigned int totalVoxels = (unsigned int)SDFBlockNum * linBlockSize;
    if (ptr + voxelIdx >= totalVoxels) {
        printf("[ERROR] streamInHashInitKernel: ptr %u + voxelIdx %u >= totalVoxels %u\n", 
               ptr, voxelIdx, totalVoxels);
        return;
    }
    
    // Copy voxel data from input buffer to GPU SDF blocks
    hashData.d_SDFBlocks[ptr + voxelIdx] = d_SDFBlocks[blockID].data[voxelIdx];
}

extern "C" void streamInHashInitCUDA(
    const Params& params,
    const CUDAHashRef& hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks
) {
    if (numSDFBlockDescs == 0) return;
    
    // Additional safety check
    if (heapCountPrev < numSDFBlockDescs) {
        printf("[ERROR] streamInHashInitCUDA: heapCountPrev %u < numSDFBlockDescs %u\n", 
               heapCountPrev, numSDFBlockDescs);
        return;
    }
    
    if (heapCountPrev > (unsigned int)params.SDFBlockNum) {
        printf("[ERROR] streamInHashInitCUDA: heapCountPrev %u > SDFBlockNum %d\n", 
               heapCountPrev, params.SDFBlockNum);
        return;
    }
    
    const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    const dim3 gridSize(numSDFBlockDescs, 1);
    const dim3 blockSize(threadsPerBlock, 1);
    
    streamInHashInitKernel<<<gridSize, blockSize>>>(
        hashData,
        numSDFBlockDescs,
        heapCountPrev,
        d_SDFBlockDescs,
        d_SDFBlocks,
        params.SDFBlockNum  // Pass heap array size
    );
    
    // Check for kernel execution errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] streamInHashInitCUDA kernel execution failed: %s\n", 
               cudaGetErrorString(err));
    }
}

