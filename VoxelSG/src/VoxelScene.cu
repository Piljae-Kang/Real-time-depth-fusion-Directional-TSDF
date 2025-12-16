#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "../include/CUDAHashRef.h"
#include "../include/VoxelScene.h"


//#define USE_NORMAL_DIR_ALLOC // normal 방향으로 allocation(Method 2)

// CUDA 12.9에서 atomicAdd는 기본적으로 제공됨

#define HASH_SLOT_FREE      (-1)
#define HASH_BUCKET_UNLOCKED 0
#define HASH_BUCKET_LOCKED    1
#define MAX_BUCKET_SPIN    128
#define MAX_BUCKET_RETRY     4

#define T_PER_BLOCK 8

// ----------------------------------------------------------------------------
// Host-side debugging helpers
// ----------------------------------------------------------------------------

static void printGpuMemoryUsage(const char* tag) {
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaError_t err = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (err != cudaSuccess) {
        printf("[GPU MEM] %s: cudaMemGetInfo failed (%s)\n", tag, cudaGetErrorString(err));
        return;
    }

    double freeGB = static_cast<double>(freeBytes) / (1024.0 * 1024.0 * 1024.0);
    double totalGB = static_cast<double>(totalBytes) / (1024.0 * 1024.0 * 1024.0);
    double usedGB = totalGB - freeGB;

    printf("[GPU MEM] %s: used %.2f GB / total %.2f GB (free %.2f GB)\n",
        tag, usedGB, totalGB, freeGB);
}

static float computeKernelElapsedMs(cudaEvent_t startEvent, cudaEvent_t stopEvent) {
    float elapsedMs = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    if (err != cudaSuccess) {
        printf("[CUDA EVENT] Failed to compute elapsed time (%s)\n", cudaGetErrorString(err));
        return 0.0f;
    }
    return elapsedMs;
}

// ============================================================================
// CUDA Device Functions (helper functions)
// ============================================================================

// CUDA 12.9의 내장 atomicAdd를 직접 사용

/**
 * Hash function for block coordinates (like expert code)
 * Computes hash value for a 3D block coordinate
 */
__device__ __forceinline__ unsigned int hashBlockCoordinate(int3 blockCoord, int numBuckets) {
    // Simple hash function - in real implementation you might want a better one
    unsigned int hash = (blockCoord.x * 73856093) ^
        (blockCoord.y * 19349663) ^
        (blockCoord.z * 83492791);
    return hash % numBuckets; // bucket id
}

/**
 * Convert world position to SDF block coordinates (like expert code)
 * Implements worldToVirtualVoxelPos + virtualVoxelPosToSDFBlock
 */
__device__ int3 worldToSDFBlock(const float3& worldPos, float voxelSize) {
    const int blockSize = SDF_BLOCK_SIZE;
    const int blockSizeMinusOne = SDF_BLOCK_SIZE - 1;

    // worldToVirtualVoxelPos
    float3 virtualVoxelPos = make_float3(worldPos.x / voxelSize, worldPos.y / voxelSize, worldPos.z / voxelSize);
    float3 sign = make_float3(
        (virtualVoxelPos.x > 0) ? 1.0f : ((virtualVoxelPos.x < 0) ? -1.0f : 0.0f),
        (virtualVoxelPos.y > 0) ? 1.0f : ((virtualVoxelPos.y < 0) ? -1.0f : 0.0f),
        (virtualVoxelPos.z > 0) ? 1.0f : ((virtualVoxelPos.z < 0) ? -1.0f : 0.0f)
    );
    int3 virtualVoxel = make_int3(
        virtualVoxelPos.x + sign.x * 0.5f,
        virtualVoxelPos.y + sign.y * 0.5f,
        virtualVoxelPos.z + sign.z * 0.5f
    );
    
    // virtualVoxelPosToSDFBlock (handle negative coordinates)
    if (virtualVoxel.x < 0) virtualVoxel.x -= blockSizeMinusOne;
    if (virtualVoxel.y < 0) virtualVoxel.y -= blockSizeMinusOne;
    if (virtualVoxel.z < 0) virtualVoxel.z -= blockSizeMinusOne;
    
    return make_int3(virtualVoxel.x / blockSize, virtualVoxel.y / blockSize, virtualVoxel.z / blockSize);
}

/**
 * Convert SDF block coordinates to world position
 */
__device__ __forceinline__ float3 SDFBlockToWorld(const int3& sdfBlock, float voxelSize) {
    return make_float3(
        sdfBlock.x * SDF_BLOCK_SIZE * voxelSize,
        sdfBlock.y * SDF_BLOCK_SIZE * voxelSize,
        sdfBlock.z * SDF_BLOCK_SIZE * voxelSize
    );
}

/**
 * Compute sign of a float3 (like expert code)
 */
__device__ float3 computeSign(const float3& v) {
    return make_float3(
        (v.x > 0) ? 1.0f : ((v.x < 0) ? -1.0f : 0.0f),
        (v.y > 0) ? 1.0f : ((v.y < 0) ? -1.0f : 0.0f),
        (v.z > 0) ? 1.0f : ((v.z < 0) ? -1.0f : 0.0f)
    );
}

// Forward declaration (kept here instead of headers to avoid CUDA header pollution)
__device__ int allocBlockWithMeta(HashSlot* d_hashTable,
                                  unsigned int* d_heap,
                                  unsigned int* d_heapCounter,
                                  int* d_hashBucketMutex,
                                  const int3& blockCoord,
                                  int SDFBlockNum,
                                  int numBuckets,
                                  int bucketSize,
                                  int totalHashSize,
                                  bool debug = false);

/**
 * Allocate a block in hash table (like expert code)
 * Uses heap counter and heap array to get free block index
 */
__device__ int allocBlock(HashSlot* d_hashTable,
                          unsigned int* d_heap,
                          unsigned int* d_heapCounter,
                          int* d_hashBucketMutex,
                          const int3& blockCoord,
                          int SDFBlockNum,
                          int numBuckets,
                          int bucketSize,
                          int totalHashSize,
                          bool debug = false) {
    unsigned int baseBucketId = hashBlockCoordinate(blockCoord, numBuckets);
    unsigned int baseBucketStart = baseBucketId * bucketSize;

    for (int retry = 0; retry < MAX_BUCKET_RETRY; ++retry) {
        int firstEmpty = -1;

        // 1) Probe inside the hashed bucket
        for (int j = 0; j < bucketSize; ++j) {
            unsigned int slotIdx = baseBucketStart + j;
            HashSlot* slot = &d_hashTable[slotIdx];
            int ptr = slot->ptr;

            if (ptr != HASH_SLOT_FREE &&
                slot->pos.x == blockCoord.x &&
                slot->pos.y == blockCoord.y &&
                slot->pos.z == blockCoord.z) {
                if (debug) printf("    -> Block already allocated at slot %u\n", slotIdx);
                return ptr;
            }

            if (firstEmpty == -1 && ptr == HASH_SLOT_FREE) {
                firstEmpty = slotIdx;
            }
        }

        // 2) Fallback: linear probe over entire table (simple open addressing)
        if (firstEmpty == -1) {
            for (int i = bucketSize; i < totalHashSize; ++i) {
                unsigned int slotIdx = (baseBucketStart + i) % totalHashSize;
                HashSlot* slot = &d_hashTable[slotIdx];
                int ptr = slot->ptr;

                if (ptr != HASH_SLOT_FREE &&
                    slot->pos.x == blockCoord.x &&
                    slot->pos.y == blockCoord.y &&
                    slot->pos.z == blockCoord.z) {
                    if (debug) printf("    -> Block already allocated at slot %u (fallback)\n", slotIdx);
                    return ptr;
                }

                if (ptr == HASH_SLOT_FREE) {
                    firstEmpty = slotIdx;
                    break;
                }
            }
        }

        if (firstEmpty == -1) {
            if (debug) printf("    -> No empty slot found for coord=(%d,%d,%d)\n",
                              blockCoord.x, blockCoord.y, blockCoord.z);
            return -1;
        }

        unsigned int targetBucketId = firstEmpty / bucketSize;

        bool lockAcquired = false;
        for (int spin = 0; spin < MAX_BUCKET_SPIN; ++spin) {
            int previous = atomicExch(&d_hashBucketMutex[targetBucketId], HASH_BUCKET_LOCKED);
            if (previous == HASH_BUCKET_UNLOCKED) {
                lockAcquired = true;
                break;
            }
        }

        if (!lockAcquired) {
            continue; // retry with a new scan
        }

        HashSlot* targetSlot = &d_hashTable[firstEmpty];
        int currentPtr = targetSlot->ptr;

        // Slot might have been filled while we were spinning; double-check
        if (currentPtr != HASH_SLOT_FREE) {
            if (targetSlot->pos.x == blockCoord.x &&
                targetSlot->pos.y == blockCoord.y &&
                targetSlot->pos.z == blockCoord.z) {
                // if (debug) printf("    -> Block already allocated at slot %d after locking\n", firstEmpty);
                d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
                return currentPtr;
            }

            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            continue; // try again to find another empty slot
        }

        unsigned int heapIndex = atomicSub((unsigned int*)d_heapCounter, 1);
        if (heapIndex > 0 && heapIndex < (unsigned int)SDFBlockNum) {
            unsigned int blockIndex = d_heap[heapIndex];
            targetSlot->pos = blockCoord;
            targetSlot->offset = 0;
            __threadfence();
            targetSlot->ptr = static_cast<int>(blockIndex);
            __threadfence();

            // if (debug) {
            //     printf("    -> Allocated block #%u from heap[%u] at slot %d, coord=(%d,%d,%d)\n",
            //            blockIndex, heapIndex, firstEmpty,
            //            blockCoord.x, blockCoord.y, blockCoord.z);
            // }

            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            return static_cast<int>(blockIndex);
        } else {
            // Out of memory - restore counter and release lock
            atomicAdd((unsigned int*)d_heapCounter, 1);
            // if (debug) {
            //     printf("    -> Out of memory! heapIndex=%u (must be > 0 and < %d)\n",
            //            heapIndex, SDFBlockNum);
            // }
            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            return -1;
        }
    }

    // if (debug) {
    //     printf("    -> Failed to acquire bucket lock after %d retries for coord=(%d,%d,%d)\n",
    //            MAX_BUCKET_RETRY, blockCoord.x, blockCoord.y, blockCoord.z);
    // }

    return -1;
}

// ============================================================================
// CUDA Kernels (executed on GPU)
// ============================================================================

/**
 * Initialize hash table kernel
 */
__global__ void initHashTableKernel(HashSlot* d_hashTable, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hashTable[idx].pos = make_int3(-1, -1, -1);
        d_hashTable[idx].ptr = -1;
        d_hashTable[idx].offset = 0;
    }
}

/**
 * Initialize heap kernel (like expert code)
 * Fills heap with block indices in reverse order
 * heap[0] = numBlocks-1, heap[1] = numBlocks-2, ..., heap[numBlocks-1] = 0
 */
__global__ void initHeapKernel(unsigned int* d_heap, int numBlocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBlocks) {
        d_heap[idx] = numBlocks - idx - 1;  // Reverse order: last element first
    }
}

/**
 * Improved allocation kernel using 3D DDA algorithm (like expert code)
 * Each thread processes one pixel and efficiently allocates blocks along the ray
 */
 /**
  * Helper function to allocate blocks along a ray with parent tracking
  * Used by both Method 1 and Method 2 allocation kernels
  */
__device__ void allocateBlocksAlongRay(
    HashSlot* d_hashTable,
    unsigned int* d_heap,
    unsigned int* d_heapCounter,
    int* d_hashBucketMutex,
    int2* d_blockParentUV,
    unsigned char* d_blockAllocationMethod,
    float3 rayStart,
    float3 rayEnd,
    float voxelSize,
    int SDFBlockNum,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    int parentPixelIdx,
    int parentPixelX,
    int parentPixelY,
    uchar allocationMethod
) {
    // Debug: Only print for first few pixels to avoid spam
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        // Using first thread as debug
    }

    int3 idCurrentTotalVoxel = make_int3(
        (int)floorf(rayStart.x / voxelSize),
        (int)floorf(rayStart.y / voxelSize),
        (int)floorf(rayStart.z / voxelSize)
    );
    int3 idTotalEnd = make_int3(
        (int)floorf(rayEnd.x / voxelSize),
        (int)floorf(rayEnd.y / voxelSize),
        (int)floorf(rayEnd.z / voxelSize)
    );

    // Convert to block coordinates using helper function (like expert code)
    int3 idCurrentVoxel = worldToSDFBlock(rayStart, voxelSize);
    int3 idEnd = worldToSDFBlock(rayEnd, voxelSize);

    // Debug: print voxel IDs for first pixel
    if (parentPixelIdx == 186748) {
        printf("allocateBlocksAlongRay: parentPixel=%d, rayStart=(%.3f,%.3f,%.3f), rayEnd=(%.3f,%.3f,%.3f)\n",
            parentPixelIdx, rayStart.x, rayStart.y, rayStart.z, rayEnd.x, rayEnd.y, rayEnd.z);
        printf("  idCurrentVoxel=(%d,%d,%d), idEnd=(%d,%d,%d)\n",
            idCurrentVoxel.x, idCurrentVoxel.y, idCurrentVoxel.z, idEnd.x, idEnd.y, idEnd.z);
        printf("  idCurrentTotalVoxel=(%d,%d,%d), idTotalEnd=(%d,%d,%d)\n",
            idCurrentTotalVoxel.x, idCurrentTotalVoxel.y, idCurrentTotalVoxel.z, idTotalEnd.x, idTotalEnd.y, idTotalEnd.z);
    }

    // Debug: Check if ray is valid (has length)
    if (idCurrentVoxel.x == idEnd.x && idCurrentVoxel.y == idEnd.y && idCurrentVoxel.z == idEnd.z) {
        // Ray too short, skip
        return;
    }

    // Calculate ray direction
    float3 rayDir = make_float3(rayEnd.x - rayStart.x, rayEnd.y - rayStart.y, rayEnd.z - rayStart.z);
    float rayLength = sqrtf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
    if (rayLength < 1e-6f) return;

    rayDir.x /= rayLength;
    rayDir.y /= rayLength;
    rayDir.z /= rayLength;

    // 3D DDA algorithm setup (like expert code)
    float3 step = make_float3(
        (rayDir.x > 0) ? 1.0f : -1.0f,
        (rayDir.y > 0) ? 1.0f : -1.0f,
        (rayDir.z > 0) ? 1.0f : -1.0f
    );
    
    // Expert code: boundaryPos = SDFBlockToWorld(idCurrentVoxel + clamp(step, 0, 1)) - 0.5f * voxelSize
    float blockExtent = static_cast<float>(SDF_BLOCK_SIZE) * voxelSize;
    float3 stepClamped = make_float3(
        (step.x > 0) ? 1.0f : 0.0f,
        (step.y > 0) ? 1.0f : 0.0f,
        (step.z > 0) ? 1.0f : 0.0f
    );
    float3 boundaryPos = make_float3(
        (idCurrentVoxel.x + stepClamped.x) * blockExtent - 0.5f * voxelSize,
        (idCurrentVoxel.y + stepClamped.y) * blockExtent - 0.5f * voxelSize,
        (idCurrentVoxel.z + stepClamped.z) * blockExtent - 0.5f * voxelSize
    );

    float3 tMax = make_float3(
        (boundaryPos.x - rayStart.x) / rayDir.x,
        (boundaryPos.y - rayStart.y) / rayDir.y,
        (boundaryPos.z - rayStart.z) / rayDir.z
    );

    float3 tDelta = make_float3(
        (step.x * blockExtent) / rayDir.x,
        (step.y * blockExtent) / rayDir.y,
        (step.z * blockExtent) / rayDir.z
    );

    // Handle zero direction components
    if (rayDir.x == 0.0f) { tMax.x = 1e6f; tDelta.x = 1e6f; }
    if (rayDir.y == 0.0f) { tMax.y = 1e6f; tDelta.y = 1e6f; }
    if (rayDir.z == 0.0f) { tMax.z = 1e6f; tDelta.z = 1e6f; }

    // Debug: Track world position of current block center
    float3 currentWorldPos = make_float3(
        (idCurrentVoxel.x + 0.5f) * blockExtent,
        (idCurrentVoxel.y + 0.5f) * blockExtent,
        (idCurrentVoxel.z + 0.5f) * blockExtent
    );

    // Debug flag: log for first few pixels to see allocation pattern
    // pixelIdx is the linear index (y * width + x)
    bool shouldDebug = (parentPixelIdx == 186748);

    // if (shouldDebug) {
    //     printf("[Pixel %d] Ray traversal: pixel(%d,%d), rayStart(%.3f,%.3f,%.3f) -> rayEnd(%.3f,%.3f,%.3f)\n",
    //         parentPixelIdx, parentPixelX, parentPixelY, rayStart.x, rayStart.y, rayStart.z, rayEnd.x, rayEnd.y, rayEnd.z);
    //     printf("  Block coords: start(%d,%d,%d) -> end(%d,%d,%d), rayDir(%.3f,%.3f,%.3f)\n",
    //         idCurrentVoxel.x, idCurrentVoxel.y, idCurrentVoxel.z,
    //         idEnd.x, idEnd.y, idEnd.z,
    //         rayDir.x, rayDir.y, rayDir.z);
    // }

    // 3D DDA traversal - walk along the ray efficiently
    int maxIterations = 1024; // Safety limit
    for (int iter = 0; iter < maxIterations; iter++) {
        // Allocate current block
        int3 blockCoord = idCurrentVoxel;
        
        // if (shouldDebug && iter < 10) {
        //     currentWorldPos = SDFBlockToWorld(blockCoord, voxelSize);
        //     printf("  [Iter %d] Block(%d,%d,%d) World(%.3f,%.3f,%.3f)\n",
        //            iter, blockCoord.x, blockCoord.y, blockCoord.z,
        //            currentWorldPos.x, currentWorldPos.y, currentWorldPos.z);
        // }
        
        // Allocate current block (using helper function, like expert code)
        // if (shouldDebug) {
        //     printf("  Before allocBlock: iter=%d, blockCoord=(%d,%d,%d)\n", iter, blockCoord.x, blockCoord.y, blockCoord.z);
        // }
        int blockPtr = allocBlockWithMeta(d_hashTable,
                   d_heap,
                   d_heapCounter,
                   d_hashBucketMutex,
                   blockCoord,
                   SDFBlockNum,
                   numBuckets,
                   bucketSize,
                   totalHashSize,
                   shouldDebug);

        if (allocationMethod == 1 && blockPtr >= 0 &&
            d_blockParentUV != nullptr && d_blockAllocationMethod != nullptr) {
            d_blockParentUV[blockPtr] = make_int2(parentPixelX, parentPixelY);
            d_blockAllocationMethod[blockPtr] = allocationMethod;
        }
        
        // Note: Expert code doesn't allocate neighboring blocks in the main allocation loop
        
        // Check if we've reached the end
        if (idCurrentVoxel.x == idEnd.x && idCurrentVoxel.y == idEnd.y && idCurrentVoxel.z == idEnd.z) {
            // if (shouldDebug) {
            //     printf("  Reached end at iter %d\n", iter);
            // }
            break;
        }

        // Find next block using 3D DDA
        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            tMax.x += tDelta.x;
            idCurrentVoxel.x += (int)step.x;
        }
        else if (tMax.y < tMax.z) {
            tMax.y += tDelta.y;
            idCurrentVoxel.y += (int)step.y;
        }
        else {
            tMax.z += tDelta.z;
            idCurrentVoxel.z += (int)step.z;
        }
    }

    if (shouldDebug) {
        printf("Ray traversal completed for pixel(%d,%d)\n", parentPixelX, parentPixelY);
    }
}

/**
 * Method 1: Allocate blocks using camera direction only (current implementation)
 * Each thread processes one depth map pixel and traces a ray in camera direction
 */
__global__ void allocBlocksFromDepthMapMethod1Kernel(
    HashSlot* d_hashTable,
    unsigned int* d_heap,
    unsigned int* d_heapCounter,
    int* d_hashBucketMutex,
    int2* d_blockParentUV,
    unsigned char* d_blockAllocationMethod,
    const float3* depthmap,
    int width, int height,
    float truncationDistance,
    int SDFBlockNum,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    float voxelSize,
    float3 cameraPos,
    float* cameraTransform,
    int* d_validPixelCounter
) {
    // Get pixel coordinates (one thread per pixel)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate linear pixel index from 2D coordinates
    // Depthmap is stored in row-major order (row=y, col=x)
    int pixelIdx = y * width + x;
    
    // Debug: Only process pixel 186748 for testing
    //if (pixelIdx != 186748) return;

    // Debug: Check if pixel is out of bounds
    if (pixelIdx < 0 || pixelIdx >= width * height) {
        printf("ERROR: pixelIdx=%d is out of bounds for %dx%d\n", pixelIdx, width, height);
        return;
    }

    // Null check for pointers (debug)
    if (depthmap == nullptr || d_hashTable == nullptr || d_heap == nullptr || d_heapCounter == nullptr || cameraTransform == nullptr) {
        printf("ERROR: nullptr detected in kernel\n");
        return;
    }

    float3 cameraPos_local = depthmap[pixelIdx];

    // Skip invalid pixels (check for very small or zero depth)
    if (cameraPos_local.z <= 1e-6f) {
        //printf("invalid pixel idx : %d\n", pixelIdx);
        return;
    }
    // Count valid pixels
    if (d_validPixelCounter != nullptr) {
        atomicAdd((unsigned int*)d_validPixelCounter, 1);
    }

    //printf("Valid pixel idx : %d\n", pixelIdx);

    // Transform from Camera coordinates to World coordinates
    // worldPos = cameraTransform * cameraPos_local
    float3 worldPos = make_float3(
        cameraTransform[0] * cameraPos_local.x + cameraTransform[1] * cameraPos_local.y + cameraTransform[2] * cameraPos_local.z + cameraTransform[3],
        cameraTransform[4] * cameraPos_local.x + cameraTransform[5] * cameraPos_local.y + cameraTransform[6] * cameraPos_local.z + cameraTransform[7],
        cameraTransform[8] * cameraPos_local.x + cameraTransform[9] * cameraPos_local.y + cameraTransform[10] * cameraPos_local.z + cameraTransform[11]
    );

    // Calculate ray direction from camera to surface point (both in world coordinates now)
    float3 rayDir = make_float3(
        worldPos.x - cameraPos.x,
        worldPos.y - cameraPos.y,
        worldPos.z - cameraPos.z
    );

    float rayLength = sqrtf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
    if (rayLength < 1e-6f) return;

    // Normalize ray direction
    rayDir.x /= rayLength;
    rayDir.y /= rayLength;
    rayDir.z /= rayLength;

    // Debug: only process first pixel for detailed logging
    bool isDebugPixel = (pixelIdx == 186748);
    
    if (isDebugPixel) {
        printf("RAY Debug for pixel 186748:\n");
        printf("  cameraPos_local=(%.3f,%.3f,%.3f)\n", cameraPos_local.x, cameraPos_local.y, cameraPos_local.z);
        printf("  worldPos=(%.3f,%.3f,%.3f)\n", worldPos.x, worldPos.y, worldPos.z);
        printf("  cameraPos=(%.3f,%.3f,%.3f)\n", cameraPos.x, cameraPos.y, cameraPos.z);
        printf("  rayDir=(%.3f,%.3f,%.3f)\n", rayDir.x, rayDir.y, rayDir.z);
        printf("  rayLength=%.3f\n", rayLength);
        printf("  truncationDistance=%.3f\n", truncationDistance);
        printf("  voxel size : %f\n", voxelSize);
    }

    // Calculate rayMin and rayMax in camera space, then transform to world
    // Like expert code: rayMin = depth - truncation, rayMax = depth + truncation
    float3 cameraRayMin = make_float3(
        cameraPos_local.x,
        cameraPos_local.y,
        cameraPos_local.z - truncationDistance  // Towards camera
    );
    float3 cameraRayMax = make_float3(
        cameraPos_local.x,
        cameraPos_local.y,
        cameraPos_local.z + truncationDistance  // Away from camera
    );

    // Transform to world coordinates
    float3 rayMin = make_float3(
        cameraTransform[0] * cameraRayMin.x + cameraTransform[1] * cameraRayMin.y + cameraTransform[2] * cameraRayMin.z + cameraTransform[3],
        cameraTransform[4] * cameraRayMin.x + cameraTransform[5] * cameraRayMin.y + cameraTransform[6] * cameraRayMin.z + cameraTransform[7],
        cameraTransform[8] * cameraRayMin.x + cameraTransform[9] * cameraRayMin.y + cameraTransform[10] * cameraRayMin.z + cameraTransform[11]
    );
    float3 rayMax = make_float3(
        cameraTransform[0] * cameraRayMax.x + cameraTransform[1] * cameraRayMax.y + cameraTransform[2] * cameraRayMax.z + cameraTransform[3],
        cameraTransform[4] * cameraRayMax.x + cameraTransform[5] * cameraRayMax.y + cameraTransform[6] * cameraRayMax.z + cameraTransform[7],
        cameraTransform[8] * cameraRayMax.x + cameraTransform[9] * cameraRayMax.y + cameraTransform[10] * cameraRayMax.z + cameraTransform[11]
    );

    // Debug: print ray endpoints for first pixel
    if (pixelIdx == 186748) {
        printf("  rayMin=(%.3f,%.3f,%.3f), rayMax=(%.3f,%.3f,%.3f)\n",
            rayMin.x, rayMin.y, rayMin.z, rayMax.x, rayMax.y, rayMax.z);
    }

    // Use helper function to allocate blocks along ray (Method 1)
    allocateBlocksAlongRay(
        d_hashTable,
        d_heap,
        d_heapCounter,
        d_hashBucketMutex,
        d_blockParentUV,
        d_blockAllocationMethod,
        rayMin,
        rayMax,
        voxelSize,
        SDFBlockNum,
        numBuckets,
        bucketSize,
        totalHashSize,
        pixelIdx,
        x,
        y,
        0
    ); // Method 1 (camera direction, both sides)
}

// Helper function to allocate a block if it doesn't exist. Returns block index or -1 on failure.
__device__ int allocBlockWithMeta(
    HashSlot* d_hashTable,
    unsigned int* d_heap,
    unsigned int* d_heapCounter,
    int* d_hashBucketMutex,
    const int3& blockCoord,
    int SDFBlockNum,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    bool debug) {
    unsigned int baseBucketId = hashBlockCoordinate(blockCoord, numBuckets);
    unsigned int baseBucketStart = baseBucketId * bucketSize;

    for (int retry = 0; retry < MAX_BUCKET_RETRY; ++retry) {
        int firstEmpty = -1;

        // Probe within hashed bucket
        for (int j = 0; j < bucketSize; ++j) {
            unsigned int slotIdx = baseBucketStart + j;
            HashSlot* slot = &d_hashTable[slotIdx];
            int ptr = slot->ptr;

            if (ptr != HASH_SLOT_FREE &&
                slot->pos.x == blockCoord.x &&
                slot->pos.y == blockCoord.y &&
                slot->pos.z == blockCoord.z) {
                return ptr;
            }

            if (firstEmpty == -1 && ptr == HASH_SLOT_FREE) {
                firstEmpty = slotIdx;
            }
        }

        // Fallback: linear probe entire table
        if (firstEmpty == -1) {
            for (int i = bucketSize; i < totalHashSize; ++i) {
                unsigned int slotIdx = (baseBucketStart + i) % totalHashSize;
                HashSlot* slot = &d_hashTable[slotIdx];
                int ptr = slot->ptr;

                if (ptr != HASH_SLOT_FREE &&
                    slot->pos.x == blockCoord.x &&
                    slot->pos.y == blockCoord.y &&
                    slot->pos.z == blockCoord.z) {
                    return ptr;
                }

                if (ptr == HASH_SLOT_FREE) {
                    firstEmpty = slotIdx;
                    break;
                }
            }
        }

        if (firstEmpty == -1) {
            if (debug) {
                printf("allocBlock: No empty slot for coord=(%d,%d,%d)\n",
                    blockCoord.x, blockCoord.y, blockCoord.z);
            }
            return -1;
        }

        unsigned int targetBucketId = firstEmpty / bucketSize;
        bool lockAcquired = false;
        for (int spin = 0; spin < MAX_BUCKET_SPIN; ++spin) {
            int previous = atomicExch(&d_hashBucketMutex[targetBucketId], HASH_BUCKET_LOCKED);
            if (previous == HASH_BUCKET_UNLOCKED) {
                lockAcquired = true;
                break;
            }
        }

        if (!lockAcquired) {
            continue;
        }

        HashSlot* targetSlot = &d_hashTable[firstEmpty];
        int currentPtr = targetSlot->ptr;

        if (currentPtr != HASH_SLOT_FREE) {
            if (targetSlot->pos.x == blockCoord.x &&
                targetSlot->pos.y == blockCoord.y &&
                targetSlot->pos.z == blockCoord.z) {
                d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
                return currentPtr;
            }

            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            continue;
        }

        unsigned int heapIndex = atomicSub((unsigned int*)d_heapCounter, 1);
        if (heapIndex > 0 && heapIndex < (unsigned int)SDFBlockNum) {
            unsigned int blockIndex = d_heap[heapIndex];
            targetSlot->pos = blockCoord;
            targetSlot->offset = 0;
            __threadfence();
            targetSlot->ptr = static_cast<int>(blockIndex);
            __threadfence();

            if (debug) {
                printf("allocBlock: Alloc block #%u at slot %d (coord=%d,%d,%d)\n",
                    blockIndex, firstEmpty, blockCoord.x, blockCoord.y, blockCoord.z);
            }

            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            return static_cast<int>(blockIndex);
        } else {
            atomicAdd((unsigned int*)d_heapCounter, 1);
            if (debug) {
                printf("allocBlock: Out of memory for coord=(%d,%d,%d)\n",
                    blockCoord.x, blockCoord.y, blockCoord.z);
            }
            d_hashBucketMutex[targetBucketId] = HASH_BUCKET_UNLOCKED;
            return -1;
        }
    }

    if (debug) {
        printf("allocBlock: Failed after retries for coord=(%d,%d,%d)\n",
            blockCoord.x, blockCoord.y, blockCoord.z);
    }

    return -1;
}

/**
 * Method 2: Allocate blocks using normal direction for inside surface
 * Each thread processes one depth map pixel and traces rays in both camera and normal directions
 */
__global__ void allocBlocksFromDepthMapMethod2Kernel(
    HashSlot* d_hashTable,
    unsigned int* d_heap,
    unsigned int* d_heapCounter,
    int* d_hashBucketMutex,
    int2* d_blockParentUV,
    unsigned char* d_blockAllocationMethod,
    const float3* depthmap,
    const float3* normalmap,
    int width, int height,
    float truncationDistance,
    int SDFBlockNum,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    float voxelSize,
    float3 cameraPos,
    float* cameraTransform
) {
    // Get thread index (each thread processes one depth map pixel)
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalPixels = width * height;

    if (idx >= totalPixels) return;

    // Null check for pointers (debug)
    if (depthmap == nullptr || normalmap == nullptr || d_hashTable == nullptr || d_heap == nullptr || d_heapCounter == nullptr || cameraTransform == nullptr) {
        printf("ERROR: nullptr detected in Method 2 kernel\n");
        return;
    }

    // Calculate 2D coordinates from linear index
    int x = idx % width;
    int y = idx / width;

    // pixelIdx equals idx for row-major storage
    int pixelIdx = idx;  // For consistency with method 1
    float3 cameraPos_local = depthmap[pixelIdx];
    float3 normal_camera = normalmap[pixelIdx];

    // Skip invalid pixels (no depth data)
    if (cameraPos_local.x == 0.0f && cameraPos_local.y == 0.0f && cameraPos_local.z == 0.0f) return;

    // Skip invalid normals
    float normalLength = sqrtf(normal_camera.x * normal_camera.x + normal_camera.y * normal_camera.y + normal_camera.z * normal_camera.z);
    if (normalLength < 0.1f) return; // Invalid normal

    // Normalize normal vector
    normal_camera.x /= normalLength;
    normal_camera.y /= normalLength;
    normal_camera.z /= normalLength;

    // Transform from Camera coordinates to World coordinates
    // worldPos = cameraTransform * cameraPos_local
    float3 worldPos = make_float3(
        cameraTransform[0] * cameraPos_local.x + cameraTransform[1] * cameraPos_local.y + cameraTransform[2] * cameraPos_local.z + cameraTransform[3],
        cameraTransform[4] * cameraPos_local.x + cameraTransform[5] * cameraPos_local.y + cameraTransform[6] * cameraPos_local.z + cameraTransform[7],
        cameraTransform[8] * cameraPos_local.x + cameraTransform[9] * cameraPos_local.y + cameraTransform[10] * cameraPos_local.z + cameraTransform[11]
    );

    // Transform normal from Camera to World (only rotation part)
    // normal_world = cameraTransform[0:2, 0:2] * normal_camera
    float3 normal = make_float3(
        cameraTransform[0] * normal_camera.x + cameraTransform[1] * normal_camera.y + cameraTransform[2] * normal_camera.z,
        cameraTransform[4] * normal_camera.x + cameraTransform[5] * normal_camera.y + cameraTransform[6] * normal_camera.z,
        cameraTransform[8] * normal_camera.x + cameraTransform[9] * normal_camera.y + cameraTransform[10] * normal_camera.z
    );

    // Normalize again after transformation
    normalLength = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normalLength > 1e-6f) {
        normal.x /= normalLength;
        normal.y /= normalLength;
        normal.z /= normalLength;
    }

    // Method 2: Trace rays in both camera direction (+truncation) and normal direction (-truncation)

    // 1. Camera direction ray (from camera to surface + truncation, both in world coordinates)
    float3 rayDir = make_float3(worldPos.x - cameraPos.x, worldPos.y - cameraPos.y, worldPos.z - cameraPos.z);
    float rayLength = sqrtf(rayDir.x * rayDir.x + rayDir.y * rayDir.y + rayDir.z * rayDir.z);
    if (rayLength > 0.0f) {
        rayDir.x /= rayLength;
        rayDir.y /= rayLength;
        rayDir.z /= rayLength;

        // Trace camera direction ray (surface + truncation)
        float3 rayStart = worldPos;
        float3 rayEnd = make_float3(
            worldPos.x + rayDir.x * truncationDistance,
            worldPos.y + rayDir.y * truncationDistance,
            worldPos.z + rayDir.z * truncationDistance
        );

    // Allocate blocks along camera direction ray
    allocateBlocksAlongRay(
        d_hashTable,
        d_heap,
        d_heapCounter,
        d_hashBucketMutex,
        d_blockParentUV,
        d_blockAllocationMethod,
        rayStart,
        rayEnd,
        voxelSize,
        SDFBlockNum,
        numBuckets,
        bucketSize,
        totalHashSize,
        pixelIdx,
        x,
        y,
        0
    ); // Method 1 (camera direction)
    }

    // 2. Normal direction ray (from surface into object -truncation)
    float3 normalRayStart = worldPos;
    float3 normalRayEnd = make_float3(
        worldPos.x - normal.x * truncationDistance,
        worldPos.y - normal.y * truncationDistance,
        worldPos.z - normal.z * truncationDistance
    );

    // Allocate blocks along normal direction ray
    allocateBlocksAlongRay(
        d_hashTable,
        d_heap,
        d_heapCounter,
        d_hashBucketMutex,
        d_blockParentUV,
        d_blockAllocationMethod,
        normalRayStart,
        normalRayEnd,
        voxelSize,
        SDFBlockNum,
        numBuckets,
        bucketSize,
        totalHashSize,
        pixelIdx,
        x,
        y,
        1
    ); // Method 2 (normal direction)
}

/**
 * Improved integration kernel using reverse projection (like expert code)
 * Each thread processes one voxel within a block (SDF_BLOCK_SIZE^3 threads per block)
 */
__global__ void integrateDepthMapIntoBlocksKernel(
    HashSlot* d_hashCompactified,
    VoxelData* d_SDFBlocks,
    const int2* d_blockParentUV,
    const unsigned char* d_blockAllocationMethod,
    const float3* depthmap,
    const uchar3* colormap,
    const float3* normalmap,
    int width, int height,
    float truncationDistance,
    int numActiveBlocks,
    float voxelSize,
    float3 cameraPos,
    float* cameraTransform,
    float fx, float fy, float cx, float cy
) {
    // Get block index from thread (each CUDA block processes one SDF block)
    int activeBlockIdx = blockIdx.x;

    // Check if we have a valid block to process
    if (activeBlockIdx >= numActiveBlocks) return;

    // Get active block from compactified array (like expert code)
    HashSlot slot = d_hashCompactified[activeBlockIdx];

    int3 blockCoord = slot.pos;
    const int blockSize = SDF_BLOCK_SIZE;
    const int voxelsPerBlock = blockSize * blockSize * blockSize;
    // Each SDF block contains SDF_BLOCK_SIZE^3 voxels; slot.ptr is the block index
    VoxelData* block = d_SDFBlocks + (slot.ptr * voxelsPerBlock);

    // Each thread processes one voxel within the block (SDF_BLOCK_SIZE^3 voxels)
    uint voxelIdx = threadIdx.x;
    if (voxelIdx >= voxelsPerBlock) return;

    // Convert linear voxel index to 3D coordinates within block
    int localX = voxelIdx % blockSize;
    int localY = (voxelIdx / blockSize) % blockSize;
    int localZ = voxelIdx / (blockSize * blockSize);

    // Calculate world position of this voxel
    float3 voxelWorldPos = make_float3(
        (blockCoord.x * blockSize + localX) * voxelSize,
        (blockCoord.y * blockSize + localY) * voxelSize,
        (blockCoord.z * blockSize + localZ) * voxelSize
    );

    // Apply inverse camera transform to transform voxel world position to camera space
    // cameraTransform is Camera->World [R|T], so inverse is [R^T|-R^T*T]
    // For rotation, R_inv = R^T (transpose)
    // For translation, T_inv = -R^T * T
    // voxelCameraPos = R^T * (voxelWorldPos - T)
    float3 worldPosRelative = make_float3(
        voxelWorldPos.x - cameraTransform[3],
        voxelWorldPos.y - cameraTransform[7],
        voxelWorldPos.z - cameraTransform[11]
    );

    float3 voxelCameraPos = make_float3(
        cameraTransform[0] * worldPosRelative.x + cameraTransform[4] * worldPosRelative.y + cameraTransform[8] * worldPosRelative.z,
        cameraTransform[1] * worldPosRelative.x + cameraTransform[5] * worldPosRelative.y + cameraTransform[9] * worldPosRelative.z,
        cameraTransform[2] * worldPosRelative.x + cameraTransform[6] * worldPosRelative.y + cameraTransform[10] * worldPosRelative.z
    );

    // Project to screen coordinates using camera intrinsics (like expert code)
    // Expert code: pos.x*fx/pos.z + cx, pos.y*fy/pos.z + cy
    float2 screenPosFloat = make_float2(
        voxelCameraPos.x * fx / voxelCameraPos.z + cx,
        voxelCameraPos.y * fy / voxelCameraPos.z + cy
    );

    // Convert to integer coordinates (like expert code: +0.5f for rounding)
    int2 screenPos = make_int2(
        (int)(screenPosFloat.x + 0.5f),
        (int)(screenPosFloat.y + 0.5f)
    );

    //if (voxelIdx <100) {

    //    printf("------------\n");
    //    printf("voxelIdx: %d\n", voxelIdx);
    //    printf("blockCoord : %d %d %d\n", blockCoord.x, blockCoord.y, blockCoord.z);
    //    printf("voxelWorldPos : %f %f %f\n", voxelWorldPos.x, voxelWorldPos.y, voxelWorldPos.z);
    //    printf("voxelCameraPos : %f %f %f\n", voxelCameraPos.x, voxelCameraPos.y, voxelCameraPos.z);
    //    printf("screenPos : %d %d\n", screenPos.x, screenPos.y);
    //    printf("------------\n");

    //}

    //else {
    //    return;
    //}

    int pixelIdx = -1;
    float3 depthPos_camera = make_float3(0.0f, 0.0f, 0.0f);
    bool pixelValid = false;

    int blockPtrIndex = slot.ptr;
    int2 parentUV = make_int2(-1, -1);
    unsigned char blockAllocMethod = 0;
    if (d_blockParentUV != nullptr && blockPtrIndex >= 0) {
        parentUV = d_blockParentUV[blockPtrIndex];
    }
    if (d_blockAllocationMethod != nullptr && blockPtrIndex >= 0) {
        blockAllocMethod = d_blockAllocationMethod[blockPtrIndex];
    }

    bool hasStoredParent = (blockAllocMethod == 1 &&
        parentUV.x >= 0 && parentUV.x < width &&
        parentUV.y >= 0 && parentUV.y < height);

    if (hasStoredParent) {
        pixelIdx = parentUV.y * width + parentUV.x;
        depthPos_camera = depthmap[pixelIdx];
        if (depthPos_camera.x != 0.0f || depthPos_camera.y != 0.0f || depthPos_camera.z != 0.0f) {
            pixelValid = true;
        }
    }

    if (!pixelValid && screenPos.x >= 0 && screenPos.x < width && screenPos.y >= 0 && screenPos.y < height) {
        pixelIdx = screenPos.y * width + screenPos.x;
        depthPos_camera = depthmap[pixelIdx];
        if (depthPos_camera.x != 0.0f || depthPos_camera.y != 0.0f || depthPos_camera.z != 0.0f) {
            pixelValid = true;
        }
    }

    if (!pixelValid) {
        return;
    }

    float3 depthPos_world = make_float3(
        cameraTransform[0] * depthPos_camera.x + cameraTransform[1] * depthPos_camera.y + cameraTransform[2] * depthPos_camera.z + cameraTransform[3],
        cameraTransform[4] * depthPos_camera.x + cameraTransform[5] * depthPos_camera.y + cameraTransform[6] * depthPos_camera.z + cameraTransform[7],
        cameraTransform[8] * depthPos_camera.x + cameraTransform[9] * depthPos_camera.y + cameraTransform[10] * depthPos_camera.z + cameraTransform[11]
    );

    float3 sdfDir = make_float3(voxelWorldPos.x - depthPos_world.x, voxelWorldPos.y - depthPos_world.y, voxelWorldPos.z - depthPos_world.z);
    float sdfValue = sqrtf(sdfDir.x * sdfDir.x + sdfDir.y * sdfDir.y + sdfDir.z * sdfDir.z);

    float3 cameraDir = make_float3(depthPos_world.x - cameraPos.x, depthPos_world.y - cameraPos.y, depthPos_world.z - cameraPos.z);
    float cameraDist = sqrtf(cameraDir.x * cameraDir.x + cameraDir.y * cameraDir.y + cameraDir.z * cameraDir.z);
    float voxelDistToCamera = sqrtf((voxelWorldPos.x - cameraPos.x) * (voxelWorldPos.x - cameraPos.x) +
        (voxelWorldPos.y - cameraPos.y) * (voxelWorldPos.y - cameraPos.y) +
        (voxelWorldPos.z - cameraPos.z) * (voxelWorldPos.z - cameraPos.z));

    if (voxelDistToCamera < cameraDist) {
        sdfValue = -sdfValue;
    }

    if (sdfValue < -truncationDistance || sdfValue > truncationDistance) {
        return;
    }

    sdfValue = fmaxf(-truncationDistance, fminf(truncationDistance, sdfValue));

    uchar3 color = colormap[pixelIdx];
    float3 normal_camera = normalmap[pixelIdx];
    float normalLength = sqrtf(normal_camera.x * normal_camera.x + normal_camera.y * normal_camera.y + normal_camera.z * normal_camera.z);
    if (normalLength > 0.0f) {
        normal_camera.x /= normalLength;
        normal_camera.y /= normalLength;
        normal_camera.z /= normalLength;
    }

    float3 normal = make_float3(
        cameraTransform[0] * normal_camera.x + cameraTransform[1] * normal_camera.y + cameraTransform[2] * normal_camera.z,
        cameraTransform[4] * normal_camera.x + cameraTransform[5] * normal_camera.y + cameraTransform[6] * normal_camera.z,
        cameraTransform[8] * normal_camera.x + cameraTransform[9] * normal_camera.y + cameraTransform[10] * normal_camera.z
    );

    normalLength = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normalLength > 1e-6f) {
        normal.x /= normalLength;
        normal.y /= normalLength;
        normal.z /= normalLength;
    }

    float weightUpdate = 1.0f;
    VoxelData* voxel = &block[voxelIdx];
    float oldSDF = voxel->sdf;
    float oldWeight = voxel->weight;
    float newWeight = oldWeight + weightUpdate;

    if (newWeight > 0.0f) {
        voxel->sdf = (oldSDF * oldWeight + sdfValue * weightUpdate) / newWeight;
        voxel->weight = (uchar)fminf(255.0f, newWeight);

        float3 oldColor = make_float3(voxel->color.x, voxel->color.y, voxel->color.z);
        float3 newColor = make_float3(
            (oldColor.x * oldWeight + color.x * weightUpdate) / newWeight,
            (oldColor.y * oldWeight + color.y * weightUpdate) / newWeight,
            (oldColor.z * oldWeight + color.z * weightUpdate) / newWeight
        );
        voxel->color = make_uchar3(
            (unsigned char)newColor.x,
            (unsigned char)newColor.y,
            (unsigned char)newColor.z
        );

        voxel->isUpdated = 1;

        if ((oldSDF > 0.0f && voxel->sdf <= 0.0f) ||
            (oldSDF <= 0.0f && voxel->sdf > 0.0f)) {
            voxel->isZeroCrossing = 1;
        }
    }

    //if (voxelIdx > 0) {

    //    //printf("------------\n");
    //    //printf("voxelIdx: %d\n", voxelIdx);
    //    //printf("blockCoord : %d %d %d\n", blockCoord.x, blockCoord.y, blockCoord.z);
    //    //printf("voxelWorldPos : %f %f %f\n", voxelWorldPos.x, voxelWorldPos.y, voxelWorldPos.z);
    //    //printf("voxelCameraPos : %f %f %f\n", voxelCameraPos.x, voxelCameraPos.y, voxelCameraPos.z);
    //    //printf("screenPos : %d %d\n", screenPos.x, screenPos.y);
    //    //printf("sdfValue : %f\n", block[voxelIdx].sdf);
    //    //printf("------------\n");

    //    //if (block[voxelIdx].sdf > 0.0f) {
    //    //    printf("%f %f %f\n", voxelWorldPos.x, voxelWorldPos.y, voxelWorldPos.z);
    //    //}
    //    //if (block[voxelIdx].sdf > 0.0f) {
    //    //    printf("%f %f %f\n", voxelWorldPos.x, voxelWorldPos.y, voxelWorldPos.z);
    //    //}

    //}

}


__global__ void resetHashMutexKernel(CUDAHashRef hashData, int numBlocks) {

    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numBlocks) {
        hashData.d_hashBucketMutex[idx] = HASH_BUCKET_UNLOCKED;
    }

}

// ============================================================================
// Host Functions (executed on CPU, call kernels)
// ============================================================================

/**
 * Reset HashData (initialize heap, hash table, set all to 0)
 */
extern "C" void resetHashDataCUDA(CUDAHashRef & hashData, const Params & params) {
    printf("resetHashDataCUDA: Resetting hash data...\n");
    
    // Initialize heap with free block indices
    if (hashData.d_heap && hashData.d_heapCounter) {
        int blockSize = 256;
        int numBlocks = (params.SDFBlockNum + blockSize - 1) / blockSize;
        
        // Launch kernel
        initHeapKernel << <numBlocks, blockSize >> > (hashData.d_heap, params.SDFBlockNum);
        
        // Wait for kernel to complete before setting counter
        cudaDeviceSynchronize();
        
        // Set heap counter to max (all blocks are free)
        unsigned int maxCount = params.SDFBlockNum - 1;
        cudaMemcpy(hashData.d_heapCounter, &maxCount, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    
    // Initialize hash table to empty
    if (hashData.d_hashTable) {
        int blockSize = 256;
        int numBlocks = (params.totalHashSize + blockSize - 1) / blockSize;
        
        // Launch kernel
        initHashTableKernel << <numBlocks, blockSize >> > (hashData.d_hashTable, params.totalHashSize);
    }

    // Reset bucket mutexes
    if (hashData.d_hashBucketMutex) {
        cudaMemset(hashData.d_hashBucketMutex, HASH_BUCKET_UNLOCKED, params.hashSlotNum * sizeof(int));
    }
    
    // Initialize compactified hash table
    if (hashData.d_CompactifiedHashTable) {
        int blockSize = 256;
        int numBlocks = (params.totalHashSize + blockSize - 1) / blockSize;
        
        initHashTableKernel << <numBlocks, blockSize >> > (hashData.d_CompactifiedHashTable, params.totalHashSize);
    }
    
    // Reset counters
    if (hashData.d_hashCompactifiedCounter) {
        cudaMemset(hashData.d_hashCompactifiedCounter, 0, sizeof(int));
    }
    
    // Synchronize
    cudaDeviceSynchronize();
    
    printf("resetHashDataCUDA: Complete\n");
}

/**
 * Allocate SDF blocks needed for depth map
 */
extern "C" void allocBlocksCUDA(CUDAHashRef & hashData, const Params & params,
    const DepthCameraData & depthCameraData,
    const DepthCameraParams & depthCameraParams) {

    // TODO: Implement block allocation kernel
    // This should:
    // 1. Ray cast through depth map
    // 2. Find voxel blocks that need to be allocated
    // 3. Allocate blocks from heap
    // 4. Insert into hash table
    printf("allocBlocksCUDA: Not fully implemented yet\n");
}

/**
 * Integrate depth map into SDF
 */
extern "C" void integrateDepthMapCUDA(CUDAHashRef & hashData, const Params & params,
    const DepthCameraData & depthCameraData,
    const DepthCameraParams & depthCameraParams) {
    // TODO: Implement depth integration kernel
    // This should:
    // 1. For each active voxel block
    // 2. Project voxel into depth map
    // 3. Update SDF value and weight
    printf("integrateDepthMapCUDA: Not fully implemented yet\n");
}

/**
 * Compact active hash entries
 */
 /**
  * Compactify hash entries - remove empty slots and create contiguous array
  * This improves memory access patterns and reduces wasted space
  */
__global__ void compactifyHashKernel(
    HashSlot* d_hashTable,
    HashSlot* d_CompactifiedHashTable,
    int* d_compactifiedCounter,
    int slotCount
) {
    // Get thread index
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < slotCount) {
        HashSlot slot = d_hashTable[idx];

        // Check if slot has a valid block (like expert code: ptr != FREE_ENTRY)
        if (slot.ptr != -1) {
            // Simple atomic approach (like expert code's COMPACTIFY_HASH_SIMPLE)
            int compactIdx = (int)atomicAdd((unsigned int*)d_compactifiedCounter, 1);
            d_CompactifiedHashTable[compactIdx] = slot;
            
            // Debug: Print first few active blocks
            // if (compactIdx < 10) {
            //     printf("Active block #%d: coord=(%d,%d,%d), ptr=%d, hashSlotIdx=%d\n",
            //            compactIdx, slot.pos.x, slot.pos.y, slot.pos.z, slot.ptr, idx);
            // }
        }
    }
}

/**
 * Compactify hash entries (like expert code)
 * Removes empty slots and creates a contiguous array of active blocks
 */
extern "C" float compactifyHashCUDA(CUDAHashRef & hashData, const Params & params) {
    // printf("compactifyHashCUDA: Starting hash compaction...\n");
    // printf("  Total slots: %d\n", params.totalHashSize);

    // Reset compactified counter
    cudaMemset(hashData.d_hashCompactifiedCounter, 0, sizeof(int));

    // Set up grid and block dimensions
    const unsigned int threadsPerBlock = 256;
    const dim3 gridSize((params.totalHashSize + threadsPerBlock - 1) / threadsPerBlock, 1);
    const dim3 blockSize(threadsPerBlock, 1);

    // printf("  Grid size: %d, Block size: %d\n", gridSize.x, blockSize.x);

    printGpuMemoryUsage("Before compactifyHash kernel");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    compactifyHashKernel<<<gridSize, blockSize>>>(
        hashData.d_hashTable,
        hashData.d_CompactifiedHashTable,
        hashData.d_hashCompactifiedCounter,
        params.totalHashSize
    );

    cudaEventRecord(stopEvent);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("  ERROR: CUDA compactification kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    cudaError_t eventSyncErr = cudaEventSynchronize(stopEvent);
    if (eventSyncErr != cudaSuccess) {
        printf("  ERROR: CUDA compactification event sync failed: %s\n", cudaGetErrorString(eventSyncErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    float kernelElapsedMs = computeKernelElapsedMs(startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    printGpuMemoryUsage("After compactifyHash kernel");

    // Check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA compactification kernel error: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }

    // Get number of compactified entries
    int numCompactified = 0;
    cudaMemcpy(&numCompactified, hashData.d_hashCompactifiedCounter,
        sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate total allocated blocks from heap counter
    unsigned int heapCounter = 0;
    cudaMemcpy(&heapCounter, hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int initialHeapCounter = params.SDFBlockNum - 1;
    unsigned int totalAllocatedBlocks = initialHeapCounter - heapCounter;
    
    // Calculate hash table usage (how many slots are used)
    float hashTableUsagePercent = (float)numCompactified / params.totalHashSize * 100.0f;
    
    // Calculate allocated blocks usage (how many allocated blocks are active)
    float allocatedBlocksUsagePercent = 0.0f;
    if (totalAllocatedBlocks > 0) {
        allocatedBlocksUsagePercent = (float)numCompactified / totalAllocatedBlocks * 100.0f;
    }

    // printf("compactifyHashCUDA: Compaction completed!\n");
    // printf("  Kernel execution time: %.3f ms\n", kernelElapsedMs);
    // printf("  Active blocks: %d\n", numCompactified);
    // printf("  Total allocated blocks: %u (max: %d, used: %.1f%%)\n",
    //     totalAllocatedBlocks, params.SDFBlockNum,
    //     totalAllocatedBlocks > 0 ? (float)totalAllocatedBlocks / params.SDFBlockNum * 100.0f : 0.0f);
    // printf("  Active / Allocated: %d / %u (%.1f%%)\n",
    //     numCompactified, totalAllocatedBlocks, allocatedBlocksUsagePercent);
    // printf("  Hash table usage: %d / %d (%.1f%%)\n",
    //     numCompactified, params.totalHashSize, hashTableUsagePercent);
    
    return kernelElapsedMs;
}

/**
 * Allocate blocks for depthmap-based integration
 */
extern "C" float allocBlocksFromDepthMapMethod1CUDA(
    CUDAHashRef & hashData,
    const Params & params,
    const float3 * depthmap,
    int width,
    int height,
    float truncationDistance,
    float3 cameraPos,
    float* cameraTransform
) {
    // printf("allocBlocksFromDepthMapCUDA: Starting block allocation...\n");
    // printf("  Depth map size: %dx%d\n", width, height);
    // printf("  Truncation distance: %f\n", truncationDistance);

    // Debug: Check pointers
    // printf("  Pointer check: depthmap=%p, d_hashTable=%p, d_heapCounter=%p, cameraTransform=%p\n",
    //     depthmap, hashData.d_hashTable, hashData.d_heapCounter, cameraTransform);

    // Set up grid and block dimensions for depthmap pixels
    // gridSize.x covers width, gridSize.y covers height
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    // printf("  Grid size: (%d, %d), Block size: (%d, %d)\n",
    //     gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    // printf("  Launching kernel with %d threads total (width=%d, height=%d)\n",
    //     gridSize.x * gridSize.y * blockSize.x * blockSize.y, width, height);

    // Check for CUDA errors before kernel launch
    cudaError_t preErr = cudaGetLastError();
    if (preErr != cudaSuccess) {
        printf("  WARNING: CUDA error before kernel launch: %s\n", cudaGetErrorString(preErr));
    }

    printGpuMemoryUsage("Before Method1 allocation");

    // Read heap counter BEFORE allocation to track newly allocated blocks
    unsigned int heapCounterBefore = 0;
    cudaMemcpy(&heapCounterBefore, hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Calculate total allocated blocks so far (initial value is SDFBlockNum - 1)
    unsigned int initialHeapCounter = params.SDFBlockNum - 1;
    unsigned int totalAllocatedBefore = initialHeapCounter - heapCounterBefore;
    
    // Allocate and initialize valid pixel counter
    int* d_validPixelCounter = nullptr;
    cudaMalloc(&d_validPixelCounter, sizeof(int));
    cudaMemset(d_validPixelCounter, 0, sizeof(int));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    // Launch allocation kernel (Method 1)
    allocBlocksFromDepthMapMethod1Kernel<<<gridSize, blockSize>>>(
        hashData.d_hashTable,
        hashData.d_heap,
        hashData.d_heapCounter,
        hashData.d_hashBucketMutex,
        hashData.d_blockParentUV,
        hashData.d_blockAllocationMethod,
        depthmap,
        width,
        height,
        truncationDistance,
        params.SDFBlockNum,
        params.hashSlotNum,           // numBuckets
        params.slotSize,              // bucketSize
        params.totalHashSize,         // totalHashSize
        params.voxelSize,                        // voxelSize (temporary)
        cameraPos,
        cameraTransform,
        d_validPixelCounter
    );

    cudaEventRecord(stopEvent);

    // Check for errors (check kernel launch error first)
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("  ERROR: CUDA kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        cudaFree(d_validPixelCounter);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    cudaError_t eventSyncErr = cudaEventSynchronize(stopEvent);
    if (eventSyncErr != cudaSuccess) {
        printf("  ERROR: CUDA event sync failed: %s\n", cudaGetErrorString(eventSyncErr));
        cudaFree(d_validPixelCounter);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    float kernelElapsedMs = computeKernelElapsedMs(startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Synchronize and check for execution errors (extra safety)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  ERROR: CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_validPixelCounter);
        return 0.0f;
    }
    
    // printf("  Kernel execution completed successfully (%.3f ms)\n", kernelElapsedMs);

    // Read back valid pixel counter
    int numValidPixels = 0;
    cudaMemcpy(&numValidPixels, d_validPixelCounter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_validPixelCounter);

    // Read back heap counter AFTER allocation
    unsigned int heapCounterAfter = 0;
    cudaMemcpy(&heapCounterAfter, hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Calculate newly allocated blocks in this frame
    // heapCounter decreases when blocks are allocated, so new blocks = before - after
    unsigned int newlyAllocatedBlocks = 0;
    if (heapCounterBefore >= heapCounterAfter) {
        newlyAllocatedBlocks = heapCounterBefore - heapCounterAfter;
    } else {
        // Overflow occurred (shouldn't happen in normal operation)
        printf("  ERROR: Heap counter overflow detected! (before=%u, after=%u)\n", 
               heapCounterBefore, heapCounterAfter);
        newlyAllocatedBlocks = 0;
    }
    
    // Calculate total allocated blocks so far
    unsigned int totalAllocatedAfter = initialHeapCounter - heapCounterAfter;
    
    // printf("allocBlocksFromDepthMapMethod1CUDA: Block allocation completed!\n");
    // printf("  Valid pixels processed: %d\n", numValidPixels);
    // printf("  Heap counter: before=%u, after=%u\n", heapCounterBefore, heapCounterAfter);
    // printf("  NEWLY allocated blocks in this frame: %u\n", newlyAllocatedBlocks);
    // printf("  NEWLY allocated voxels in this frame: %u (blocks * %d)\n", newlyAllocatedBlocks * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE), SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
    // printf("  Total allocated blocks: %u (max: %d, used: %.1f%%)\n",
    //     totalAllocatedAfter, params.SDFBlockNum, (float)totalAllocatedAfter / params.SDFBlockNum * 100.0f);

    // Debug: Check if we're hitting memory limits
    if (heapCounterAfter == 0 || heapCounterAfter > initialHeapCounter) {
        printf("  WARNING: Heap counter invalid! (counter=%u, initial=%u, max=%d)\n", 
               heapCounterAfter, initialHeapCounter, params.SDFBlockNum);
    }
    if (totalAllocatedAfter >= params.SDFBlockNum) {
        printf("  WARNING: Reached maximum block limit! (%u >= %d)\n", totalAllocatedAfter, params.SDFBlockNum);
    }

    printGpuMemoryUsage("After Method1 allocation");
    
    return kernelElapsedMs;
}

/**
 * Method 2: Allocate blocks using normal direction for inside surface
 */
extern "C" float allocBlocksFromDepthMapMethod2CUDA(
    CUDAHashRef & hashData,
    const Params & params,
    const float3 * depthmap,
    const float3 * normalmap,
    int width,
    int height,
    float truncationDistance,
    float3 cameraPos,
    float* cameraTransform
) {
    // printf("allocBlocksFromDepthMapMethod2CUDA: Starting normal direction allocation...\n");
    // printf("  Depth map size: %dx%d\n", width, height);
    // printf("  Truncation distance: %f\n", truncationDistance);

    // Set up grid and block dimensions for depthmap pixels
    const unsigned int threadsPerBlock = 256;
    const unsigned int totalPixels = width * height;
    const dim3 gridSize((totalPixels + threadsPerBlock - 1) / threadsPerBlock, 1);
    const dim3 blockSize(threadsPerBlock, 1);

    // printf("  Grid size: %d, Block size: %d\n", gridSize.x, blockSize.x);

    printGpuMemoryUsage("Before Method2 allocation");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    // Launch Method 2 allocation kernel
    allocBlocksFromDepthMapMethod2Kernel<<<gridSize, blockSize>>>(
        hashData.d_hashTable,
        hashData.d_heap,
        hashData.d_heapCounter,
        hashData.d_hashBucketMutex,
        hashData.d_blockParentUV,
        hashData.d_blockAllocationMethod,
        depthmap,
        normalmap,
        width,
        height,
        truncationDistance,
        params.SDFBlockNum,
        params.hashSlotNum,       // numBuckets
        params.slotSize,          // bucketSize
        params.totalHashSize,     // totalHashSize
        params.voxelSize,                    // voxelSize (temporary)
        cameraPos,
        cameraTransform
    );

    cudaEventRecord(stopEvent);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("CUDA Method 2 allocation kernel launch error: %s\n", cudaGetErrorString(launchErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    cudaError_t eventSyncErr = cudaEventSynchronize(stopEvent);
    if (eventSyncErr != cudaSuccess) {
        printf("CUDA Method 2 allocation event sync error: %s\n", cudaGetErrorString(eventSyncErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    float kernelElapsedMs = computeKernelElapsedMs(startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Method 2 allocation kernel error: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }

    // printf("allocBlocksFromDepthMapMethod2CUDA: Normal direction allocation completed! (%.3f ms)\n", kernelElapsedMs);
    printGpuMemoryUsage("After Method2 allocation");
    
    return kernelElapsedMs;
}

/**
 * Legacy function for backward compatibility (calls Method 1)
 */
extern "C" float allocBlocksFromDepthMapCUDA(
    CUDAHashRef & hashData,
    const Params & params,
    const float3 * depthmap,
    const float3 * normalmap,
    int width,
    int height,
    float truncationDistance,
    float3 cameraPos,
    float* cameraTransform
) {

#ifndef USE_NORMAL_DIR_ALLOC

    // Method 1: Camera-direction allocation
    return allocBlocksFromDepthMapMethod1CUDA(hashData, params, depthmap, width, height,
        truncationDistance, cameraPos, cameraTransform);



#else

    // Method 2: Normal-direction allocation (if normal map available)
    if (normalmap != nullptr) {
        return allocBlocksFromDepthMapMethod2CUDA(
            hashData,
            params,
            depthmap,
            normalmap,
            width,
            height,
            truncationDistance,
            cameraPos,
            cameraTransform
        );
    }
    else {
        printf("allocBlocksFromDepthMapCUDA: Skipping Method 2 (normalmap is nullptr)\n");
        return 0.0f;
    }



#endif // !#define USE_NORMAL_DIR_ALLOC

}

/**
 * Integrate depth map into allocated blocks (improved version)
 */
extern "C" float integrateDepthMapIntoBlocksCUDA(
    CUDAHashRef & hashData,
    const Params & params,
    const float3 * depthmap,
    const uchar3 * colormap,
    const float3 * normalmap,
    int width,
    int height,
    float truncationDistance,
    float3 cameraPos,
    float* cameraTransform,
    float fx, float fy, float cx, float cy
) {
    // printf("integrateDepthMapIntoBlocksCUDA: Starting improved integration...\n");
    // printf("  Depth map size: %dx%d\n", width, height);
    // printf("  Truncation distance: %f\n", truncationDistance);

    // Get number of active blocks from compactification
    int numActiveBlocks = 0;
    cudaMemcpy(&numActiveBlocks, hashData.d_hashCompactifiedCounter,
        sizeof(int), cudaMemcpyDeviceToHost);

    if (numActiveBlocks == 0) {
        printf("No active blocks found for integration!\n");
        return 0.0f;
    }

    // Set up grid and block dimensions (like expert code)
    // Each CUDA block processes one SDF block, with SDF_BLOCK_SIZE^3 threads
    const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    dim3 gridSize(numActiveBlocks, 1);
    dim3 blockSize(threadsPerBlock, 1);

    // printf("  Active blocks: %d\n", numActiveBlocks);
    // printf("  Grid size: %d, Block size: %d (%d threads per block)\n", gridSize.x, blockSize.x, threadsPerBlock);

    printGpuMemoryUsage("Before integrateDepthMapIntoBlocks kernel");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent);

    // Launch integration kernel (like expert code)
    integrateDepthMapIntoBlocksKernel<<<gridSize, blockSize>>>(
        hashData.d_CompactifiedHashTable,
        hashData.d_SDFBlocks,
        hashData.d_blockParentUV,
        hashData.d_blockAllocationMethod,
        depthmap,
        colormap,
        normalmap,
        width,
        height,
        truncationDistance,
        numActiveBlocks,
        params.voxelSize,  // Use fixed voxel size for now
        cameraPos,
        cameraTransform,
        fx, fy, cx, cy
    );

    cudaEventRecord(stopEvent);

    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        printf("CUDA integration kernel launch error: %s\n", cudaGetErrorString(launchErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    cudaError_t eventSyncErr = cudaEventSynchronize(stopEvent);
    if (eventSyncErr != cudaSuccess) {
        printf("CUDA integration kernel event sync error: %s\n", cudaGetErrorString(eventSyncErr));
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return 0.0f;
    }

    float kernelElapsedMs = computeKernelElapsedMs(startEvent, stopEvent);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    // Check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA integration kernel error: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }

    // printf("integrateDepthMapIntoBlocksCUDA: Integration completed successfully! (%.3f ms)\n", kernelElapsedMs);
    // printf("  Integrated SDF data into %d active blocks\n", numActiveBlocks);
    // printf("  Total voxels updated: %d (%d voxels per block)\n", numActiveBlocks * (int)threadsPerBlock, threadsPerBlock);
    printGpuMemoryUsage("After integrateDepthMapIntoBlocks kernel");
    
    return kernelElapsedMs;
}



extern "C" void resetHasMutexCUDA(CUDAHashRef & hashData, const Params & hashParams) {

    const dim3 gridSize((hashParams.hashSlotNum + (T_PER_BLOCK * T_PER_BLOCK) - 1) / (T_PER_BLOCK * T_PER_BLOCK), 1);
    const dim3 blockSize((T_PER_BLOCK * T_PER_BLOCK), 1);

    resetHashMutexKernel << <gridSize, blockSize >> > (hashData, hashParams.hashSlotNum);



}