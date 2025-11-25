#pragma once

#include "CUDAHashData.h"
#include <cuda_runtime.h>
#include "utils.h"

/**
 * HashSlot: Spatial hash entry structure
 * Stores 3D position and pointer to voxel block
 */
struct HashSlot {
	int3 pos;      // 3D block position
	int ptr;       // Pointer to voxel block
	uint offset;   // Offset in heap
};

/**
 * VoxelData: Per-voxel data structure
 * Stores color, SDF value, weight, and directional TSDF data
 */

//__align__(16)
struct VoxelData {
    uchar3 color;          // RGB color
    uchar weight;          // Integration confidence/weight
    // float prev_sdf;        // Previous SDF distance (unused)
    float sdf;             // Current SDF distance
    // uchar prev_weight;     // Previous weight (unused)
    uchar isUpdated;       // Update flag (unused)
    uchar isZeroCrossing;  // Surface flag (unused)
    // uchar patch_bit;       // Patch flag (unused)

    // Directional TSDF fields (unused)
    // uchar dir_id0;
    // uchar dir_w0;
    // float dtsdf0;
    // uchar dir_id1;
    // uchar dir_w1;
    // float dtsdf1;
    
    // Parent tracking for Method 2 (unused)
    // int parentPixelX;
    // int parentPixelY;
    // uchar allocationMethod;
};

/**
 * CUDAHashRef: GPU-side hash data structure
 * Manages all GPU memory for voxel hashing
 */
class CUDAHashRef {
public:
    CUDAHashRef();
    
    void HashDataAllocation(const Params params);

    // Hash table structures
    HashSlot* d_hashTable;                    // Main spatial hash table
    HashSlot* d_CompactifiedHashTable;        // Compacted active blocks
    int* d_hashCompactifiedCounter;           // Active block counter

    // Voxel data
    VoxelData* d_SDFBlocks;                   // Voxel data blocks
    int2* d_blockParentUV;                    // Per-block parent pixel (u,v)
    unsigned char* d_blockAllocationMethod;   // Per-block allocation method (0=cam,1=normal)
    int* d_hashBucketMutex;                   // Mutex for hash buckets

    // Memory management
    unsigned int* d_heap;                     // Free block heap
    unsigned int* d_heapCounter;              // Heap counter

    // Surface extraction
    int3* d_voxelZeroCross;                   // Surface voxel positions
    int* d_voxelZeroCrossCounter;             // Surface voxel counter

    // Compactification helpers
    int* d_hashDecision;                      // Decision array for compactification
    int* d_hashDecisionPrefix;                // Prefix sum for compactification
};
