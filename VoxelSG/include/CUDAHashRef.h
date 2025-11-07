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
struct VoxelData {
    uchar3 color;          // RGB color
    uchar weight;          // Integration confidence/weight
    float prev_sdf;        // Previous SDF distance
    float sdf;             // Current SDF distance
    uchar prev_weight;     // Previous weight
    uchar isUpdated;       // Update flag
    uchar isZeroCrossing;  // Surface flag (zero crossing)
    uchar patch_bit;       // Patch flag

    // Directional TSDF fields
    uchar dir_id0;         // Primary direction ID
    uchar dir_w0;          // Primary direction weight
    float dtsdf0;          // Primary directional distance
    uchar dir_id1;         // Secondary direction ID
    uchar dir_w1;          // Secondary direction weight
    float dtsdf1;          // Secondary directional distance
    
    // Parent tracking for Method 2 (normal direction allocation)
    int parentPixelX;      // Parent depth pixel X coordinate
    int parentPixelY;      // Parent depth pixel Y coordinate
    uchar allocationMethod; // 0: Method 1 (camera direction), 1: Method 2 (normal direction)
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
