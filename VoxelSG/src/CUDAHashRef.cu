#include <cuda_runtime.h>
#include <cstdio>
#include "../include/CUDAHashRef.h"

// Utility to check CUDA errors in debug builds
static inline void checkCuda(cudaError_t result) {
#if !defined(NDEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    }
#else
    (void)result;
#endif
}

static void logDeviceAlloc(const char* name, size_t bytes) {
    const double toGB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    printf("[GPU ALLOC] %-24s : %zu bytes (%.3f GB)\n",
        name,
        bytes,
        bytes * toGB);
}

CUDAHashRef::CUDAHashRef()
    : d_hashTable(nullptr),
      d_CompactifiedHashTable(nullptr),
      d_hashCompactifiedCounter(nullptr),
      d_SDFBlocks(nullptr),
      d_hashBucketMutex(nullptr),
      d_heap(nullptr),
      d_heapCounter(nullptr),
      d_voxelZeroCross(nullptr),
      d_voxelZeroCrossCounter(nullptr),
      d_hashDecision(nullptr),
      d_hashDecisionPrefix(nullptr) {}

// NOTE: Ensure the header declares this as: `void HashDataAllocation(const Params);`
void CUDAHashRef::HashDataAllocation(const Params params) {
    // The actual sizes depend on your hashing scheme. Here we allocate
    // conservative buffers based on available Params fields.
    size_t totalAllocatedBytes = 0;

    // Hash table storing slots
    size_t hash_table_bytes = static_cast<size_t>(params.totalHashSize) * sizeof(HashSlot);
    if (hash_table_bytes > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_hashTable), hash_table_bytes));
        checkCuda(cudaMemset(d_hashTable, 0, hash_table_bytes));
        logDeviceAlloc("HashTable", hash_table_bytes);
        totalAllocatedBytes += hash_table_bytes;
    }

    // Compactified (active) hash entries — size can vary; mirror totalHashSize for now
    size_t compact_table_bytes = static_cast<size_t>(params.totalHashSize) * sizeof(HashSlot);
    if (compact_table_bytes > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_CompactifiedHashTable), compact_table_bytes));
        checkCuda(cudaMemset(d_CompactifiedHashTable, 0, compact_table_bytes));
        logDeviceAlloc("CompactifiedHashTable", compact_table_bytes);
        totalAllocatedBytes += compact_table_bytes;
    }

    // Counter for compactified entries
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_hashCompactifiedCounter), sizeof(int)));
    checkCuda(cudaMemset(d_hashCompactifiedCounter, 0, sizeof(int)));
    logDeviceAlloc("HashCompactifiedCounter", sizeof(int));
    totalAllocatedBytes += sizeof(int);

    // Voxel blocks — using SDFBlockSize as an upper bound for demonstration
    size_t voxel_block_count = static_cast<size_t>(params.totalBlockSize);
    if (voxel_block_count > 0) {
        size_t voxel_bytes = voxel_block_count * sizeof(VoxelData);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_SDFBlocks), voxel_bytes));
        checkCuda(cudaMemset(d_SDFBlocks, 0, voxel_bytes));
        logDeviceAlloc("SDFBlocks", voxel_bytes);
        totalAllocatedBytes += voxel_bytes;
    }

    // Mutex array for hash buckets — assume one per bucket
    size_t bucket_count = static_cast<size_t>(params.hashSlotNum);
    if (bucket_count > 0) {
        size_t mutex_bytes = bucket_count * sizeof(int);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_hashBucketMutex), mutex_bytes));
        checkCuda(cudaMemset(d_hashBucketMutex, 0, mutex_bytes));
        logDeviceAlloc("HashBucketMutex", mutex_bytes);
        totalAllocatedBytes += mutex_bytes;
    }

    // Heap and heap counter — sizes are project-specific. Allocate minimal placeholders.
    // Adjust to your allocator design.
    const size_t default_heap_capacity = static_cast<size_t>(params.SDFBlockNum);
    if (default_heap_capacity > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_heap), default_heap_capacity * sizeof(unsigned int)));
        checkCuda(cudaMemset(d_heap, 0, default_heap_capacity * sizeof(unsigned int)));
        size_t heapBytes = default_heap_capacity * sizeof(unsigned int);
        logDeviceAlloc("Heap", heapBytes);
        totalAllocatedBytes += heapBytes;
    }
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_heapCounter), sizeof(unsigned int)));
    checkCuda(cudaMemset(d_heapCounter, 0, sizeof(unsigned int)));
    logDeviceAlloc("HeapCounter", sizeof(unsigned int));
    totalAllocatedBytes += sizeof(unsigned int);

    // Zero crossing buffers — allocate lazily with 0 length for now
    d_voxelZeroCross = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_voxelZeroCrossCounter), sizeof(int)));
    checkCuda(cudaMemset(d_voxelZeroCrossCounter, 0, sizeof(int)));
    logDeviceAlloc("VoxelZeroCrossCounter", sizeof(int));
    totalAllocatedBytes += sizeof(int);

    // Decision arrays for compactification
    size_t decision_bytes = static_cast<size_t>(params.totalHashSize) * sizeof(int);
    if (decision_bytes > 0) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_hashDecision), decision_bytes));
        checkCuda(cudaMemset(d_hashDecision, 0, decision_bytes));
        logDeviceAlloc("HashDecision", decision_bytes);
        totalAllocatedBytes += decision_bytes;
        
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_hashDecisionPrefix), decision_bytes));
        checkCuda(cudaMemset(d_hashDecisionPrefix, 0, decision_bytes));
        logDeviceAlloc("HashDecisionPrefix", decision_bytes);
        totalAllocatedBytes += decision_bytes;
    }

    const double toGB = 1.0 / (1024.0 * 1024.0 * 1024.0);
    printf("[GPU ALLOC] %-24s : %zu bytes (%.3f GB)\n",
        "Total (HashDataAllocation)",
        totalAllocatedBytes,
        totalAllocatedBytes * toGB);

    size_t freeBytes = 0;
    size_t totalBytes = 0;
    cudaError_t infoErr = cudaMemGetInfo(&freeBytes, &totalBytes);
    if (infoErr == cudaSuccess) {
        double usedGB = (totalBytes - freeBytes) * toGB;
        printf("[GPU MEM ] Device total %.3f GB, used %.3f GB, free %.3f GB\n",
            totalBytes * toGB,
            usedGB,
            freeBytes * toGB);
    } else {
        printf("[GPU MEM ] cudaMemGetInfo failed (%s)\n", cudaGetErrorString(infoErr));
    }
}


