#include "../include/VoxelStreamingManager.h"

#define _USE_MATH_DEFINES  // For M_PI on Windows
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "../include/CUDAHashRef.h"
#include "../include/CUDAHashData.h"

// Define to enable saving active blocks to xyz files after streaming in
// Set to 0 to disable xyz file saving (saves disk I/O)
#define SAVE_STREAM_IN_ACTIVE_BLOCKS_TO_XYZ 1

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

extern "C" void streamInHashAllocCUDA(
    const Params& params,
    const CUDAHashRef& hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks,
    StreamInDebugInfo * d_debugInfo
);

extern "C" void streamInHashInitCUDA(
    const Params& params,
    const CUDAHashRef& hashData,
    unsigned int numSDFBlockDescs,
    unsigned int heapCountPrev,
    const SDFBlockInfo* d_SDFBlockDescs,
    const SDFBlock* d_SDFBlocks
);




DWORD WINAPI AuxiliaryStreamingFunc(LPVOID lParam) {
    VoxelStreamingManager* chunkStreamingManager = (VoxelStreamingManager*)lParam;
    
    printf("[AUXILIARY THREAD] Started!\n");
    fflush(stdout);
    
    // Auxiliary Thread: CPU 작업 (stream out/in의 CPU 부분 처리)
    while (true) {
        printf("[AUXILIARY THREAD] Loop iteration - calling streamOutCopyToChunkGrid\n");
        fflush(stdout);
        
        // Stream Out: GPU에서 찾은 블록들을 CPU chunk grid에 저장
        chunkStreamingManager->streamOutCopyToChunkGrid(true);
        
        printf("[AUXILIARY THREAD] After streamOutCopyToChunkGrid, calling streamInCopyToGPUBuffer\n");
        fflush(stdout);
        
        // Stream In: CPU chunk grid에서 GPU로 복사할 블록 찾기
        // (main thread에서 camera position과 radius를 설정해야 함)
        // 여기서는 posCamera_와 radius_를 사용
        chunkStreamingManager->streamInCopyToGPUBuffer(
            chunkStreamingManager->posCamera_,
            chunkStreamingManager->radius_,
            true,  // useParts
            true   // multiThreaded
        );
        
        // Thread 종료 체크 (static member variable)
        if (chunkStreamingManager->s_terminateThread) {
            printf("[AUXILIARY THREAD] Terminating...\n");
            fflush(stdout);
            return 0;
        }
    }
    
    return 0;
}



VoxelStreamingManager::VoxelStreamingManager(VoxelScene& sceneHashRef, Params& params, const vec3f& voxelExtends, 
    const vec3i& gridDimensions, const vec3i& minGridPos,
    unsigned int initialChunkListSize, bool streamingEnabled) {

    sceneHashRef_ = &sceneHashRef;
    params_ = params;

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
    m_frameNumber = 0;  // Initialize frame number
    m_outputDirectory = "";  // Initialize output directory

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

    if (streamingEnabled) {
        initializeCriticalSection();
        startMultiThreading();
    }
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
    // Deprecated: Use startMultiThreading() instead
    startMultiThreading();
}

void VoxelStreamingManager::startMultiThreading() {
    if (!s_terminateThread) {
        // Already running
        printf("[STREAMING] Thread already running\n");
        return;
    }
    
    s_terminateThread = false;
    
    printf("[STREAMING] Creating thread... (s_terminateThread = %s)\n", s_terminateThread ? "true" : "false");
    fflush(stdout);
    
    hStreamingThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)AuxiliaryStreamingFunc, (LPVOID)this, 0, &dwStreamingThreadID);
    
    if (hStreamingThread == NULL) {
        printf("[ERROR] Streaming thread could not be created (error: %lu)\n", GetLastError());
    } else {
        printf("[STREAMING] Multi-threading started (Thread ID: %lu, Handle: %p)\n", dwStreamingThreadID, hStreamingThread);
        fflush(stdout);
        
        // Give the thread a moment to start (optional, for debugging)
        Sleep(10);  // 10ms delay to let thread start
        printf("[STREAMING] After thread creation delay\n");
        fflush(stdout);
    }
}

void VoxelStreamingManager::stopMultiThreading() {
    if (s_terminateThread) {
        // Already stopped
        return;
    }
    
    s_terminateThread = true;
    
    // Signal all events to wake up waiting threads
    SetEvent(hEventOutProduce);
    SetEvent(hEventOutConsume);
    SetEvent(hEventInProduce);
    SetEvent(hEventInConsume);
    
    // Wait for thread to finish
    if (hStreamingThread != NULL) {
        WaitForSingleObject(hStreamingThread, INFINITE);
        CloseHandle(hStreamingThread);
        hStreamingThread = NULL;
    }
    
    deleteCriticalSection();
    
    printf("[STREAMING] Multi-threading stopped\n");
    fflush(stdout);
}

void VoxelStreamingManager::initializeCriticalSection() {
    hMutexOut = CreateMutex(NULL, FALSE, NULL);
    hEventOutProduce = CreateEvent(NULL, FALSE, TRUE, NULL);   // Initially signaled (producer can start)
    hEventOutConsume = CreateEvent(NULL, FALSE, FALSE, NULL);  // Initially non-signaled
    
    hMutexIn = CreateMutex(NULL, FALSE, NULL);
    hEventInProduce = CreateEvent(NULL, FALSE, TRUE, NULL);   // Initially signaled
    hEventInConsume = CreateEvent(NULL, FALSE, FALSE, NULL);  // Initially non-signaled
    
    hMutexSetTransform = CreateMutex(NULL, FALSE, NULL);
    hEventSetTransformProduce = CreateEvent(NULL, FALSE, TRUE, NULL);
    hEventSetTransformConsume = CreateEvent(NULL, FALSE, FALSE, NULL);
}

void VoxelStreamingManager::deleteCriticalSection() {
    if (hMutexOut != NULL) CloseHandle(hMutexOut);
    if (hEventOutProduce != NULL) CloseHandle(hEventOutProduce);
    if (hEventOutConsume != NULL) CloseHandle(hEventOutConsume);
    
    if (hMutexIn != NULL) CloseHandle(hMutexIn);
    if (hEventInProduce != NULL) CloseHandle(hEventInProduce);
    if (hEventInConsume != NULL) CloseHandle(hEventInConsume);
    
    if (hMutexSetTransform != NULL) CloseHandle(hMutexSetTransform);
    if (hEventSetTransformProduce != NULL) CloseHandle(hEventSetTransformProduce);
    if (hEventSetTransformConsume != NULL) CloseHandle(hEventSetTransformConsume);
}

  
void VoxelStreamingManager::streamOutFindBlocksOnGPU(const vec3f& sphereCenter, float radius, bool useParts, bool multiThreaded) { // Main Thread 1 : Find Block for Stream Out

    printf("[STREAM OUT FIND] Main thread 1 : Waiting\n");
    fflush(stdout);

    if (multiThreaded) {
        WaitForSingleObject(hEventOutProduce, INFINITE);
        WaitForSingleObject(hMutexOut, INFINITE);
    }

    printf("[STREAM OUT FIND] Main thread 1 : Start\n");
    fflush(stdout);


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
        // Convert vec3f to float3 (sphereCenter: typically 100m ahead of camera)
        float3 sphereCenterFloat3 = make_float3(sphereCenter.x, sphereCenter.y, sphereCenter.z);
        
        // Launch OPTIMIZED kernel (scans only active blocks, not entire hash!)
        streamOutFindBlocksCUDA_OPTIMIZED(
            sceneHashRef_->getHashData(),
            sceneHashRef_->getParams(),
            numActiveBlocks,
            radius,
            sphereCenterFloat3,  // Use sphere center (100m ahead of camera)
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
            
            // Save block world positions to .xyz file (Main thread 1)
            // Create streaming_debugging directory if it doesn't exist
            std::string debugDir = m_outputDirectory + "/streaming_debugging";
            if (!m_outputDirectory.empty()) {
                std::filesystem::create_directories(debugDir);
            }
            
            // Copy block info from GPU to CPU
            std::vector<SDFBlockInfo> h_blockInfos(nSDFBlockDescs);
            cudaMemcpy(h_blockInfos.data(), d_SDFBlockInfoOutput, 
                      sizeof(SDFBlockInfo) * nSDFBlockDescs, 
                      cudaMemcpyDeviceToHost);
            
            std::stringstream filename;
            if (!m_outputDirectory.empty()) {
                filename << debugDir << "/stream_out_block_positions_frame_" << m_frameNumber << ".xyz";
            } else {
                filename << "stream_out_block_positions_frame_" << m_frameNumber << ".xyz";
            }
            std::ofstream xyzFile(filename.str());
            
            for (unsigned int i = 0; i < nSDFBlockDescs; i++) {
                vec3i pos = h_blockInfos[i].pos;
                vec3f blockWorldPos = vec3f(
                    pos.x * SDF_BLOCK_SIZE * params_.voxelSize,
                    pos.y * SDF_BLOCK_SIZE * params_.voxelSize,
                    pos.z * SDF_BLOCK_SIZE * params_.voxelSize
                );
                xyzFile << blockWorldPos.x << " " << blockWorldPos.y << " " << blockWorldPos.z << "\n";
            }
            
            xyzFile.close();
            printf("[STREAM OUT FIND] Saved %u block positions to %s\n", nSDFBlockDescs, filename.str().c_str());
        }
    } else {
        printf("[STREAM OUT] No active blocks to scan\n");
    }

    // Task 3 :  Copy SDFBlocks to output buffer
    unsigned int nSDFBlockDescs = getSDFBlockCounter();
    
    if (nSDFBlockDescs > 0) {
        // Get heap counter before stream out for logging
        unsigned int heapCounterBefore = 0;
        cudaMemcpy(&heapCounterBefore, sceneHashRef_->getHashData().d_heapCounter,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int initialHeapCounter = sceneHashRef_->getParams().SDFBlockNum - 1;
        unsigned int totalAllocatedBefore = initialHeapCounter - heapCounterBefore;
        
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
        
        // Get heap counter after stream out for logging
        unsigned int heapCounterAfter = 0;
        cudaMemcpy(&heapCounterAfter, sceneHashRef_->getHashData().d_heapCounter,
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        // Calculate totalAllocatedAfter with overflow protection
        // If heapCounterAfter > initialHeapCounter, it means we have more free blocks than initially,
        // which shouldn't happen in normal operation. Clamp to 0.
        int totalAllocatedAfterSigned = static_cast<int>(initialHeapCounter) - static_cast<int>(heapCounterAfter);
        unsigned int totalAllocatedAfter = (totalAllocatedAfterSigned > 0) ? static_cast<unsigned int>(totalAllocatedAfterSigned) : 0;
        
        if (heapCounterAfter > initialHeapCounter) {
            std::cerr << "[WARNING] STREAM_OUT: HeapCounter (" << heapCounterAfter 
                      << ") > InitialCounter (" << initialHeapCounter 
                      << "). This indicates a potential heap counter corruption!" << std::endl;
        }
        
        // Log stream out to file
        if (!m_outputDirectory.empty()) {
            std::filesystem::path logPath = std::filesystem::path(m_outputDirectory) / "heap_block_log.txt";
            std::ofstream logFile(logPath, std::ios::app);
            if (logFile.is_open()) {
                unsigned int freeBlocks = heapCounterAfter + 1;
                logFile << m_frameNumber << "\tSTREAM_OUT\t" 
                        << heapCounterAfter << "\t" << initialHeapCounter << "\t"
                        << totalAllocatedAfter << "\t0\t" << nSDFBlockDescs << "\t" << freeBlocks << "\n";
                logFile.close();
            }
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
        

        //printf("[STREAM OUT] Repacking compactified array to remove gaps (scanned %d blocks, not %u total slots)...\n", 
        //       numActiveBlocks, sceneHashRef_->getParams().totalHashSize);
        //float repackTime = repackCompactifiedCUDA(sceneHashRef_->getHashData(), numActiveBlocks);
        //printf("[STREAM OUT] Compactified array repacked in %.3f ms\n", repackTime);


    } else {
        nStreamdOutBlocks_ = 0;
    }

    printf("[STREAM OUT FIND] Main thread 1 : End\n");
    fflush(stdout);

    if (multiThreaded) {
        SetEvent(hEventOutConsume);
        ReleaseMutex(hMutexOut);
    }

}


void VoxelStreamingManager::streamOutCopyToChunkGrid(bool multiThreaded) {        // Auxiliary: CPU

    printf("[STREAM OUT COPY] Auxiliary thread 1 : Waiting\n");
    fflush(stdout);

    if (multiThreaded) {
        WaitForSingleObject(hEventOutConsume, INFINITE);
        WaitForSingleObject(hMutexOut, INFINITE);
    }

    printf("[STREAM OUT COPY] Auxiliary thread 1 : Start\n");
    fflush(stdout);

    updateChunkGrid((int*)h_SDFBlockInfoOutput, (int*)h_SDFBlockOutput, nStreamdOutBlocks_);

    printf("[STREAM OUT COPY] Auxiliary thread 1 : End\n");
    fflush(stdout);

    if (multiThreaded) {
        SetEvent(hEventOutProduce);
        ReleaseMutex(hMutexOut);
    }
}


void VoxelStreamingManager::streamInCopyToGPUBuffer(const vec3f& sphereCenter, float radius, bool useParts, bool multiThreaded) {         // Auxiliary: CPU→GPU Copy

    // Searching active voxel in Chunk grid --> 

    printf("[STREAM IN COPY] Auxiliary thread 2 : Waiting\n");
    fflush(stdout);

    if (multiThreaded) {
        WaitForSingleObject(hEventInProduce, INFINITE);
        WaitForSingleObject(hMutexIn, INFINITE);
        if (s_terminateThread) {
            return;	//avoid duplicate insertions when stop multi-threading is called
        }
    }

    printf("[STREAM IN COPY] Auxiliary thread 2 : Start\n");
    fflush(stdout);

    unsigned int nSDFBlockDescs = streamInFindChunk(sphereCenter, radius, useParts);

    nStreamdInBlocks_ = nSDFBlockDescs;

    printf("[STREAM IN COPY] Moving Chunk block num : %d\n", nStreamdInBlocks_);
    fflush(stdout);

    printf("[STREAM IN COPY] Auxiliary thread 2 : End\n");
    fflush(stdout);


    if (multiThreaded) {
        SetEvent(hEventInConsume);
        ReleaseMutex(hMutexIn);
    }
}


void VoxelStreamingManager::streamInInsertToHashTable(bool multiThreaded) {       // Main: Hash table에 삽입

    printf("[STREAM IN INSERT] Main thread 2 : Waiting\n");
    fflush(stdout);

    if (multiThreaded) {
        WaitForSingleObject(hEventInConsume, INFINITE);
        WaitForSingleObject(hMutexIn, INFINITE);
    }

    printf("[STREAM IN INSERT] Main thread 2 : Start\n");
    fflush(stdout);

    bool debug_flag = false;


    if (nStreamdInBlocks_ != 0 && debug_flag) {
        //-------------------------------------------------------
        // Pass 1: Alloc memory for chunks (hash table에 블록 할당)
        //-------------------------------------------------------
        
        // Get current heap counter (free block count)
        unsigned int heapCountPrev = 0;
        CUDAHashRef& hashData = sceneHashRef_->getHashData();
        const Params& params = sceneHashRef_->getParams();
        
        cudaError_t err = cudaMemcpy(&heapCountPrev, hashData.d_heapCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("[ERROR] Failed to read heap counter: %s\n", cudaGetErrorString(err));
            if (multiThreaded) {
                SetEvent(hEventInProduce);
                ReleaseMutex(hMutexIn);
            }
            return;
        }
        
        printf("[STREAM IN INSERT] heapCountPrev=%u, nStreamdInBlocks_=%u, SDFBlockNum=%d\n", 
               heapCountPrev, nStreamdInBlocks_, params.SDFBlockNum);
        fflush(stdout);
        
        // Check if we have enough free blocks
        if (heapCountPrev < nStreamdInBlocks_) {
            printf("[ERROR] Not enough free blocks in heap: have %u, need %u\n", 
                   heapCountPrev, nStreamdInBlocks_);
            if (multiThreaded) {
                SetEvent(hEventInProduce);
                ReleaseMutex(hMutexIn);
            }
            return;
        }
        
        if (heapCountPrev > (unsigned int)params.SDFBlockNum) {
            printf("[ERROR] heapCountPrev %u > SDFBlockNum %d (heap corruption?)\n", 
                   heapCountPrev, params.SDFBlockNum);
            if (multiThreaded) {
                SetEvent(hEventInProduce);
                ReleaseMutex(hMutexIn);
            }
            return;
        }
        
        printf("[STREAM IN INSERT] Calling streamInHashAllocCUDA...\n");
        fflush(stdout);
        
        // Allocate debug info array for logging
        StreamInDebugInfo* d_debugInfo = nullptr;
        StreamInDebugInfo* h_debugInfo = nullptr;
        if (nStreamdInBlocks_ > 0) {
            size_t debugInfoSize = sizeof(StreamInDebugInfo) * nStreamdInBlocks_;
            cudaMalloc(&d_debugInfo, debugInfoSize);
            cudaMemset(d_debugInfo, 0, debugInfoSize);
            h_debugInfo = new StreamInDebugInfo[nStreamdInBlocks_];
        }
        
        // Allocate blocks in hash table
        streamInHashAllocCUDA(
            params,
            hashData,
            nStreamdInBlocks_,
            heapCountPrev,
            d_SDFBlockInfoInput,
            d_SDFBlockInput,
            d_debugInfo  // Pass debug info array
        );
        
        // Check for errors after Pass 1
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] streamInHashAllocCUDA failed: %s\n", cudaGetErrorString(err));
            if (d_debugInfo) cudaFree(d_debugInfo);
            if (h_debugInfo) delete[] h_debugInfo;
            if (multiThreaded) {
                SetEvent(hEventInProduce);
                ReleaseMutex(hMutexIn);
            }
            return;
        }
        
        // Copy debug info from device to host
        unsigned int actuallyInsertedBlocks = 0;  // Count blocks that were actually inserted (not already existing)
        if (d_debugInfo && h_debugInfo) {
            cudaMemcpy(h_debugInfo, d_debugInfo, sizeof(StreamInDebugInfo) * nStreamdInBlocks_, 
                      cudaMemcpyDeviceToHost);
            
            // Count actually inserted blocks (NEW_SLOT or NEW_OPEN, excluding EXISTS and FAIL)
            for (unsigned int i = 0; i < nStreamdInBlocks_; i++) {
                if (h_debugInfo[i].result == INSERT_RESULT_NEW_SLOT || 
                    h_debugInfo[i].result == INSERT_RESULT_NEW_OPEN) {
                    actuallyInsertedBlocks++;
                }
            }
            
            // Write debug info to file
            if (!m_outputDirectory.empty()) {
                std::filesystem::path logPath = std::filesystem::path(m_outputDirectory) / "stream_in_insert_log.txt";
                std::ofstream logFile(logPath, std::ios::app);
                if (logFile.is_open()) {
                    // Write header for first entry
                    static bool headerWritten = false;
                    if (!headerWritten) {
                        logFile << "Frame\tBlockID\tBlockCoord_X\tBlockCoord_Y\tBlockCoord_Z\t"
                               << "Result\tSlotIdx\tBucketId\n";
                        headerWritten = true;
                    }
                    
                    // Write debug info for each block
                    for (unsigned int i = 0; i < nStreamdInBlocks_; i++) {
                        const char* resultStr = "";
                        switch (h_debugInfo[i].result) {
                            case INSERT_RESULT_NEW_SLOT: resultStr = "NEW_SLOT"; break;
                            case INSERT_RESULT_EXISTS_PRIMARY: resultStr = "EXISTS_PRIMARY"; break;
                            case INSERT_RESULT_NEW_OPEN: resultStr = "NEW_OPEN"; break;
                            case INSERT_RESULT_EXISTS_OPEN: resultStr = "EXISTS_OPEN"; break;
                            case INSERT_RESULT_FAIL: resultStr = "FAIL"; break;
                            default: resultStr = "UNKNOWN"; break;
                        }
                        
                        logFile << m_frameNumber << "\t" << i << "\t"
                               << h_debugInfo[i].blockCoord.x << "\t"
                               << h_debugInfo[i].blockCoord.y << "\t"
                               << h_debugInfo[i].blockCoord.z << "\t"
                               << resultStr << "\t"
                               << h_debugInfo[i].slotIdx << "\t"
                               << h_debugInfo[i].bucketId << "\n";
                    }
                    logFile.close();
                }
            }
            
            // Free debug info memory
            cudaFree(d_debugInfo);
            delete[] h_debugInfo;
        } else {
            // If debug info is not available, assume all blocks were inserted (fallback)
            actuallyInsertedBlocks = nStreamdInBlocks_;
        }
        
        printf("[STREAM IN INSERT] Pass 1 completed, calling streamInHashInitCUDA...\n");
        fflush(stdout);

        //-------------------------------------------------------
        // Pass 2: Initialize corresponding SDFBlocks (데이터 복사)
        //-------------------------------------------------------

        if (false) {

            streamInHashInitCUDA(
                params,
                hashData,
                nStreamdInBlocks_,
                heapCountPrev,
                d_SDFBlockInfoInput,
                d_SDFBlockInput
            );

            // Check for errors after Pass 2
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[ERROR] streamInHashInitCUDA failed: %s\n", cudaGetErrorString(err));
                if (multiThreaded) {
                    SetEvent(hEventInProduce);
                    ReleaseMutex(hMutexIn);
                }
                return;
            }

            // Update heap counter (decrease only by number of blocks actually inserted, not already existing ones)
            // IMPORTANT: Blocks that already exist (EXISTS_PRIMARY, EXISTS_OPEN) don't use heap, so don't decrement counter
            unsigned int heapCountNew = heapCountPrev - actuallyInsertedBlocks;
            printf("[STREAM IN INSERT] Updating heap counter: %u -> %u (inserted: %u, total: %u, existing: %u)\n",
                heapCountPrev, heapCountNew, actuallyInsertedBlocks, nStreamdInBlocks_,
                nStreamdInBlocks_ - actuallyInsertedBlocks);
            fflush(stdout);

            err = cudaMemcpy(hashData.d_heapCounter, &heapCountNew, sizeof(unsigned int), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                printf("[ERROR] Failed to update heap counter: %s\n", cudaGetErrorString(err));
            }

            // Final error check
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                printf("[ERROR] streamInInsertToHashTable final sync failed: %s\n", cudaGetErrorString(err));
            }

            printf("[STREAM IN INSERT] Completed successfully\n");
            fflush(stdout);

        }
    }
    
#if SAVE_STREAM_IN_ACTIVE_BLOCKS_TO_XYZ
    // After streaming in, compactify and save all active blocks to xyz file
    if (nStreamdInBlocks_ > 0) {
        CUDAHashRef& hashData = sceneHashRef_->getHashData();
        const Params& params = sceneHashRef_->getParams();
        
        // Compactify hash entries to get active blocks
        compactifyHashCUDA(hashData, params);
        
        // Get active blocks count
        int numActiveBlocks = 0;
        cudaMemcpy(&numActiveBlocks, hashData.d_hashCompactifiedCounter,
                   sizeof(int), cudaMemcpyDeviceToHost);
        
        if (numActiveBlocks > 0 && !m_outputDirectory.empty()) {
            // Read current active blocks from compactified hash table
            std::vector<HashSlot> currentActiveBlocks(numActiveBlocks);
            cudaMemcpy(currentActiveBlocks.data(), hashData.d_CompactifiedHashTable,
                      sizeof(HashSlot) * numActiveBlocks, cudaMemcpyDeviceToHost);
            
            // Save all active blocks to xyz file
            std::filesystem::path debugDir = std::filesystem::path(m_outputDirectory) / "streaming_debugging";
            std::filesystem::create_directories(debugDir);
            
            std::stringstream filename;
            filename << debugDir.string() << "/stream_in_all_active_blocks_frame_" 
                     << m_frameNumber << ".xyz";
            
            std::ofstream xyzFile(filename.str());
            if (xyzFile.is_open()) {
                for (int i = 0; i < numActiveBlocks; i++) {
                    const HashSlot& slot = currentActiveBlocks[i];
                    if (slot.ptr != -1) {  // Valid block
                        // Convert block coordinates to world position
                        float blockExtent = static_cast<float>(SDF_BLOCK_SIZE) * params.voxelSize;
                        float worldX = slot.pos.x * blockExtent;
                        float worldY = slot.pos.y * blockExtent;
                        float worldZ = slot.pos.z * blockExtent;
                        
                        xyzFile << worldX << " " << worldY << " " << worldZ << "\n";
                    }
                }
                xyzFile.close();
                printf("[STREAM IN INSERT] Saved %d all active block positions after streaming in to %s\n", 
                       numActiveBlocks, filename.str().c_str());
            }
        }
    }
#endif // SAVE_STREAM_IN_ACTIVE_BLOCKS_TO_XYZ

    printf("[STREAM IN INSERT] Main thread 2 : End\n");
    fflush(stdout);

    if (multiThreaded) {
        SetEvent(hEventInProduce);
        ReleaseMutex(hMutexIn);
    }
}

void VoxelStreamingManager::updateChunkGrid(const int* blockInfos, const int* blocks, unsigned int nSDFBlocks) {
    // Convert int pointers to proper types
    const SDFBlockInfo* desc = reinterpret_cast<const SDFBlockInfo*>(blockInfos);
    const SDFBlock* block = reinterpret_cast<const SDFBlock*>(blocks);

    std::cout << "------------------------------\n";

    // Create streaming_debugging directory if it doesn't exist
    std::string debugDir = m_outputDirectory + "/streaming_debugging";
    if (!m_outputDirectory.empty()) {
        std::filesystem::create_directories(debugDir);
    }

    // Save world positions to .xyz file (Auxiliary thread 1)
    std::stringstream filename;
    if (!m_outputDirectory.empty()) {
        filename << debugDir << "/stream_out_world_positions_frame_" << m_frameNumber << ".xyz";
    } else {
        filename << "stream_out_world_positions_frame_" << m_frameNumber << ".xyz";
    }
    std::ofstream xyzFile(filename.str());

    for (unsigned int i = 0; i < nSDFBlocks; i++) {
        // Get block position (SDF block coordinates)
        vec3i pos = desc[i].pos;
        
        // Convert SDF block coordinates to world position
        // posWorld = pos * SDF_BLOCK_SIZE * voxelSize
        vec3f posWorld = vec3f(
            pos.x * SDF_BLOCK_SIZE * params_.voxelSize,
            pos.y * SDF_BLOCK_SIZE * params_.voxelSize,
            pos.z * SDF_BLOCK_SIZE * params_.voxelSize
        );

        //std::cout << i << " world : " << posWorld.x << " " << posWorld.y << " " << posWorld.z << "\n";
        
        // Write to .xyz file
        xyzFile << posWorld.x << " " << posWorld.y << " " << posWorld.z << "\n";
        
        // Convert world position to chunk coordinates
        vec3i chunk = worldToChunks(posWorld);

        if (!isValidChunk(chunk)) {
            printf("[WARNING] Chunk out of bounds\n");
            continue;
        }

        // std::cout << i << " chunk : " << chunk.x << " " << chunk.y << " " << chunk.z << "\n";


        unsigned int index = linearizeChunkPos(chunk);

        if (m_grid[index] == NULL) // Allocate memory for chunk
        {
            m_grid[index] = new chunkData(m_initialChunkListSize);
        }

        // Add element
        m_grid[index]->addSDFBlock(desc[i], block[i]);
        m_bitMask.setBit(index);
    }

    xyzFile.close();
    printf("[STREAM OUT COPY] Saved %u world positions to %s\n", nSDFBlocks, filename.str().c_str());

    std::cout << "------------------------------\n";
}

// Helper function: Convert world position to chunk coordinates
vec3i VoxelStreamingManager::worldToChunks(const vec3f& posWorld) const {
    vec3f p;
    p.x = posWorld.x / m_voxelExtents.x;
    p.y = posWorld.y / m_voxelExtents.y;
    p.z = posWorld.z / m_voxelExtents.z;
    
    vec3f s;
    s.x = (p.x > 0.0f) ? 1.0f : ((p.x < 0.0f) ? -1.0f : 0.0f);
    s.y = (p.y > 0.0f) ? 1.0f : ((p.y < 0.0f) ? -1.0f : 0.0f);
    s.z = (p.z > 0.0f) ? 1.0f : ((p.z < 0.0f) ? -1.0f : 0.0f);
    
    return vec3i(
        static_cast<int>(p.x + s.x * 0.5f),
        static_cast<int>(p.y + s.y * 0.5f),
        static_cast<int>(p.z + s.z * 0.5f)
    );
}

// Helper function: Check if chunk is within valid bounds
bool VoxelStreamingManager::isValidChunk(const vec3i& chunk) const {
    if (chunk.x < m_minGridPos.x || chunk.y < m_minGridPos.y || chunk.z < m_minGridPos.z) {
        return false;
    }
    if (chunk.x >= m_maxGridPos.x || chunk.y >= m_maxGridPos.y || chunk.z >= m_maxGridPos.z) {
        return false;
    }
    return true;
}

// Helper function: Convert chunk coordinates to linearized index
unsigned int VoxelStreamingManager::linearizeChunkPos(const vec3i& chunkPos) const {
    vec3i p = chunkPos - m_minGridPos;
    
    return p.z * m_gridDimensions.x * m_gridDimensions.y +
           p.y * m_gridDimensions.x +
           p.x;
}

// Helper function: Convert linearized index back to chunk coordinates
vec3i VoxelStreamingManager::delinearizeChunkIndex(unsigned int idx) const {
    unsigned int x = idx % m_gridDimensions.x;
    unsigned int y = (idx % (m_gridDimensions.x * m_gridDimensions.y)) / m_gridDimensions.x;
    unsigned int z = idx / (m_gridDimensions.x * m_gridDimensions.y);
    
    return m_minGridPos + vec3i(x, y, z);
}

// Helper function: Convert radius (in meters) to number of chunks (ceiling)
vec3i VoxelStreamingManager::meterToNumberOfChunksCeil(float radius) const {
    return vec3i(
        static_cast<int>(ceil(radius / m_voxelExtents.x)),
        static_cast<int>(ceil(radius / m_voxelExtents.y)),
        static_cast<int>(ceil(radius / m_voxelExtents.z))
    );
}

// Helper function: Check if chunk is within sphere (camera radius)
bool VoxelStreamingManager::isChunkInSphere(const vec3i& chunk, const vec3f& center, float radius) const {
    // Convert chunk coordinates to world position (chunk center)
    vec3f chunkWorldPos = vec3f(
        chunk.x * m_voxelExtents.x,
        chunk.y * m_voxelExtents.y,
        chunk.z * m_voxelExtents.z
    );
    
    // Calculate chunk radius (diagonal half-length)
    float chunkExt = std::max(std::max(m_voxelExtents.x, m_voxelExtents.y), m_voxelExtents.z);
    float chunkRadius = 0.5f * chunkExt * sqrtf(3.0f);
    
    // Calculate distance from chunk center to camera
    float dx = chunkWorldPos.x - center.x;
    float dy = chunkWorldPos.y - center.y;
    float dz = chunkWorldPos.z - center.z;
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    
    // Check if chunk is completely within sphere
    // If distance + chunkRadius <= radius, chunk is inside
    if (distance + chunkRadius <= radius) {
        return true;
    }
    
    // Conservative check: if distance - chunkRadius <= radius, chunk intersects sphere
    // For simplicity, we'll use the conservative check
    return (distance - chunkRadius <= radius);
}



unsigned int VoxelStreamingManager::streamInFindChunk(const vec3f& sphereCenter, float radius, bool useParts) {

    useParts = false;

    // Convert sphere center position to chunk coordinates
    vec3i camChunk = worldToChunks(sphereCenter);
    
    // Convert radius (in meters) to number of chunks
    vec3i chunkRadius = meterToNumberOfChunksCeil(radius);
    
    // Calculate chunk range to search (clamped to grid bounds)
    vec3i startChunk = vec3i(
        std::max(camChunk.x - chunkRadius.x, m_minGridPos.x),
        std::max(camChunk.y - chunkRadius.y, m_minGridPos.y),
        std::max(camChunk.z - chunkRadius.z, m_minGridPos.z)
    );
    vec3i endChunk = vec3i(
        std::min(camChunk.x + chunkRadius.x, m_maxGridPos.x - 1),
        std::min(camChunk.y + chunkRadius.y, m_maxGridPos.y - 1),
        std::min(camChunk.z + chunkRadius.z, m_maxGridPos.z - 1)
    );
    
    unsigned int nSDFBlocks = 0;


    std::cout << "------------ Find chunk grid ------------\n";

    std::cout << "camChunk : " << camChunk.x << " " << camChunk.y << " " << camChunk.z << "\n";
    std::cout << "chunkRadius : " << chunkRadius.x << " " << chunkRadius.y << " " << chunkRadius.z << "\n";
    std::cout << "startChunk : " << startChunk.x << " " << startChunk.y << " " << startChunk.z << "\n";
    std::cout << "endChunk : " << endChunk.x << " " << endChunk.y << " " << endChunk.z << "\n";
    
    // Calculate total chunks to search
    int totalChunksX = endChunk.x - startChunk.x + 1;
    int totalChunksY = endChunk.y - startChunk.y + 1;
    int totalChunksZ = endChunk.z - startChunk.z + 1;
    int totalChunks = totalChunksX * totalChunksY * totalChunksZ;
    std::cout << "Total chunks to search: " << totalChunksX << " x " << totalChunksY << " x " << totalChunksZ << " = " << totalChunks << "\n";
    std::cout << "  (Note: chunkRadius=" << chunkRadius.x << " means range from -" << chunkRadius.x << " to +" << chunkRadius.x << " = " << (chunkRadius.x * 2 + 1) << " chunks per axis)\n";

    std::cout << "------------------------------\n";

    // Create streaming_debugging directory if it doesn't exist
    std::string debugDir = m_outputDirectory + "/streaming_debugging";
    if (!m_outputDirectory.empty()) {
        std::filesystem::create_directories(debugDir);
    }

    // Save chunk positions to .xyz file (Auxiliary thread)
    std::stringstream filename;
    if (!m_outputDirectory.empty()) {
        filename << debugDir << "/stream_in_chunk_positions_frame_" << m_frameNumber << ".xyz";
    } else {
        filename << "stream_in_chunk_positions_frame_" << m_frameNumber << ".xyz";
    }
    std::ofstream xyzFile(filename.str());

    // Save actual copied block positions to .xyz file (world coordinates)
    std::stringstream blockFilename;
    if (!m_outputDirectory.empty()) {
        blockFilename << debugDir << "/stream_in_block_world_positions_frame_" << m_frameNumber << ".xyz";
    } else {
        blockFilename << "stream_in_block_world_positions_frame_" << m_frameNumber << ".xyz";
    }
    std::ofstream blockXyzFile(blockFilename.str());

    int cnt = 0;
    
    // Iterate through chunks in the radius
    for (int x = startChunk.x; x <= endChunk.x; x++) {
        for (int y = startChunk.y; y <= endChunk.y; y++) {
            for (int z = startChunk.z; z <= endChunk.z; z++) {
                vec3i chunkPos = vec3i(x, y, z);

                //std::cout << cnt << " : chunkPos : " << chunkPos.x << " " << chunkPos.y << " " << chunkPos.z << "\n";
                // std::cout << chunkPos.x << " " << chunkPos.y << " " << chunkPos.z << "\n";
                
                // Write to .xyz file
                xyzFile << chunkPos.x << " " << chunkPos.y << " " << chunkPos.z << "\n";

                cnt++;

                unsigned int index = linearizeChunkPos(chunkPos);
                
                // Check if chunk exists and has streamed out blocks
                if (m_grid[index] != NULL && m_grid[index]->isStreamedOut()) {
                    // Check if chunk is within sphere (centered at sphereCenter, typically 100m ahead of camera)
                    if (isChunkInSphere(chunkPos, sphereCenter, radius)) {
                        unsigned int nBlock = m_grid[index]->getNElements();
                        
                        // Check buffer overflow
                        if (nBlock + nSDFBlocks > m_maxNumberOfSDFBlocksIntegrateFromGlobalHash) {
                            printf("[ERROR] Not enough memory allocated for stream-in buffer (wants %u blocks, max %u)\n",
                                   nBlock + nSDFBlocks, m_maxNumberOfSDFBlocksIntegrateFromGlobalHash);
                            return nSDFBlocks; // Return what we have so far
                        }
                        
                        // Copy chunk data to GPU buffers
                        const std::vector<SDFBlockInfo>& blockDescs = m_grid[index]->getSDFBlockDescs();
                        
                        // Save actual copied block positions to .xyz file (before copying to GPU)
                        for (unsigned int i = 0; i < nBlock; i++) {
                            vec3i blockPos = blockDescs[i].pos;
                            // Convert block coordinate to world position
                            vec3f posWorld = vec3f(
                                blockPos.x * SDF_BLOCK_SIZE * params_.voxelSize,
                                blockPos.y * SDF_BLOCK_SIZE * params_.voxelSize,
                                blockPos.z * SDF_BLOCK_SIZE * params_.voxelSize
                            );
                            if (blockXyzFile.is_open()) {
                                blockXyzFile << posWorld.x << " " << posWorld.y << " " << posWorld.z << "\n";
                            }
                        }

                        
                        cudaMemcpy(d_SDFBlockInfoInput + nSDFBlocks,
                                   &(blockDescs[0]),
                                   sizeof(SDFBlockInfo) * nBlock,
                                   cudaMemcpyHostToDevice);
                        
                        cudaMemcpy(d_SDFBlockInput + nSDFBlocks,
                                   &(m_grid[index]->getSDFBlocks()[0]),
                                   sizeof(SDFBlock) * nBlock,
                                   cudaMemcpyHostToDevice);
                        
                        // Check for copy errors
                        cudaError_t err = cudaGetLastError();
                        if (err != cudaSuccess) {
                            printf("[ERROR] GPU copy failed: %s\n", cudaGetErrorString(err));
                            return nSDFBlocks;
                        }
                        
                        // Clear chunk data from CPU (streamed in, so remove from CPU storage)
                        m_grid[index]->clear();
                        m_bitMask.resetBit(index);
                        
                        nSDFBlocks += nBlock;
                        
                        // If using parts, only process one chunk per frame
                        if (useParts) {
                            return nSDFBlocks;
                        }
                    }
                }
            }
        }
    }
    
    xyzFile.close();
    blockXyzFile.close();
    printf("[STREAM IN COPY] Saved %d chunk positions to %s\n", cnt, filename.str().c_str());
    printf("[STREAM IN COPY] Saved %u actual copied block world positions to %s\n", nSDFBlocks, blockFilename.str().c_str());
    
    // Save sphere surface points to .xyz file
    if (!m_outputDirectory.empty()) {
        std::stringstream sphereFilename;
        sphereFilename << debugDir << "/stream_in_sphere_surface_frame_" << m_frameNumber << ".xyz";
        std::ofstream sphereFile(sphereFilename.str());
        
        if (sphereFile.is_open()) {
            // Generate sphere surface points using spherical coordinates
            // theta: 0 ~ PI (latitude, from north pole to south pole)
            // phi: 0 ~ 2*PI (longitude, full rotation)
            const int numTheta = 30;  // Number of latitude divisions
            const int numPhi = 60;    // Number of longitude divisions
            
            int spherePointCount = 0;
            for (int i = 0; i <= numTheta; i++) {
                float theta = M_PI * i / numTheta;  // 0 to PI
                
                // For poles, only one point
                int phiSteps = (i == 0 || i == numTheta) ? 1 : numPhi;
                
                for (int j = 0; j < phiSteps; j++) {
                    float phi = 2.0f * M_PI * j / numPhi;  // 0 to 2*PI
                    
                    // Convert spherical to Cartesian coordinates
                    float x = sphereCenter.x + radius * sinf(theta) * cosf(phi);
                    float y = sphereCenter.y + radius * sinf(theta) * sinf(phi);
                    float z = sphereCenter.z + radius * cosf(theta);
                    
                    sphereFile << x << " " << y << " " << z << "\n";
                    spherePointCount++;
                }
            }
            
            sphereFile.close();
            printf("[STREAM IN COPY] Saved %d sphere surface points to %s\n", 
                   spherePointCount, sphereFilename.str().c_str());
        }
    }
    
    return nSDFBlocks;
}