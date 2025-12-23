#include "../include/VoxelScene.h"
//#include "../include/DepthCameraData.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>
#include <cuda_runtime.h>
#include <filesystem>
#include <system_error>
#include <set>
#include <tuple>
#include <vector>

// Define to enable saving active blocks to xyz files
// Set to 0 to disable xyz file saving (saves disk I/O)
#define SAVE_ACTIVE_BLOCKS_TO_XYZ 1

namespace fs = std::filesystem;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        } \
    } while(0)

// Helper function to write heap/block log to file
static void writeHeapBlockLog(const std::string& outputDir, int frameNumber, 
                              const std::string& phase, 
                              unsigned int heapCounter, unsigned int initialHeapCounter,
                              unsigned int totalAllocated, unsigned int activeBlocks,
                              unsigned int newlyAllocated = 0) {
    if (outputDir.empty()) return;
    
    fs::path logPath = fs::path(outputDir) / "heap_block_log.txt";
    std::ofstream logFile(logPath, std::ios::app);  // Append mode
    
    if (!logFile.is_open()) {
        std::cerr << "Failed to open log file: " << logPath << std::endl;
        return;
    }
    
    // Write header for first entry
    static bool headerWritten = false;
    if (!headerWritten) {
        logFile << "Frame\tPhase\tHeapCounter\tInitialCounter\tTotalAllocated\tActiveBlocks\tNewlyAllocated\tFreeBlocks\n";
        headerWritten = true;
    }
    
    unsigned int freeBlocks = heapCounter + 1;
    logFile << frameNumber << "\t" << phase << "\t" 
            << heapCounter << "\t" << initialHeapCounter << "\t"
            << totalAllocated << "\t" << activeBlocks << "\t"
            << newlyAllocated << "\t" << freeBlocks << "\n";
    
    logFile.close();
}

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

    initializeOutputDirectory();
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

void VoxelScene::initializeOutputDirectory() {
    if (!m_outputDirectory.empty()) {
        return;
    }

    std::error_code ec;
    const fs::path baseDir("output");
    fs::create_directories(baseDir, ec);
    if (ec) {
        std::cerr << "VoxelScene: Failed to create base output directory 'output' ("
                  << ec.message() << "). Using current working directory instead." << std::endl;
        m_outputDirectory = fs::current_path().string();
        return;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTm{};
#ifdef _WIN32
    localtime_s(&localTm, &nowTime);
#else
    localtime_r(&nowTime, &localTm);
#endif

    char buffer[32];
    std::string folderName;
    if (std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &localTm) == 0) {
        std::ostringstream fallback;
        fallback << "run_" << static_cast<long long>(nowTime);
        folderName = fallback.str();
    } else {
        folderName = buffer;
    }

    fs::path runDir = baseDir / folderName;
    fs::create_directories(runDir, ec);
    if (ec) {
        std::cerr << "VoxelScene: Failed to create run output directory '" << runDir.string()
                  << "' (" << ec.message() << "). Falling back to base directory." << std::endl;
        m_outputDirectory = baseDir.string();
    } else {
        m_outputDirectory = runDir.string();
    }

    std::cout << "VoxelScene: Output directory set to '" << m_outputDirectory << "'" << std::endl;
    writeGlobalParametersLog();
}

void VoxelScene::writeGlobalParametersLog() const {
    if (m_outputDirectory.empty()) {
        return;
    }

    const auto& gpc = GlobalParamsConfig::get();
    const fs::path logPath = fs::path(m_outputDirectory) / "global_parameters.txt";

    std::ofstream out(logPath);
    if (!out.is_open()) {
        std::cerr << "VoxelScene: Failed to write global parameters log at "
                  << logPath.string() << std::endl;
        return;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTm{};
#ifdef _WIN32
    localtime_s(&localTm, &nowTime);
#else
    localtime_r(&localTm, &nowTime);
#endif

    out << "# VoxelScene Global Parameters" << std::endl;
    out << "timestamp: " << std::put_time(&localTm, "%Y-%m-%d %H:%M:%S") << std::endl;
    out << "output_directory: " << m_outputDirectory << std::endl;
    out << "folder_name: " << fs::path(m_outputDirectory).filename().string() << std::endl;
    out << "Data_name: " << gpc.g_data_name << std::endl;

    out << "\n[Params]" << std::endl;
    out << "SDFBlockSize: " << m_params.SDFBlockSize << std::endl;
    out << "SDFBlockNum: " << m_params.SDFBlockNum << std::endl;
    out << "TotalHashSize: " << m_params.totalHashSize << std::endl;
    out << "HashSlotNum: " << m_params.hashSlotNum << std::endl;
    out << "HashBucketSize: " << m_params.slotSize << std::endl;
    out << "TotalVoxelCount: " << m_params.totalBlockSize << std::endl;
    out << "VoxelSize: " << m_params.voxelSize << std::endl;
    out << "TruncationScale: " << m_params.truncationScale << std::endl;
    out << "TruncationDistance: " << m_params.truncation << std::endl;

    out << "\n[GlobalParamsConfig]" << std::endl;
    out << "g_hashNumSlots: " << gpc.g_hashNumSlots << std::endl;
    out << "g_hashNumSDFBlocks: " << gpc.g_hashNumSDFBlocks << std::endl;
    out << "g_hashNumBuckets: " << gpc.g_hashNumBuckets << std::endl;
    out << "g_hashSlotSize: " << gpc.g_hashSlotSize << std::endl;
    out << "g_SDFVoxelSize: " << gpc.g_SDFVoxelSize << std::endl;
    out << "g_SDFTruncation: " << gpc.g_SDFTruncation << std::endl;
    out << "g_SDFTruncationScale: " << gpc.g_SDFTruncationScale << std::endl;
    out << "g_SDFMarchingCubeThreshFactor: " << gpc.g_SDFMarchingCubeThreshFactor << std::endl;

    out.close();
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
    if (m_hashData.d_blockParentUV) CUDA_CHECK(cudaFree(m_hashData.d_blockParentUV));
    if (m_hashData.d_blockAllocationMethod) CUDA_CHECK(cudaFree(m_hashData.d_blockAllocationMethod));
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
    m_hashData.d_blockParentUV = nullptr;
    m_hashData.d_blockAllocationMethod = nullptr;
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

    if (m_hashData.d_blockParentUV && m_params.SDFBlockNum > 0) {
        size_t parentBytes = static_cast<size_t>(m_params.SDFBlockNum) * sizeof(int2);
        CUDA_CHECK(cudaMemset(m_hashData.d_blockParentUV, 0xFF, parentBytes));
    }
    if (m_hashData.d_blockAllocationMethod && m_params.SDFBlockNum > 0) {
        size_t methodBytes = static_cast<size_t>(m_params.SDFBlockNum) * sizeof(unsigned char);
        CUDA_CHECK(cudaMemset(m_hashData.d_blockAllocationMethod, 0, methodBytes));
    }

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

std::string VoxelScene::getOutputDirectory() const {
    if (m_outputDirectory.empty()) {
        const_cast<VoxelScene*>(this)->initializeOutputDirectory();
    }
    return m_outputDirectory;
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
    
    if (m_outputDirectory.empty()) {
        initializeOutputDirectory();
    }

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
    
    // ===================================================================
    // Kernel Timing Measurement (kernel execution time only)
    // ===================================================================
    float totalKernelTime = 0.0f;
    float allocTime = 0.0f;
    float compactTime = 0.0f;
    float integrateTime = 0.0f;
    
    // Get heap counter before allocation for logging
    unsigned int heapCounterBeforeAlloc = 0;
    CUDA_CHECK(cudaMemcpy(&heapCounterBeforeAlloc, m_hashData.d_heapCounter, 
                          sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int initialHeapCounter = m_params.SDFBlockNum - 1;
    unsigned int totalAllocatedBefore = initialHeapCounter - heapCounterBeforeAlloc;
    
    // Step 1: Allocate blocks based on depthmap
    std::cout << "  Step 1: Allocating blocks..." << std::endl;
    allocTime = allocBlocksFromDepthMapCUDA(
        m_hashData,
        m_params,
        depthmap,
        normalmap,
        width,
        height,
        truncationDistance,
        cameraPos,
        d_transform
    );
    totalKernelTime += allocTime;
    
    // Get heap counter after allocation for logging
    unsigned int heapCounterAfterAlloc = 0;
    CUDA_CHECK(cudaMemcpy(&heapCounterAfterAlloc, m_hashData.d_heapCounter, 
                          sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int totalAllocatedAfter = initialHeapCounter - heapCounterAfterAlloc;
    // NOTE: newlyAllocated from heap counter is approximate (includes streaming in blocks)
    // Actual newly allocated blocks will be calculated after compactify by comparing active blocks
    unsigned int newlyAllocated = heapCounterBeforeAlloc - heapCounterAfterAlloc;
    
    // Log allocation to file (with approximate newlyAllocated, will be updated after compactify)
    writeHeapBlockLog(m_outputDirectory, static_cast<int>(m_numIntegratedFrames), 
                     "ALLOCATION", heapCounterAfterAlloc, initialHeapCounter,
                     totalAllocatedAfter, 0, newlyAllocated);
    
    // Step 2: Compactify hash entries (optional, for efficiency)
    std::cout << "  Step 2: Compactifying hash entries..." << std::endl;
    compactTime = compactifyHashCUDA(m_hashData, m_params);
    totalKernelTime += compactTime;
    
    // Get active blocks count and heap counter after compactify for logging
    int numActiveBlocks = 0;
    CUDA_CHECK(cudaMemcpy(&numActiveBlocks, m_hashData.d_hashCompactifiedCounter,
                          sizeof(int), cudaMemcpyDeviceToHost));
    unsigned int heapCounterAfterCompact = 0;
    CUDA_CHECK(cudaMemcpy(&heapCounterAfterCompact, m_hashData.d_heapCounter,
                          sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int totalAllocatedAfterCompact = initialHeapCounter - heapCounterAfterCompact;
    
    // Log compactify to file
    writeHeapBlockLog(m_outputDirectory, static_cast<int>(m_numIntegratedFrames),
                     "COMPACTIFY", heapCounterAfterCompact, initialHeapCounter,
                     totalAllocatedAfterCompact, static_cast<unsigned int>(numActiveBlocks), 0);
    
    // Save newly allocated blocks to xyz file
    if (numActiveBlocks > 0 && !m_outputDirectory.empty()) {
        // Read current active blocks from compactified hash table
        std::vector<HashSlot> currentActiveBlocks(numActiveBlocks);
        CUDA_CHECK(cudaMemcpy(currentActiveBlocks.data(), m_hashData.d_CompactifiedHashTable,
                              sizeof(HashSlot) * numActiveBlocks, cudaMemcpyDeviceToHost));
        
        // Static variable to store previous frame's active blocks
        static std::set<std::tuple<int, int, int>> previousActiveBlocks;
        
        // Find newly allocated blocks (blocks that exist in current but not in previous)
        std::vector<std::tuple<int, int, int>> newlyAllocatedBlocks;
        for (int i = 0; i < numActiveBlocks; i++) {
            const HashSlot& slot = currentActiveBlocks[i];
            if (slot.ptr != -1) {  // Valid block
                std::tuple<int, int, int> blockKey(slot.pos.x, slot.pos.y, slot.pos.z);
                if (previousActiveBlocks.find(blockKey) == previousActiveBlocks.end()) {
                    // This block is new (not in previous frame)
                    newlyAllocatedBlocks.push_back(blockKey);
                }
            }
        }
        
        // Calculate actual newly allocated blocks count (excluding streaming in blocks)
        // This is the accurate count, not the heap counter difference
        unsigned int actualNewlyAllocated = static_cast<unsigned int>(newlyAllocatedBlocks.size());
        
        // Update ALLOCATION log with actual newly allocated blocks count
        // (This is more accurate than heap counter difference, which includes streaming in blocks)
        if (m_numIntegratedFrames > 0) {
            // Re-read the log file and update the last ALLOCATION entry
            fs::path logPath = fs::path(m_outputDirectory) / "heap_block_log.txt";
            if (fs::exists(logPath)) {
                // Read all lines
                std::vector<std::string> lines;
                std::ifstream logFile(logPath);
                std::string line;
                while (std::getline(logFile, line)) {
                    lines.push_back(line);
                }
                logFile.close();
                
                // Find and update the last ALLOCATION entry
                for (int i = static_cast<int>(lines.size()) - 1; i >= 0; i--) {
                    if (lines[i].find("ALLOCATION") != std::string::npos && 
                        lines[i].find("\t" + std::to_string(m_numIntegratedFrames) + "\t") != std::string::npos) {
                        // Parse the line and replace NewlyAllocated column
                        std::istringstream iss(lines[i]);
                        std::vector<std::string> tokens;
                        std::string token;
                        while (std::getline(iss, token, '\t')) {
                            tokens.push_back(token);
                        }
                        
                        if (tokens.size() >= 7) {
                            // Update NewlyAllocated column (index 6)
                            tokens[6] = std::to_string(actualNewlyAllocated);
                            
                            // Reconstruct the line
                            std::ostringstream oss;
                            for (size_t j = 0; j < tokens.size(); j++) {
                                if (j > 0) oss << "\t";
                                oss << tokens[j];
                            }
                            lines[i] = oss.str();
                        }
                        break;
                    }
                }
                
                // Write back all lines
                std::ofstream logFileOut(logPath);
                for (const auto& l : lines) {
                    logFileOut << l << "\n";
                }
                logFileOut.close();
            }
        }
        
#if SAVE_ACTIVE_BLOCKS_TO_XYZ
        // Save newly allocated blocks to xyz file
        if (!newlyAllocatedBlocks.empty()) {
            fs::path debugDir = fs::path(m_outputDirectory) / "allocation_debugging";
            fs::create_directories(debugDir);
            
            std::stringstream filename;
            filename << debugDir.string() << "/allocation_new_blocks_frame_" 
                     << m_numIntegratedFrames << ".xyz";
            
            std::ofstream xyzFile(filename.str());
            if (xyzFile.is_open()) {
                for (const auto& blockKey : newlyAllocatedBlocks) {
                    int x = std::get<0>(blockKey);
                    int y = std::get<1>(blockKey);
                    int z = std::get<2>(blockKey);
                    
                    // Convert block coordinates to world position
                    float blockExtent = static_cast<float>(SDF_BLOCK_SIZE) * m_params.voxelSize;
                    float worldX = x * blockExtent;
                    float worldY = y * blockExtent;
                    float worldZ = z * blockExtent;
                    
                    xyzFile << worldX << " " << worldY << " " << worldZ << "\n";
                }
                xyzFile.close();
                printf("[ALLOCATION] Saved %zu newly allocated block positions to %s (heap counter diff: %u, actual: %u)\n", 
                       newlyAllocatedBlocks.size(), filename.str().c_str(),
                       heapCounterBeforeAlloc - heapCounterAfterAlloc, actualNewlyAllocated);
            }
        }
        
        // Save all active blocks to xyz file (for debugging/visualization)
        {
            fs::path debugDir = fs::path(m_outputDirectory) / "allocation_debugging";
            fs::create_directories(debugDir);
            
            std::stringstream filename;
            filename << debugDir.string() << "/all_active_blocks_frame_" 
                     << m_numIntegratedFrames << ".xyz";
            
            std::ofstream xyzFile(filename.str());
            if (xyzFile.is_open()) {
                for (int i = 0; i < numActiveBlocks; i++) {
                    const HashSlot& slot = currentActiveBlocks[i];
                    if (slot.ptr != -1) {  // Valid block
                        // Convert block coordinates to world position
                        float blockExtent = static_cast<float>(SDF_BLOCK_SIZE) * m_params.voxelSize;
                        float worldX = slot.pos.x * blockExtent;
                        float worldY = slot.pos.y * blockExtent;
                        float worldZ = slot.pos.z * blockExtent;
                        
                        xyzFile << worldX << " " << worldY << " " << worldZ << "\n";
                    }
                }
                xyzFile.close();
                printf("[ALLOCATION] Saved %d all active block positions to %s\n", 
                       numActiveBlocks, filename.str().c_str());
            }
        }
#endif // SAVE_ACTIVE_BLOCKS_TO_XYZ
        
        // Update previous active blocks for next frame
        previousActiveBlocks.clear();
        for (int i = 0; i < numActiveBlocks; i++) {
            const HashSlot& slot = currentActiveBlocks[i];
            if (slot.ptr != -1) {
                previousActiveBlocks.insert(std::make_tuple(slot.pos.x, slot.pos.y, slot.pos.z));
            }
        }
    }
    
    // Step 3: Integrate depth map into allocated blocks
    std::cout << "  Step 3: Integrating depth map..." << std::endl;
    integrateTime = integrateDepthMapIntoBlocksCUDA(
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
    totalKernelTime += integrateTime;
    
    // Print kernel timing summary
    std::cout << "\n=== Kernel Execution Time Summary ===" << std::endl;
    std::cout << "  1. Voxel Allocation:      " << std::fixed << std::setprecision(3) << allocTime << " ms" << std::endl;
    std::cout << "  2. Hash Compactification: " << std::fixed << std::setprecision(3) << compactTime << " ms" << std::endl;
    std::cout << "  3. Voxelwise Integration:  " << std::fixed << std::setprecision(3) << integrateTime << " ms" << std::endl;
    std::cout << "  Total Kernel Time:         " << std::fixed << std::setprecision(3) << totalKernelTime << " ms" << std::endl;
    std::cout << "=====================================\n" << std::endl;
    
    // Calculate active voxel count and memory usage (numActiveBlocks already retrieved above)
    const int voxelsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE; // 8^3 = 512
    int numActiveVoxels = numActiveBlocks * voxelsPerBlock;
    size_t activeVoxelDataBytes = static_cast<size_t>(numActiveVoxels) * sizeof(VoxelData);
    
    // Print memory statistics
    std::cout << "\n=== Active Voxel Memory Statistics ===" << std::endl;
    std::cout << "  Active blocks: " << numActiveBlocks << std::endl;
    std::cout << "  Active voxels: " << numActiveVoxels 
              << " (blocks × " << voxelsPerBlock << " voxels/block)" << std::endl;
    std::cout << "  Active voxel data memory: " 
              << std::fixed << std::setprecision(2) 
              << (activeVoxelDataBytes / (1024.0 * 1024.0)) << " MB"
              << " (" << activeVoxelDataBytes << " bytes)" << std::endl;
    
    // Calculate total allocated blocks for comparison (initialHeapCounter already declared above)
    unsigned int totalAllocatedBlocks = initialHeapCounter - heapCounterAfterCompact;
    
    if (totalAllocatedBlocks > 0) {
        int totalAllocatedVoxels = totalAllocatedBlocks * voxelsPerBlock;
        size_t totalAllocatedVoxelDataBytes = static_cast<size_t>(totalAllocatedVoxels) * sizeof(VoxelData);
        float activeRatio = (float)numActiveBlocks / totalAllocatedBlocks * 100.0f;
        
        std::cout << "  Total allocated blocks: " << totalAllocatedBlocks << std::endl;
        std::cout << "  Total allocated voxels: " << totalAllocatedVoxels << std::endl;
        std::cout << "  Total allocated voxel data memory: "
                  << std::fixed << std::setprecision(2)
                  << (totalAllocatedVoxelDataBytes / (1024.0 * 1024.0)) << " MB"
                  << " (" << totalAllocatedVoxelDataBytes << " bytes)" << std::endl;
        std::cout << "  Active / Allocated ratio: " 
                  << std::fixed << std::setprecision(1) << activeRatio << "%" << std::endl;
    }
    std::cout << "=====================================\n" << std::endl;


    
#ifdef SAVE_VOXEL_TO_XYZ
    // Download compactified hash table from GPU

    static int save_idx = 0;
    //int save_idx = 0;

    const float blockSpan = static_cast<float>(m_params.SDFBlockSize) * m_params.voxelSize;
    HashSlot* h_compactified = new HashSlot[numActiveBlocks];
    cudaError_t err_save = cudaMemcpy(h_compactified, m_hashData.d_CompactifiedHashTable, 
                                  numActiveBlocks * sizeof(HashSlot), 
                                  cudaMemcpyDeviceToHost);

    if (err_save == cudaSuccess && save_idx == 90) {
        std::cout << "  Saving updated voxel positions to updated_voxel_positions.xyz..." << std::endl;
        
        // Download voxel data from GPU
        const int voxelsPerBlock = m_params.SDFBlockSize * m_params.SDFBlockSize * m_params.SDFBlockSize;
        VoxelData* h_SDFBlocks = new VoxelData[numActiveBlocks * voxelsPerBlock];
        cudaError_t err2 = cudaMemcpy(h_SDFBlocks, m_hashData.d_SDFBlocks,
                                     numActiveBlocks * voxelsPerBlock * sizeof(VoxelData),
                                     cudaMemcpyDeviceToHost);
        
        if (err2 == cudaSuccess) {
            fs::path outputDir(m_outputDirectory);
            fs::path xyzPath = outputDir / ("updated_voxel_positions_" + std::to_string(save_idx) + ".xyz");
            fs::path fullPath = outputDir / "updated_voxel_full.txt";

            FILE* fp_xyz = fopen(xyzPath.string().c_str(), "w");
            FILE* fp_full = fopen(fullPath.string().c_str(), "w");
            
            if (fp_xyz && fp_full) {
                fprintf(fp_full, "# Voxel data: x, y, z, sdf, weight, r, g, b\n");
                
                int totalVoxels = 0;
                int updatedVoxels = 0;
                
                for (int i = 0; i < numActiveBlocks; i++) {
                    int3 blockCoord = h_compactified[i].pos;
                    
                    // Save block center/corner instead of all voxels in the block
                    float3 blockCenter = make_float3(
                        blockCoord.x * blockSpan,
                        blockCoord.y * blockSpan,
                        blockCoord.z * blockSpan
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

    save_idx++;
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