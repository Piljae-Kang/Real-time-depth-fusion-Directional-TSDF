#pragma once

#include <unordered_map>
#include <vector>
#include <Windows.h>

#include "CUDAHashData.h"
#include "CUDAHashRef.h"
#include "VoxelScene.h"
#include "BitArray.h"

// Debug info structure for stream in insertion logging
enum StreamInInsertResult {
    INSERT_RESULT_NEW_SLOT = 0,           // Found new slot in primary bucket
    INSERT_RESULT_EXISTS_PRIMARY = 1,     // Already exists in primary bucket
    INSERT_RESULT_NEW_OPEN = 2,           // Found new slot via open addressing
    INSERT_RESULT_EXISTS_OPEN = 3,        // Already exists via open addressing
    INSERT_RESULT_FAIL = 4                 // Failed to insert (hash table full)
};

struct StreamInDebugInfo {
    int3 blockCoord;
    int result;  // StreamInInsertResult
    unsigned int slotIdx;
    unsigned int bucketId;
};



struct SDFBlock {

	VoxelData data[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];

	static vec3ui delinearizeVoxelIndex(uint idx) {
		uint x = idx % SDF_BLOCK_SIZE;
		uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
		uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
		return vec3ui(x, y, z);
	}
};

struct SDFBlockInfo {

	__host__ __device__ SDFBlockInfo() : pos(0, 0, 0), ptr(-1) {

	}

	__host__ __device__ SDFBlockInfo(const HashSlot& hashSlot) {

		pos.x = hashSlot.pos.x;
		pos.y = hashSlot.pos.y;
		pos.z = hashSlot.pos.z;

		ptr = hashSlot.ptr;
	}

	bool operator<(const SDFBlockInfo& other) const {
		if (pos.x == other.pos.x) {
			if (pos.y == other.pos.y) {
				return pos.z < other.pos.z;
			}
			return pos.y < other.pos.y;
		}
		return pos.x < other.pos.x;
	}

	bool operator==(const SDFBlockInfo& other) const {
		return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z;
	}


	vec3i pos;
	int ptr;


};

class chunkData {

public:

	chunkData(unsigned int initialChunkListSize) {
		m_SDFBlocks = std::vector<SDFBlock>();
		m_SDFBlocks.reserve(initialChunkListSize);

		m_ChunkInfos = std::vector<SDFBlockInfo>();
		m_ChunkInfos.reserve(initialChunkListSize);
	}

	void addSDFBlock(const SDFBlockInfo& desc, const SDFBlock& data) {

		m_ChunkInfos.push_back(desc);
		m_SDFBlocks.push_back(data);
	}

	unsigned int getNElements() {
		return (unsigned int)m_SDFBlocks.size();
	}

	SDFBlockInfo& getChunkInfo(unsigned int i) {
		return m_ChunkInfos[i];
	}

	SDFBlock& getSDFBlock(unsigned int i) {
		return m_SDFBlocks[i];
	}

	void clear() {
		m_ChunkInfos.clear();
		m_SDFBlocks.clear();
	}

	bool isStreamedOut() const {
		return m_SDFBlocks.size() > 0;
	}

	std::vector<SDFBlockInfo>& getSDFBlockDescs() {
		return m_ChunkInfos;
	}

	std::vector<SDFBlock>& getSDFBlocks() {
		return m_SDFBlocks;
	}

	const std::vector<SDFBlockInfo>& getSDFBlockDescs() const {
		return m_ChunkInfos;
	}

	const std::vector<SDFBlock>& getSDFBlocks() const {
		return m_SDFBlocks;
	}

private:

	std::vector<SDFBlock>		m_SDFBlocks;
	std::vector<SDFBlockInfo>	m_ChunkInfos;

};



class VoxelStreamingManager {


public:

    VoxelStreamingManager(VoxelScene& sceneHashRef, Params& params, const vec3f& voxelExtends, const vec3i& gridDimensions, const vec3i& minGridPos,
		unsigned int initialChunkListSize, bool streamingEnabled);


    ~VoxelStreamingManager();

	// Chunk Grid allocation and initialization

	void initialize(const vec3f& voxelExtends, const vec3i& gridDimensions, const vec3i& minGridPos,
		unsigned int initialChunkListSize, bool streamingEnabled);


	void startAuxiliaryThread();  // Deprecated: Use startMultiThreading() instead
	void startMultiThreading();
	void stopMultiThreading();
	void initializeCriticalSection();
	void deleteCriticalSection();


	//-------------------------------------------------------
	// Streaming Function
	//-------------------------------------------------------


	void streamOut(VoxelScene& scene, const float3& cameraPos, float radius);

	void streamIn(VoxelScene& scene, const float3& cameraPos, float radius);



	// GPU to CPU (Stream Out)
	//-------------------------------------------------------
	void streamOutFindBlocksOnGPU(const vec3f& sphereCenter, float radius, bool useParts, bool multiThreaded /*= true*/);  // sphereCenter: center of sphere for stream out (typically 100m ahead of camera)        // Main: GPU���� ã��

	void streamOutCopyToChunkGrid(bool multiThreaded);        // Auxiliary: CPU�� ����
	//-------------------------------------------------------

	// CPU to GPU (Stream In)
	//-------------------------------------------------------
	void streamInCopyToGPUBuffer(const vec3f& sphereCenter, float radius, bool useParts, bool multiThreaded);  // sphereCenter: center of sphere for stream-in (typically 100m ahead of camera)        // Auxiliary: GPU ���۷� ����

	void streamInInsertToHashTable(bool multiThreaded);       // Main: Hash table�� ����
	//-------------------------------------------------------




	void updateChunkGrid(const int* blockInfos, const int* blocks, unsigned int nSDFBlocks);

	unsigned int streamInFindChunk(const vec3f& posCamera, float radius, bool useParts);


	void clearSDFBlockCounter() {
		unsigned int src = 0;
		cudaMemcpy(d_SDFBlockCounter, &src, sizeof(unsigned int), cudaMemcpyHostToDevice);
	}

	unsigned int getSDFBlockCounter() const {
		unsigned int dest;
		cudaMemcpy(&dest, d_SDFBlockCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		return dest;
	}


private:

    // Helper functions for chunk grid
    vec3i worldToChunks(const vec3f& posWorld) const;
    bool isValidChunk(const vec3i& chunk) const;
    unsigned int linearizeChunkPos(const vec3i& chunkPos) const;
    vec3i delinearizeChunkIndex(unsigned int idx) const;
    vec3i meterToNumberOfChunksCeil(float radius) const;
    bool isChunkInSphere(const vec3i& chunk, const vec3f& center, float radius) const;

    bool isInsideRadius(const float3& blockWorldCenter,
                        const float3& cameraPos,
                        float radius) const;

    float3 blockCoordToWorld(const int3& blockCoord) const;

	//-------------------------------------------------------
	// member valuables
	//-------------------------------------------------------

	unsigned int m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;

	SDFBlockInfo* h_SDFBlockInfoOutput;
	SDFBlock* h_SDFBlockOutput;

	SDFBlockInfo* d_SDFBlockInfoOutput;
	SDFBlockInfo* d_SDFBlockInfoInput;
	SDFBlock* d_SDFBlockOutput;
	SDFBlock* d_SDFBlockInput;
	unsigned int* d_SDFBlockCounter;

	unsigned int* d_bitMask;

	//-------------------------------------------------------
	// Chunk Grid
	//-------------------------------------------------------

	vec3f m_voxelExtents;		// extend of the voxels in meters
	vec3i m_gridDimensions;	    // number of voxels in each dimension

	vec3i m_minGridPos;
	vec3i m_maxGridPos;

	unsigned int m_initialChunkListSize;	 // initial size for vectors in the Chunk

	std::vector<chunkData*>	m_grid; // Grid data
	BitArray<unsigned int>	m_bitMask;



	//-------------------------------------------------------
	// Multi threading
	//-------------------------------------------------------

	// Multi-threading
	HANDLE hStreamingThread;
	DWORD dwStreamingThreadID;

	// Mutex
	HANDLE hMutexOut;
	HANDLE hEventOutProduce;
	HANDLE hEventOutConsume;

	HANDLE hMutexIn;
	HANDLE hEventInProduce;
	HANDLE hEventInConsume;

	HANDLE hMutexSetTransform;
	HANDLE hEventSetTransformProduce;
	HANDLE hEventSetTransformConsume;


	//-------------------------------------------------------
	// Runtime variables
	//-------------------------------------------------------

public:
	vec3f			posCamera_;  // Sphere center for stream-in (typically 100m ahead of camera, set by main thread for auxiliary thread)
	float			radius_;     // Streaming radius for stream-in (typically 10m, set by main thread for auxiliary thread)
	bool			s_terminateThread;  // Thread termination flag (accessed by AuxiliaryStreamingFunc)
	unsigned int	m_frameNumber;  // Current frame number (for .xyz file naming)
	std::string		m_outputDirectory;  // Output directory path (set by main thread, e.g., "output/시간")

private:
	unsigned int	nStreamdInBlocks_;
	unsigned int	nStreamdOutBlocks_;

	VoxelScene* sceneHashRef_;
	Params params_;


};