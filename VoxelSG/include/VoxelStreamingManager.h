#pragma once

#include <unordered_map>
#include <vector>
#include <Windows.h>

#include "CUDAHashData.h"
#include "CUDAHashRef.h"
#include "VoxelScene.h"
#include "BitArray.h"



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
		unsigned int initialChunkListSize, bool streamingEnabled, unsigned int streamOutParts);


    ~VoxelStreamingManager();

	// Chunk Grid allocation and initialization

	void initialize(const vec3f& voxelExtends, const vec3i& gridDimensions, const vec3i& minGridPos,
		unsigned int initialChunkListSize, bool streamingEnabled);


	void startAuxiliaryThread();


	//-------------------------------------------------------
	// Streaming Function
	//-------------------------------------------------------


	void streamOut(VoxelScene& scene, const float3& cameraPos, float radius);

	void streamIn(VoxelScene& scene, const float3& cameraPos, float radius);



	// GPU �� CPU (Stream Out)
	void streamOutFindBlocksOnGPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded /*= true*/);        // Main: GPU���� ã��

	void streamOutCopyToChunkGrid(bool multiThreaded = true);        // Auxiliary: CPU�� ����

	// CPU �� GPU (Stream In)  
	void streamInCopyToGPUBuffer();         // Auxiliary: GPU ���۷� ����

	void streamInInsertToHashTable();       // Main: Hash table�� ����




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

	bool s_terminateThread;

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

	unsigned int m_currentPart;
	unsigned int m_streamOutParts;



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


	vec3f			posCamera_;
	float			radius_;
	unsigned int	nStreamdInBlocks_;
	unsigned int	nStreamdOutBlocks_;
	bool			terminateThread_;

	VoxelScene* sceneHashRef_;
	Params params_;


};