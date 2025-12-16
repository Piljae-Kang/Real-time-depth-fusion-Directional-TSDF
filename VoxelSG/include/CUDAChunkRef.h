#pragma once

#include "CUDAHashRef.h"
#include "CUDAHashData.h"
#include "BitArray.h"
#include "VoxelScene.h"

#include <vector>

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


	SDFBlockInfo() : pos(0, 0, 0), ptr(-1) {
	
	}

	SDFBlockInfo(const HashSlot& hashSlot) {

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


class CUDAChunkRef {


public:

	CUDAChunkRef();
	~CUDAChunkRef();


	// get data from hash data

	// streaming function

	const CUDAHashRef& getHashData() const {
		return sceneHashRef_->getHashData();
	}
	const Params& getHashParams() const {
		return sceneHashRef_->getParams();
	}


	// Stream Out
	void streamOutToCPUAll();
	void streamOutToCPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamOutToCPUPass0GPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
	void streamOutToCPUPass1CPU(bool multiThreaded = true);
	void integrateInChunkGrid(const int* desc, const int* block, unsigned int nSDFBlocks);

	// Stream In
	void streamInToGPUAll();
	void streamInToGPUAll(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamInToGPUChunk(const vec3i& chunkPos);
	void streamInToGPUChunkNeighborhood(const vec3i& chunkPos, int kernelRadius);
	void streamInToGPU(const vec3f& posCamera, float radius, bool useParts, unsigned int& nStreamedBlocks);

	void streamInToGPUPass0CPU(const vec3f& posCamera, float radius, bool useParts, bool multiThreaded = true);
	void streamInToGPUPass1GPU(bool multiThreaded = true);

	unsigned int integrateInHash(const vec3f& posCamera, float radius, bool useParts);







	// member valuables

	unsigned int m_maxNumberOfSDFBlocksIntegrateFromGlobalHash;

	SDFBlockInfo* h_SDFBlockDescOutput;
	SDFBlock* h_SDFBlockOutput;

	SDFBlockInfo* d_SDFBlockDescOutput;
	SDFBlockInfo* d_SDFBlockDescInput;
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

	unsigned int m_initialChunkDescListSize;	 // initial size for vectors in the ChunkDesc

	std::vector<chunkData*>	m_grid; // Grid data
	BitArray<unsigned int>	m_bitMask;

	unsigned int m_currentPart;
	unsigned int m_streamOutParts;


	vec3f			posCamera_;
	float			radius_;
	unsigned int	nStreamdInBlocks_;
	unsigned int	nStreamdOutBlocks_;
	bool			terminateThread_;

	VoxelScene* sceneHashRef_;

};