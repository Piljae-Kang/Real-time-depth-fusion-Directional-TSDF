#pragma once

#include "globalPramasConfig.h"

/**
 * Params: Configuration parameters for voxel hashing
 * Derived from GlobalParamsConfig
 */
struct Params {

	Params(GlobalParamsConfig gpc) {
		
		SDFBlockNum = gpc.g_hashNumSDFBlocks;
		hashSlotNum = gpc.g_hashNumSlots;

		SDFBlockSize = SDF_BLOCK_SIZE;          // Size of each voxel block (8x8x8)
		slotSize = HASH_BUCKET_SIZE;            // Hash bucket size

		totalHashSize = gpc.g_hashNumSlots * HASH_BUCKET_SIZE;  // Total hash entries
		totalBlockSize = gpc.g_hashNumSDFBlocks * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;  // Total voxels: N x (8x8x8)
		
		voxelSize = gpc.g_SDFVoxelSize;
		//voxelSize = 0.004;
		truncationScale = gpc.g_SDFTruncation;
		truncation = gpc.g_SDFTruncationScale;
	}

	int hashSlotNum;      // Number of hash slots
	int SDFBlockNum;      // Number of SDF blocks
	int SDFBlockSize;     // Size of each SDF block (8)
	int totalHashSize;    // Total number of hash entries
	int slotSize;         // Bucket size

	int totalBlockSize;   // Total number of voxels

	float voxelSize;      // Size of a single voxel in world units
	float truncationScale;  // SDF truncation distance
	float truncation;       // SDF truncation scale
};
