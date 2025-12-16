#pragma once

#include "globalPramasConfig.h"

/**
 * Params: Configuration parameters for voxel hashing
 * Derived from GlobalParamsConfig
 */
struct Params {

	// Default constructor (required for some containers)
	Params() = default;

	Params(GlobalParamsConfig gpc) {
		
		SDFBlockNum = gpc.g_hashNumSDFBlocks;
		hashSlotNum = gpc.g_hashNumSlots;

		SDFBlockSize = SDF_BLOCK_SIZE;          // Size of each voxel block (SDF_BLOCK_SIZE^3)
		slotSize = HASH_BUCKET_SIZE;            // Hash bucket size

		totalHashSize = gpc.g_hashNumSlots * HASH_BUCKET_SIZE;  // Total hash entries
		totalBlockSize = gpc.g_hashNumSDFBlocks * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;  // Total voxels: N x (SDF_BLOCK_SIZE^3)
		
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
