#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include "../include/CUDAHashRef.h"
#include "../include/VoxelScene.h"

// (Removed unused RayCastParams constant memory from expert code)

// ============================================================================
// CUDA Device Functions (helper functions for ray casting)
// ============================================================================

// Fractional part helper
__device__ inline float frac01(float v) { return v - floorf(v); }

// Convert world position to virtual voxel position (center-aligned, like expert code)
__device__ inline int3 worldToVirtualVoxelPos(const float3& pos, float voxelSize) {
    float3 p = make_float3(pos.x / voxelSize, pos.y / voxelSize, pos.z / voxelSize);
    float3 s = make_float3((p.x > 0) ? 1.0f : ((p.x < 0) ? -1.0f : 0.0f),
                           (p.y > 0) ? 1.0f : ((p.y < 0) ? -1.0f : 0.0f),
                           (p.z > 0) ? 1.0f : ((p.z < 0) ? -1.0f : 0.0f));
    return make_int3((int)floorf(p.x + 0.5f * s.x),
                     (int)floorf(p.y + 0.5f * s.y),
                     (int)floorf(p.z + 0.5f * s.z));
}

// Convert virtual voxel pos to SDF block with negative correction
__device__ inline int3 virtualVoxelPosToSDFBlock(int3 v) {
    const int blockSizeMinusOne = SDF_BLOCK_SIZE - 1;
    if (v.x < 0) v.x -= blockSizeMinusOne;
    if (v.y < 0) v.y -= blockSizeMinusOne;
    if (v.z < 0) v.z -= blockSizeMinusOne;
    return make_int3(v.x / SDF_BLOCK_SIZE, v.y / SDF_BLOCK_SIZE, v.z / SDF_BLOCK_SIZE);
}

// Local index inside block (robust modulo)
__device__ inline int3 virtualVoxelPosToLocal(int3 v) {
    int lx = ((v.x % SDF_BLOCK_SIZE) + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
    int ly = ((v.y % SDF_BLOCK_SIZE) + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
    int lz = ((v.z % SDF_BLOCK_SIZE) + SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
    return make_int3(lx, ly, lz);
}

/**
 * Helper function to get voxel at world position
 */
__device__ __forceinline__ unsigned int hashBlockCoordinateRC(int3 blockCoord) {
    // Same constants as allocation side
    unsigned int h = (unsigned int)(blockCoord.x * 73856093) ^
                     (unsigned int)(blockCoord.y * 19349663) ^
                     (unsigned int)(blockCoord.z * 83492791);
    return h;
}

__device__ VoxelData getVoxel(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    const float3& worldPos,
    float voxelSize
) {
    // Compute exact block and local coords like integration/allocation
    int3 vpos = worldToVirtualVoxelPos(worldPos, voxelSize);
    int3 blockCoord = virtualVoxelPosToSDFBlock(vpos);
    int3 local = virtualVoxelPosToLocal(vpos);
    
    // Hash to bucket
    unsigned int bucketId = hashBlockCoordinateRC(blockCoord) % numBuckets;
    unsigned int hp = bucketId * bucketSize;

    //if (hp != 1284370) {
    //    return;
    //}

    //printf("worldPos : %f %f %f\n", worldPos.x, worldPos.y, worldPos.z);
    //printf("blockCoord : %d %d %d\n", blockCoord.x, blockCoord.y, blockCoord.z);
    //printf("hp : %d\n", hp);
    
    // Probe inside the bucket first (do not early break)
    for (int j = 0; j < bucketSize; j++) {
        const HashSlot& slot = d_hashTable[hp + j];

        if (slot.ptr == -1) continue;

        if (slot.pos.x == blockCoord.x && slot.pos.y == blockCoord.y && slot.pos.z == blockCoord.z) {
            int voxelIdx = local.x + local.y * SDF_BLOCK_SIZE + local.z * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);

            //printf("---------------\n");
            //printf("j : %d, slot.ptr : %d\n", j, slot.ptr);
            //printf("worldPos : %f %f %f\n", worldPos.x, worldPos.y, worldPos.z);
            //printf("blockCoord : %d %d %d\n", blockCoord.x, blockCoord.y, blockCoord.z);
            //printf("hp : %d\n", hp);
            //printf("sdf : %f\n", d_SDFBlocks[slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE) + voxelIdx].sdf);
            //printf("weight : %f\n", d_SDFBlocks[slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE) + voxelIdx].weight);



            //printf("---------------\n");

            //printf("%f %f %f\n", worldPos.x, worldPos.y, worldPos.z);




            return d_SDFBlocks[slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE) + voxelIdx];
        }
    }
    
    // No global linear probe: if not found in bucket, treat as empty (fast fail)
    
    // Return empty voxel if not found
    VoxelData empty;
    empty.sdf = 0.0f;
    empty.weight = 0;
    empty.color.x = 0;
    empty.color.y = 0;
    empty.color.z = 0;
    return empty;
}

/**
 * Trilinear interpolation for SDF sampling (from expert code)
 * Samples 8 surrounding voxels and interpolates SDF and color values
 */
__device__ bool trilinearInterpolationSimpleFastFast(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    const float3& worldPos,
    float& dist,
    uchar3& color,
    float voxelSize
) {
    // Calculate dual grid position (shifted by half voxel)
    const float offset = voxelSize;
    const float3 posDual = make_float3(
        worldPos.x - offset / 2.0f,
        worldPos.y - offset / 2.0f,
        worldPos.z - offset / 2.0f
    );
    
    // Convert to voxel coordinates
    float3 voxelPosFloat = make_float3(
        posDual.x / voxelSize,
        posDual.y / voxelSize,
        posDual.z / voxelSize
    );
    
    // Calculate fractional weights (helper defined outside; cannot define nested functions in CUDA)
    float3 weight = make_float3(
        frac01(voxelPosFloat.x),
        frac01(voxelPosFloat.y),
        frac01(voxelPosFloat.z)
    );
    
    dist = 0.0f;
    float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
    float validWeightSum = 0.0f;
    
    // Sample 8 surrounding voxels
    for (int i = 0; i < 8; i++) {
        float addX = ((i & 1) ? offset : 0.0f);
        float addY = ((i & 2) ? offset : 0.0f);
        float addZ = ((i & 4) ? offset : 0.0f);
        float3 offsetPos = make_float3(
            posDual.x + addX,
            posDual.y + addY,
            posDual.z + addZ
        );

        //printf("%f %f %f\n", dist, offsetPos.x, offsetPos.y, offsetPos.z);


        // Get voxel at this position (reverted)
        VoxelData voxel = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, offsetPos, voxelSize);
        if (voxel.weight == 0) {
            continue;
        }



        float3 vColor = make_float3(voxel.color.x, voxel.color.y, voxel.color.z);
        
        // Calculate weight for this corner
        float w = ((i & 1) ? weight.x : (1.0f - weight.x)) *
                  ((i & 2) ? weight.y : (1.0f - weight.y)) *
                  ((i & 4) ? weight.z : (1.0f - weight.z));
        
        dist += w * voxel.sdf;
        colorFloat.x += w * vColor.x;
        colorFloat.y += w * vColor.y;
        colorFloat.z += w * vColor.z;
        validWeightSum += w;
    }
    
    if (validWeightSum <= 1e-6f) {
        return false;
    }
    
    float invWeight = 1.0f / validWeightSum;
    dist *= invWeight;
    colorFloat.x *= invWeight;
    colorFloat.y *= invWeight;
    colorFloat.z *= invWeight;
    
    color = make_uchar3((unsigned char)colorFloat.x, (unsigned char)colorFloat.y, (unsigned char)colorFloat.z);

    //printf("dist : %f, validWeightSum : %f\n", dist, validWeightSum);

    return true;
}

/**
 * Linear intersection finding
 */
__device__ float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) {
    return tNear + (dNear / (dNear - dFar)) * (tFar - tNear);
}

/**
 * Bisection method for finding zero-crossing (from expert code)
 */
static const unsigned int nIterationsBisection = 3;

__device__ bool findIntersectionBisection(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    const float3& worldCamPos,
    const float3& worldDir,
    float d0, float r0,
    float d1, float r1,
    float& alpha,
    uchar3& color,
    float voxelSize
) {
    float a = r0; float aDist = d0;
    float b = r1; float bDist = d1;
    float c = 0.0f;
    
    #pragma unroll 1
    for (unsigned int i = 0; i < nIterationsBisection; i++) {
        c = findIntersectionLinear(a, b, aDist, bDist);
        
        float cDist;
        float3 samplePos = make_float3(
            worldCamPos.x + c * worldDir.x,
            worldCamPos.y + c * worldDir.y,
            worldCamPos.z + c * worldDir.z
        );
        if (!trilinearInterpolationSimpleFastFast(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, 
                                                 samplePos, cDist, color, voxelSize)) {
            return false;
        }
        
        if (aDist * cDist > 0.0f) {
            a = c; aDist = cDist;
        } else {
            b = c; bDist = cDist;
        }
    }
    
    alpha = c;
    return true;
}

/**
 * Traverse coarse grid and sample SDF (expert code algorithm)
 */
__device__ void traverseCoarseGridSimpleSampleAll(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    const float3& worldCamPos,
    const float3& worldDir,
    const float3& camDir,
    const int3& dTid,
    float minInterval, float maxInterval,
    float* d_depth,
    float4* d_depth4,
    float4* d_colors,
    float4* d_normals,
    int width, int height,
    float minDepth, float maxDepth,
    float voxelSize,
    int* d_rayHitCount,
    int* d_rayMissCount
) {
    // Last Sample
    float lastSdf = 0.0f;
    float lastAlpha = 0.0f;
    unsigned int lastWeight = 0;
    
    const float depthToRayLength = 1.0f / camDir.z; // scale factor to convert from depth to ray length
    
    float rayCurrent = depthToRayLength * fmaxf(minDepth, minInterval);  // Convert depth to raylength
    float rayEnd = depthToRayLength * fminf(maxDepth, maxInterval);      // Convert depth to raylength
    
    #pragma unroll 1
    while (rayCurrent < rayEnd) {
        float3 currentPosWorld = make_float3(
            worldCamPos.x + rayCurrent * worldDir.x,
            worldCamPos.y + rayCurrent * worldDir.y,
            worldCamPos.z + rayCurrent * worldDir.z
        );
        float dist;
        uchar3 color;

        //printf("%f %f %f\n", currentPosWorld.x, currentPosWorld.y, currentPosWorld.z);
        
        if (trilinearInterpolationSimpleFastFast(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks,
                                                currentPosWorld, dist, color, voxelSize)) {
            if (lastWeight > 0 && lastSdf < 0.0f && dist > 0.0f) {

                //printf("-----------------------------\n");
                //printf("-----------change---------\n");
                //printf("%f %f %f\n", currentPosWorld.x, currentPosWorld.y, currentPosWorld.z);




                float alpha;
                uchar3 color2;
                bool b = findIntersectionBisection(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks,
                                                   worldCamPos, worldDir, lastSdf, lastAlpha,
                                                   dist, rayCurrent, alpha, color2, voxelSize);
                
                float3 currentIso = make_float3(
                    worldCamPos.x + alpha * worldDir.x,
                    worldCamPos.y + alpha * worldDir.y,
                    worldCamPos.z + alpha * worldDir.z
                );

                //if (b == false) {

                //    printf("pixel id : %d\n", dTid.y * width + dTid.x);

                //}
                //else {
                //    printf("b : %d\n", b);
                //    printf("dist gap : %f\n", fabsf(lastSdf - dist));
                //    printf("depth : %f\n", alpha / depthToRayLength);
                //    printf("current pos : %f %f %f\n", currentPosWorld.x, currentPosWorld.y, currentPosWorld.z);
                //    printf("depth pos : %f %f %f\n", currentIso.x, currentIso.y, currentIso.z);
                //    printf("-----------------------------\n");
                //}

                
                // TODO: Check thresholds and write results
                // For now, just store the result


                //float th = 0.05f;
                float th = voxelSize * 3;
                
                //if(b)
                //{
                //    printf("[RayCast] voxelSize=%.4f, threshold=%.4f, dist : %f\n", voxelSize, th, dist);
                //}
                
                
                if (b && fabsf(lastSdf - dist) < th) {
                    if (fabsf(dist) < th) {
                        float depth = alpha / depthToRayLength; // Convert ray length to depth
                        
                        int pixelIdx = dTid.y * width + dTid.x;
                        d_depth[pixelIdx] = depth;
                        d_depth4[pixelIdx] = make_float4(currentIso.x, currentIso.y, currentIso.z, 1.0f);
                        d_colors[pixelIdx] = make_float4(color2.x / 255.0f, color2.y / 255.0f, color2.z / 255.0f, 1.0f);
                        
                        // Count successful surface hit
                        atomicAdd(d_rayHitCount, 1);
                        return;
                    }
                }
            }
            
            lastSdf = dist;
            lastAlpha = rayCurrent;
            lastWeight = 1;
            rayCurrent += voxelSize * 0.5f; // rayIncrement
        } else {
            lastWeight = 0;
            rayCurrent += voxelSize * 0.5f;
        }
    }
    
    // Count missed surface (ray end reached without finding surface)
    atomicAdd(d_rayMissCount, 1);
}

// ============================================================================
// CUDA Kernels (executed on GPU)
// ============================================================================

/**
 * Ray casting kernel for rendering (expert code algorithm)
 */
__global__ void renderKernel(
    HashSlot* d_hashTable,
    VoxelData* d_SDFBlocks,
    int numActiveBlocks,
    float* d_depth,
    float4* d_depth4,
    float4* d_colors,
    float4* d_normals,
    int width,
    int height,
    float3 cameraPos,
    float* cameraTransform,
    float fx, float fy, float cx, float cy,
    float minDepth,
    float maxDepth,
    float voxelSize,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    int* d_rayHitCount,
    int* d_rayMissCount
) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        const float MINF = 1e10f; // Large invalid value
        
        int pixelIdx = y * width + x;

        //if (pixelIdx != 120200) {
        //    return;
        //}

        //printf("x, y : %d, %d\n", x, y);
        
        // Initialize output to invalid
        d_depth[pixelIdx] = MINF;
        d_depth4[pixelIdx] = make_float4(MINF, MINF, MINF, MINF);
        d_colors[pixelIdx] = make_float4(MINF, MINF, MINF, MINF);
        d_normals[pixelIdx] = make_float4(MINF, MINF, MINF, MINF);
        
        // Calculate ray direction from camera (like expert code)
        float3 camDir = make_float3(
            (x - cx) / fx,
            (y - cy) / fy,
            1.0f
        );

        float camDirLength = sqrtf(camDir.x * camDir.x + camDir.y * camDir.y + camDir.z * camDir.z);
        if (camDirLength > 1e-6f) {
            camDir.x /= camDirLength;
            camDir.y /= camDirLength;
            camDir.z /= camDirLength;
        }

        //float3 worldPos = make_float3(
        //    cameraTransform[0] * cameraPos_local.x + cameraTransform[1] * cameraPos_local.y + cameraTransform[2] * cameraPos_local.z + cameraTransform[3],
        //    cameraTransform[4] * cameraPos_local.x + cameraTransform[5] * cameraPos_local.y + cameraTransform[6] * cameraPos_local.z + cameraTransform[7],
        //    cameraTransform[8] * cameraPos_local.x + cameraTransform[9] * cameraPos_local.y + cameraTransform[10] * cameraPos_local.z + cameraTransform[11]
        //);
        
        // Transform ray direction to world space
        //float3 worldDir = make_float3(
        //    cameraTransform[0] * camDir.x + cameraTransform[4] * camDir.y + cameraTransform[8] * camDir.z,
        //    cameraTransform[1] * camDir.x + cameraTransform[5] * camDir.y + cameraTransform[9] * camDir.z,
        //    cameraTransform[2] * camDir.x + cameraTransform[6] * camDir.y + cameraTransform[10] * camDir.z
        //);

        float3 worldDir = make_float3(
            cameraTransform[0] * camDir.x + cameraTransform[1] * camDir.y + cameraTransform[2] * camDir.z,
            cameraTransform[4] * camDir.x + cameraTransform[5] * camDir.y + cameraTransform[6] * camDir.z,
            cameraTransform[8] * camDir.x + cameraTransform[9] * camDir.y + cameraTransform[10] * camDir.z
        );
        
        float worldDirLength = sqrtf(worldDir.x * worldDir.x + worldDir.y * worldDir.y + worldDir.z * worldDir.z);
        if (worldDirLength > 1e-6f) {
            worldDir.x /= worldDirLength;
            worldDir.y /= worldDirLength;
            worldDir.z /= worldDirLength;
        }
        
        // World camera position
        float3 worldCamPos = cameraPos;
        //printf("fx: %f, fy: %f, Cx: %f, Cy: %f\n", fx, fy, cx, cy);
        //printf("cam pos world : %f %f %f\n", worldCamPos.x, worldCamPos.y, worldCamPos.z);

        //for (int i = 0; i < 30; i++) {
        //      
        //    printf("%f %f %f\n", worldCamPos.x + 100 * i* worldDir.x, worldCamPos.y + 100 * i * worldDir.y, worldCamPos.z + 100 * i * worldDir.z);
   
        //}
        
        // Ray intervals
        float minInterval = minDepth;
        float maxInterval = maxDepth;
        
        if (minInterval == 0 || minInterval == MINF) return;
        if (maxInterval == 0 || maxInterval == MINF) return;
        
        // Traverse grid (expert code algorithm)
        traverseCoarseGridSimpleSampleAll(
            d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks,
            worldCamPos, worldDir, camDir,
            make_int3(x, y, 1), minInterval, maxInterval,
            d_depth, d_depth4, d_colors, d_normals,
            width, height, minDepth, maxDepth, voxelSize,
            d_rayHitCount, d_rayMissCount
        );
    }
}

// ============================================================================
// Host Functions (executed on CPU, call kernels)
// ============================================================================

/**
 * Ray cast rendering host function
 */
extern "C" void rayCastRenderCUDA(
    CUDAHashRef& hashData,
    int width,
    int height,
    float* d_depth,
    float4* d_depth4,
    float4* d_colors,
    float4* d_normals,
    float3 cameraPos,
    float* cameraTransform,
    float fx, float fy, float cx, float cy,
    float minDepth,
    float maxDepth,
    float voxelSize,
    int numBuckets,
    int bucketSize,
    int totalHashSize
) {
    printf("rayCastRenderCUDA: Starting ray casting...\n");
    printf("  Image size: %dx%d\n", width, height);
    printf("  Depth range: [%.3f, %.3f]\n", minDepth, maxDepth);
    
    // Get number of active blocks
    int numActiveBlocks = 0;
    cudaMemcpy(&numActiveBlocks, hashData.d_hashCompactifiedCounter,
               sizeof(int), cudaMemcpyDeviceToHost);
    
    if (numActiveBlocks == 0) {
        printf("  No active blocks to render!\n");
        return;
    }
    
    printf("  Active blocks: %d\n", numActiveBlocks);
    
    // Allocate debug counters
    int* d_rayHitCount = nullptr;
    int* d_rayMissCount = nullptr;
    cudaMalloc(&d_rayHitCount, sizeof(int));
    cudaMalloc(&d_rayMissCount, sizeof(int));
    cudaMemset(d_rayHitCount, 0, sizeof(int));
    cudaMemset(d_rayMissCount, 0, sizeof(int));
    
    // Set up grid and block dimensions
    const int T_PER_BLOCK = 8; // Like expert code
    dim3 gridSize((width + T_PER_BLOCK - 1) / T_PER_BLOCK,
                  (height + T_PER_BLOCK - 1) / T_PER_BLOCK);
    dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);
    
    // Launch ray casting kernel
    renderKernel << <gridSize, blockSize >> > (
        hashData.d_hashTable,
        hashData.d_SDFBlocks,
        numActiveBlocks,
        d_depth,
        d_depth4,
        d_colors,
        d_normals,
        width,
        height,
        cameraPos,
        cameraTransform,
        fx, fy, cx, cy,
        minDepth,
        maxDepth,
        voxelSize,
        numBuckets,
        bucketSize,
        totalHashSize,
        d_rayHitCount,
        d_rayMissCount
    );
    
    // Check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA ray cast kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_rayHitCount);
        cudaFree(d_rayMissCount);
        return;
    }
    
    // Read back debug counters
    int rayHitCount = 0, rayMissCount = 0;
    cudaMemcpy(&rayHitCount, d_rayHitCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&rayMissCount, d_rayMissCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("rayCastRenderCUDA: Ray casting completed!\n");
    printf("  Pixels with surface found: %d (%.1f%%)\n", rayHitCount, 
           (width * height > 0) ? (float)rayHitCount / (width * height) * 100.0f : 0.0f);
    printf("  Pixels without surface: %d (%.1f%%)\n", rayMissCount,
           (width * height > 0) ? (float)rayMissCount / (width * height) * 100.0f : 0.0f);
    
    cudaFree(d_rayHitCount);
    cudaFree(d_rayMissCount);
}

// ============================================================================
// Voxelwise Surface Point Extraction
// ============================================================================

/**
 * Compute normal from SDF gradient (finite difference)
 */
__device__ float3 computeNormalFromSDF(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    const float3& worldPos,
    float voxelSize
) {
    const float eps = voxelSize * 0.5f;
    
    float3 grad = make_float3(0.0f, 0.0f, 0.0f);
    
    // X gradient
    float3 posX = make_float3(worldPos.x + eps, worldPos.y, worldPos.z);
    float3 posXNeg = make_float3(worldPos.x - eps, worldPos.y, worldPos.z);
    VoxelData voxX = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posX, voxelSize);
    VoxelData voxXNeg = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posXNeg, voxelSize);
    if (voxX.weight > 0 && voxXNeg.weight > 0) {
        grad.x = (voxX.sdf - voxXNeg.sdf) / (2.0f * eps);
    }
    
    // Y gradient
    float3 posY = make_float3(worldPos.x, worldPos.y + eps, worldPos.z);
    float3 posYNeg = make_float3(worldPos.x, worldPos.y - eps, worldPos.z);
    VoxelData voxY = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posY, voxelSize);
    VoxelData voxYNeg = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posYNeg, voxelSize);
    if (voxY.weight > 0 && voxYNeg.weight > 0) {
        grad.y = (voxY.sdf - voxYNeg.sdf) / (2.0f * eps);
    }
    
    // Z gradient
    float3 posZ = make_float3(worldPos.x, worldPos.y, worldPos.z + eps);
    float3 posZNeg = make_float3(worldPos.x, worldPos.y, worldPos.z - eps);
    VoxelData voxZ = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posZ, voxelSize);
    VoxelData voxZNeg = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize, d_SDFBlocks, posZNeg, voxelSize);
    if (voxZ.weight > 0 && voxZNeg.weight > 0) {
        grad.z = (voxZ.sdf - voxZNeg.sdf) / (2.0f * eps);
    }
    
    // Normalize gradient to get normal
    float len = sqrtf(grad.x * grad.x + grad.y * grad.y + grad.z * grad.z);
    if (len > 1e-6f) {
        grad.x /= len;
        grad.y /= len;
        grad.z /= len;
    }
    
    return grad;
}

/**
 * Kernel: Extract surface points from voxel grid
 * Each thread processes one hash slot
 */
__global__ void kernelExtractSurfacePoints(
    const HashSlot* d_hashTable,
    int numBuckets,
    int bucketSize,
    int totalHashSize,
    VoxelData* d_SDFBlocks,
    float voxelSize,
    float minSDF,
    float maxSDF,
    unsigned int minWeight,
    float4* d_outputPositions,
    float4* d_outputColors,
    float4* d_outputNormals,
    int maxPoints,
    int* d_pointCount
) {
    int slotIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (slotIdx >= totalHashSize) return;
    
    const HashSlot& slot = d_hashTable[slotIdx];
    
    // Skip empty slots
    if (slot.ptr == -1) return;
    
    // Get block position (SDF block coordinates)
    int3 blockPos = slot.pos;
    
    // Calculate block's first voxel corner position in world space
    // blockWorldPos is the corner of the first voxel (0,0,0) in this block
    float3 blockWorldPos = make_float3(
        blockPos.x * SDF_BLOCK_SIZE * voxelSize,
        blockPos.y * SDF_BLOCK_SIZE * voxelSize,
        blockPos.z * SDF_BLOCK_SIZE * voxelSize
    );
    
    // Iterate through all 8x8x8 = 512 voxels in this SDF block
    for (int z = 0; z < SDF_BLOCK_SIZE; z++) {
        for (int y = 0; y < SDF_BLOCK_SIZE; y++) {
            for (int x = 0; x < SDF_BLOCK_SIZE; x++) {
                // Calculate local voxel index within block (0-511)
                int voxelIdx = x + y * SDF_BLOCK_SIZE + z * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
                
                // Calculate global voxel index in d_SDFBlocks array
                // slot.ptr points to the start of this block's voxel data
                // Each block has SDF_BLOCK_SIZE^3 = 512 voxels
                int globalVoxelIdx = slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE) + voxelIdx;
                
                // Get voxel data from global array
                VoxelData voxel = d_SDFBlocks[globalVoxelIdx];
                
                // Skip invalid voxels or voxels with weight below threshold
                if (voxel.weight == 0 || voxel.weight < minWeight) continue;
                
                // Check if SDF is within surface range (near zero)
                // Use wider range to capture more surface points
                if (voxel.sdf < minSDF || voxel.sdf > maxSDF) continue;
                
                // Calculate voxel center position in world space
                float3 voxelCenter = make_float3(
                    blockWorldPos.x + (x + 0.5f) * voxelSize,
                    blockWorldPos.y + (y + 0.5f) * voxelSize,
                    blockWorldPos.z + (z + 0.5f) * voxelSize
                );
                
                // Find iso-surface position using SDF values only (no normal needed)
                // Check 6 neighbors to find zero-crossing
                float3 isoPos = make_float3(0.0f, 0.0f, 0.0f);
                float3 accumulatedPos = make_float3(0.0f, 0.0f, 0.0f);
                int validNeighbors = 0;
                
                // Check 6 neighbors (x+, x-, y+, y-, z+, z-)
                int offsets[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};
                float3 directions[6] = {
                    make_float3(1.0f, 0.0f, 0.0f),
                    make_float3(-1.0f, 0.0f, 0.0f),
                    make_float3(0.0f, 1.0f, 0.0f),
                    make_float3(0.0f, -1.0f, 0.0f),
                    make_float3(0.0f, 0.0f, 1.0f),
                    make_float3(0.0f, 0.0f, -1.0f)
                };
                
                for (int dir = 0; dir < 6; dir++) {
                    int nx = x + offsets[dir][0];
                    int ny = y + offsets[dir][1];
                    int nz = z + offsets[dir][2];
                    
                    // Check if neighbor is within block bounds
                    if (nx < 0 || nx >= SDF_BLOCK_SIZE ||
                        ny < 0 || ny >= SDF_BLOCK_SIZE ||
                        nz < 0 || nz >= SDF_BLOCK_SIZE) {
                        // Neighbor is in different block - need to query via hash table
                        float3 neighborPos = make_float3(
                            voxelCenter.x + directions[dir].x * voxelSize,
                            voxelCenter.y + directions[dir].y * voxelSize,
                            voxelCenter.z + directions[dir].z * voxelSize
                        );
                        VoxelData neighborVoxel = getVoxel(d_hashTable, numBuckets, bucketSize, totalHashSize,
                                                           d_SDFBlocks, neighborPos, voxelSize);
                        if (neighborVoxel.weight == 0) continue;
                        
                        // Check for zero-crossing: different signs
                        if ((voxel.sdf < 0.0f && neighborVoxel.sdf > 0.0f) ||
                            (voxel.sdf > 0.0f && neighborVoxel.sdf < 0.0f)) {
                            // Linear interpolation to find SDF=0 position
                            float t = voxel.sdf / (voxel.sdf - neighborVoxel.sdf);
                            float3 zeroPos = make_float3(
                                voxelCenter.x + directions[dir].x * voxelSize * t,
                                voxelCenter.y + directions[dir].y * voxelSize * t,
                                voxelCenter.z + directions[dir].z * voxelSize * t
                            );
                            accumulatedPos.x += zeroPos.x;
                            accumulatedPos.y += zeroPos.y;
                            accumulatedPos.z += zeroPos.z;
                            validNeighbors++;
                        }
                    } else {
                        // Neighbor is in same block
                        int neighborVoxelIdx = nx + ny * SDF_BLOCK_SIZE + nz * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
                        int neighborGlobalIdx = slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE) + neighborVoxelIdx;
                        VoxelData neighborVoxel = d_SDFBlocks[neighborGlobalIdx];
                        if (neighborVoxel.weight == 0) continue;
                        
                        // Check for zero-crossing: different signs
                        if ((voxel.sdf < 0.0f && neighborVoxel.sdf > 0.0f) ||
                            (voxel.sdf > 0.0f && neighborVoxel.sdf < 0.0f)) {
                            // Linear interpolation to find SDF=0 position
                            float t = voxel.sdf / (voxel.sdf - neighborVoxel.sdf);
                            float3 zeroPos = make_float3(
                                voxelCenter.x + directions[dir].x * voxelSize * t,
                                voxelCenter.y + directions[dir].y * voxelSize * t,
                                voxelCenter.z + directions[dir].z * voxelSize * t
                            );
                            accumulatedPos.x += zeroPos.x;
                            accumulatedPos.y += zeroPos.y;
                            accumulatedPos.z += zeroPos.z;
                            validNeighbors++;
                        }
                    }
                }
                
                // If no zero-crossing found, use voxel center if SDF is very close to zero
                // Also allow points even if zero-crossing is not found but SDF is within range
                if (validNeighbors == 0) {
                    if (fabsf(voxel.sdf) < 0.001f) {

                    //if (fabsf(voxel.sdf) <= fmaxf(fabsf(minSDF), fabsf(maxSDF))) {
                        isoPos = voxelCenter;
                    } else {
                        continue; // No surface in this voxel
                    }
                } else {
                    // Average of all zero-crossing positions
                    isoPos.x = accumulatedPos.x / validNeighbors;
                    isoPos.y = accumulatedPos.y / validNeighbors;
                    isoPos.z = accumulatedPos.z / validNeighbors;
                }
                
                // Compute normal from SDF gradient (for output, not for position calculation)
                float3 normal = computeNormalFromSDF(
                    d_hashTable, numBuckets, bucketSize, totalHashSize,
                    d_SDFBlocks, isoPos, voxelSize
                );
                
                // Atomically get next output index
                int outputIdx = atomicAdd(d_pointCount, 1);
                
                if (outputIdx >= maxPoints) {
                    // Buffer full, skip this point
                    return;
                }
                
                // Write output
                d_outputPositions[outputIdx] = make_float4(isoPos.x, isoPos.y, isoPos.z, 1.0f);
                d_outputColors[outputIdx] = make_float4(
                    (float)voxel.color.x / 255.0f,
                    (float)voxel.color.y / 255.0f,
                    (float)voxel.color.z / 255.0f,
                    1.0f
                );
                d_outputNormals[outputIdx] = make_float4(normal.x, normal.y, normal.z, 0.0f);
            }
        }
    }
}

/**
 * Host function: Extract surface points from voxel grid
 */
extern "C" void extractSurfacePointsCUDA(
    const CUDAHashRef& hashData,
    const Params& params,
    float minSDF,
    float maxSDF,
    unsigned int minWeight,
    float4* d_outputPositions,
    float4* d_outputColors,
    float4* d_outputNormals,
    int maxPoints,
    int* d_pointCount
) {
    // Reset point count
    cudaMemset(d_pointCount, 0, sizeof(int));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int numBlocks = (params.totalHashSize + threadsPerBlock - 1) / threadsPerBlock;
    
    kernelExtractSurfacePoints<<<numBlocks, threadsPerBlock>>>(
        hashData.d_hashTable,
        params.hashSlotNum,
        params.slotSize,
        params.totalHashSize,
        hashData.d_SDFBlocks,
        params.voxelSize,
        minSDF,
        maxSDF,
        minWeight,
        d_outputPositions,
        d_outputColors,
        d_outputNormals,
        maxPoints,
        d_pointCount
    );
    
    cudaDeviceSynchronize();
}

