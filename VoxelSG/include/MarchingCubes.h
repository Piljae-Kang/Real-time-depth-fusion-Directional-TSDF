#pragma once

#include <cuda_runtime.h>
#include "CUDAHashRef.h"
#include "CUDAHashData.h"

struct MarchingCubesParams {
    float isoValue;
    float voxelSize;
    int maxTriangles;

    MarchingCubesParams()
        : isoValue(0.0f)
        , voxelSize(1.0f)
        , maxTriangles(0) {}
};

struct MarchingCubesOutput {
    float3* d_vertices;
    float3* d_normals;
    int* d_triangleCount;
    int maxTriangles;

    MarchingCubesOutput()
        : d_vertices(nullptr)
        , d_normals(nullptr)
        , d_triangleCount(nullptr)
        , maxTriangles(0) {}
};

/**
 * Allocate GPU buffers for marching cubes output.
 */
void allocateMarchingCubesOutput(MarchingCubesOutput& output, int maxTriangles);

/**
 * Release GPU buffers associated with marching cubes output.
 */
void freeMarchingCubesOutput(MarchingCubesOutput& output);

/**
 * Run marching cubes over the current hash data using the compactified active block list.
 * The resulting triangle vertices and normals are written to the output buffers.
 *
 * @param hashData Hash data reference (with compactified tables populated)
 * @param params   Marching cubes parameters (iso value, voxel size, etc.)
 * @param output   Output buffers to fill with triangles
 */
void runMarchingCubesCUDA(const CUDAHashRef& hashData,
    const Params& params,
    const MarchingCubesParams& paramsMC,
    MarchingCubesOutput& output,
    cudaStream_t stream = nullptr);

/**
 * Copy the triangle count from GPU to CPU.
 */
int copyMarchingCubesTriangleCount(const MarchingCubesOutput& output);

