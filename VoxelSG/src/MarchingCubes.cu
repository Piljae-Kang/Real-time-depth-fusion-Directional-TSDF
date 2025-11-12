#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/MarchingCubes.h"
#include <cstdio>

// ----------------------------------------------------------------------------
// Marching cubes lookup tables (host-side constants)
// ----------------------------------------------------------------------------
namespace {

// Standard edge table for marching cubes (256 entries)
static const int MC_EDGE_TABLE_HOST[256] = {
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

// Triangle table (256 x 16 entries, -1 terminates)
static const int MC_TRI_TABLE_HOST[256][16] = {
    {-1},
    {0, 8, 3, -1},
    {0, 1, 9, -1},
    {1, 8, 3, 9, 8, 1, -1},
    {1, 2, 10, -1},
    {0, 8, 3, 1, 2, 10, -1},
    {9, 2, 10, 0, 2, 9, -1},
    {2, 8, 3, 2, 10, 8, 10, 9, 8, -1},
    {3, 11, 2, -1},
    {0, 11, 2, 8, 11, 0, -1},
    {1, 9, 0, 2, 3, 11, -1},
    {1, 11, 2, 1, 9, 11, 9, 8, 11, -1},
    {3, 10, 1, 11, 10, 3, -1},
    {0, 10, 1, 0, 8, 10, 8, 11, 10, -1},
    {3, 9, 0, 3, 11, 9, 11, 10, 9, -1},
    {9, 8, 10, 10, 8, 11, -1},
    {4, 7, 8, -1},
    {4, 3, 0, 7, 3, 4, -1},
    {0, 1, 9, 8, 4, 7, -1},
    {4, 1, 9, 4, 7, 1, 7, 3, 1, -1},
    {1, 2, 10, 8, 4, 7, -1},
    {3, 4, 7, 3, 0, 4, 1, 2, 10, -1},
    {9, 2, 10, 9, 0, 2, 8, 4, 7, -1},
    {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1},
    {8, 4, 7, 3, 11, 2, -1},
    {11, 4, 7, 11, 2, 4, 2, 0, 4, -1},
    {9, 0, 1, 8, 4, 7, 2, 3, 11, -1},
    {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1},
    {3, 10, 1, 3, 11, 10, 7, 8, 4, -1},
    {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1},
    {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1},
    {4, 7, 11, 4, 11, 9, 9, 11, 10, -1},
    {9, 5, 4, -1},
    {9, 5, 4, 0, 8, 3, -1},
    {0, 5, 4, 1, 5, 0, -1},
    {8, 5, 4, 8, 3, 5, 3, 1, 5, -1},
    {1, 2, 10, 9, 5, 4, -1},
    {3, 0, 8, 1, 2, 10, 4, 9, 5, -1},
    {5, 2, 10, 5, 4, 2, 4, 0, 2, -1},
    {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1},
    {9, 5, 4, 2, 3, 11, -1},
    {0, 11, 2, 0, 8, 11, 4, 9, 5, -1},
    {0, 5, 4, 0, 1, 5, 2, 3, 11, -1},
    {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1},
    {10, 3, 11, 10, 1, 3, 9, 5, 4, -1},
    {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1},
    {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1},
    {5, 4, 8, 5, 8, 10, 10, 8, 11, -1},
    {9, 7, 8, 5, 7, 9, -1},
    {9, 3, 0, 9, 5, 3, 5, 7, 3, -1},
    {0, 7, 8, 0, 1, 7, 1, 5, 7, -1},
    {1, 5, 3, 3, 5, 7, -1},
    {9, 7, 8, 9, 5, 7, 10, 1, 2, -1},
    {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1},
    {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1},
    {2, 10, 5, 2, 5, 3, 3, 5, 7, -1},
    {7, 9, 5, 7, 8, 9, 3, 11, 2, -1},
    {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1},
    {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1},
    {11, 2, 1, 11, 1, 7, 7, 1, 5, -1},
    {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1},
    {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
    {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
    {11, 10, 5, 7, 11, 5, -1},
    {10, 6, 5, -1},
    {0, 8, 3, 5, 10, 6, -1},
    {9, 0, 1, 5, 10, 6, -1},
    {1, 8, 3, 1, 9, 8, 5, 10, 6, -1},
    {1, 6, 5, 2, 6, 1, -1},
    {1, 6, 5, 1, 2, 6, 3, 0, 8, -1},
    {9, 6, 5, 9, 0, 6, 0, 2, 6, -1},
    {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1},
    {2, 3, 11, 10, 6, 5, -1},
    {11, 0, 8, 11, 2, 0, 10, 6, 5, -1},
    {0, 1, 9, 2, 3, 11, 5, 10, 6, -1},
    {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1},
    {6, 3, 11, 6, 5, 3, 5, 1, 3, -1},
    {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1},
    {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1},
    {6, 5, 9, 6, 9, 11, 11, 9, 8, -1},
    {5, 10, 6, 4, 7, 8, -1},
    {4, 3, 0, 4, 7, 3, 6, 5, 10, -1},
    {1, 9, 0, 5, 10, 6, 8, 4, 7, -1},
    {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1},
    {6, 1, 2, 6, 5, 1, 4, 7, 8, -1},
    {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1},
    {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1},
    {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
    {3, 11, 2, 7, 8, 4, 10, 6, 5, -1},
    {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1},
    {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1},
    {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
    {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1},
    {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
    {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
    {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1},
    {10, 4, 9, 6, 4, 10, -1},
    {4, 10, 6, 4, 9, 10, 0, 8, 3, -1},
    {10, 0, 1, 10, 6, 0, 6, 4, 0, -1},
    {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1},
    {1, 4, 9, 1, 2, 4, 2, 6, 4, -1},
    {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1},
    {0, 2, 4, 4, 2, 6, -1},
    {8, 3, 2, 8, 2, 4, 4, 2, 6, -1},
    {10, 4, 9, 10, 6, 4, 11, 2, 3, -1},
    {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1},
    {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1},
    {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
    {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1},
    {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
    {3, 11, 6, 3, 6, 0, 0, 6, 4, -1},
    {6, 4, 8, 11, 6, 8, -1},
    {7, 10, 6, 7, 8, 10, 8, 9, 10, -1},
    {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1},
    {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1},
    {10, 6, 7, 10, 7, 1, 1, 7, 3, -1},
    {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1},
    {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
    {7, 8, 0, 7, 0, 6, 6, 0, 2, -1},
    {7, 3, 2, 6, 7, 2, -1},
    {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1},
    {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
    {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
    {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1},
    {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
    {0, 9, 1, 11, 6, 7, -1},
    {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1},
    {7, 11, 6, -1},
    {7, 6, 11, -1},
    {3, 0, 8, 11, 7, 6, -1},
    {0, 1, 9, 11, 7, 6, -1},
    {8, 1, 9, 8, 3, 1, 11, 7, 6, -1},
    {10, 1, 2, 6, 11, 7, -1},
    {1, 2, 10, 3, 0, 8, 6, 11, 7, -1},
    {2, 9, 0, 2, 10, 9, 6, 11, 7, -1},
    {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1},
    {7, 2, 3, 6, 2, 7, -1},
    {7, 0, 8, 7, 6, 0, 6, 2, 0, -1},
    {2, 7, 6, 2, 3, 7, 0, 1, 9, -1},
    {1, 9, 8, 1, 8, 2, 2, 8, 7, 6, 2, 7, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 1, 8, 1, 3, -1},
    {7, 6, 10, 7, 10, 0, 7, 0, 8, 0, 10, 1, -1},
    {3, 7, 6, 3, 6, 0, 0, 6, 10, 0, 10, 9, -1},
    {7, 6, 10, 7, 10, 8, 8, 10, 9, -1},
    {6, 8, 4, 11, 8, 6, -1},
    {3, 6, 11, 3, 0, 6, 0, 4, 6, -1},
    {8, 6, 11, 8, 4, 6, 9, 0, 1, -1},
    {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1},
    {6, 8, 4, 6, 11, 8, 2, 10, 1, -1},
    {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1},
    {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1},
    {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
    {8, 2, 3, 8, 4, 2, 4, 6, 2, -1},
    {0, 4, 2, 4, 6, 2, -1},
    {1, 9, 0, 2, 3, 8, 2, 8, 4, 2, 4, 6, -1},
    {1, 9, 4, 1, 4, 2, 2, 4, 6, -1},
    {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1},
    {10, 1, 0, 10, 0, 6, 6, 0, 4, -1},
    {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
    {10, 9, 4, 6, 10, 4, -1},
    {4, 9, 5, 7, 6, 11, -1},
    {0, 8, 3, 4, 9, 5, 11, 7, 6, -1},
    {5, 0, 1, 5, 4, 0, 7, 6, 11, -1},
    {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1},
    {9, 5, 4, 10, 1, 2, 7, 6, 11, -1},
    {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1},
    {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1},
    {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
    {7, 2, 3, 7, 6, 2, 5, 4, 9, -1},
    {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1},
    {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1},
    {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
    {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1},
    {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
    {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
    {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1},
    {6, 9, 5, 6, 11, 9, 11, 8, 9, -1},
    {3, 6, 11, 3, 0, 6, 0, 5, 6, 0, 9, 5, -1},
    {0, 1, 5, 0, 5, 8, 8, 5, 6, 8, 6, 11, -1},
    {6, 11, 3, 6, 3, 5, 5, 3, 1, -1},
    {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1},
    {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
    {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
    {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1},
    {5, 8, 9, 5, 6, 8, 6, 11, 8, -1},
    {5, 0, 9, 5, 6, 0, 6, 11, 0, 3, 0, 11, -1},
    {0, 1, 8, 1, 6, 8, 1, 5, 6, 11, 8, 6, -1},
    {1, 5, 6, 2, 1, 6, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, 3, 8, 0, -1},
    {6, 10, 5, 8, 0, 3, -1},
    {6, 10, 5, 3, 8, 2, 8, 5, 2, 5, 6, 2, -1},
    {2, 3, 11, 10, 5, 6, -1},
    {11, 0, 8, 11, 2, 0, 10, 5, 6, -1},
    {0, 1, 9, 2, 3, 11, 5, 6, 10, -1},
    {5, 6, 10, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1},
    {6, 10, 5, 3, 11, 1, 11, 0, 1, 11, 8, 0, -1},
    {1, 0, 9, 5, 6, 10, -1},
    {0, 5, 6, 0, 6, 3, 3, 6, 11, -1},
    {6, 10, 5, 3, 11, 8, -1},
    {5, 6, 11, 5, 11, 9, 9, 11, 8, -1},
    {0, 5, 9, 0, 6, 5, 0, 11, 6, 0, 3, 11, -1},
    {0, 1, 5, 0, 5, 8, 8, 5, 6, 8, 6, 11, -1},
    {1, 5, 6, 2, 1, 6, -1},
    {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
    {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, 3, 8, 0, -1},
    {0, 3, 8, 5, 6, 10, -1},
    {10, 5, 6, -1},
    {-1}
};

static const int MC_EDGE_CONNECTION_HOST[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7}
};

static const float MC_EDGE_DIRECTION_HOST[12][3] = {
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {-1.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {-1.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f}
};

static const int MC_CORNER_OFFSET_HOST[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
};

} // namespace

// ----------------------------------------------------------------------------
// Constant memory copies of lookup tables
// ----------------------------------------------------------------------------
__device__ __constant__ int MC_EDGE_TABLE_DEVICE[256];
__device__ __constant__ int MC_TRI_TABLE_DEVICE[256][16];
__device__ __constant__ int MC_EDGE_CONNECTION_DEVICE[12][2];
__device__ __constant__ float MC_EDGE_DIRECTION_DEVICE[12][3];
__device__ __constant__ int MC_CORNER_OFFSET_DEVICE[8][3];

// ----------------------------------------------------------------------------
// Utility device helpers
// ----------------------------------------------------------------------------
__device__ inline int voxelIndex(int x, int y, int z) {
    return (z * SDF_BLOCK_SIZE + y) * SDF_BLOCK_SIZE + x;
}

__device__ inline float3 make_float3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 make_float3_scale(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float3 vertexInterp(float iso, const float3& p1, const float3& p2, float valp1, float valp2) {
    float denom = valp2 - valp1;
    if (fabsf(denom) < 1e-6f) {
        return make_float3(
            0.5f * (p1.x + p2.x),
            0.5f * (p1.y + p2.y),
            0.5f * (p1.z + p2.z));
    }
    float mu = (iso - valp1) / denom;
    mu = fminf(fmaxf(mu, 0.0f), 1.0f);
    return make_float3(
        p1.x + mu * (p2.x - p1.x),
        p1.y + mu * (p2.y - p1.y),
        p1.z + mu * (p2.z - p1.z));
}

__device__ float3 computeNormal(const float3& v0, const float3& v1, const float3& v2) {
    float3 a = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 b = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 n = make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
    float len = sqrtf(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-6f) {
        float invLen = 1.0f / len;
        n.x *= invLen;
        n.y *= invLen;
        n.z *= invLen;
    }
    else {
        n = make_float3(0.0f, 0.0f, 0.0f);
    }
    return n;
}

// ----------------------------------------------------------------------------
// Marching cubes kernel
// ----------------------------------------------------------------------------
__global__ void marchingCubesKernel(
    const HashSlot* __restrict__ compactifiedHash,
    const VoxelData* __restrict__ sdfBlocks,
    int numActiveBlocks,
    float voxelSize,
    float isoValue,
    float3* __restrict__ outVertices,
    float3* __restrict__ outNormals,
    int* __restrict__ triangleCounter,
    int maxTriangles)
{
    int blockIdxGlobal = blockIdx.x;
    if (blockIdxGlobal >= numActiveBlocks) return;

    const HashSlot slot = compactifiedHash[blockIdxGlobal];
    if (slot.ptr < 0) return;

    const VoxelData* block = sdfBlocks + slot.ptr * (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);

    int localIdx = threadIdx.x;
    const int voxelsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    if (localIdx >= voxelsPerBlock) return;

    int localX = localIdx % SDF_BLOCK_SIZE;
    int localY = (localIdx / SDF_BLOCK_SIZE) % SDF_BLOCK_SIZE;
    int localZ = localIdx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);

    if (localX >= SDF_BLOCK_SIZE - 1 || localY >= SDF_BLOCK_SIZE - 1 || localZ >= SDF_BLOCK_SIZE - 1) return;

    float sdfValues[8];
    float3 cornerPos[8];

    float blockExtent = static_cast<float>(SDF_BLOCK_SIZE) * voxelSize;
    float3 blockBase = make_float3(
        static_cast<float>(slot.pos.x) * blockExtent,
        static_cast<float>(slot.pos.y) * blockExtent,
        static_cast<float>(slot.pos.z) * blockExtent);

    for (int i = 0; i < 8; ++i) {
        int offsetX = MC_CORNER_OFFSET_DEVICE[i][0];
        int offsetY = MC_CORNER_OFFSET_DEVICE[i][1];
        int offsetZ = MC_CORNER_OFFSET_DEVICE[i][2];
        int vx = localX + offsetX;
        int vy = localY + offsetY;
        int vz = localZ + offsetZ;
        int linearIdx = voxelIndex(vx, vy, vz);
        const VoxelData& voxel = block[linearIdx];
        sdfValues[i] = voxel.sdf;

        cornerPos[i] = make_float3(
            blockBase.x + (static_cast<float>(vx) + 0.5f) * voxelSize,
            blockBase.y + (static_cast<float>(vy) + 0.5f) * voxelSize,
            blockBase.z + (static_cast<float>(vz) + 0.5f) * voxelSize);
    }

    int cubeIndex = 0;
    if (sdfValues[0] < isoValue) cubeIndex |= 1;
    if (sdfValues[1] < isoValue) cubeIndex |= 2;
    if (sdfValues[2] < isoValue) cubeIndex |= 4;
    if (sdfValues[3] < isoValue) cubeIndex |= 8;
    if (sdfValues[4] < isoValue) cubeIndex |= 16;
    if (sdfValues[5] < isoValue) cubeIndex |= 32;
    if (sdfValues[6] < isoValue) cubeIndex |= 64;
    if (sdfValues[7] < isoValue) cubeIndex |= 128;

    int edgeMask = MC_EDGE_TABLE_DEVICE[cubeIndex];
    if (edgeMask == 0) return;

    float3 vertexList[12];

    for (int e = 0; e < 12; ++e) {
        if (!(edgeMask & (1 << e))) continue;

        int c0 = MC_EDGE_CONNECTION_DEVICE[e][0];
        int c1 = MC_EDGE_CONNECTION_DEVICE[e][1];

        vertexList[e] = vertexInterp(isoValue, cornerPos[c0], cornerPos[c1], sdfValues[c0], sdfValues[c1]);
    }

    int triCount = 0;
    while (MC_TRI_TABLE_DEVICE[cubeIndex][triCount * 3] != -1) {
        ++triCount;
    }

    if (triCount == 0) return;

    int triBase = atomicAdd(triangleCounter, triCount);
    if (triBase >= maxTriangles) {
        // Counter overflow; clamp back
        atomicMax(triangleCounter, maxTriangles);
        return;
    }

    for (int t = 0; t < triCount; ++t) {
        if (triBase + t >= maxTriangles) break;

        int idx0 = MC_TRI_TABLE_DEVICE[cubeIndex][t * 3 + 0];
        int idx1 = MC_TRI_TABLE_DEVICE[cubeIndex][t * 3 + 1];
        int idx2 = MC_TRI_TABLE_DEVICE[cubeIndex][t * 3 + 2];

        float3 v0 = vertexList[idx0];
        float3 v1 = vertexList[idx1];
        float3 v2 = vertexList[idx2];

        float3 n = computeNormal(v0, v1, v2);

        int outIdx = (triBase + t) * 3;
        outVertices[outIdx + 0] = v0;
        outVertices[outIdx + 1] = v1;
        outVertices[outIdx + 2] = v2;

        outNormals[outIdx + 0] = n;
        outNormals[outIdx + 1] = n;
        outNormals[outIdx + 2] = n;
    }
}

// ----------------------------------------------------------------------------
// Host-side helpers
// ----------------------------------------------------------------------------
void allocateMarchingCubesOutput(MarchingCubesOutput& output, int maxTriangles) {
    if (maxTriangles <= 0) {
        printf("allocateMarchingCubesOutput: maxTriangles must be positive.\n");
        return;
    }

    freeMarchingCubesOutput(output);

    size_t vertexBytes = static_cast<size_t>(maxTriangles) * 3 * sizeof(float3);
    cudaMalloc(&output.d_vertices, vertexBytes);
    cudaMalloc(&output.d_normals, vertexBytes);
    cudaMalloc(&output.d_triangleCount, sizeof(int));
    cudaMemset(output.d_triangleCount, 0, sizeof(int));

    output.maxTriangles = maxTriangles;
}

void freeMarchingCubesOutput(MarchingCubesOutput& output) {
    if (output.d_vertices) cudaFree(output.d_vertices);
    if (output.d_normals) cudaFree(output.d_normals);
    if (output.d_triangleCount) cudaFree(output.d_triangleCount);

    output.d_vertices = nullptr;
    output.d_normals = nullptr;
    output.d_triangleCount = nullptr;
    output.maxTriangles = 0;
}

int copyMarchingCubesTriangleCount(const MarchingCubesOutput& output) {
    int count = 0;
    if (output.d_triangleCount) {
        cudaMemcpy(&count, output.d_triangleCount, sizeof(int), cudaMemcpyDeviceToHost);
    }
    return count;
}

void runMarchingCubesCUDA(const CUDAHashRef& hashData,
    const Params& params,
    const MarchingCubesParams& paramsMC,
    MarchingCubesOutput& output,
    cudaStream_t stream)
{
    static bool tablesUploaded = false;
    if (!tablesUploaded) {
        cudaMemcpyToSymbol(MC_EDGE_TABLE_DEVICE, MC_EDGE_TABLE_HOST, sizeof(MC_EDGE_TABLE_HOST));
        cudaMemcpyToSymbol(MC_TRI_TABLE_DEVICE, MC_TRI_TABLE_HOST, sizeof(MC_TRI_TABLE_HOST));
        cudaMemcpyToSymbol(MC_EDGE_CONNECTION_DEVICE, MC_EDGE_CONNECTION_HOST, sizeof(MC_EDGE_CONNECTION_HOST));
        cudaMemcpyToSymbol(MC_EDGE_DIRECTION_DEVICE, MC_EDGE_DIRECTION_HOST, sizeof(MC_EDGE_DIRECTION_HOST));
        cudaMemcpyToSymbol(MC_CORNER_OFFSET_DEVICE, MC_CORNER_OFFSET_HOST, sizeof(MC_CORNER_OFFSET_HOST));
        tablesUploaded = true;
    }

    if (hashData.d_CompactifiedHashTable == nullptr || hashData.d_SDFBlocks == nullptr) {
        printf("runMarchingCubesCUDA: Hash data not initialized.\n");
        return;
    }

    if (output.d_vertices == nullptr || output.d_normals == nullptr || output.d_triangleCount == nullptr) {
        printf("runMarchingCubesCUDA: Output buffers not allocated.\n");
        return;
    }

    int numActiveBlocks = 0;
    cudaMemcpy(&numActiveBlocks, hashData.d_hashCompactifiedCounter, sizeof(int), cudaMemcpyDeviceToHost);
    if (numActiveBlocks == 0) {
        cudaMemset(output.d_triangleCount, 0, sizeof(int));
        return;
    }

    cudaMemsetAsync(output.d_triangleCount, 0, sizeof(int), stream);

    float voxelSize = paramsMC.voxelSize > 0.0f ? paramsMC.voxelSize : params.voxelSize;
    float isoValue = paramsMC.isoValue;

    const int voxelsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    dim3 blockDim(voxelsPerBlock, 1, 1);
    dim3 gridDim(numActiveBlocks, 1, 1);

    marchingCubesKernel<<<gridDim, blockDim, 0, stream>>>(
        hashData.d_CompactifiedHashTable,
        hashData.d_SDFBlocks,
        numActiveBlocks,
        voxelSize,
        isoValue,
        output.d_vertices,
        output.d_normals,
        output.d_triangleCount,
        output.maxTriangles);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("runMarchingCubesCUDA: kernel launch failed (%s)\n", cudaGetErrorString(err));
        return;
    }

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("runMarchingCubesCUDA: stream sync failed (%s)\n", cudaGetErrorString(err));
    }
}

