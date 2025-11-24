#ifndef BUILD_FOR_CPU
#include "H_MarchingCubes.h"
#include "H_NoiseFilter.h"
#include "H_reg_TestPrint.h" // 테스트 프린트를 할지 여부
#include "HColorEstimation.h"
#include "HMesh.h"
#include "HSerialization.h"
#include <Cuda/CudaHashManager_2.h>
#include <Cuda/cudaHelp/helper_cuda.h>
#include <Cuda/CudaNvtxRange.h>
#include <Device/HDSConst.h>
#include <Device/HDSStruct.h>
#include <Eigen/Core>
#include <qdebug.h>
#include <thrust/system/cuda/memory.h>

#ifndef VOXEL_GRID_MODULE
#include "HVETM.h"
#endif

#ifdef AARON_TEST
#include "HPrimitivePresenter.h"
typedef HPrimitivePresenter HPP;
#include <ctime>
#include <iomanip>
#define USE_CLUSTERING
#endif

#include "HMemoryDebug.h" // 메모리 릭 체크용

#endif // BUILD_FOR_CPU

using namespace std;
using namespace Huvitz;
using namespace MarchingCubesKernel;

//해당 포인트가 카메라 시점에서 어디에 찍혀 있을지 계산하여, 해당 위치의 뎁스맵에 내용을 업데이트한다.
//이렇게 했을 경우 동일 뎁스맵 위치에 뎁스 정보가 누적이 되어 불필요한 오차가 발생하게 되는 것을 최소화 할 수 있다.

//	mscho	@20250313
//#define USING_PROJECTION_DEPTHMAP
//#define SEGMENTATION_COLOR_DEBUG

#ifdef BUILD_FOR_CPU
#ifdef CUDA_MANAGER
#undef CUDA_MANAGER // CUDA_MANAGER should not be used in this file
#endif
#endif

#ifndef BUILD_FOR_CPU
#define kernel_return return
#else
#define kernel_return continue
#endif

void plyFileWrite_Ex(const std::string& filename, int pointSize, const Eigen::Vector3f* points, const Eigen::Vector3f* normals, const unsigned int* colors, const float* alphas = nullptr, const VoxelExtraAttrib* extraAttribs = nullptr, const float* downSampleCnt = nullptr, const Eigen::Vector3f* hsvs = nullptr);
void plyFileWrite_Ex(const std::string& filename, int pointSize, const Eigen::Vector3f* points, const Eigen::Vector3f* normals, const Eigen::Vector3b* colors = nullptr, const float* alphas = nullptr, const VoxelExtraAttrib* extraAttribs = nullptr, const float* downSampleCnt = nullptr, const Eigen::Vector3f* hsvs = nullptr, const int* flag = nullptr);
void plyFileWrite_Ex(const std::string& filename, int pointSize, const Eigen::Vector3f* points, const Eigen::Vector3f* normals, const Eigen::Vector3f* colors, const float* alphas = nullptr, const VoxelExtraAttrib* extraAttribs = nullptr, const float* downSampleCnt = nullptr, const Eigen::Vector3f* hsvs = nullptr);
void plyFileWrite_Ex_deviceData(const std::string& filename, cudaStream_t stream, int pointSize, const Eigen::Vector3f* dev_points, const Eigen::Vector3f* dev_normals = nullptr, const Eigen::Vector3b* dev_colors = nullptr, const float* dev_alphas = nullptr, const VoxelExtraAttrib* dev_extraAttribs = nullptr, const float* dev_downSampleCnt = nullptr, const Eigen::Vector3f* dev_hsvs = nullptr, const int* flag = nullptr);

#ifndef BUILD_FOR_CPU
// cached_allocator  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
cached_allocator::cached_allocator() {
	allocated_blocks_size = 0;
}

cached_allocator::~cached_allocator()
{
	free_all();
}

char* cached_allocator::allocate(std::ptrdiff_t num_bytes)
{
#if DEBUG &&TEST_PRINT
	//std::cout << "cached_allocator::allocate() : num_bytes == " << num_bytes << std::endl;
#endif
	char* result = 0;
	if ((ptrdiff_t)allocated_blocks_size < num_bytes)
	{
		try
		{
			if (allocated_blocks_size != 0)
				thrust::cuda::free(thrust::cuda::pointer<char>(pre_result));
#if DEBUG && TEST_PRINT
			std::cout << "main_allocator::allocate() : allocating new block " << allocated_blocks_size << " -> " << num_bytes << std::endl;
#endif
			result = thrust::cuda::malloc<char>(num_bytes * 1.5).get();
			allocated_blocks_size = num_bytes * 1.5;

			pre_result = result;
		}
		catch (thrust::system_error& e)
		{
			(void)e;
#if DEBUG && TEST_PRINT
			std::cerr << "CUDA error during some_function: " << e.what() << std::endl;
#endif
		}
		catch (std::bad_alloc& e)
		{
			(void)e;
#if DEBUG && TEST_PRINT
			std::cerr << "Bad memory allocation during some_function: " << e.what() << std::endl;
#endif
		}
		catch (std::runtime_error& e)
		{
			(void)e;
#if DEBUG && TEST_PRINT
			std::cerr << "Runtime error during some_function: " << e.what() << std::endl;
#endif
			throw;
		}
	}
	else {
		//std::cout << "main_allocator::allocate() : not allocating " << allocated_blocks_size << " -> " << num_bytes << std::endl;
		result = pre_result;
	}
	return result;
}

void cached_allocator::deallocate(char* ptr, size_t)
{
}

void cached_allocator::free_all()
{
#if DEBUG &&TEST_PRINT
	std::cout << "main_allocator::free_all()" << std::endl;
#endif
	if (allocated_blocks_size > 0)
		thrust::cuda::free(thrust::cuda::pointer<char>(pre_result));
}
// cached_allocator  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

namespace MarchingCubesKernel
{
	__constant__ AIToleranceControl devAITolerance;
	__constant__ AIToleranceControl devAIViewControl;
	__constant__ AIToleranceControl devAILambPower;
	__constant__ AIToleranceControl devAIMaterialBrightness;
	__constant__ AIInterpolationControl devAIInterpolation;

	__constant__ int dev_cam_cx;
	__constant__ int dev_cam_cy;

	__constant__ int dev_cam_w;
	__constant__ int dev_cam_h;

	__constant__ float dev_cam_cfx;
	__constant__ float dev_cam_cfy;

	__constant__ float dev_cam_ccx;
	__constant__ float dev_cam_ccy;

	//    mscho    @20240319
	__constant__ float sc_dark_corner_up;
	__constant__ float sc_dark_corner_left;
	__constant__ float sc_dark_corner_ul_begin;
	__constant__ float sc_dark_corner_right;
	__constant__ float sc_dark_corner_down;
	__constant__ int sc_dark_corner_enable;

	//	mscho	@20241015 ==> @20241022
	__constant__ int		sc_corner_v3_en; // 신규 팁용 Dark corner 제외 기능 활성화 여부 sc_corner_v3_enabled

	__constant__ int		sc_corner1_en;
	__constant__ float		sc_corner1_up;
	__constant__ float		sc_corner1_left;
	__constant__ float		sc_corner1_begin;
	__constant__ int		sc_corner2_en;
	__constant__ float		sc_corner2_left;
	__constant__ float		sc_corner2_bottom;
	__constant__ float		sc_corner2_begin;
	__constant__ int		sc_corner3_en;
	__constant__ float		sc_corner3_top;
	__constant__ float		sc_corner3_bottom;
	__constant__ float		sc_corner3_begin;
	__constant__ int		sc_corner4_en;
	__constant__ float		sc_corner4_top;
	__constant__ float		sc_corner4_bottom;
	__constant__ float		sc_corner4_begin;



	__constant__ float sc_dark_corner_left_limit;
	__constant__ float sc_dark_corner_right_limit;
	__constant__ float sc_dark_corner_top_limit;
	__constant__ float sc_dark_corner_bottom_limit;

	__constant__ bool sc_dark_usetip;

	//	mscho	@20240523
	__constant__ int   sc_voxel_neighbor_id[7 * 7 * 7 * 3];

	//__constant__ float reconColorBrightness = 2.0f;

} // namespace MarchingCubesKernel

void MarchingCubes::FrameBegin()
{
	//if (0 < captureFlag) captureCount++;
}

void MarchingCubes::FrameEnd()
{
	//captureFlag = 0;
}

// Bruce: 디버그용 코드 주석처리
int MarchingCubes::GetCaptureFlag() { return captureFlag; }
void MarchingCubes::SetCaptureFlag(int flag) { captureFlag = flag; }
void MarchingCubes::ToggleCaptureMode() { if (captureFlag & 1) captureFlag ^= 1; else captureFlag |= 1; }
int MarchingCubes::GetRecordFlag() { return recordFlag; }
void MarchingCubes::SetRecordFlag(int flag) { recordFlag = flag; }
void MarchingCubes::ToggleRecordMode() { if (recordFlag == 0) recordFlag = 1; else captureFlag = 0; }

// Bruce: 더이상 사용되지 않음
int MarchingCubes::GetMeshFilterMode() { return MeshFilterMode; }
void MarchingCubes::SetMeshFilterMode(int mode) { MeshFilterMode = mode; }

void MarchingCubes::ToggleMeshFilterMode()
{
	MeshFilterMode++;
	MeshFilterMode = MeshFilterMode % 3;

	qDebug("Mesh Filter Mode : %d", MeshFilterMode);
}

float MarchingCubes::GetMeshFilterZBeginOffset() { return meshFilterZBeginOffset; }
void MarchingCubes::SetMeshFilterZBeginOffset(float offset) { meshFilterZBeginOffset = offset; }

float MarchingCubes::GetMeshFilterZEnd() { return meshFilterZEnd; }
void MarchingCubes::SetMeshFilterZEnd(float end) { meshFilterZEnd = end; }

bool MarchingCubes::GetUseOutlierRemoval()
{
	return useOutlierRemoval;
}

void MarchingCubes::SetUseOutlierRemoval(bool use)
{
	useOutlierRemoval = use;
}

#ifndef BUILD_FOR_CPU
__device__ float __signf(float x) { return x / fabs(x); }
#endif

__device__ Eigen::Vector3f ComputeEigenvector0(const Eigen::Matrix3f& A,
	float eval0) {
	Eigen::Vector3f row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
	Eigen::Vector3f row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
	Eigen::Vector3f row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
	Eigen::Vector3f rxr[3];
	rxr[0] = row0.cross(row1);
	rxr[1] = row0.cross(row2);
	rxr[2] = row1.cross(row2);
	Eigen::Vector3f d;
	d[0] = rxr[0].dot(rxr[0]);
	d[1] = rxr[1].dot(rxr[1]);
	d[2] = rxr[2].dot(rxr[2]);

	int imax;
	d.maxCoeff(&imax);
	return rxr[imax] / sqrtf(d[imax]);
}

__device__ Eigen::Vector3f ComputeEigenvector1(const Eigen::Matrix3f& A,
	const Eigen::Vector3f& evec0,
	float eval1) {
	float max_evec0_abs = max(fabs(evec0(0)), fabs(evec0(1)));
	float inv_length =
		1 / sqrtf(max_evec0_abs * max_evec0_abs + evec0(2) * evec0(2));
	Eigen::Vector3f U = (fabs(evec0(0)) > fabs(evec0(1)))
		? Eigen::Vector3f(-evec0(2), 0, evec0(0))
		: Eigen::Vector3f(0, evec0(2), -evec0(1));
	U *= inv_length;
	Eigen::Vector3f V = evec0.cross(U);

	Eigen::Vector3f AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
		A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
		A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

	Eigen::Vector3f AV = { A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
							A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
							A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2) };

	float m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
	float m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
	float m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

	float absM00 = fabs(m00);
	float absM01 = fabs(m01);
	float absM11 = fabs(m11);
	float max_abs_comp0 = max(absM00, absM11);
	float max_abs_comp = max(max_abs_comp0, absM01);
	float coef2 = std::min<float>(max_abs_comp0, absM01) / std::max<float>(max_abs_comp, 1.0e-6);
	float coef1 = 1.0 / sqrtf(1.0 + coef2 * coef2);
	if (absM00 >= absM11) {
		coef2 *= coef1 * __signf(m00) * __signf(m01);
		return (max_abs_comp0 >= absM01) ? coef2 * U - coef1 * V
			: coef1 * U - coef2 * V;
	}
	else {
		coef2 *= coef1 * __signf(m11) * __signf(m01);
		return (max_abs_comp0 >= absM01) ? coef1 * U - coef2 * V
			: coef2 * U - coef1 * V;
	}
}

__device__ thrust::tuple<Eigen::Vector3f, Eigen::Matrix3f> FastEigen3x3(Eigen::Matrix3f& A) {
	float max_coeff = A.maxCoeff();
	if (max_coeff == 0) {
		return thrust::make_tuple(Eigen::Vector3f::Zero(), Eigen::Matrix3f::Identity());
	}
	A /= max_coeff;

	float norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);

	if (norm > 0) {
		Eigen::Vector3f eval;
		Eigen::Matrix3f evec;

		float q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

		float b00 = A(0, 0) - q;
		float b11 = A(1, 1) - q;
		float b22 = A(2, 2) - q;

		float p = sqrtf((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

		float c00 = b11 * b22 - A(1, 2) * A(1, 2);
		float c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
		float c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
		float det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

		float half_det = det * 0.5;
		half_det = std::min<float>(std::max<float>(half_det, -1.0), 1.0);

		float angle = acos(half_det) / (float)3;
		float const two_thirds_pi = 2.09439510239319549;
		float beta2 = cos(angle) * 2;
		float beta0 = cos(angle + two_thirds_pi) * 2;
		float beta1 = -(beta0 + beta2);

		eval(0) = q + p * beta0;
		eval(1) = q + p * beta1;
		eval(2) = q + p * beta2;

		if (half_det >= 0) {
			evec.col(2) = ComputeEigenvector0(A, eval(2));
			evec.col(1) = ComputeEigenvector1(A, evec.col(2), eval(1));
			evec.col(0) = evec.col(1).cross(evec.col(2));
			A *= max_coeff;
			return thrust::make_tuple(eval, evec);
		}
		else {
			evec.col(0) = ComputeEigenvector0(A, eval(0));
			evec.col(1) = ComputeEigenvector1(A, evec.col(0), eval(1));
			evec.col(2) = evec.col(0).cross(evec.col(1));
			A *= max_coeff;
			return thrust::make_tuple(eval, evec);
		}
	}
	else {
		A *= max_coeff;
		return thrust::make_tuple(A.diagonal(), Eigen::Matrix3f::Identity());
	}
}

#ifndef BUILD_FOR_CPU
__device__ inline thrust::tuple<Eigen::Vector3f, Eigen::Vector3f> FastEigen3x3MinMaxVec(Eigen::Matrix3f& A) {
	auto eig_val_vec = FastEigen3x3(A);
	int min_id, max_id;
	thrust::get<0>(eig_val_vec).minCoeff(&min_id);
	thrust::get<0>(eig_val_vec).maxCoeff(&max_id);
	return thrust::make_tuple(thrust::get<1>(eig_val_vec).col(min_id), thrust::get<1>(eig_val_vec).col(max_id));
	//return thrust::make_tuple(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
}

__device__ inline Eigen::Vector3f FastEigen3x3Val(Eigen::Matrix3f& A) {
	auto eig_val_vec = FastEigen3x3(A);
	int min_id, max_id;
	float minv = thrust::get<0>(eig_val_vec).minCoeff(&min_id);
	float maxv = thrust::get<0>(eig_val_vec).maxCoeff(&max_id);
	return Eigen::Vector3f(minv, thrust::get<0>(eig_val_vec).sum() - minv - maxv, maxv);
}
#endif

#pragma region Region
void Region::SetGlobalMinMax(const Eigen::Vector3f& gmin, const Eigen::Vector3f& gmax, bool update)
{
	globalMin = gmin;
	globalMax = gmax;

	if (update)
	{
		Update();
	}
}

void Region::SetLocalMinMax(const Eigen::Vector3f& lmin, const Eigen::Vector3f& lmax, bool update)
{
	localMin = lmin;
	localMax = lmax;

	if (update)
	{
		Update();
	}
}

void Region::Update()
{
	localMinGlobalIndexX = (size_t)floorf(localMin.x() / voxelSize - (globalMin.x() / voxelSize));
	localMinGlobalIndexY = (size_t)floorf(localMin.y() / voxelSize - (globalMin.y() / voxelSize));
	localMinGlobalIndexZ = (size_t)floorf(localMin.z() / voxelSize - (globalMin.z() / voxelSize));

	localMaxGlobalIndexX = (size_t)floorf(localMax.x() / voxelSize - (globalMin.x() / voxelSize));
	localMaxGlobalIndexY = (size_t)floorf(localMax.y() / voxelSize - (globalMin.y() / voxelSize));
	localMaxGlobalIndexZ = (size_t)floorf(localMax.z() / voxelSize - (globalMin.z() / voxelSize));

	voxelCountX = localMaxGlobalIndexX - localMinGlobalIndexX;
	voxelCountY = localMaxGlobalIndexY - localMinGlobalIndexY;
	voxelCountZ = localMaxGlobalIndexZ - localMinGlobalIndexZ;
	voxelCount = voxelCountX * voxelCountY * voxelCountZ;
}

__host__ __device__
bool Region::Contains(size_t globalIndexX, size_t globalIndexY, size_t globalIndexZ) const
{
	if (localMinGlobalIndexX > globalIndexX) return false;
	if (localMaxGlobalIndexX <= globalIndexX) return false;
	if (localMinGlobalIndexY > globalIndexY) return false;
	if (localMaxGlobalIndexY <= globalIndexY) return false;
	if (localMinGlobalIndexZ > globalIndexZ) return false;
	if (localMaxGlobalIndexZ <= globalIndexZ) return false;

	return true;
	//(localMinGlobalIndexX <= globalIndexX && globalIndexX < localMaxGlobalIndexX) &&
	//(localMinGlobalIndexY <= globalIndexY && globalIndexY < localMaxGlobalIndexY) &&
	//(localMinGlobalIndexZ <= globalIndexZ && globalIndexZ < localMaxGlobalIndexZ);
}

__host__ __device__
bool Region::ContainsWithMargin(size_t globalIndexX, size_t globalIndexY, size_t globalIndexZ, int margin) const
{
	size_t minx = localMinGlobalIndexX - margin <= 0 ? 0 : localMinGlobalIndexX - margin;
	size_t miny = localMinGlobalIndexY - margin <= 0 ? 0 : localMinGlobalIndexY - margin;
	size_t minz = localMinGlobalIndexZ - margin <= 0 ? 0 : localMinGlobalIndexZ - margin;

	size_t maxx = localMaxGlobalIndexX + margin >= localMinGlobalIndexX + voxelCountX ? localMinGlobalIndexX + voxelCountX : localMaxGlobalIndexX + margin;
	size_t maxy = localMaxGlobalIndexY + margin >= localMinGlobalIndexY + voxelCountY ? localMinGlobalIndexY + voxelCountY : localMaxGlobalIndexY + margin;
	size_t maxz = localMaxGlobalIndexZ + margin >= localMinGlobalIndexZ + voxelCountZ ? localMinGlobalIndexZ + voxelCountZ : localMaxGlobalIndexZ + margin;
	//printf("min %llu, %llu, %llu | max %llu, %llu, %llu | min %llu, %llu, %llu | max %llu, %llu, %llu | countXYZ %llu, %llu, %llu \n", localMinGlobalIndexX, localMinGlobalIndexY, localMinGlobalIndexZ, localMaxGlobalIndexX, localMaxGlobalIndexY, localMaxGlobalIndexZ, minx, miny,minz, maxx, maxy, maxz, voxelCountX, voxelCountY, voxelCountZ);
	if (minx > globalIndexX) return false;
	if (maxx <= globalIndexX) return false;
	if (miny > globalIndexY) return false;
	if (maxy <= globalIndexY) return false;
	if (minz > globalIndexZ) return false;
	if (maxz <= globalIndexZ) return false;

	return true;
}

//	mscho	@20240805
__host__ __device__
bool Region::ContainsWithMargin_v2(size_t globalIndexX, size_t globalIndexY, size_t globalIndexZ, int margin) const
{
	size_t minx = localMinGlobalIndexX - margin <= 0 ? 0 : localMinGlobalIndexX - margin;
	if (minx > globalIndexX) return false;
	size_t miny = localMinGlobalIndexY - margin <= 0 ? 0 : localMinGlobalIndexY - margin;
	if (miny > globalIndexY) return false;
	size_t minz = localMinGlobalIndexZ - margin <= 0 ? 0 : localMinGlobalIndexZ - margin;
	if (minz > globalIndexZ) return false;

	size_t maxx = localMaxGlobalIndexX + margin >= localMinGlobalIndexX + voxelCountX ? localMinGlobalIndexX + voxelCountX : localMaxGlobalIndexX + margin;
	if (maxx <= globalIndexX) return false;
	size_t maxy = localMaxGlobalIndexY + margin >= localMinGlobalIndexY + voxelCountY ? localMinGlobalIndexY + voxelCountY : localMaxGlobalIndexY + margin;
	if (maxy <= globalIndexY) return false;
	size_t maxz = localMaxGlobalIndexZ + margin >= localMinGlobalIndexZ + voxelCountZ ? localMinGlobalIndexZ + voxelCountZ : localMaxGlobalIndexZ + margin;
	if (maxz <= globalIndexZ) return false;

	return true;
}
__host__ __device__
bool Region::ContainsWithMargin(size_t globalIndexX, size_t globalIndexY, size_t globalIndexZ, int marginMin, int marginMax) const
{
	size_t minx = localMinGlobalIndexX - marginMin <= 0 ? 0 : localMinGlobalIndexX - marginMin;
	size_t miny = localMinGlobalIndexY - marginMin <= 0 ? 0 : localMinGlobalIndexY - marginMin;
	size_t minz = localMinGlobalIndexZ - marginMin <= 0 ? 0 : localMinGlobalIndexZ - marginMin;

	size_t maxx = localMaxGlobalIndexX + marginMax >= localMinGlobalIndexX + voxelCountX ? localMinGlobalIndexX + voxelCountX : localMaxGlobalIndexX + marginMax;
	size_t maxy = localMaxGlobalIndexY + marginMax >= localMinGlobalIndexY + voxelCountY ? localMinGlobalIndexY + voxelCountY : localMaxGlobalIndexY + marginMax;
	size_t maxz = localMaxGlobalIndexZ + marginMax >= localMinGlobalIndexZ + voxelCountZ ? localMinGlobalIndexZ + voxelCountZ : localMaxGlobalIndexZ + marginMax;
	//printf("min %llu, %llu, %llu | max %llu, %llu, %llu | min %llu, %llu, %llu | max %llu, %llu, %llu | countXYZ %llu, %llu, %llu \n", localMinGlobalIndexX, localMinGlobalIndexY, localMinGlobalIndexZ, localMaxGlobalIndexX, localMaxGlobalIndexY, localMaxGlobalIndexZ, minx, miny,minz, maxx, maxy, maxz, voxelCountX, voxelCountY, voxelCountZ);
	if (minx > globalIndexX) return false;
	if (maxx <= globalIndexX) return false;
	if (miny > globalIndexY) return false;
	if (maxy <= globalIndexY) return false;
	if (minz > globalIndexZ) return false;
	if (maxz <= globalIndexZ) return false;

	return true;
}

__host__ __device__
size_t Region::GetLocalIndexX(size_t flattenIndex) const
{
	return (flattenIndex % (voxelCountX * voxelCountY)) % voxelCountX;
}

__host__ __device__
size_t Region::GetLocalIndexY(size_t flattenIndex) const
{
	return (flattenIndex % (voxelCountX * voxelCountY)) / voxelCountX;
}

__host__ __device__
size_t Region::GetLocalIndexZ(size_t flattenIndex) const
{
	return flattenIndex / (voxelCountX * voxelCountY);
}

__host__ __device__
size_t Region::GetGlobalIndexX(size_t index) const
{
	return localMinGlobalIndexX + index;
}

__host__ __device__
size_t Region::GetGlobalIndexY(size_t index) const
{
	return localMinGlobalIndexY + index;
}

__host__ __device__
size_t Region::GetGlobalIndexZ(size_t index) const
{
	return localMinGlobalIndexZ + index;
}

__host__ __device__
Eigen::Vector3f Region::GetGlobalPosition(size_t xIndex, size_t yIndex, size_t zIndex) const
{
	return Eigen::Vector3f(
		(float)GetGlobalIndexX(xIndex) * voxelSize + 0.5f * voxelSize + globalMin.x(),
		(float)GetGlobalIndexY(yIndex) * voxelSize + 0.5f * voxelSize + globalMin.y(),
		(float)GetGlobalIndexZ(zIndex) * voxelSize + 0.5f * voxelSize + globalMin.z());
}

void Region::Dump() const
{
	qDebug("voxelSize : %f", voxelSize);
	qDebug("globalMin : %f, %f, %f", globalMin.x(), globalMin.y(), globalMin.z());
	qDebug("globalMax : %f, %f, %f", globalMax.x(), globalMax.y(), globalMax.z());
	qDebug("localMinGlboalIndex : %llu, %llu, %llu", localMinGlobalIndexX, localMinGlobalIndexY, localMinGlobalIndexZ);
	qDebug("localMaxGlboalIndex : %llu, %llu, %llu", localMaxGlobalIndexX, localMaxGlobalIndexY, localMaxGlobalIndexZ);
	qDebug("voxelCount : %llu = %llu x %llu x %llu", voxelCount, voxelCountX, voxelCountY, voxelCountZ);
}
#pragma endregion

#pragma region ExecutionInfo

void MarchingCubes::ExecutionInfo::Dump()
{
	qDebug("-global");
	global.Dump();

	qDebug("-local");
	local.Dump();

	qDebug("-cache");
	cache.Dump();

	qDebug("globalHashInfo : @%p", globalHashInfo);
	qDebug("globalHash : @%p", globalHash);
	//qDebug("globalHashValue : @%p", globalHashValue);

	qDebug("blockIndex : %llu", blockIndex);
	qDebug("blockSize : %llu", blockSize);
	qDebug("blockRemainder : %llu", blockRemainder);
	qDebug("dataSize : %llu", dataSize);

	qDebug("voxelValues : @%p", voxelValues);
	qDebug("gridSlotIndexCache : @%p", gridSlotIndexCache);
	qDebug("gridSlotIndexCache_pts : @%p", gridSlotIndexCache_pts);
}

#pragma endregion

void MarchingCubes::initialize_CameraRT(
	Eigen::Matrix4f& camRT_matrix,
	CamInfo_& cam,
	const CutCorner& cutCorner,
	bool useTip,
	CUstream_st* st)
{
	// CUDA static memory에 저장한다.
	//checkCudaErrors(cudaMalloc((void**)&dev_camRT, sizeof(Eigen::Matrix4f)));
	//checkCudaErrors(cudaMalloc((void**)&dev_cam_tilt, sizeof(Eigen::Matrix3f)));
	//checkCudaErrors(cudaMalloc((void**)&dev_cam_tilt_inv, sizeof(Eigen::Matrix3f)));
	//checkCudaErrors(cudaMemcpyAsync(dev_camRT, &camRT_matrix, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice, st));
	//checkCudaErrors(cudaMemcpyAsync(dev_cam_tilt, &cam.matTilt, sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, st));
	//checkCudaErrors(cudaMemcpyAsync(dev_cam_tilt_inv, &cam.invMatTilt, sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice, st));

	_dev_camRT = camRT_matrix;
	this->camInfo = cam;

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_w), &cam.img_width, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_h), &cam.img_height, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_cx), &cam.cx, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_cy), &cam.cy, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_cfx), &cam.cfx, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_cfy), &cam.cfy, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_ccx), &cam.ccx, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&dev_cam_ccy), &cam.ccy, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	// CUDA 쪽과 자료형 맞춰주기
	float        cpu_corner_up = cutCorner.cut_corner_top;
	float        cpu_corner_left = cutCorner.cut_corner_left;
	float        cpu_corner_ul_begin = cutCorner.cut_corner_begin;
	float        cpu_corner_right = cutCorner.cut_corner_right;
	float        cpu_corner_down = cutCorner.cut_corner_bottom;
	int          cpu_corner_enable = cutCorner.cut_corner_en;

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_enable), &cpu_corner_enable, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_up), &cpu_corner_up, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_left), &cpu_corner_left, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_ul_begin), &cpu_corner_ul_begin, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_right), &cpu_corner_right, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_down), &cpu_corner_down, sizeof(float), 0, cudaMemcpyHostToDevice, st));

	//============================================================================================================================
	//	mscho	@20241015 ==> @20241022
	//	Corner

	// CUDA 쪽과 자료형 맞춰주기
	int			cpu_corner_v3_en = cutCorner.cut_corner_v2_en;

	int			cpu_corner1_en = cutCorner.cut_corner1_en;
	float		cpu_corner1_up = cutCorner.cut_corner1_top;
	float		cpu_corner1_left = cutCorner.cut_corner1_left;
	float		cpu_corner1_begin = cutCorner.cut_corner1_begin;

	int			cpu_corner2_en = cutCorner.cut_corner2_en;
	float		cpu_corner2_bot = cutCorner.cut_corner2_bot;
	float		cpu_corner2_left = cutCorner.cut_corner2_left;
	float		cpu_corner2_begin = cutCorner.cut_corner2_begin;

	int			cpu_corner3_en = cutCorner.cut_corner3_en;
	float		cpu_corner3_up = cutCorner.cut_corner3_top;
	float		cpu_corner3_down = cutCorner.cut_corner3_bot;
	float		cpu_corner3_begin = cutCorner.cut_corner3_begin;

	int			cpu_corner4_en = cutCorner.cut_corner4_en;
	float		cpu_corner4_up = cutCorner.cut_corner4_top;
	float		cpu_corner4_down = cutCorner.cut_corner4_bot;
	float		cpu_corner4_begin = cutCorner.cut_corner4_begin;

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner_v3_en), &cpu_corner_v3_en, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner1_en), &cpu_corner1_en, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner1_up), &cpu_corner1_up, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner1_left), &cpu_corner1_left, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner1_begin), &cpu_corner1_begin, sizeof(float), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner2_en), &cpu_corner2_en, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner2_bottom), &cpu_corner2_bot, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner2_left), &cpu_corner2_left, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner2_begin), &cpu_corner2_begin, sizeof(float), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner3_en), &cpu_corner3_en, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner3_top), &cpu_corner3_up, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner3_bottom), &cpu_corner3_down, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner3_begin), &cpu_corner3_begin, sizeof(float), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner4_en), &cpu_corner4_en, sizeof(int), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner4_top), &cpu_corner4_up, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner4_bottom), &cpu_corner4_down, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_corner4_begin), &cpu_corner4_begin, sizeof(float), 0, cudaMemcpyHostToDevice, st));

	//============================================================================================================================
	// CUDA 쪽과 자료형 맞춰주기
	float cpu_top_limit = cutCorner.cut_corner_top_limit;
	float cpu_bottom_limit = cutCorner.cut_corner_bottom_limit;
	float cpu_left_limit = cutCorner.cut_corner_left_limit;
	float cpu_right_limit = cutCorner.cut_corner_right_limit;

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_top_limit), &cpu_top_limit, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_bottom_limit), &cpu_bottom_limit, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_left_limit), &cpu_left_limit, sizeof(float), 0, cudaMemcpyHostToDevice, st));
	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_corner_right_limit), &cpu_right_limit, sizeof(int), 0, cudaMemcpyHostToDevice, st));

	checkCudaErrors(cudaMemcpyToSymbolAsync(reinterpret_cast<const void*>(&sc_dark_usetip), &useTip, sizeof(int), 0, cudaMemcpyHostToDevice, st));

#ifndef BUILD_FOR_CPU
	if (tgt_cnt == nullptr)
	{
		//	mscho	@20240530
		//	cudaMallocAsync => cudaMalloc
		cudaMalloc(&tgt_cnt, sizeof(uint32_t));
	}
	checkCudaSync(st);
#endif
}

// Aaron @240614
/*
{ // 생성 코드
	int cx = 0;			// 중심 좌표 X
	int cy = 0;			// 중심 좌표 Y
	int cz = 0;			// 중심 좌표 Z
	int offset = 3;		// 중심으로 부터의 개수 ex) 3이면 중심 +- 3 이므로 7 x 7 x 7로 동작
	int currentOffset = 0;
	while (currentOffset <= offset)
	{
		for (int z = -currentOffset; z <= currentOffset; z++)
		{
			for (int y = -currentOffset; y <= currentOffset; y++)
			{
				for (int x = -currentOffset; x <= currentOffset; x++)
				{
					if ((x == -currentOffset || x == currentOffset) ||
						(y == -currentOffset || y == currentOffset) ||
						(z == -currentOffset || z == currentOffset))
					{
						qDebug("%d, %d, %d,", cx + x, cy + y, cz + z);
					}
				}
			}
		}
		currentOffset++;
	}
}
*/
static int32_t _neighbor_id[7 * 7 * 7 * 3] =
{
	0, 0, 0,
	-1, -1, -1,
	0, -1, -1,
	1, -1, -1,
	-1, 0, -1,
	0, 0, -1,
	1, 0, -1,
	-1, 1, -1,
	0, 1, -1,
	1, 1, -1,
	-1, -1, 0,
	0, -1, 0,
	1, -1, 0,
	-1, 0, 0,
	1, 0, 0,
	-1, 1, 0,
	0, 1, 0,
	1, 1, 0,
	-1, -1, 1,
	0, -1, 1,
	1, -1, 1,
	-1, 0, 1,
	0, 0, 1,
	1, 0, 1,
	-1, 1, 1,
	0, 1, 1,
	1, 1, 1,
	-2, -2, -2,
	-1, -2, -2,
	0, -2, -2,
	1, -2, -2,
	2, -2, -2,
	-2, -1, -2,
	-1, -1, -2,
	0, -1, -2,
	1, -1, -2,
	2, -1, -2,
	-2, 0, -2,
	-1, 0, -2,
	0, 0, -2,
	1, 0, -2,
	2, 0, -2,
	-2, 1, -2,
	-1, 1, -2,
	0, 1, -2,
	1, 1, -2,
	2, 1, -2,
	-2, 2, -2,
	-1, 2, -2,
	0, 2, -2,
	1, 2, -2,
	2, 2, -2,
	-2, -2, -1,
	-1, -2, -1,
	0, -2, -1,
	1, -2, -1,
	2, -2, -1,
	-2, -1, -1,
	2, -1, -1,
	-2, 0, -1,
	2, 0, -1,
	-2, 1, -1,
	2, 1, -1,
	-2, 2, -1,
	-1, 2, -1,
	0, 2, -1,
	1, 2, -1,
	2, 2, -1,
	-2, -2, 0,
	-1, -2, 0,
	0, -2, 0,
	1, -2, 0,
	2, -2, 0,
	-2, -1, 0,
	2, -1, 0,
	-2, 0, 0,
	2, 0, 0,
	-2, 1, 0,
	2, 1, 0,
	-2, 2, 0,
	-1, 2, 0,
	0, 2, 0,
	1, 2, 0,
	2, 2, 0,
	-2, -2, 1,
	-1, -2, 1,
	0, -2, 1,
	1, -2, 1,
	2, -2, 1,
	-2, -1, 1,
	2, -1, 1,
	-2, 0, 1,
	2, 0, 1,
	-2, 1, 1,
	2, 1, 1,
	-2, 2, 1,
	-1, 2, 1,
	0, 2, 1,
	1, 2, 1,
	2, 2, 1,
	-2, -2, 2,
	-1, -2, 2,
	0, -2, 2,
	1, -2, 2,
	2, -2, 2,
	-2, -1, 2,
	-1, -1, 2,
	0, -1, 2,
	1, -1, 2,
	2, -1, 2,
	-2, 0, 2,
	-1, 0, 2,
	0, 0, 2,
	1, 0, 2,
	2, 0, 2,
	-2, 1, 2,
	-1, 1, 2,
	0, 1, 2,
	1, 1, 2,
	2, 1, 2,
	-2, 2, 2,
	-1, 2, 2,
	0, 2, 2,
	1, 2, 2,
	2, 2, 2,
	-3, -3, -3,
	-2, -3, -3,
	-1, -3, -3,
	0, -3, -3,
	1, -3, -3,
	2, -3, -3,
	3, -3, -3,
	-3, -2, -3,
	-2, -2, -3,
	-1, -2, -3,
	0, -2, -3,
	1, -2, -3,
	2, -2, -3,
	3, -2, -3,
	-3, -1, -3,
	-2, -1, -3,
	-1, -1, -3,
	0, -1, -3,
	1, -1, -3,
	2, -1, -3,
	3, -1, -3,
	-3, 0, -3,
	-2, 0, -3,
	-1, 0, -3,
	0, 0, -3,
	1, 0, -3,
	2, 0, -3,
	3, 0, -3,
	-3, 1, -3,
	-2, 1, -3,
	-1, 1, -3,
	0, 1, -3,
	1, 1, -3,
	2, 1, -3,
	3, 1, -3,
	-3, 2, -3,
	-2, 2, -3,
	-1, 2, -3,
	0, 2, -3,
	1, 2, -3,
	2, 2, -3,
	3, 2, -3,
	-3, 3, -3,
	-2, 3, -3,
	-1, 3, -3,
	0, 3, -3,
	1, 3, -3,
	2, 3, -3,
	3, 3, -3,
	-3, -3, -2,
	-2, -3, -2,
	-1, -3, -2,
	0, -3, -2,
	1, -3, -2,
	2, -3, -2,
	3, -3, -2,
	-3, -2, -2,
	3, -2, -2,
	-3, -1, -2,
	3, -1, -2,
	-3, 0, -2,
	3, 0, -2,
	-3, 1, -2,
	3, 1, -2,
	-3, 2, -2,
	3, 2, -2,
	-3, 3, -2,
	-2, 3, -2,
	-1, 3, -2,
	0, 3, -2,
	1, 3, -2,
	2, 3, -2,
	3, 3, -2,
	-3, -3, -1,
	-2, -3, -1,
	-1, -3, -1,
	0, -3, -1,
	1, -3, -1,
	2, -3, -1,
	3, -3, -1,
	-3, -2, -1,
	3, -2, -1,
	-3, -1, -1,
	3, -1, -1,
	-3, 0, -1,
	3, 0, -1,
	-3, 1, -1,
	3, 1, -1,
	-3, 2, -1,
	3, 2, -1,
	-3, 3, -1,
	-2, 3, -1,
	-1, 3, -1,
	0, 3, -1,
	1, 3, -1,
	2, 3, -1,
	3, 3, -1,
	-3, -3, 0,
	-2, -3, 0,
	-1, -3, 0,
	0, -3, 0,
	1, -3, 0,
	2, -3, 0,
	3, -3, 0,
	-3, -2, 0,
	3, -2, 0,
	-3, -1, 0,
	3, -1, 0,
	-3, 0, 0,
	3, 0, 0,
	-3, 1, 0,
	3, 1, 0,
	-3, 2, 0,
	3, 2, 0,
	-3, 3, 0,
	-2, 3, 0,
	-1, 3, 0,
	0, 3, 0,
	1, 3, 0,
	2, 3, 0,
	3, 3, 0,
	-3, -3, 1,
	-2, -3, 1,
	-1, -3, 1,
	0, -3, 1,
	1, -3, 1,
	2, -3, 1,
	3, -3, 1,
	-3, -2, 1,
	3, -2, 1,
	-3, -1, 1,
	3, -1, 1,
	-3, 0, 1,
	3, 0, 1,
	-3, 1, 1,
	3, 1, 1,
	-3, 2, 1,
	3, 2, 1,
	-3, 3, 1,
	-2, 3, 1,
	-1, 3, 1,
	0, 3, 1,
	1, 3, 1,
	2, 3, 1,
	3, 3, 1,
	-3, -3, 2,
	-2, -3, 2,
	-1, -3, 2,
	0, -3, 2,
	1, -3, 2,
	2, -3, 2,
	3, -3, 2,
	-3, -2, 2,
	3, -2, 2,
	-3, -1, 2,
	3, -1, 2,
	-3, 0, 2,
	3, 0, 2,
	-3, 1, 2,
	3, 1, 2,
	-3, 2, 2,
	3, 2, 2,
	-3, 3, 2,
	-2, 3, 2,
	-1, 3, 2,
	0, 3, 2,
	1, 3, 2,
	2, 3, 2,
	3, 3, 2,
	-3, -3, 3,
	-2, -3, 3,
	-1, -3, 3,
	0, -3, 3,
	1, -3, 3,
	2, -3, 3,
	3, -3, 3,
	-3, -2, 3,
	-2, -2, 3,
	-1, -2, 3,
	0, -2, 3,
	1, -2, 3,
	2, -2, 3,
	3, -2, 3,
	-3, -1, 3,
	-2, -1, 3,
	-1, -1, 3,
	0, -1, 3,
	1, -1, 3,
	2, -1, 3,
	3, -1, 3,
	-3, 0, 3,
	-2, 0, 3,
	-1, 0, 3,
	0, 0, 3,
	1, 0, 3,
	2, 0, 3,
	3, 0, 3,
	-3, 1, 3,
	-2, 1, 3,
	-1, 1, 3,
	0, 1, 3,
	1, 1, 3,
	2, 1, 3,
	3, 1, 3,
	-3, 2, 3,
	-2, 2, 3,
	-1, 2, 3,
	0, 2, 3,
	1, 2, 3,
	2, 2, 3,
	3, 2, 3,
	-3, 3, 3,
	-2, 3, 3,
	-1, 3, 3,
	0, 3, 3,
	1, 3, 3,
	2, 3, 3,
	3, 3, 3
};

//	mscho	@20240523
void MarchingCubes::initial_voxel_neighbor_idx(CUstream_st* st)
{
#ifndef BUILD_FOR_CPU
	checkCudaErrors(cudaMemcpyToSymbolAsync(sc_voxel_neighbor_id, &_neighbor_id, sizeof(int) * (7 * 7 * 7 * 3), 0, cudaMemcpyHostToDevice, st));
	checkCudaSync(st);
#else
	memcpy(sc_voxel_neighbor_id, &_neighbor_id, sizeof(int) * (7 * 7 * 7 * 3));
#endif
}

void MarchingCubes::freeMarchingCube() {
#ifndef BUILD_FOR_CPU
	if (tgt_cnt != nullptr)
		cudaFree(tgt_cnt);

	//	mscho	@20240805
	if (used_cnt_HashVoxel_h != nullptr)
		cudaFreeHost(used_cnt_HashVoxel_h);
	if (used_cnt_Extract_h != nullptr)
		cudaFreeHost(used_cnt_Extract_h);

	if (used_cnt_HashVoxel != nullptr)
		cudaFree(used_cnt_HashVoxel);
	if (used_cnt_localContains != nullptr)
		cudaFree(used_cnt_localContains);
	if (used_cnt_Extract != nullptr)
		cudaFree(used_cnt_Extract);
	if (cnt_averageRes != nullptr)
		cudaFree(cnt_averageRes);

	//	mscho	@20250207
	if (host_Count_HashTableUsed != nullptr)
		cudaFreeHost(host_Count_HashTableUsed);
	if (host_Count_AvePoints != nullptr)
		cudaFreeHost(host_Count_AvePoints);

	//	mscho	@20250228
	if (host_used_cnt_HashVoxel != nullptr)
		cudaFreeHost(host_used_cnt_HashVoxel);
	if (host_Count_avgArea != nullptr)
		cudaFreeHost(host_Count_avgArea);
#else
	delete[] exeInfo.gridSlotIndexCache;
	delete[] exeInfo.gridSlotIndexCache_pts;

	delete used_cnt_HashVoxel;
	delete used_cnt_HashVoxel_h;
	delete used_cnt_Extract_h;
	delete used_cnt_localContains;

	delete used_cnt_Extract;
	delete cnt_averageRes;
#endif

}

MarchingCubes::MarchingCubes(HSettingsAdapter* pSettings) :
	pSettings{ pSettings },
	m_voxelDataBundle(
		m_MC_voxelValues
		, m_MC_voxelValueCounts
		, m_MC_voxelPositions
		, m_MC_voxelNormals
		, m_MC_voxelColors
		, m_MC_voxelColorScores
		, m_MC_voxelSegmentations
		, m_MC_voxelExtraAttribs
#ifdef USE_EXPERIMENTAL_COLOR_OPT2
		, m_MC_voxelColorReconData
#endif//USE_EXPERIMENTAL_COLOR_OPT2
	)
{
	pHashManager = new CudaHashManager;
}

MarchingCubes::~MarchingCubes() {
#ifdef BUILD_FOR_CPU
	freeMarchingCube();
#endif
	delete pHashManager;

	if (nullptr != noiseFilter)
	{
		noiseFilter->Terminate();
		delete noiseFilter;
		noiseFilter = nullptr;
	}
}

#ifndef BUILD_FOR_CPU
namespace MarchingCubesKernel {
	__device__ int edgeTable[256] = {
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0 };

	__device__ int triTable[256][16] = {
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} };
} // marchnigCubesKernel
#endif

#pragma region Math
#define DOT(a, b) (a).x() * (b).x() + (a).y() * (b).y() + (a).z() * (b).z()
#define CROSS(a, b) Eigen::Vector3f((a).y() * (b).z() - (b).y() * (a).z(), (a).z() * (b).x() - (b).z() * (a).x(), (a).x() * (b).y() - (b).x() * (a).y())
#define LENGTHSQUARED(a) DOT((a), (a))
#define LENGTH(a) __fsqrt_rn(LENGTHSQUARED(a))
#define DISTANCESQUARED(a, b) LENGTHSQUARED((a) - (b))
#define DISTANCE(a, b) __fsqrt_rn(DISTANCESQUARED((a), (b)))
#define NORMALIZE(a) (a) / (LENGTH(a))

#ifndef BUILD_FOR_CPU
bool __device__ RayTriangleIntersect(const Eigen::Vector3f& ray_origin, const Eigen::Vector3f& ray_direction,
	const Eigen::Vector3f& v0, const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, bool enable_backculling, float& distance)
{
	using Eigen::Vector3f;
	const float epsilon = 1e-7f;

	const Vector3f v0v1 = v1 - v0;
	const Vector3f v0v2 = v2 - v0;

	const Vector3f pvec = ray_direction.cross(v0v2);

	const float det = v0v1.dot(pvec);

	if (enable_backculling)
	{
		// If det is negative, the triangle is back-facing.
		// If det is close to 0, the ray misses the triangle.
		if (det < epsilon)
			return false;
	}
	else
	{
		// If det is close to 0, the ray and triangle are parallel.
		if (std::abs(det) < epsilon)
			return false;
	}
	const float inv_det = 1 / det;

	const Vector3f tvec = ray_origin - v0;
	const auto u = tvec.dot(pvec) * inv_det;
	if (u < 0 || u > 1)
		return false;

	const Vector3f qvec = tvec.cross(v0v1);
	const auto v = ray_direction.dot(qvec) * inv_det;
	if (v < 0 || u + v > 1)
		return false;

	const auto t = v0v2.dot(qvec) * inv_det;

	distance = t;
	return true;
}

__device__
float DotProduct(const float3& a, const float3& b)
{
	return __fadd_rn(__fadd_rn(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y)), __fmul_rn(a.z, b.z));
}

__device__
float3 CrossProduct(const float3& a, const float3& b)
{
	return
	{
		__fsub_rn(__fmul_rn(a.y, b.z), __fmul_rn(a.z, b.y)),
		__fsub_rn(__fmul_rn(a.z, b.x), __fmul_rn(a.x, b.z)),
		__fsub_rn(__fmul_rn(a.x, b.y), __fmul_rn(a.y, b.x))
	};
}

__device__
float3 Normalize(const float3& a)
{
	auto norm = norm3df(a.x, a.y, a.z);
	return { __fdiv_rn(a.x, norm), __fdiv_rn(a.y, norm), __fdiv_rn(a.z, norm) };
}

__device__
float GetDistance(const float3& p0, const float3& p1)
{
	float x = __fsub_rn(p1.x, p0.x);
	float y = __fsub_rn(p1.y, p0.y);
	float z = __fsub_rn(p1.z, p0.z);

	return norm3df(x, y, z);
}
__device__
float GetDistance(const Eigen::Vector3f& p0, const Eigen::Vector3f& p1)
{
	float x = __fsub_rn(p1.x(), p0.x());
	float y = __fsub_rn(p1.y(), p0.y());
	float z = __fsub_rn(p1.z(), p0.z());

	return norm3df(x, y, z);
}
__device__
float3 GetDirection(const float3& from, const float3& to)
{
	return { __fsub_rn(to.x, from.x), __fsub_rn(to.y, from.y), __fsub_rn(to.z, from.z) };
}

__device__
float3 GetNormalizedDirection(const float3& from, const float3& to)
{
	float3 d = { __fsub_rn(to.x, from.x), __fsub_rn(to.y, from.y), __fsub_rn(to.z, from.z) };
	auto norm = norm3df(d.x, d.y, d.z);
	return { __fdiv_rn(d.x, norm), __fdiv_rn(d.y, norm), __fdiv_rn(d.z, norm) };
}

__device__
float3 GetCentroid(const float3& p0, const float3& p1, const float3& p2)
{
	auto x = __fdiv_rn(__fadd_rn(__fadd_rn(p0.x, p1.x), p2.x), 3.0f);
	auto y = __fdiv_rn(__fadd_rn(__fadd_rn(p0.y, p1.y), p2.y), 3.0f);
	auto z = __fdiv_rn(__fadd_rn(__fadd_rn(p0.z, p1.z), p2.z), 3.0f);
	return { x, y, z };
}

__device__
float3 GetTriangleNormal(const float3& p0, const float3& p1, const float3& p2)
{
	auto d01 = GetDirection(p0, p1);
	auto d02 = GetDirection(p0, p2);
	return Normalize(CrossProduct(d01, d02));
}

__device__
float GetTriangleArea(const float3& p0, const float3& p1, const float3& p2)
{
	auto d01 = GetDirection(p0, p1);
	auto d02 = GetDirection(p0, p2);
	auto c = CrossProduct(d01, d02);
	return __fdiv_rn(norm3df(c.x, c.y, c.z), 2.0f);
}

__device__
void GetTriangleNormalAndArea(const float3& p0, const float3& p1, const float3& p2, float3& normal, float& area)
{
	auto d01 = GetDirection(p0, p1);
	auto d02 = GetDirection(p0, p2);
	auto c = CrossProduct(d01, d02);
	auto norm = norm3df(c.x, c.y, c.z);
	normal.x = __fdiv_rn(c.x, norm);
	normal.y = __fdiv_rn(c.y, norm);
	normal.z = __fdiv_rn(c.z, norm);
	area = __fdiv_rn(norm, 2.0f);
}

__device__
ulonglong3 GetHashKey(const float3& position, const float3& globalMin, float voxelSize)
{
	auto xIndex = __float2ull_rn(floorf(__fdiv_rn(__fsub_rn(position.x, globalMin.x), voxelSize)));
	auto yIndex = __float2ull_rn(floorf(__fdiv_rn(__fsub_rn(position.y, globalMin.y), voxelSize)));
	auto zIndex = __float2ull_rn(floorf(__fdiv_rn(__fsub_rn(position.z, globalMin.z), voxelSize)));
	return { xIndex, yIndex, zIndex };
}

__device__
float3 GetPosition(const ulonglong3& hashKey, const float3& globalMin, float voxelSize)
{
	float halfVoxelSize = __fmul_rn(voxelSize, 0.5f);
	float x = __fadd_rn(globalMin.x, __fmaf_rn(hashKey.x, voxelSize, halfVoxelSize));
	float y = __fadd_rn(globalMin.y, __fmaf_rn(hashKey.y, voxelSize, halfVoxelSize));
	float z = __fadd_rn(globalMin.z, __fmaf_rn(hashKey.z, voxelSize, halfVoxelSize));
	return { x, y, z };
}
#endif
#pragma endregion

__device__ Eigen::Vector3f Interpolation(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
{
	if (false == FLT_VALID(valp1))
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

	if (false == FLT_VALID(valp2))
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

	float mu;
	Eigen::Vector3f p;

	if (fabsf(isolevel - valp1) < 0.00001f)
		return(p1);
	if (fabsf(isolevel - valp2) < 0.00001f)
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	if (fabsf(valp1 - valp2) < 0.00001f)
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	mu = (isolevel - valp1) / (valp2 - valp1);
	p.x() = p1.x() + mu * (p2.x() - p1.x());
	p.y() = p1.y() + mu * (p2.y() - p1.y());
	p.z() = p1.z() + mu * (p2.z() - p1.z());
	return p;
}

#ifdef BUILD_FOR_CPU
// Equivalent function to CUDA __fmaf_rd
float __fmaf_rd(float a, float b, float c) {
	// Compute the fused multiply-add
	float result = a * b + c;

	// Round towards negative infinity
	result = std::floor(result);

	return result;
}
#endif

#ifndef BUILD_FOR_CPU
__device__ Eigen::Vector3f getCameraDirFromPixelCoord(float u, float v) {
	float px = float(dev_cam_w) * u;
	float py = float(dev_cam_h) * v;
	float rx = (px + 0.5f - dev_cam_ccx) / dev_cam_cfx;
	float ry = (py + 0.5f - dev_cam_ccy) / dev_cam_cfy;
	return Eigen::Vector3f(rx, ry, 1.0f).normalized();
}
#endif

// 카메라의 Pixel 좌표계에서의 위치를 계산. 카메라 이미지 크기 안의 정수 좌표(0 ~ ImageSize - 1)를 반환
__device__ bool ColorUtil::getPixelCoord_pos(
	const Eigen::Vector3f local_pts, Eigen::Vector<int, 2>& cam_pos, const Eigen::Matrix4f dev_camRT, const Eigen::Matrix3f dev_cam_tilt)
{
	auto inPos = Eigen::Vector4f(local_pts.x(), local_pts.y(), local_pts.z(), 1.0f);
	auto camPos = dev_camRT * inPos;

	float rx = camPos[0] / camPos[2];
	float ry = camPos[1] / camPos[2];

	Eigen::Vector3f CamPos3f(rx, ry, 1);

	const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
	float tx = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
	float ty = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();

	cam_pos.x() = (int)((tx * dev_cam_cfx + dev_cam_ccx) - 0.5);
	cam_pos.y() = (int)((ty * dev_cam_cfy + dev_cam_ccy) - 0.5);

	if (cam_pos.x() < 0 || cam_pos.x() > dev_cam_w - 1 || cam_pos.y() < 0 || cam_pos.y() > dev_cam_h - 1)
		return false;
	else
		return true;
}

// 카메라의 Pixel 좌표계에서의 위치를 계산. 카메라 이미지 크기 안에서 크기에 상대적인 좌표(0.0 ~ 1.0) 값을 반환
__device__ bool ColorUtil::getPixelCoord_relative(
	const Eigen::Vector3f local_pts, Eigen::Vector2f& cam_pos, const Eigen::Matrix4f dev_camRT, const Eigen::Matrix3f dev_cam_tilt)
{
	auto inPos = Eigen::Vector4f(local_pts.x(), local_pts.y(), local_pts.z(), 1.0f);
	auto camPos = dev_camRT * inPos;

	float rx = camPos[0] / camPos[2];
	float ry = camPos[1] / camPos[2];

	Eigen::Vector3f CamPos3f(rx, ry, 1);

	const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
	float tx = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
	float ty = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();

	cam_pos.x() = ((tx * dev_cam_cfx + dev_cam_ccx) - 0.5) / float(dev_cam_w);
	cam_pos.y() = ((ty * dev_cam_cfy + dev_cam_ccy) - 0.5) / float(dev_cam_h);

	if (cam_pos.x() < 0 || cam_pos.x() >= 1.0f || cam_pos.y() <= 0.0f || cam_pos.y() >= 1.0f)
		return false;
	else
		return true;
}

//    mscho    @20240130
//  true : dark corner이므로, 사용하면 안된다
__device__ bool ColorUtil::_Checking_Darkcorner_image_v2(
	const float        pnt_z,
	const Eigen::Vector2i input_coord,
	bool _usetip
)
{
	Eigen::Vector3f Local_pos;

	int img_last_pixX = CU_CX_SIZE_MAX - 1;
	int img_last_pixY = CU_CY_SIZE_MAX - 1;

	if (input_coord.x() < sc_dark_corner_left_limit || input_coord.x() > img_last_pixX - sc_dark_corner_right_limit || input_coord.y() < sc_dark_corner_bottom_limit || input_coord.y() > img_last_pixY - sc_dark_corner_top_limit)
		return true;

	if (sc_dark_corner_enable < 1) return false;

	Local_pos = Eigen::Vector3f((float)input_coord.x(), (float)input_coord.y(), pnt_z);

	if (sc_dark_usetip == false)
	{
		Local_pos.x() = img_last_pixX - Local_pos.x();        // x
		Local_pos.y() = Local_pos.y();
	}
	else
	{
		Local_pos.x() = img_last_pixX - Local_pos.x();        // x
		Local_pos.y() = img_last_pixY - Local_pos.y();        // y
	}
	const float img_x = Local_pos.x();
	const float img_y = img_last_pixY - Local_pos.y();   // 1차 직선의 방정식으로 풀이를 하기 위해, y 축을 바꾼다.
	// 위에서 바꾼건, tip의 유무에 따라서, mirror를 통과하기 때문에 바꿈

	float i_cpu_corner_up = sc_dark_corner_up - 5;
	float i_cpu_corner_left = sc_dark_corner_left - 5;
	float i_cpu_corner_ul_begin = sc_dark_corner_ul_begin + 3;
	float i_cpu_corner_right = sc_dark_corner_right + 5;
	float i_cpu_corner_down = sc_dark_corner_down + 5;

	float dx = i_cpu_corner_up;
	float dy = i_cpu_corner_left;    // y절편

	if (dx != 0.0f)
	{
		float ul_slope = i_cpu_corner_left / i_cpu_corner_up;
		float ul_x0 = img_last_pixY - dy;
		float ul_y_value = __fmaf_rd(ul_slope, img_x, ul_x0);// ul_slope* img_x + ul_x0;
		if (img_y >= ul_y_value)
			return true;
	}

	if ((img_last_pixX - dx) != 0.0f) {
		dx = i_cpu_corner_down;
		dy = (img_last_pixY - i_cpu_corner_right);
		float dr_slope = dy / (img_last_pixX - dx);
		float dr_x0 = -dr_slope * dx;

		float dr_y_value = __fmaf_rd(dr_slope, img_x, dr_x0);// dr_slope* img_x + dr_x0;

		if (img_y <= dr_y_value)
			return true;
	}

	return false;
}


//	mscho	@20241015 ==> @20241022
//  true : dark corner이므로, 사용하면 안된다
__device__ bool ColorUtil::_Checking_Darkcorner_image_v3(
	const float        pnt_z,
	const Eigen::Vector2i input_coord,
	bool _usetip
)
{
	Eigen::Vector3f Local_pos;

	int img_last_pixX = CU_CX_SIZE_MAX - 1;
	int img_last_pixY = CU_CY_SIZE_MAX - 1;

	if (input_coord.x() < sc_dark_corner_left_limit || input_coord.x() > img_last_pixX - sc_dark_corner_right_limit || input_coord.y() < sc_dark_corner_bottom_limit || input_coord.y() > img_last_pixY - sc_dark_corner_top_limit)
		return true;

	if (sc_dark_corner_enable < 1) return false;

	Local_pos = Eigen::Vector3f((float)input_coord.x(), (float)input_coord.y(), pnt_z);

	if (sc_dark_usetip == false)
	{
		Local_pos.x() = img_last_pixX - Local_pos.x();        // x
		Local_pos.y() = Local_pos.y();
	}
	else
	{
		Local_pos.x() = img_last_pixX - Local_pos.x();        // x
		Local_pos.y() = img_last_pixY - Local_pos.y();        // y
	}
	const float img_x = Local_pos.x();
	const float img_y = img_last_pixY - Local_pos.y();   // 1차 직선의 방정식으로 풀이를 하기 위해, y 축을 바꾼다.
	// 위에서 바꾼건, tip의 유무에 따라서, mirror를 통과하기 때문에 바꿈

	float		i_cpu_corner_up;
	float		i_cpu_corner_left;
	float		i_cpu_corner1_begin;
	float		i_cpu_corner_down;
	float		i_cpu_corner2_begin;

	float		i_cpu_corner_top;
	float		i_cpu_corner_bottom;
	float		i_cpu_corner3_begin;

	//float i_cpu_corner_up = sc_dark_corner_up - 5;
	//float i_cpu_corner_left = sc_dark_corner_left - 5;
	//float i_cpu_corner_ul_begin = sc_dark_corner_ul_begin + 3;
	//float i_cpu_corner_right = sc_dark_corner_right + 5;
	//float i_cpu_corner_down = sc_dark_corner_down + 5;




	i_cpu_corner_up = sc_corner1_up - 5;
	i_cpu_corner_left = sc_corner1_left - 5;
	float dx = i_cpu_corner_up;
	float dy = i_cpu_corner_left;    // y절편

	float	corner_slope = i_cpu_corner_left / i_cpu_corner_up;	// 기울기 값은 설정 top / left 값에 결정된다.
	float	limit_z = sc_corner1_begin + 3;
	float	dof_bottom = -11.0;

	float	delta_z = fabs((limit_z - pnt_z) / (limit_z - dof_bottom));
	delta_z = min(delta_z, 1.0f);


	if (pnt_z < limit_z)
	{
		delta_z = max(delta_z, 0.0f);

		float	dy = i_cpu_corner_left * delta_z;	// y절편(cam 좌표계)

		float	corner1_slope = i_cpu_corner_left / i_cpu_corner_up;	// 기울기 값은 설정 top / left 값에 결정된다.
		float	corner1_x0 = img_last_pixY - dy;				// y절편은 z값에 가변된다.(y반전 좌표계)
		float	corner1_y_value = __fmaf_rd(corner1_slope, img_x, corner1_x0);// ul_slope* img_x + ul_x0;

		if (img_y > corner1_y_value)
		{
			return true;
		}
	}

	i_cpu_corner_down = sc_corner2_bottom - 5;
	i_cpu_corner_left = sc_corner2_left + 5;

	limit_z = sc_corner2_begin + 3;
	dof_bottom = -11.0;

	delta_z = fabs((limit_z - pnt_z) / (limit_z - dof_bottom));
	delta_z = min(delta_z, 1.0f);

	if (pnt_z < limit_z)
	{
		delta_z = max(delta_z, 0.0f);

		float	dy = img_last_pixY - (i_cpu_corner_left * delta_z);	// y절편(cam 좌표계)

		float	corner2_slope = (float)i_cpu_corner_left / (float)i_cpu_corner_down * -1.0;
		float	corner2_x0 = img_last_pixY - dy;// .(y반전 좌표계)..y절편
		float	corner2_y_value = __fmaf_rd(corner2_slope, img_x, corner2_x0);// dr_slope* img_x + dr_x0;

		if (img_y < corner2_y_value)
			return true;
	}


	i_cpu_corner_up = sc_corner3_top;
	i_cpu_corner_down = sc_corner3_bottom;

	float	delta_x = fabsf(i_cpu_corner_up - i_cpu_corner_down);

	limit_z = sc_corner3_begin;
	float dof_top = +11.0;

	delta_z = fabs((pnt_z - limit_z) / (dof_top - limit_z));
	delta_z = min(delta_z, 1.0f);


	if (pnt_z > limit_z)
	{
		delta_z = max(delta_z, 0.0f);

		float	corner3_y0 = img_last_pixX - (img_last_pixX - i_cpu_corner_down) * delta_z;	// x 절편
		float	corner3_slope = CU_CY_SIZE_MAX / delta_x;

		float	corner3_x0 = -1.0f * corner3_slope * corner3_y0;	// y절편
		float	corner3_y_value = corner3_slope * img_x + corner3_x0;

		if (img_y < corner3_y_value)
			return true;
	}

	i_cpu_corner_up = sc_corner4_top;
	i_cpu_corner_down = sc_corner4_bottom;

	delta_x = fabsf(i_cpu_corner_up - i_cpu_corner_down);

	limit_z = sc_corner4_begin;
	dof_bottom = -11.0;

	delta_z = fabs((limit_z - pnt_z) / (limit_z - dof_bottom));
	delta_z = min(delta_z, 1.0f);


	if (pnt_z > limit_z)
	{
		delta_z = max(delta_z, 0.0f);

		float	corner4_up = i_cpu_corner_up * delta_z;	// x 절편
		float	corner4_slope = CU_CY_SIZE_MAX / delta_x;

		float	corner4_x0 = img_last_pixX - corner4_slope * corner4_up;	// y절편
		float	corner4_y_value = corner4_slope * img_x + corner4_x0;

		if (img_y > corner4_y_value)
			return true;
	}
	return false;
}

#ifndef BUILD_FOR_CPU
// 0도 , 45도 모두 읽어서, 평균값으로 texture color를 만들어 내는 방법
__device__ Eigen::Vector3b ColorUtil::getPixelCoord_pos_Mix(
	const Eigen::Vector3f vertex,
	const unsigned char* img0_,
	const unsigned char* img45_,
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector3b& defaultColor
)
{
	Eigen::Vector<int, 2> img_pixel_pos_0;
	Eigen::Vector<int, 2> img_pixel_pos_45;
	Eigen::Vector3b cam_pixel;

	// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
	bool bimg_area_0 = getPixelCoord_pos(vertex, img_pixel_pos_0, dev_camRT, dev_cam_tilt);
	bool bimg_area_45 = false;// getPixelCoord_pos(vertex, img_pixel_pos_45, dev_camRT, dev_cam_tilt);

	if (_Checking_Darkcorner_image_v2(vertex.z(), img_pixel_pos_0, sc_dark_usetip)) return defaultColor;

	if (bimg_area_0 && bimg_area_45)
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		cam_pixel.x() = (unsigned char)(((int)img0_[img_index_offset_0] + (int)img45_[img_index_offset_45]) / 2);
		cam_pixel.y() = (unsigned char)(((int)img0_[img_index_offset_0 + 1] + (int)img45_[img_index_offset_45 + 1]) / 2);
		cam_pixel.z() = (unsigned char)(((int)img0_[img_index_offset_0 + 2] + (int)img45_[img_index_offset_45 + 2]) / 2);
	}
	else
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		if (bimg_area_0)
		{
			cam_pixel.x() = img0_[img_index_offset_0];
			cam_pixel.y() = img0_[img_index_offset_0 + 1];
			cam_pixel.z() = img0_[img_index_offset_0 + 2];
		}
		else 	if (bimg_area_45)
		{
			cam_pixel.x() = img45_[img_index_offset_45];
			cam_pixel.y() = img45_[img_index_offset_45 + 1];
			cam_pixel.z() = img45_[img_index_offset_45 + 2];
		}
		else
		{
			// 45도에서도 벗어난 다면
// Error 처리
			cam_pixel = defaultColor;
		}
	}

	return cam_pixel;
}

// 0도 , 45도 모두 읽어서, 평균값으로 texture color를 만들어 내는 방법
__device__ Eigen::Vector3b ColorUtil::getPixelCoord_pos_Mix(
	const Eigen::Vector3f vertex,
	const Eigen::Vector3f vertex45,
	const unsigned char* img0_,
	const unsigned char* img45_,
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector3b& defaultColor
)
{

	Eigen::Vector<int, 2> img_pixel_pos_0;
	Eigen::Vector<int, 2> img_pixel_pos_45;
	Eigen::Vector3b cam_pixel;

	// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
	bool bimg_area_0 = getPixelCoord_pos(vertex, img_pixel_pos_0, dev_camRT, dev_cam_tilt);
	//	mscho	@20240625
	//	45도 color를 제외하였으므로, 하기와 같이 처리한다..
	//  지금은 속도를 위해서, 제거하였지만.. 향후 추가하면 변경하도록 한다.
	bool bimg_area_45 = false;// getPixelCoord_pos(vertex45, img_pixel_pos_45, dev_camRT, dev_cam_tilt);

	if (bimg_area_0) bimg_area_0 = !_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_0, sc_dark_usetip);
	if (bimg_area_45) bimg_area_45 = !_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_45, sc_dark_usetip);

	if (bimg_area_0 && bimg_area_45)
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		cam_pixel.x() = (unsigned char)(((int)img0_[img_index_offset_0] + (int)img45_[img_index_offset_45]) / 2);
		cam_pixel.y() = (unsigned char)(((int)img0_[img_index_offset_0 + 1] + (int)img45_[img_index_offset_45 + 1]) / 2);
		cam_pixel.z() = (unsigned char)(((int)img0_[img_index_offset_0 + 2] + (int)img45_[img_index_offset_45 + 2]) / 2);
	}
	else
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		if (bimg_area_0)
		{
			cam_pixel.x() = img0_[img_index_offset_0];
			cam_pixel.y() = img0_[img_index_offset_0 + 1];
			cam_pixel.z() = img0_[img_index_offset_0 + 2];
		}
		else 	if (bimg_area_45)
		{
			cam_pixel.x() = img45_[img_index_offset_45];
			cam_pixel.y() = img45_[img_index_offset_45 + 1];
			cam_pixel.z() = img45_[img_index_offset_45 + 2];
		}
		else
		{
			// 45도에서도 벗어난 다면
// Error 처리
			cam_pixel = defaultColor;
		}
	}

	return cam_pixel;
}

#endif // !BUILD_FOR_CPU

//	mscho	@20240611
// 0도 , 45도 모두 읽어서, 평균값으로 texture color를 만들어 내는 방법.
//	 Camera image 의 암부에 해당하는 위치가 오면, color 를 리턴하는 것이 아니라, false를 리턴
//	color는 parameter를 통해서 전달하는 함수로 변경한다.
__device__ bool ColorUtil::getPixelCoord_pos_Mix_v2(
	Eigen::Vector3b& pixel_color,
	const Eigen::Vector3f vertex,
	const Eigen::Vector3f vertex45,
	const unsigned char* img0_,
	const unsigned char* img45_,
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector2f camSpd,
	const Eigen::Vector3b& defaultColor
)
{
	Eigen::Vector<int, 2> img_pixel_pos_0;
	Eigen::Vector<int, 2> img_pixel_pos_45;
	Eigen::Vector3b cam_pixel;
	bool	bInsideImage = true;
	// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
	bool bimg_area_0 = getPixelCoord_pos(vertex, img_pixel_pos_0, dev_camRT, dev_cam_tilt);

	//	mscho	@20240625
	//	4세트의 패턴을 적용하면서, 45도 color를 제거하였다..
	//	향후에 이를 다시 적용하려면, 하기부분을 수정하여야 한다.
	bool bimg_area_45 = false;// getPixelCoord_pos(vertex45, img_pixel_pos_45, dev_camRT, dev_cam_tilt);

	//	mscho	@20241015 ==> @20241022
	if (sc_corner_v3_en)
	{	// 신규 팁용
		if (bimg_area_0) bimg_area_0 = !_Checking_Darkcorner_image_v3(vertex.z(), img_pixel_pos_0, sc_dark_usetip);
		if (bimg_area_45) bimg_area_45 = !_Checking_Darkcorner_image_v3(vertex.z(), img_pixel_pos_45, sc_dark_usetip);
	}
	else
	{	// 기존 팁용
		if (bimg_area_0) bimg_area_0 = !_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_0, sc_dark_usetip);
		if (bimg_area_45) bimg_area_45 = !_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_45, sc_dark_usetip);
	}

	if (!bimg_area_0 && !bimg_area_45) return false;

	if (bimg_area_0 && bimg_area_45)
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		cam_pixel.x() = (unsigned char)(((int)img0_[img_index_offset_0] + (int)img45_[img_index_offset_45]) / 2);
		cam_pixel.y() = (unsigned char)(((int)img0_[img_index_offset_0 + 1] + (int)img45_[img_index_offset_45 + 1]) / 2);
		cam_pixel.z() = (unsigned char)(((int)img0_[img_index_offset_0 + 2] + (int)img45_[img_index_offset_45 + 2]) / 2);
	}
	else
	{
		Eigen::Vector2i img_pixel_pos_0_g = Eigen::Vector2i(img_pixel_pos_0.x() + camSpd.x(), img_pixel_pos_0.y() + camSpd.y());
		Eigen::Vector2i img_pixel_pos_0_r = Eigen::Vector2i(img_pixel_pos_0.x() + camSpd.x() * 2.0f, img_pixel_pos_0.y() + camSpd.y() * 2.0f);

		//	mscho	@20241015 ==> @20241022
		if (sc_corner_v3_en)
		{ // 신규 팁용
			if (_Checking_Darkcorner_image_v3(vertex.z(), img_pixel_pos_0_g, sc_dark_usetip))
				return false;
			if (_Checking_Darkcorner_image_v3(vertex.z(), img_pixel_pos_0_r, sc_dark_usetip))
				return false;
		}
		else
		{ // 기존 팁용
			if (_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_0_g, sc_dark_usetip))
				return false;
			if (_Checking_Darkcorner_image_v2(0.0f, img_pixel_pos_0_r, sc_dark_usetip))
				return false;
		}

		size_t  img_index_offset_0_b = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t img_index_offset_0_g = (img_pixel_pos_0_g.y() * dev_cam_w + img_pixel_pos_0_g.x()) * 4;
		size_t img_index_offset_0_r = (img_pixel_pos_0_r.y() * dev_cam_w + img_pixel_pos_0_r.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		if (bimg_area_0)
		{
			cam_pixel.x() = img0_[img_index_offset_0_r];
			cam_pixel.y() = img0_[img_index_offset_0_g + 1];
			cam_pixel.z() = img0_[img_index_offset_0_b + 2];
		}
		else 	if (bimg_area_45)
		{
			cam_pixel.x() = img45_[img_index_offset_45];
			cam_pixel.y() = img45_[img_index_offset_45 + 1];
			cam_pixel.z() = img45_[img_index_offset_45 + 2];
		}
		else
		{
			// 45도에서도 벗어난 다면
// Error 처리
			cam_pixel = defaultColor;
			bInsideImage = false;
		}
	}
	pixel_color = cam_pixel;
	return bInsideImage;
}

#ifndef BUILD_FOR_CPU
//  mscho   @20240205
//  color map을 만드는 기능을 추가한다.
// normal map을 만드는 기능을 추가한다
__global__ void MarchingCubesKernel::kernel_gen_depth_map_ave_v4(
	Eigen::Vector3f* depth_map,
	Eigen::Vector3f* normal_map,
	unsigned int* depth_color,
	unsigned int* depth_map_cnt,
	const int voxel_x,        // 200
	const int voxel_y        // 250
)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
{
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	int _idxy = threadid / voxel_x;
	int _idxx = threadid % voxel_x;

	if (threadid > voxel_x * voxel_y - 1)    return;

	if (depth_map_cnt[threadid] > 0)
	{
		depth_map[threadid].x() /= depth_map_cnt[threadid];
		depth_map[threadid].y() /= depth_map_cnt[threadid];
		depth_map[threadid].z() /= depth_map_cnt[threadid];


		//  mscho   @20240408
		//	normalize?? device ????????? ???????.
		float	dist = norm3df(normal_map[threadid].x(), normal_map[threadid].y(), normal_map[threadid].z());

		normal_map[threadid].x() /= dist;
		normal_map[threadid].y() /= dist;
		normal_map[threadid].z() /= dist;

		depth_color[threadid * 3] /= depth_map_cnt[threadid];
		depth_color[threadid * 3 + 1] /= depth_map_cnt[threadid];
		depth_color[threadid * 3 + 2] /= depth_map_cnt[threadid];
	}
}
#endif

//  mscho   @20240205 ==> @20240527
//  color map을 만드는 기능을 추가한다.
// normal map을 만드는 기능을 추가한다
__global__ void MarchingCubesKernel::kernel_gen_depth_map_ave_v6(
	Eigen::Vector3f* depth_map,
	Eigen::Vector3f* normal_map,
	float* alpha_map,
	float* specular_map,
	unsigned int* depth_color,
	unsigned int* depth_map_cnt,
	const int voxel_x,        // 200
	const int voxel_y        // 250
)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
{
#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > voxel_x * voxel_y - 1)    return;
	{
#else
	const int threadCount = voxel_x * voxel_y;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif
		int count = depth_map_cnt[threadid];
		if (count > 0)
		{
			alpha_map[threadid] /= count;
			specular_map[threadid] /= count;
			depth_map[threadid] /= count;

			//  mscho   @20240408
			//	normalize?? device ????????? ???????.
#ifndef BUILD_FOR_CPU
			float	dist = norm3df(normal_map[threadid].x(), normal_map[threadid].y(), normal_map[threadid].z());
			normal_map[threadid] /= dist;
#else
			normal_map[threadid].normalize();
#endif
			depth_color[threadid * 3] /= count;
			depth_color[threadid * 3 + 1] /= count;
			depth_color[threadid * 3 + 2] /= count;
		}
	}
	}

__global__ void MarchingCubesKernel::kernel_gen_depth_map_blank_v2(
	unsigned int* depth_color,
	unsigned int* depth_map_cnt,	//	mscho	@20240308

	const int voxel_x,        // 200
	const int voxel_y        // 250
)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
{
#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > voxel_x * voxel_y - 1)    return;
	{
#else
	const int threadCount = voxel_x * voxel_y;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif
		if (depth_map_cnt[threadid] > 0)
			kernel_return;
		int _idxy = threadid / voxel_x;
		int _idxx = threadid % voxel_x;

		int   ave_range = 1;
		int	sx = _idxx - ave_range;
		int	ex = _idxx + ave_range;
		int	sy = _idxy - ave_range;
		int	ey = _idxy + ave_range;

		if (sx < 0) sx = 0;
		if (ex > voxel_x - 1) ex = voxel_x - 1;
		if (sy < 0) sy = 0;
		if (ey > voxel_y - 1)  ex = voxel_y - 1;

		unsigned int		map_cnt = 0;
		unsigned int		blank_color[3] = { 0, };

		if (depth_map_cnt[threadid] == 0)
		{
			for (int i = sy; i <= ey; i++)
			{
				for (int k = sx; k <= ex; k++)
				{
					int  p_idx = i * voxel_x + k;
					if (depth_map_cnt[p_idx] > 0)
					{
						blank_color[0] += depth_color[p_idx * 3];
						blank_color[1] += depth_color[p_idx * 3 + 1];
						blank_color[2] += depth_color[p_idx * 3 + 2];
						map_cnt++;
					}
				}
			}
		}
		if (map_cnt > 0)
		{
			depth_color[threadid * 3] = blank_color[0] / map_cnt;
			depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
			depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
		}
		else
		{
			int   ave_range = 2;
			int	sx = _idxx - ave_range;
			int	ex = _idxx + ave_range;
			int	sy = _idxy - ave_range;
			int	ey = _idxy + ave_range;

			if (sx < 0) sx = 0;
			if (ex > voxel_x - 1) ex = voxel_x - 1;
			if (sy < 0) sy = 0;
			if (ey > voxel_y - 1)  ex = voxel_y - 1;

			unsigned int		map_cnt = 0;
			unsigned int		blank_color[3] = { 0, };

			for (int i = sy; i <= ey; i++)
			{
				for (int k = sx; k <= ex; k++)
				{
					int  p_idx = i * voxel_x + k;
					if (depth_map_cnt[p_idx] > 0)
					{
						blank_color[0] += depth_color[p_idx * 3];
						blank_color[1] += depth_color[p_idx * 3 + 1];
						blank_color[2] += depth_color[p_idx * 3 + 2];
						map_cnt++;
					}
				}
			}
			if (map_cnt > 0)
			{
				depth_color[threadid * 3] = blank_color[0] / map_cnt;
				depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
				depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
			}
			else
			{
				int   ave_range = 3;
				int	sx = _idxx - ave_range;
				int	ex = _idxx + ave_range;
				int	sy = _idxy - ave_range;
				int	ey = _idxy + ave_range;

				if (sx < 0) sx = 0;
				if (ex > voxel_x - 1) ex = voxel_x - 1;
				if (sy < 0) sy = 0;
				if (ey > voxel_y - 1)  ex = voxel_y - 1;

				unsigned int		map_cnt = 0;
				unsigned int		blank_color[3] = { 0, };

				for (int i = sy; i <= ey; i++)
				{
					for (int k = sx; k <= ex; k++)
					{
						int  p_idx = i * voxel_x + k;
						if (depth_map_cnt[p_idx] > 0)
						{
							blank_color[0] += depth_color[p_idx * 3];
							blank_color[1] += depth_color[p_idx * 3 + 1];
							blank_color[2] += depth_color[p_idx * 3 + 2];
							map_cnt++;
						}
					}
				}
				if (map_cnt > 0)
				{
					depth_color[threadid * 3] = blank_color[0] / map_cnt;
					depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
					depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
				}
			}
		}
	}
	}

//	mscho	@20240423	==> @20240524
//	depthmap mask 를 만드는 기능을 추가한다
//	주어진 depth_map_cnt에서 실제 point가 있는 지점과 그 주위를 1로 설정하는 depthmap_mask 를 생성한다.
__global__ void MarchingCubesKernel::kernel_gen_depth_map_mask(
	const unsigned int* depth_map_cnt	//	mscho	@20240308
	, unsigned short* depthmap_mask
	, const int voxel_x       // 500
	, const int voxel_y        // 700
	, const int ave_width
	, const int ave_height
)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
{
#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > voxel_x * voxel_y - 1)    return;
	{
#else
	const int threadCount = voxel_x * voxel_y;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif
		bool done = false;

		int _idxy = threadid / voxel_x;
		int _idxx = threadid % voxel_x;

		if (depth_map_cnt[threadid] > 0)
		{
			depthmap_mask[threadid] = 1;
			kernel_return;
		}


		int   ave_range = ave_height >= ave_width ? ave_height : ave_width;

		//	mscho	@20240524
		//  왜 반복을 하고 있었는지 알수 없으나, 의미없이 반복하고 있어서
		// 반복문을 제거한다.
		//for (int k = 1; k <= ave_range; k++)
		{
			int	sy = _idxy - ave_height;
			int	ey = _idxy + ave_height;

			int	sx = _idxx - ave_width;
			int	ex = _idxx + ave_width;

			if (sx < 0) sx = 0;
			if (ex > voxel_x - 1) ex = voxel_x - 1;
			if (sy < 0) sy = 0;
			if (ey > voxel_y - 1)  ex = voxel_y - 1;

			for (int i = sy; i <= ey; i++)
			{
				for (int m = sx; m <= ex; m++)
				{
					if (depth_map_cnt[i * voxel_x + m] > 0)
					{
						depthmap_mask[threadid] = 1;
						done = true;
						break;
					}
				}
				if (done)
					break;
			}
		}
		if (!done)
			depthmap_mask[threadid] = 0;
	}
	}

/* no londer used
__global__ void kernel_gen_depth_map_blank_v3(
	unsigned int* depth_color,
	unsigned int* depth_map_cnt,	//	mscho	@20240308
	const int voxel_x,        // 200
	const int voxel_y        // 250
)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
{
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;

	int _idxy = threadid / voxel_x;
	int _idxx = threadid % voxel_x;

	if (threadid > voxel_x * voxel_y - 1)    return;
	if (depth_map_cnt[threadid] > 0)    return;
	int   ave_range = 1;
	int	sx = _idxx - ave_range;
	int	ex = _idxx + ave_range;
	int	sy = _idxy - ave_range;
	int	ey = _idxy + ave_range;

	if (sx < 0) sx = 0;
	if (ex > voxel_x - 1) ex = voxel_x - 1;
	if (sy < 0) sy = 0;
	if (ey > voxel_y - 1)  ex = voxel_y - 1;

	unsigned int		map_cnt = 0;
	unsigned int		blank_color[3] = { 0, };

	if (depth_map_cnt[threadid] == 0)
	{
		for (int i = sy; i <= ey; i++)
		{
			for (int k = sx; k <= ex; k++)
			{
				int  p_idx = i * voxel_x + k;
				if (depth_map_cnt[p_idx] > 0)
				{
					blank_color[0] += depth_color[p_idx * 3];
					blank_color[1] += depth_color[p_idx * 3 + 1];
					blank_color[2] += depth_color[p_idx * 3 + 2];
					map_cnt++;
				}
			}
		}
	}
	if (map_cnt > 0)
	{
		depth_color[threadid * 3] = blank_color[0] / map_cnt;
		depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
		depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
	}
	else
	{
		int   ave_range = 2;
		int	sx = _idxx - ave_range;
		int	ex = _idxx + ave_range;
		int	sy = _idxy - ave_range;
		int	ey = _idxy + ave_range;

		if (sx < 0) sx = 0;
		if (ex > voxel_x - 1) ex = voxel_x - 1;
		if (sy < 0) sy = 0;
		if (ey > voxel_y - 1)  ex = voxel_y - 1;

		unsigned int		map_cnt = 0;
		unsigned int		blank_color[3] = { 0, };

		for (int i = sy; i <= ey; i++)
		{
			for (int k = sx; k <= ex; k++)
			{
				int  p_idx = i * voxel_x + k;
				if (depth_map_cnt[p_idx] > 0)
				{
					blank_color[0] += depth_color[p_idx * 3];
					blank_color[1] += depth_color[p_idx * 3 + 1];
					blank_color[2] += depth_color[p_idx * 3 + 2];
					map_cnt++;
				}
			}
		}
		if (map_cnt > 0)
		{
			depth_color[threadid * 3] = blank_color[0] / map_cnt;
			depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
			depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
		}
		else
		{
			int   ave_range = 3;
			int	sx = _idxx - ave_range;
			int	ex = _idxx + ave_range;
			int	sy = _idxy - ave_range;
			int	ey = _idxy + ave_range;

			if (sx < 0) sx = 0;
			if (ex > voxel_x - 1) ex = voxel_x - 1;
			if (sy < 0) sy = 0;
			if (ey > voxel_y - 1)  ex = voxel_y - 1;

			unsigned int		map_cnt = 0;
			unsigned int		blank_color[3] = { 0, };

			for (int i = sy; i <= ey; i++)
			{
				for (int k = sx; k <= ex; k++)
				{
					int  p_idx = i * voxel_x + k;
					if (depth_map_cnt[p_idx] > 0)
					{
						blank_color[0] += depth_color[p_idx * 3];
						blank_color[1] += depth_color[p_idx * 3 + 1];
						blank_color[2] += depth_color[p_idx * 3 + 2];
						map_cnt++;
					}
				}
			}
			if (map_cnt > 0)
			{
				depth_color[threadid * 3] = blank_color[0] / map_cnt;
				depth_color[threadid * 3 + 1] = blank_color[1] / map_cnt;
				depth_color[threadid * 3 + 2] = blank_color[2] / map_cnt;
			}
		}
	}

}



//  mscho   @20240118
//  45도의 경우, 0도 좌표계로 이동하는 기능을 추가한다
	__global__ void kernel_gen_depth_map_v4(
		const Eigen::Vector3f* pts,
		const Eigen::Matrix4f& transform_45_to_0,
		Eigen::Vector3f* depth_map,
		unsigned char* depth_map_cnt,
		const int voxel_x,        // 200
		const int voxel_y,        // 250
		const int phase_x,        // 256
		const int phase_y,        // 480
		const int x_step,         // 2
		const int y_step,         // 12
		const int x_offset,       // 0도 - 0, 45도 - 1
		const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
		const int y_scan,
		const int in_size)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	{
		int threadid = blockIdx.x * blockDim.x + threadIdx.x;

		int _x_width = phase_x / x_step;
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;

		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;

		if (_pidx > phase_x * phase_y - 1)    return;

		float   invalid_float_ = FLT_MAX / 2.0;
		for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx + i * phase_x;
			if (x_offset)
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0 * p45;
					//  변환된 자표를 이용해서,
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//  Depth map에 사용될 좌표를 만들어 낸다.
					int  px = (int)(new_pts.x() / 0.1) + voxel_x / 2;
					int  py = (int)(new_pts.y() / 0.1) + voxel_y / 2;
					//  이후에는 0도와 동일한 방법으로 동작하도록 한다.
					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();
						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();
						depth_map_cnt[m_idx]++;
					}
				}
			}
			else
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					int  px = (int)(pts[phase_idx].x() / 0.1) + voxel_x / 2;
					int  py = (int)(pts[phase_idx].y() / 0.1) + voxel_y / 2;

					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();
						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();
						depth_map_cnt[m_idx]++;
					}
				}
			}
		}
	}

	__global__ void kernel_gen_depth_normal_map_v6(
		unsigned char* current_img_0,
		unsigned char* current_img_45,
		const Eigen::Matrix4f dev_camRT,
		const Eigen::Matrix3f dev_cam_tilt,
		const Eigen::Vector3f* pts,
		const Eigen::Vector3f* normals,
		const Eigen::Matrix4f transform_0_pts,
		const Eigen::Matrix4f transform_0_normal,
		const Eigen::Matrix4f transform_45_to_0_pts,
		const Eigen::Matrix4f transform_45_to_0_normal,
		const Eigen::Vector3b* pts_color,
		Eigen::Vector3f* depth_map,
		Eigen::Vector3f* normal_map, // ½A±O±a´E
		unsigned int* depth_color,
		unsigned int* depth_map_cnt,
		const int voxel_x,        // 200 ==> 500
		const int voxel_y,        // 250 ==> 700
		const float x_unit,
		const float y_unit,
		const int phase_x,        // 256
		const int phase_y,        // 480
		const int x_step,         // 2
		const int y_step,         // 12
		const int x_offset,       // 0도 - 0, 45도 - 1
		const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
		const int y_scan,
		const int in_size)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	{
		int threadid = blockIdx.x * blockDim.x + threadIdx.x;

		int _x_width = phase_x / x_step;
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;

		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;

		if (_pidx > phase_x * phase_y - 1)    return;

		float   invalid_float_ = FLT_MAX / 2.0;

		for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx + i * phase_x;

			if (x_offset)
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0_pts * p45;
					//  변환된 자표를 이용해서,
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//auto new_pts = pts[phase_idx];
					auto color = ColorUtil::getPixelCoord_pos_Mix(new_pts, current_img_0, current_img_45, dev_camRT, dev_cam_tilt);

					int  px = (int)((floorf)(new_pts.x() / x_unit) + voxel_x / 2);
					int  py = (int)((floorf)(new_pts.y() / y_unit) + voxel_y / 2);

					auto normal_transformed = transform_45_to_0_normal * Eigen::Vector4f(normals[phase_idx].x(), normals[phase_idx].y(), normals[phase_idx].z(), 0.f);
					auto new_normal = Eigen::Vector3f(normal_transformed.x(), normal_transformed.y(), normal_transformed.z());
					new_normal.normalize();

					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();

						normal_map[m_idx].x() = new_normal.x();
						normal_map[m_idx].y() = new_normal.y();
						normal_map[m_idx].z() = new_normal.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();

						normal_map[m_idx].x() += new_normal.x();
						normal_map[m_idx].y() += new_normal.y();
						normal_map[m_idx].z() += new_normal.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}

				}
			}
			else
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					int  px = (int)(floorf(pts[phase_idx].x() / x_unit)) + voxel_x / 2;
					int  py = (int)(floorf(pts[phase_idx].y() / y_unit)) + voxel_y / 2;
					auto normal_zero = normals[phase_idx];
					auto color = ColorUtil::getPixelCoord_pos_Mix(pts[phase_idx], current_img_0, current_img_45, dev_camRT, dev_cam_tilt);
					normal_zero.normalize();

					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();

						normal_map[m_idx].x() = normal_zero.x();
						normal_map[m_idx].y() = normal_zero.y();
						normal_map[m_idx].z() = normal_zero.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();

						normal_map[m_idx].x() += normal_zero.x();
						normal_map[m_idx].y() += normal_zero.y();
						normal_map[m_idx].z() += normal_zero.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}

				}
			}

		}
	}

	//  mscho   @20240408
//  color map?? ????? ????? ??????.
//  45???? ???, 0?? ?????? ?????? ????? ??????
//  normal map?? ????? ????? ??????.
	__global__ void kernel_gen_depth_normal_map_v5(
		unsigned char* current_img_0,
		unsigned char* current_img_45,
		const Eigen::Matrix4f dev_camRT,
		const Eigen::Matrix3f dev_cam_tilt,
		const Eigen::Vector3f* pts,
		const Eigen::Vector3f* normals,
		const Eigen::Matrix4f transform_0_pts,
		const Eigen::Matrix4f transform_0_normal,
		const Eigen::Matrix4f transform_45_to_0_pts,
		const Eigen::Matrix4f transform_45_to_0_normal,
		const Eigen::Vector3b* pts_color,
		Eigen::Vector3f* depth_map,
		Eigen::Vector3f* normal_map, // ?????
		unsigned int* depth_color,
		unsigned int* depth_map_cnt,
		const int voxel_x,        // 200
		const int voxel_y,        // 250
		const int phase_x,        // 256
		const int phase_y,        // 480
		const int x_step,         // 2
		const int y_step,         // 12
		const int x_offset,       // 0도 - 0, 45도 - 1
		const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
		const int y_scan,
		const int in_size)        // 256 * 480 / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	{
		int threadid = blockIdx.x * blockDim.x + threadIdx.x;

		int _x_width = phase_x / x_step;
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;

		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;

		if (_pidx > phase_x * phase_y - 1)    return;

		float   invalid_float_ = FLT_MAX / 2.0;

		for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx + i * phase_x;

			if (x_offset)
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0_pts * p45;
					//  변환된 자표를 이용해서,
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//auto new_pts = pts[phase_idx];
					auto color = ColorUtil::getPixelCoord_pos_Mix(new_pts, current_img_0, current_img_45, dev_camRT, dev_cam_tilt);

					int  px = (int)((floorf)(new_pts.x() / 0.1) + voxel_x / 2);
					int  py = (int)((floorf)(new_pts.y() / 0.1) + voxel_y / 2);


					auto n45 = Eigen::Vector3f(normals[phase_idx].x(), normals[phase_idx].y(), normals[phase_idx].z());
					auto normal_transformed = transform_45_to_0_normal * Eigen::Vector4f(n45.x(), n45.y(), n45.z(), 0.f);
					auto normal_to_zero = Eigen::Vector3f(normal_transformed.x(), normal_transformed.y(), normal_transformed.z());

					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();

						normal_map[m_idx].x() = normal_to_zero.x();
						normal_map[m_idx].y() = normal_to_zero.y();
						normal_map[m_idx].z() = normal_to_zero.z();

						depth_color[m_idx * 3] = color.x();// pts_color[phase_idx].x();
						depth_color[m_idx * 3 + 1] = color.y();//pts_color[phase_idx].y();
						depth_color[m_idx * 3 + 2] = color.z();//pts_color[phase_idx].z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();

						normal_map[m_idx].x() += normal_to_zero.x();
						normal_map[m_idx].y() += normal_to_zero.y();
						normal_map[m_idx].z() += normal_to_zero.z();

						depth_color[m_idx * 3] += color.x();// pts_color[phase_idx].x();
						depth_color[m_idx * 3 + 1] += color.y();//pts_color[phase_idx].y();
						depth_color[m_idx * 3 + 2] += color.z();//pts_color[phase_idx].z();

						depth_map_cnt[m_idx]++;
					}

				}
			}
			else
			{
				if (pts[phase_idx].x() < invalid_float_)
				{
					int  px = (int)(floorf(pts[phase_idx].x() / 0.1)) + voxel_x / 2;
					int  py = (int)(floorf(pts[phase_idx].y() / 0.1)) + voxel_y / 2;
					auto normal_zero = normals[phase_idx];
					auto color = ColorUtil::getPixelCoord_pos_Mix(pts[phase_idx], current_img_0, current_img_45, dev_camRT, dev_cam_tilt);

					int     m_idx = py * voxel_x + px;
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();

						normal_map[m_idx].x() = normal_zero.x();
						normal_map[m_idx].y() = normal_zero.y();
						normal_map[m_idx].z() = normal_zero.z();

						depth_color[m_idx * 3] = color.x();// pts_color[phase_idx].x();
						depth_color[m_idx * 3 + 1] = color.y();//pts_color[phase_idx].y();
						depth_color[m_idx * 3 + 2] = color.z();//pts_color[phase_idx].z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();

						normal_map[m_idx].x() += normal_zero.x();
						normal_map[m_idx].y() += normal_zero.y();
						normal_map[m_idx].z() += normal_zero.z();

						depth_color[m_idx * 3] += color.x();// pts_color[phase_idx].x();
						depth_color[m_idx * 3 + 1] += color.y();//pts_color[phase_idx].y();
						depth_color[m_idx * 3 + 2] += color.z();//pts_color[phase_idx].z();

						depth_map_cnt[m_idx]++;
					}

				}
			}

		}
	}
	*/

	//	mscho	@20240422
#define	DEPTHMAP_OIT_ENABLE

	//	mscho	@20240422
	//  mutex lock
	//  CUDA  에서 default로 제공하는 mutex가 없으므로 kernel 에서 race condition 을 제어하기 위해서
	//  atomic 함수를 응용한 mutex함수를 만들어서 사용한다.
#ifndef BUILD_FOR_CPU
__device__ void cu_mutex_lock(uint32_t * mutex) {
	while (atomicCAS(mutex, 0, 1) != 0);
}

//	mscho	@20240422
//	mutex unlock
__device__ void cu_mutex_unlock(uint32_t * mutex) {
	atomicExch(mutex, 0);
}
#else 
inline void cu_mutex_lock(std::mutex * mutex) {
	mutex->lock();
}

inline void cu_mutex_unlock(std::mutex * mutex) {
	mutex->unlock();
}
#endif // BUILD_FOR_CPU
//특정 지점(ptPos)가 카메라 시점이나 dlp 시점에서 잘 보이지 않는 위치에 있을 경우 해당 포인트를 걸러주도록 하고
//반사광과 시점간의 각도(specAngle)을 계산해 주는 함수
__device__ bool pointFiltering(Eigen::Vector3f camPos, Eigen::Vector3f dlpPos, Eigen::Vector3f ptPos, Eigen::Vector3f ptNormal, float& alpha, float& specAngle) {
	auto camDir = (camPos - ptPos).normalized();
	auto camAlpha = camDir.dot(ptNormal);
	if (camAlpha < 0.0f) return false;
	auto dlpDir = (dlpPos - ptPos).normalized();
	auto dlpAlpha = dlpDir.dot(ptNormal);
	if (dlpAlpha < 0.0f) return false;

	alpha = dlpAlpha;

	Eigen::Vector3f	reflectVector = -1 * dlpDir + 2.f * ptNormal * ptNormal.dot(dlpDir);
	reflectVector.normalize();
	auto viewDotLight = reflectVector.dot(camDir);
	specAngle = acosf(viewDotLight) / DEF_PI * 180.0;
	return true;
}

__device__ Eigen::Vector3b getConfidenceColor(float value, float gap) {
	float r = 0;
	float g = 0;
	float b = 0;

	auto count = value;
	if (count < 0)
	{
		r = 0;
		g = 0;
		b = 255;
	}
	else if (0 <= count && count < gap)
	{
		r = 0;
		g = count * (255 / gap);
		b = 255;
	}
	else if (gap <= count && count < gap * 2)
	{
		r = 0;
		g = 255;
		b = 255 - (count - gap) * (255 / gap);
	}
	else if (gap * 2 <= count && count < gap * 3)
	{
		r = (count - gap * 2) * (255 / gap);
		g = 255;
		b = 0;
	}
	else if (gap * 3 <= count && count < gap * 4)
	{
		r = 255;
		g = 255 - (count - gap * 3) * (255 / gap);
		b = 0;
	}
	else
	{
		r = 255;
		g = 0;
		b = 0;
	}
	return Eigen::Vector3b(r, g, b);
}

//	mscho	@20250627
//	atomicExch 함수를 이용해서, mutex  를 만들고 이를 사용해서  race condition 을 방지한다.

__global__ void MarchingCubesKernel::kernel_gen_depth_normal_map_v11(
	int  line_idx,	//	mscho	@20250724
	int  y_div,
	int  phase_begin,
	int  phase_width,
	bool bDeepLearningEnable,				// true : DeepLearning enable & initial OK
	const unsigned short* deeplearning_inference,	//	deep learning 추론의 결과가 저장되어 있다.. 400x480
	const unsigned char* current_img_0,
	const unsigned char* current_img_45,
	const Eigen::Vector3f dlp_pos,	// dlp position
	const Eigen::Vector3f cam_pos,	// camera position
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector3f * pts,
	const Eigen::Vector3f * normals,
	const float* confidencemap,
	const Eigen::Matrix4f transform_0_pts,
	const Eigen::Matrix3f transform_0_normal,
	const Eigen::Matrix4f transform_45_to_0_pts,
	const Eigen::Matrix3f transform_45_to_0_normal,
	const Eigen::Vector3b * pts_color,
	Eigen::Vector3f * depth_map,
	Eigen::Vector3f * normal_map, // 노말맵
	float* alpha_map, // 알파맵
	float* specular_map,	// 광원의 반사가 카메라와 이루는 각도 in degree
	unsigned int* depth_color,
	unsigned int* depth_map_cnt, // depthMap의 포인트에 몇 개의 point들이 겹쳐져 있는가. (DepthMap은 패치의 크기와 다른 크기로 생성될 수 있으므로 depth를 작게 만들 때는 포인트들이 겹쳐질 수 있다)
	unsigned short* material_map,
#ifdef BUILD_FOR_CPU
	std::mutex * depth_map_mutex,
#else
	unsigned int* depth_map_mutex,
#endif
	const int voxel_x,        // 200 ==> 500
	const int voxel_y,        // 250 ==> 700	
	const float x_unit,
	const float y_unit,
	const int phase_x,        // PHASE_CX_SIZE_MAX
	const int phase_y,        // PHASE_CY_SIZE_MAX
	const int x_step,         // 2
	const int y_step,         // 12
	const int x_offset,       // 0도 - 0, 45도 - 1
	const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
	const int y_scan,
	const int in_size,        // PHASE_CX_SIZE_MAX * PHASE_CY_SIZE_MAX / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	const float point_mag,
	const Eigen::Vector2f channelSpd)
{
	const float   invalid_float_ = FLT_MAX / 2.0;

#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	{
		//	mscho	@20250724
		if (line_idx < 0)
		{
			if (threadid > phase_x * phase_y - 1)
				kernel_return;
		}
		else
		{
			if (threadid > (phase_width / x_step) * (phase_y / y_div) - 1)
				kernel_return;
		}

		int _x_width = phase_width / x_step;
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;

		int _y;

		if (line_idx < 0)
		{
			_y = _idxy * y_step + y_offset;
		}
		else
		{
			//	mscho	@20250724
			_y = (_idxy * y_div + line_idx) * y_step + y_offset;
		}
		int _x = _idxx * x_step + phase_begin + x_offset;

		int _pidx = _y * phase_x + _x;


#else
	const int threadCount = phase_x * phase_y / x_step / y_step; //phase_x * phase_y / 2 / 6 / 2;
	const int _x_width = phase_x / x_step;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;
		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;
#endif

		Eigen::Vector3b img_color;
		Eigen::Vector3b color;

		for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx + i * phase_x;

			if (x_offset)
			{ // 45도
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음                
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0_pts * p45;
					//  변환된 자표를 이용해서, 
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, new_pts / point_mag, Eigen::Vector3f(p45.head(3)) / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)
						continue;

#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(new_pts / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//    mscho    @20240527
					int  px = (int)((floorf)(new_pts.x() / x_unit + 2500.f) - 2500 + voxel_x / 2);
					int  py = (int)((floorf)(new_pts.y() / y_unit + 2500.f) - 2500 + voxel_y / 2);
#endif//USING_PROJECTION_DEPTHMAP
					//	mscho	@20240611
					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       continue;
					m_idx = py * voxel_x + px;

					Eigen::Vector3f new_normal = transform_45_to_0_normal * normals[phase_idx];// auto 를 쓰면 CPU 용으로 컴파일 할 때 해석이 달라지므로 타입 지정해 주어야 함
					new_normal.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, new_pts, new_normal, alpha, angle_specular_degree)) continue;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = new_pts;
						normal_map[m_idx] = new_normal;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - new_pts.z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (new_pts.z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = new_pts;
								normal_map[m_idx] = new_normal;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();

							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
							/* {
								bool ok = true;
								if (prev_cnt > 2) {
									float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
									if (fabsf(ave_z - new_pts.z()) > 0.15f)
										ok = false;
								}
								if (ok) {
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}*/
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - new_pts.z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = new_pts;
										normal_map[m_idx] = new_normal;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
#error
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();

						normal_map[m_idx].x() = new_normal.x();
						normal_map[m_idx].y() = new_normal.y();
						normal_map[m_idx].z() = new_normal.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();

						normal_map[m_idx].x() += new_normal.x();
						normal_map[m_idx].y() += new_normal.y();
						normal_map[m_idx].z() += new_normal.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
			else
			{ // 0도
				if (pts[phase_idx].x() < invalid_float_)
				{
#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(pts[phase_idx] / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//	mscho	@20240527
					int  px = (int)(floorf(pts[phase_idx].x() / x_unit + 2500.f) - 2500) + voxel_x / 2;
					int  py = (int)(floorf(pts[phase_idx].y() / y_unit + 2500.f) - 2500) + voxel_y / 2;
#endif//USING_PROJECTION_DEPTHMAP

					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       continue;
					m_idx = py * voxel_x + px;

					auto normal_zero = normals[phase_idx];
					auto p0 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					auto p45 = Eigen::Vector3f((transform_45_to_0_pts.inverse() * p0).head(3));
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, pts[phase_idx] / point_mag, p45 / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)	continue;

					normal_zero.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, pts[phase_idx], normal_zero, alpha, angle_specular_degree)) continue;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = pts[phase_idx];
						normal_map[m_idx] = normal_zero;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();
						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (pts[phase_idx].z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = pts[phase_idx];
								normal_map[m_idx] = normal_zero;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();
							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = pts[phase_idx];
										normal_map[m_idx] = normal_zero;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();

						normal_map[m_idx].x() = normal_zero.x();
						normal_map[m_idx].y() = normal_zero.y();
						normal_map[m_idx].z() = normal_zero.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();

						normal_map[m_idx].x() += normal_zero.x();
						normal_map[m_idx].y() += normal_zero.y();
						normal_map[m_idx].z() += normal_zero.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
		}
	}
	}


//	mscho	@20240527
//	atomicExch 함수를 이용해서, mutex  를 만들고 이를 사용해서  race condition 을 방지한다.

__global__ void MarchingCubesKernel::kernel_gen_depth_normal_map_v9(
	bool bDeepLearningEnable,				// true : DeepLearning enable & initial OK
	const unsigned short* deeplearning_inference,	//	deep learning 추론의 결과가 저장되어 있다.. 400x480
	const unsigned char* current_img_0,
	const unsigned char* current_img_45,
	const Eigen::Vector3f dlp_pos,	// dlp position
	const Eigen::Vector3f cam_pos,	// camera position
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector3f * pts,
	const Eigen::Vector3f * normals,
	const float* confidencemap,
	const Eigen::Matrix4f transform_0_pts,
	const Eigen::Matrix3f transform_0_normal,
	const Eigen::Matrix4f transform_45_to_0_pts,
	const Eigen::Matrix3f transform_45_to_0_normal,
	const Eigen::Vector3b * pts_color,
	Eigen::Vector3f * depth_map,
	Eigen::Vector3f * normal_map, // 노말맵
	float* alpha_map, // 알파맵
	float* specular_map,	// 광원의 반사가 카메라와 이루는 각도 in degree
	unsigned int* depth_color,
	unsigned int* depth_map_cnt, // depthMap의 포인트에 몇 개의 point들이 겹쳐져 있는가. (DepthMap은 패치의 크기와 다른 크기로 생성될 수 있으므로 depth를 작게 만들 때는 포인트들이 겹쳐질 수 있다)
	unsigned short* material_map,
#ifdef BUILD_FOR_CPU
	std::mutex * depth_map_mutex,
#else
	unsigned int* depth_map_mutex,
#endif
	const int voxel_x,        // 200 ==> 500
	const int voxel_y,        // 250 ==> 700	
	const float x_unit,
	const float y_unit,
	const int phase_x,        // PHASE_CX_SIZE_MAX
	const int phase_y,        // PHASE_CY_SIZE_MAX
	const int x_step,         // 2
	const int y_step,         // 12
	const int x_offset,       // 0도 - 0, 45도 - 1
	const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
	const int y_scan,
	const int in_size,        // PHASE_CX_SIZE_MAX * PHASE_CY_SIZE_MAX / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	const float point_mag,
	const Eigen::Vector2f channelSpd)
{
	const float   invalid_float_ = FLT_MAX / 2.0;

#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	{
		int _x_width = phase_x / x_step;
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;

		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;

		if (_pidx > phase_x * phase_y - 1)
			kernel_return;
#else
	const int threadCount = phase_x * phase_y / x_step / y_step; //phase_x * phase_y / 2 / 6 / 2;
	const int _x_width = phase_x / x_step;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;
		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;
#endif

		Eigen::Vector3b img_color;
		Eigen::Vector3b color;

		for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx + i * phase_x;

			if (x_offset)
			{ // 45도
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음                
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0_pts * p45;
					//  변환된 자표를 이용해서, 
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, new_pts / point_mag, Eigen::Vector3f(p45.head(3)) / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)
						continue;

#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(new_pts / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//    mscho    @20240527
					int  px = (int)((floorf)(new_pts.x() / x_unit + 2500.f) - 2500 + voxel_x / 2);
					int  py = (int)((floorf)(new_pts.y() / y_unit + 2500.f) - 2500 + voxel_y / 2);
#endif//USING_PROJECTION_DEPTHMAP
					//	mscho	@20240611
					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       continue;
					m_idx = py * voxel_x + px;

					Eigen::Vector3f new_normal = transform_45_to_0_normal * normals[phase_idx];// auto 를 쓰면 CPU 용으로 컴파일 할 때 해석이 달라지므로 타입 지정해 주어야 함
					new_normal.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, new_pts, new_normal, alpha, angle_specular_degree)) continue;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = new_pts;
						normal_map[m_idx] = new_normal;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - new_pts.z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (new_pts.z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = new_pts;
								normal_map[m_idx] = new_normal;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();

							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
							/* {
								bool ok = true;
								if (prev_cnt > 2) {
									float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
									if (fabsf(ave_z - new_pts.z()) > 0.15f)
										ok = false;
								}
								if (ok) {
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}*/
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - new_pts.z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = new_pts;
										normal_map[m_idx] = new_normal;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
#error
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();

						normal_map[m_idx].x() = new_normal.x();
						normal_map[m_idx].y() = new_normal.y();
						normal_map[m_idx].z() = new_normal.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();

						normal_map[m_idx].x() += new_normal.x();
						normal_map[m_idx].y() += new_normal.y();
						normal_map[m_idx].z() += new_normal.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
			else
			{ // 0도
				if (pts[phase_idx].x() < invalid_float_)
				{
#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(pts[phase_idx] / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//	mscho	@20240527
					int  px = (int)(floorf(pts[phase_idx].x() / x_unit + 2500.f) - 2500) + voxel_x / 2;
					int  py = (int)(floorf(pts[phase_idx].y() / y_unit + 2500.f) - 2500) + voxel_y / 2;
#endif//USING_PROJECTION_DEPTHMAP

					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       continue;
					m_idx = py * voxel_x + px;

					auto normal_zero = normals[phase_idx];
					auto p0 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					auto p45 = Eigen::Vector3f((transform_45_to_0_pts.inverse() * p0).head(3));
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, pts[phase_idx] / point_mag, p45 / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)	continue;

					normal_zero.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, pts[phase_idx], normal_zero, alpha, angle_specular_degree)) continue;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = pts[phase_idx];
						normal_map[m_idx] = normal_zero;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();
						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (pts[phase_idx].z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = pts[phase_idx];
								normal_map[m_idx] = normal_zero;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();
							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = pts[phase_idx];
										normal_map[m_idx] = normal_zero;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();

						normal_map[m_idx].x() = normal_zero.x();
						normal_map[m_idx].y() = normal_zero.y();
						normal_map[m_idx].z() = normal_zero.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();

						normal_map[m_idx].x() += normal_zero.x();
						normal_map[m_idx].y() += normal_zero.y();
						normal_map[m_idx].z() += normal_zero.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
		}
	}
	}

//	mscho	@20250313
//	atomicExch 함수를 이용해서, mutex  를 만들고 이를 사용해서  race condition 을 방지한다.
	//	race condition을 제어하기 위해서, 4개의 구역으로 나눠서 처리했었는데.
	//	Atomic 함수를 이용해서, 하나의 kernel로 통합하도록 한다.

__global__ void MarchingCubesKernel::kernel_gen_depth_normal_map_v10(
	bool bDeepLearningEnable,				// true : DeepLearning enable & initial OK
	const unsigned short* deeplearning_inference,	//	deep learning 추론의 결과가 저장되어 있다.. 400x480
	const unsigned char* current_img_0,
	const unsigned char* current_img_45,
	const Eigen::Vector3f dlp_pos,	// dlp position
	const Eigen::Vector3f cam_pos,	// camera position
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt,
	const Eigen::Vector3f * pts,
	const Eigen::Vector3f * normals,
	const Eigen::Matrix4f transform_0_pts,
	const Eigen::Matrix3f transform_0_normal,
	const Eigen::Matrix4f transform_45_to_0_pts,
	const Eigen::Matrix3f transform_45_to_0_normal,
	const Eigen::Vector3b * pts_color,
	Eigen::Vector3f * depth_map,
	Eigen::Vector3f * normal_map, // 노말맵
	float* alpha_map, // 알파맵
	float* specular_map,	// 광원의 반사가 카메라와 이루는 각도 in degree
	unsigned int* depth_color,
	unsigned int* depth_map_cnt, // depthMap의 포인트에 몇 개의 point들이 겹쳐져 있는가. (DepthMap은 패치의 크기와 다른 크기로 생성될 수 있으므로 depth를 작게 만들 때는 포인트들이 겹쳐질 수 있다)
	unsigned short* material_map,
#ifdef BUILD_FOR_CPU
	std::mutex * depth_map_mutex,
#else
	unsigned int* depth_map_mutex,
#endif
	const int voxel_x,        // 200 ==> 500
	const int voxel_y,        // 250 ==> 700	
	const float x_unit,
	const float y_unit,
	const int phase_x,        // PHASE_CX_SIZE_MAX
	const int phase_y,        // PHASE_CY_SIZE_MAX
	const int x_step,         // 2
	const int y_step,         // 12
	const int EnQuadPatch,       // 0도 - 0, 45도 - 1
	const int y_offset,       // 0 : 0~5 라인 검색, 6 : 6~11라인 검색
	const int y_scan,
	const int in_size,        // PHASE_CX_SIZE_MAX * PHASE_CY_SIZE_MAX / 2(가로,2스텝) / 6(세로6라인 검색) / 2 (6
	const float point_mag,
	const Eigen::Vector2f channelSpd)
{
	const float   invalid_float_ = FLT_MAX / 2.0;

#ifndef BUILD_FOR_CPU
	int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	{
		int _x_width = phase_x / x_step;
		int _idxy = threadid / phase_x;
		int _idxx = threadid % phase_x;

		int _y = _idxy;
		int _x = _idxx;

		int _pidx = _y * phase_x + _x;
		bool	bQuadPatch = (_x % 2 ? true : false);

		if (_pidx > phase_x * phase_y - 1)
			kernel_return;
#else
	const int threadCount = phase_x * phase_y / x_step / y_step; //phase_x * phase_y / 2 / 6 / 2;
	const int _x_width = phase_x / x_step;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
		int _idxy = threadid / _x_width;
		int _idxx = threadid % _x_width;
		int _y = _idxy * y_step + y_offset;
		int _x = _idxx * x_step + x_offset;

		int _pidx = _y * phase_x + _x;
#endif

		Eigen::Vector3b img_color;
		Eigen::Vector3b color;

		//for (int i = 0; i < y_scan; i++)
		{
			int phase_idx = _pidx;// +i * phase_x;

			if (bQuadPatch && EnQuadPatch)
			{ // 45도
				if (pts[phase_idx].x() < invalid_float_)
				{
					//  45도의 position 을 읽어온다음                
					auto p45 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					//  45도에서 0도의 공간으로 이동할 수 있는, matrix를 곱해주고
					auto tr_p45 = transform_45_to_0_pts * p45;
					//  변환된 자표를 이용해서, 
					auto new_pts = Eigen::Vector3f(tr_p45.x(), tr_p45.y(), tr_p45.z());
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, new_pts / point_mag, Eigen::Vector3f(p45.head(3)) / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)
						kernel_return;

#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(new_pts / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//    mscho    @20240527
					int  px = (int)((floorf)(new_pts.x() / x_unit + 2500.f) - 2500 + voxel_x / 2);
					int  py = (int)((floorf)(new_pts.y() / y_unit + 2500.f) - 2500 + voxel_y / 2);

#endif//USING_PROJECTION_DEPTHMAP
					//	mscho	@20240611
					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       kernel_return;
					m_idx = py * voxel_x + px;

					Eigen::Vector3f new_normal = transform_45_to_0_normal * normals[phase_idx];// auto 를 쓰면 CPU 용으로 컴파일 할 때 해석이 달라지므로 타입 지정해 주어야 함
					new_normal.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, new_pts, new_normal, alpha, angle_specular_degree)) kernel_return;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = new_pts;
						normal_map[m_idx] = new_normal;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - new_pts.z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (new_pts.z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = new_pts;
								normal_map[m_idx] = new_normal;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();

							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
							/* {
								bool ok = true;
								if (prev_cnt > 2) {
									float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
									if (fabsf(ave_z - new_pts.z()) > 0.15f)
										ok = false;
								}
								if (ok) {
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}*/
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - new_pts.z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = new_pts;
										normal_map[m_idx] = new_normal;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += new_pts;
									normal_map[m_idx] += new_normal;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
#error
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = new_pts.x();
						depth_map[m_idx].y() = new_pts.y();
						depth_map[m_idx].z() = new_pts.z();

						normal_map[m_idx].x() = new_normal.x();
						normal_map[m_idx].y() = new_normal.y();
						normal_map[m_idx].z() = new_normal.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += new_pts.x();
						depth_map[m_idx].y() += new_pts.y();
						depth_map[m_idx].z() += new_pts.z();

						normal_map[m_idx].x() += new_normal.x();
						normal_map[m_idx].y() += new_normal.y();
						normal_map[m_idx].z() += new_normal.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(Eigen::Vector3f(tr_p45.head(3)) / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
			else
			{ // 0도
				if (pts[phase_idx].x() < invalid_float_)
				{
#ifdef USING_PROJECTION_DEPTHMAP
					Eigen::Vector2f imgPixelPos;
					ColorUtil::getPixelCoord_relative(pts[phase_idx] / point_mag, imgPixelPos, dev_camRT, dev_cam_tilt);

					int  px = (int)(float(voxel_x) * imgPixelPos.x());
					int  py = (int)(float(voxel_y) * imgPixelPos.y());
#else
					//	mscho	@20240527
					int  px = (int)(floorf(pts[phase_idx].x() / x_unit + 2500.f) - 2500) + voxel_x / 2;
					int  py = (int)(floorf(pts[phase_idx].y() / y_unit + 2500.f) - 2500) + voxel_y / 2;
#endif//USING_PROJECTION_DEPTHMAP

					int m_idx = 0;

					if (px < 0 || py < 0 || px > voxel_x - 1 || py > voxel_y - 1)       kernel_return;
					m_idx = py * voxel_x + px;

					auto normal_zero = normals[phase_idx];
					auto p0 = Eigen::Vector4f(pts[phase_idx].x(), pts[phase_idx].y(), pts[phase_idx].z(), 1.0f);
					auto p45 = Eigen::Vector3f((transform_45_to_0_pts.inverse() * p0).head(3));
					//	mscho	@20240611
					auto bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, pts[phase_idx] / point_mag, p45 / point_mag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt, channelSpd, pts_color[phase_idx]);// pts_color[phase_idx]); 
					if (!bImg)	kernel_return;

					normal_zero.normalize();

					float alpha = 0.0f;
					float angle_specular_degree = 0.0f;
					if (!pointFiltering(cam_pos, dlp_pos, pts[phase_idx], normal_zero, alpha, angle_specular_degree)) kernel_return;

#ifdef DEPTHMAP_OIT_ENABLE
					cu_mutex_lock(&depth_map_mutex[m_idx]);
					uint32_t prev_cnt = depth_map_cnt[m_idx];
					if (prev_cnt == 0)
					{
						alpha_map[m_idx] = alpha;
						specular_map[m_idx] = angle_specular_degree;
						depth_map[m_idx] = pts[phase_idx];
						normal_map[m_idx] = normal_zero;

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();
						depth_map_cnt[m_idx]++;
					}
					else
					{
						//if you use projection depthmap, there's no structural shadowed area.
						//so you should not exclude such values even though there is some differences between previous value and new value
#ifndef USING_PROJECTION_DEPTHMAP
						float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
						if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
						{
							//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
							if (pts[phase_idx].z() > ave_z)
							{
								// 새로들어온 좌표가 상단이다
								// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
								depth_map_cnt[m_idx] = 1;

								alpha_map[m_idx] = alpha;
								specular_map[m_idx] = angle_specular_degree;
								depth_map[m_idx] = pts[phase_idx];
								normal_map[m_idx] = normal_zero;

								depth_color[m_idx * 3] = color.x();
								depth_color[m_idx * 3 + 1] = color.y();
								depth_color[m_idx * 3 + 2] = color.z();
							}
							else
							{
								//	기존의 좌표가 상단이다.
								//  이 경우에는 새로들어온 좌표를 무시한다.

							}
						}
						else
#endif//USING_PROJECTION_DEPTHMAP
						{
							if (prev_cnt > 3) {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) < ONE_VOXEL_SIZE)
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
							else {
								float	ave_z = depth_map[m_idx].z() / (float)prev_cnt;
								if (fabsf(ave_z - pts[phase_idx].z()) > 1.f)
								{
									//	 저장되어 있는 z값과 새로운 z값의 차이가 1mm가 넘는 경우
									if (pts[phase_idx].z() > ave_z)
									{
										// 새로들어온 좌표가 상단이다
										// 이런 경우에는 기존에 저장되어 있는 값을 지워야 한다.
										depth_map_cnt[m_idx] = 1;

										alpha_map[m_idx] = alpha;
										specular_map[m_idx] = angle_specular_degree;
										depth_map[m_idx] = pts[phase_idx];
										normal_map[m_idx] = normal_zero;

										depth_color[m_idx * 3] = color.x();
										depth_color[m_idx * 3 + 1] = color.y();
										depth_color[m_idx * 3 + 2] = color.z();
									}
								}
								else
								{
									alpha_map[m_idx] += alpha;
									specular_map[m_idx] += angle_specular_degree;
									depth_map[m_idx] += pts[phase_idx];
									normal_map[m_idx] += normal_zero;

									depth_color[m_idx * 3] += color.x();
									depth_color[m_idx * 3 + 1] += color.y();
									depth_color[m_idx * 3 + 2] += color.z();
									depth_map_cnt[m_idx]++;
								}
							}
						}

					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
					cu_mutex_unlock(&depth_map_mutex[m_idx]);
#else // DEPTHMAP_OIT_ENABLE
					if (depth_map_cnt[m_idx] == 0)
					{
						depth_map[m_idx].x() = pts[phase_idx].x();
						depth_map[m_idx].y() = pts[phase_idx].y();
						depth_map[m_idx].z() = pts[phase_idx].z();

						normal_map[m_idx].x() = normal_zero.x();
						normal_map[m_idx].y() = normal_zero.y();
						normal_map[m_idx].z() = normal_zero.z();

						depth_color[m_idx * 3] = color.x();
						depth_color[m_idx * 3 + 1] = color.y();
						depth_color[m_idx * 3 + 2] = color.z();

						depth_map_cnt[m_idx]++;
					}
					else
					{
						depth_map[m_idx].x() += pts[phase_idx].x();
						depth_map[m_idx].y() += pts[phase_idx].y();
						depth_map[m_idx].z() += pts[phase_idx].z();

						normal_map[m_idx].x() += normal_zero.x();
						normal_map[m_idx].y() += normal_zero.y();
						normal_map[m_idx].z() += normal_zero.z();

						depth_color[m_idx * 3] += color.x();
						depth_color[m_idx * 3 + 1] += color.y();
						depth_color[m_idx * 3 + 2] += color.z();

						depth_map_cnt[m_idx]++;
					}
					if (bDeepLearningEnable && deeplearning_inference != nullptr) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(pts[phase_idx] / point_mag, imgXY, dev_camRT, dev_cam_tilt)) {
							material_map[m_idx] = deeplearning_inference[imgXY.y() * CU_CX_SIZE_MAX + imgXY.x()];
						}
					}
#endif // DEPTHMAP_OIT_ENABLE
				}
			}
		}
	}
	}


#ifdef USE_MESH_BASE
#ifndef BUILD_FOR_CPU
/// Use Local, Global CountXYZ
//	mscho	@20240221
struct BuildGridFunctor_v7
{

	//	mscho	@20240221
	HashKey64* hashinfo_voxel;
	HashEntry* hashTable_voxel;
	uint8_t* hashTable_voxel_value;
	//	mscho	@20240221
	HashKey64* hashinfo_triangle;
	HashEntry* hashTable_triangle;
	uint8_t* hashTable_triangle_value;
	//	mscho	@20240221
	HashKey64* hashinfo_vtx;
	HashEntry* hashTable_vtx;
	uint8_t* hashTable_vtx_value;

	voxel_value_t* values;
	unsigned short* voxelValueCounts;
	Eigen::Vector3b* voxelColors;

	size_t localVoxelCountX;
	size_t localVoxelCountY;
	size_t localVoxelCountZ;
	size_t globalVoxelCountX;
	size_t globalVoxelCountY;
	size_t globalVoxelCountZ;
	float localMinX;
	float localMinY;
	float localMinZ;
	float globalMinZ;
	float globalMinX;
	float globalMinY;

	float voxelSize;
	float isoValue;

	Eigen::Matrix4f transform_0;
	Eigen::Matrix4f transform_45;

	unsigned char* img_0;
	unsigned char* img_45;

	Eigen::Vector3f _cam_pos;
	Eigen::Matrix4f  _camRT;
	Eigen::Matrix3f  _cam_tilt;
	Eigen::Matrix3f  _cam_tilt_inv;

	Eigen::Vector<uint32_t, 3>* repos_triangle;
	Eigen::Vector3f* repos_vtx_pos;
	Eigen::Vector3f* repos_vtx_nm;
	Eigen::Vector3b* repos_vtx_color;
	uint32_t* repos_vtx_dupCnt;

	__device__
		void operator()(size_t index)
	{
		//printf("%d, %d, %d,%d, %f, %f, %f, %f,\n", dev_cam_w, dev_cam_h, dev_cam_cx, dev_cam_cy, dev_cam_cfx, dev_cam_cfy, dev_cam_ccx,dev_cam_ccy);
		//tri_cnt[index] = 0;

		auto z = index / (localVoxelCountX * localVoxelCountY);
		auto y = (index % (localVoxelCountX * localVoxelCountY)) / localVoxelCountX;
		auto x = (index % (localVoxelCountX * localVoxelCountY)) % localVoxelCountX;

		MarchingCubes::GRIDCELL gridcell;
		gridcell.p[0] = Eigen::Vector3f(
			localMinX + (float)x * voxelSize,
			localMinY + (float)y * voxelSize,
			localMinZ + (float)z * voxelSize);
		gridcell.p[1] = Eigen::Vector3f(
			localMinX + (float)(x + 1) * voxelSize,
			localMinY + (float)y * voxelSize,
			localMinZ + (float)z * voxelSize);
		gridcell.p[2] = Eigen::Vector3f(
			localMinX + (float)(x + 1) * voxelSize,
			localMinY + (float)y * voxelSize,
			localMinZ + (float)(z + 1) * voxelSize);
		gridcell.p[3] = Eigen::Vector3f(
			localMinX + (float)x * voxelSize,
			localMinY + (float)y * voxelSize,
			localMinZ + (float)(z + 1) * voxelSize);
		gridcell.p[4] = Eigen::Vector3f(
			localMinX + (float)x * voxelSize,
			localMinY + (float)(y + 1) * voxelSize,
			localMinZ + (float)z * voxelSize);
		gridcell.p[5] = Eigen::Vector3f(
			localMinX + (float)(x + 1) * voxelSize,
			localMinY + (float)(y + 1) * voxelSize,
			localMinZ + (float)z * voxelSize);
		gridcell.p[6] = Eigen::Vector3f(
			localMinX + (float)(x + 1) * voxelSize,
			localMinY + (float)(y + 1) * voxelSize,
			localMinZ + (float)(z + 1) * voxelSize);
		gridcell.p[7] = Eigen::Vector3f(
			localMinX + (float)x * voxelSize,
			localMinY + (float)(y + 1) * voxelSize,
			localMinZ + (float)(z + 1) * voxelSize);

		uint32_t  hs_voxel_hashIdx_core;

		for (size_t idx = 0; idx < 8; idx++)
		{
			//	mscho	@20240214
			//auto xGlobalIndex = (size_t)(floorf((gridcell.p[idx].x() - globalMinX) / voxelSize));
			//auto yGlobalIndex = (size_t)(floorf((gridcell.p[idx].y() - globalMinY) / voxelSize));
			//auto zGlobalIndex = (size_t)(floorf((gridcell.p[idx].z() - globalMinZ) / voxelSize));

			//printf("----- %llu %llu %llu\n", xGlobalIndex, yGlobalIndex, zGlobalIndex);

			auto voxel_ = 10.f;

			auto xGlobalIndex = (size_t)floorf((gridcell.p[idx].x()) * voxel_ + 2500.f);
			auto yGlobalIndex = (size_t)floorf((gridcell.p[idx].y()) * voxel_ + 2500.f);
			auto zGlobalIndex = (size_t)floorf((gridcell.p[idx].z()) * voxel_ + 2500.f);



			HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
			auto i = get_hashtable_lookup_idx_func64_v4(hashinfo_voxel, hashTable_voxel, hashTable_voxel_value, key);

			if (idx == 0)
			{
				hs_voxel_hashIdx_core = i;
				if (hs_voxel_hashIdx_core == kEmpty32)	return;
			}

			if (i != kEmpty32)
			{
				auto voxelValueCount = voxelValueCounts[i];

				//gridcell.val[idx] = VV2D(values[i]);
				gridcell.val[idx] = VV2D(values[i]) / (float)voxelValueCount;
				// 
				//printf("values[%d] : %d\tgridcell.val[%llu] : %f\n", i, values[i], idx, gridcell.val[idx]);
				//printf("GlobalIndex : %llu, %llu, %llu\tvalues[%d] : %d\t VV2D[%d] : %f\n", xGlobalIndex, yGlobalIndex, zGlobalIndex, i, values[i], values[i], VV2D(values[i]));
			}
			else
			{
				gridcell.val[idx] = FLT_MAX;
			}
		}

		/*for (size_t i = 0; i < 8; i++)
		{
			if (false == FLT_VALID(gridcell.val[i]))
			{
				return;
			}
		}*/

		int cubeindex = 0;
		float isolevel = isoValue;
		Eigen::Vector3f vertlist[12];

		if (FLT_VALID(gridcell.val[0]) && gridcell.val[0] < isolevel) cubeindex |= 1;
		if (FLT_VALID(gridcell.val[1]) && gridcell.val[1] < isolevel) cubeindex |= 2;
		if (FLT_VALID(gridcell.val[2]) && gridcell.val[2] < isolevel) cubeindex |= 4;
		if (FLT_VALID(gridcell.val[3]) && gridcell.val[3] < isolevel) cubeindex |= 8;
		if (FLT_VALID(gridcell.val[4]) && gridcell.val[4] < isolevel) cubeindex |= 16;
		if (FLT_VALID(gridcell.val[5]) && gridcell.val[5] < isolevel) cubeindex |= 32;
		if (FLT_VALID(gridcell.val[6]) && gridcell.val[6] < isolevel) cubeindex |= 64;
		if (FLT_VALID(gridcell.val[7]) && gridcell.val[7] < isolevel) cubeindex |= 128;

		if (edgeTable[cubeindex] == 0)
		{
			return;
		}

		if (edgeTable[cubeindex] & 1)
			vertlist[0] =
			VertexInterp(isolevel, gridcell.p[0], gridcell.p[1], gridcell.val[0], gridcell.val[1]);
		if (edgeTable[cubeindex] & 2)
			vertlist[1] =
			VertexInterp(isolevel, gridcell.p[1], gridcell.p[2], gridcell.val[1], gridcell.val[2]);
		if (edgeTable[cubeindex] & 4)
			vertlist[2] =
			VertexInterp(isolevel, gridcell.p[2], gridcell.p[3], gridcell.val[2], gridcell.val[3]);
		if (edgeTable[cubeindex] & 8)
			vertlist[3] =
			VertexInterp(isolevel, gridcell.p[3], gridcell.p[0], gridcell.val[3], gridcell.val[0]);
		if (edgeTable[cubeindex] & 16)
			vertlist[4] =
			VertexInterp(isolevel, gridcell.p[4], gridcell.p[5], gridcell.val[4], gridcell.val[5]);
		if (edgeTable[cubeindex] & 32)
			vertlist[5] =
			VertexInterp(isolevel, gridcell.p[5], gridcell.p[6], gridcell.val[5], gridcell.val[6]);
		if (edgeTable[cubeindex] & 64)
			vertlist[6] =
			VertexInterp(isolevel, gridcell.p[6], gridcell.p[7], gridcell.val[6], gridcell.val[7]);
		if (edgeTable[cubeindex] & 128)
			vertlist[7] =
			VertexInterp(isolevel, gridcell.p[7], gridcell.p[4], gridcell.val[7], gridcell.val[4]);
		if (edgeTable[cubeindex] & 256)
			vertlist[8] =
			VertexInterp(isolevel, gridcell.p[0], gridcell.p[4], gridcell.val[0], gridcell.val[4]);
		if (edgeTable[cubeindex] & 512)
			vertlist[9] =
			VertexInterp(isolevel, gridcell.p[1], gridcell.p[5], gridcell.val[1], gridcell.val[5]);
		if (edgeTable[cubeindex] & 1024)
			vertlist[10] =
			VertexInterp(isolevel, gridcell.p[2], gridcell.p[6], gridcell.val[2], gridcell.val[6]);
		if (edgeTable[cubeindex] & 2048)
			vertlist[11] =
			VertexInterp(isolevel, gridcell.p[3], gridcell.p[7], gridcell.val[3], gridcell.val[7]);

		MarchingCubes::TRIANGLE tris[4];
		Eigen::Vector3f nm;
		int ntriang = 0;
		for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
			auto v0 = vertlist[triTable[cubeindex][i]];
			auto v1 = vertlist[triTable[cubeindex][i + 1]];
			auto v2 = vertlist[triTable[cubeindex][i + 2]];

			tris[ntriang].p[0] = v0;
			tris[ntriang].p[1] = v1;
			tris[ntriang].p[2] = v2;
			ntriang++;
		}

		if (ntriang != 0)
		{
			//qDebug("(%d)", ntriang);
			//tri_cnt[index] = ntriang;
		}



		//auto xGlobalIndex = (size_t)(floorf(((localMinX + (float)x * voxelSize) - globalMinX) / voxelSize));
		//auto yGlobalIndex = (size_t)(floorf(((localMinY + (float)y * voxelSize) - globalMinY) / voxelSize));
		//auto zGlobalIndex = (size_t)(floorf(((localMinZ + (float)z * voxelSize) - globalMinZ) / voxelSize));

		//HashKey hs_voxel_key(xGlobalIndex, yGlobalIndex, zGlobalIndex);


		auto voxel_ = 10.f;

		auto xGlobalIndex = (size_t)floorf(((localMinX + (float)x * voxelSize)) * voxel_ + 2500.f);
		auto yGlobalIndex = (size_t)floorf(((localMinY + (float)y * voxelSize)) * voxel_ + 2500.f);
		auto zGlobalIndex = (size_t)floorf(((localMinZ + (float)z * voxelSize)) * voxel_ + 2500.f);

		HashKey hs_voxel_key(xGlobalIndex, yGlobalIndex, zGlobalIndex);

		int	iX = (int)xGlobalIndex;
		int	iY = (int)yGlobalIndex;
		int	iZ = (int)zGlobalIndex;

		Eigen::Vector3f voxel_pos_ = Eigen::Vector3f(
			(localMinX + (float)x * voxelSize), (localMinY + (float)y * voxelSize), (localMinZ + (float)z * voxelSize)
		);


		//auto hs_voxel_hashIdx = get_hashtable_insert_idx_func64(hashinfo_voxel, hashTable_voxel, hs_voxel_key);
		auto hs_voxel_hashIdx = hs_voxel_hashIdx_core;// get_hashtable_lookup_idx_func64_v4(hashinfo_voxel, hashTable_voxel, hashTable_voxel_value, hs_voxel_key);
		if (hs_voxel_hashIdx == kEmpty32)
		{
			//printf("Empty Voxel Hash slot\n");
			return;
		}
		Eigen::Vector3b _voxelColor = voxelColors[hs_voxel_hashIdx];
		//printf("%d, %d, %d\n", _voxelColor.x(), _voxelColor.y(), _voxelColor.z());
		for (size_t i = 0; i < 4; i++)
		{
			if (i < ntriang)
			{
				if (VECTOR3F_VALID_(tris[i].p[0]) && VECTOR3F_VALID_(tris[i].p[1]) && VECTOR3F_VALID_(tris[i].p[2]))
				{
					//calculateNormalfromVertices_v2(const Eigen::Vector3f cam_pos, Eigen::Vector3f* vertex, int sc_usetip)
					//nm = calculateNormalfromVertices(tris[hashSlot_idx].p);
					nm = calculateNormalfromVertices_v2(_cam_pos, tris[i].p, true);

					Eigen::Vector<uint32_t, 3> vtx_hashIdx;

					//	mscho	@20240214
					int i_vertex[3];
					for (int vtx_idx = 0; vtx_idx < 3; vtx_idx++)
					{
						//auto key = HASH_KEY_GEN_VOXEL_VERTEX_64_mm2(tris[i].p[vtx_idx].x(), tris[i].p[vtx_idx].y(), tris[i].p[vtx_idx].z(), 100.0f);


						i_vertex[0] = (int)floorf(tris[i].p[vtx_idx].x() * 100.f + 25000.f);
						i_vertex[1] = (int)floorf(tris[i].p[vtx_idx].y() * 100.f + 25000.f);
						i_vertex[2] = (int)floorf(tris[i].p[vtx_idx].z() * 100.f + 25000.f);

						HaskKey key((uint64_t)(i_vertex[0]), (uint64_t)(i_vertex[1]), (uint64_t)(i_vertex[2]));


						auto ptColor = _voxelColor;//getPixelCoord_pos_Mix(transform_0, transform_45, tris[hashSlot_idx].p[vtx_idx], img_0, img_45, _camRT, _cam_tilt);

						tris[i].p[vtx_idx].x() = (i_vertex[0] - 25000) * 0.01;
						tris[i].p[vtx_idx].y() = (i_vertex[1] - 25000) * 0.01;
						tris[i].p[vtx_idx].z() = (i_vertex[2] - 25000) * 0.01;

						//float distance = norm3df(
						//	voxel_pos_.x() - tris[i].p[vtx_idx].x(),
						//	voxel_pos_.y() - tris[i].p[vtx_idx].y(),
						//	voxel_pos_.z() - tris[i].p[vtx_idx].z()
						//);

						//if (distance > 0.2)
						//{
						//	printf("(distance : %f)Voxel Pos : %f, %f, %f ==>  vertex %f, %f, %f \n",
						//		distance,
						//		voxel_pos_.x(), voxel_pos_.y(), voxel_pos_.z(),
						//		tris[i].p[vtx_idx].x(),
						//		tris[i].p[vtx_idx].y(),
						//		tris[i].p[vtx_idx].z()
						//	);

						//}

						vtx_hashIdx[vtx_idx] = _hashtable_insert_vertex_func64_v5(
							hashinfo_vtx, hashTable_vtx, hashTable_vtx_value,
							repos_vtx_dupCnt, repos_vtx_pos, repos_vtx_nm, repos_vtx_color,
							tris[i].p[vtx_idx], nm, ptColor, true,
							key
						);
						//printf("[%lld]-vtx_hashIdx[%d] chk TriIdx %d\n", i, vtx_idx, vtx_hashIdx[vtx_idx]);
					}
					//printf("\n\t\tvertex(0) = %f, %f, %f / %f, %f, %f / %f, %f, %f \n\t\tvertex(1) = %f, %f, %f / %f, %f, %f / %f, %f, %f\n\n"
					//	, tris[i].p[0].x(), tris[i].p[0].y(), tris[i].p[0].z()
					//	, tris[i].p[1].x(), tris[i].p[1].y(), tris[i].p[1].z()
					//	, tris[i].p[2].x(), tris[i].p[2].y(), tris[i].p[2].z()
					//	, repos_vtx_pos[vtx_hashIdx.x()].x(), repos_vtx_pos[vtx_hashIdx.x()].y(), repos_vtx_pos[vtx_hashIdx.x()].z()
					//	, repos_vtx_pos[vtx_hashIdx.y()].x(), repos_vtx_pos[vtx_hashIdx.y()].y(), repos_vtx_pos[vtx_hashIdx.y()].z()
					//	, repos_vtx_pos[vtx_hashIdx.z()].x(), repos_vtx_pos[vtx_hashIdx.z()].y(), repos_vtx_pos[vtx_hashIdx.z()].z()
					//
					//);
					//printf("[%lld] chk TriIdx %d, %d, %d \n", i, vtx_hashIdx[0], vtx_hashIdx[1], vtx_hashIdx[2]);

					auto triangle_hashidx = _hashtable_insert_triangle_idx_func64_v4(
						hashinfo_triangle, hashTable_triangle, hashTable_triangle_value,
						repos_triangle, i, vtx_hashIdx,
						hs_voxel_key);
					if (triangle_hashidx == kEmpty32)	printf("Voxel slot is not occupied....\n");
				}

			}
			//printf("%f, %f, %f\n", tris[i].p->x(), tris[i].p->y(), tris[i].p->z());
		}
	}

	__device__ bool getPixelCoord_uv(const Eigen::Matrix4f icp_transform_, const Eigen::Vector3f vertex, Eigen::Vector2f& cam_pos, const Eigen::Matrix4f dev_camRT, const Eigen::Matrix3f dev_cam_tilt)
	{
		//  global 좌표계의 position을 읽어오고             
		auto p_global = Eigen::Vector4f(vertex.x(), vertex.y(), vertex.z(), 1.0f);
		//  local 좌표계로 이동하도록 inverse transform matrix를 곱해주고
		auto tr_local = icp_transform_.inverse() * p_global;
		//  변환된 자표를 이용해서, Local position을 만든다음
		auto local_pts = Eigen::Vector3f(tr_local.x(), tr_local.y(), tr_local.z());
		auto inPos = Eigen::Vector4f(local_pts.x(), local_pts.y(), local_pts.z(), 1.0f);
		auto camPos = dev_camRT * inPos;

		float rx = camPos[0] / camPos[2];
		float ry = camPos[1] / camPos[2];

		Eigen::Vector3f CamPos3f(rx, ry, 1);

		const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
		float tx = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
		float ty = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();

		cam_pos.x() = (tx * dev_cam_cfx + dev_cam_ccx) - 0.5;
		cam_pos.y() = (ty * dev_cam_cfy + dev_cam_ccy) - 0.5;

		if (cam_pos.x() < 0. || cam_pos.x() >= (float)dev_cam_w || cam_pos.y() < 0 || cam_pos.y() >= (float)dev_cam_h)
			return false;
		else
			return true;
	}

	__device__ bool getPixelCoord_pos(const Eigen::Matrix4f icp_transform_, const Eigen::Vector3f vertex, Eigen::Vector<int, 2>& cam_pos, const Eigen::Matrix4f dev_camRT, const Eigen::Matrix3f dev_cam_tilt)
	{
		//  global 좌표계의 position을 읽어오고             
		auto p_global = Eigen::Vector4f(vertex.x(), vertex.y(), vertex.z(), 1.0f);
		//  local 좌표계로 이동하도록 inverse transform matrix를 곱해주고
		auto tr_local = icp_transform_.inverse() * p_global;
		//  변환된 자표를 이용해서, Local position을 만든다음
		auto local_pts = Eigen::Vector3f(tr_local.x(), tr_local.y(), tr_local.z());
		auto inPos = Eigen::Vector4f(local_pts.x(), local_pts.y(), local_pts.z(), 1.0f);
		auto camPos = dev_camRT * inPos;


		float rx = camPos[0] / camPos[2];
		float ry = camPos[1] / camPos[2];

		Eigen::Vector3f CamPos3f(rx, ry, 1);

		const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
		float tx = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
		float ty = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();

		float u = (tx * dev_cam_cfx + dev_cam_ccx) - 0.5;
		float v = (ty * dev_cam_cfy + dev_cam_ccy) - 0.5;

		cam_pos.x() = (int)((tx * dev_cam_cfx + dev_cam_ccx) - 0.5);
		cam_pos.y() = (int)((ty * dev_cam_cfy + dev_cam_ccy) - 0.5);

		if (cam_pos.x() < 0 || cam_pos.x() > dev_cam_w - 1 || cam_pos.y() < 0 || cam_pos.y() > dev_cam_h - 1)
			return false;
		else
			return true;
	}

	// 0도 , 45도 모두 읽어서, 평균값으로 texture color를 만들어 내는 방법
	__device__ Eigen::Vector3b getPixelCoord_pos_Mix(
		const Eigen::Matrix4f icp_transform_0_,
		const Eigen::Matrix4f icp_transform_45_,
		const Eigen::Vector3f vertex,
		const unsigned char* img0_,
		const unsigned char* img45_,
		const Eigen::Matrix4f dev_camRT,
		const Eigen::Matrix3f dev_cam_tilt
	)
	{

		Eigen::Vector<int, 2> img_pixel_pos_0;
		Eigen::Vector<int, 2> img_pixel_pos_45;
		Eigen::Vector3b cam_pixel;

		// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
		bool bimg_area_0 = getPixelCoord_pos(icp_transform_0_, vertex, img_pixel_pos_0, dev_camRT, dev_cam_tilt);
		bool bimg_area_45 = getPixelCoord_pos(icp_transform_45_, vertex, img_pixel_pos_45, dev_camRT, dev_cam_tilt);
		//qDebug("img_pixel_pos_0 = %d, %d, %s / img_pixel_pos_45 = %d, %d , %s, ",
		//	img_pixel_pos_0.x(), img_pixel_pos_0.y(), bimg_area_0, img_pixel_pos_45.x(), img_pixel_pos_45.y(), bimg_area_45);

		if (bimg_area_0 && bimg_area_45)
		{
			size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
			size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

			cam_pixel.x() = (unsigned char)(((int)img0_[img_index_offset_0] + (int)img45_[img_index_offset_45]) / 2);
			cam_pixel.y() = (unsigned char)(((int)img0_[img_index_offset_0 + 1] + (int)img45_[img_index_offset_45 + 1]) / 2);
			cam_pixel.z() = (unsigned char)(((int)img0_[img_index_offset_0 + 2] + (int)img45_[img_index_offset_45 + 2]) / 2);
		}
		else
		{
			size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
			size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

			if (bimg_area_0)
			{
				cam_pixel.x() = img0_[img_index_offset_0];
				cam_pixel.y() = img0_[img_index_offset_0 + 1];
				cam_pixel.z() = img0_[img_index_offset_0 + 2];
			}
			else 	if (bimg_area_45)
			{
				cam_pixel.x() = img45_[img_index_offset_45];
				cam_pixel.y() = img45_[img_index_offset_45 + 1];
				cam_pixel.z() = img45_[img_index_offset_45 + 2];
			}
			else
			{
				// 45도에서도 벗어난 다면
	// Error 처리
				cam_pixel.x() = 0;
				cam_pixel.y() = 0;
				cam_pixel.z() = 0;
			}
		}

		return cam_pixel;
	}

	// 여기에서 Vertex정보는 global 좌표계이어야 한다..
	// 만약에 Local 좌표계라고 한다면..
	// RT 를 Identity 로 보내면 된다.
	__device__ Eigen::Vector3b getPixelCoord_pos_full(
		const Eigen::Matrix4f icp_transform_0_,
		const Eigen::Matrix4f icp_transform_45_,
		const Eigen::Vector3f vertex,
		const unsigned char* img0_,
		const unsigned char* img45_,
		Eigen::Vector<int, 2>& cam_pos,
		const Eigen::Matrix4f dev_camRT,
		const Eigen::Matrix3f dev_cam_tilt
	)
	{

		Eigen::Vector<int, 2> img_pixel_pos;

		// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
		if (getPixelCoord_pos(icp_transform_0_, vertex, img_pixel_pos, dev_camRT, dev_cam_tilt))
		{
			Eigen::Vector3b cam_pixel;
			size_t  img_index_offset = (img_pixel_pos.x() * dev_cam_w + img_pixel_pos.x()) * 4;

			cam_pixel.x() = img0_[img_index_offset];
			cam_pixel.y() = img0_[img_index_offset + 1];
			cam_pixel.z() = img0_[img_index_offset + 2];

			return cam_pixel;
		}
		else
		{
			// 0도 이미지의 좌표를 벗어난 다면, 45도 만으로 만들 수 있는 영역이므로
			// 45도 RT와 Image를 이용해서 다시 한번 수행하도록 한다.
			Eigen::Vector3b cam_pixel;
			if (getPixelCoord_pos(icp_transform_45_, vertex, img_pixel_pos, dev_camRT, dev_cam_tilt))
			{
				size_t  img_index_offset = (img_pixel_pos.x() * dev_cam_w + img_pixel_pos.x()) * 4;
				cam_pixel.x() = img45_[img_index_offset];
				cam_pixel.y() = img45_[img_index_offset + 1];
				cam_pixel.z() = img45_[img_index_offset + 2];
			}
			else
			{
				// 45도에서도 벗어난 다면
				// Error 처리
				cam_pixel.x() = 0;
				cam_pixel.y() = 0;
				cam_pixel.z() = 0;
			}
			return cam_pixel;
		}
	}
#endif

#pragma region OLD
	/*
	__device__
		Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
		//Eigen::Vector3f VertexInterp_old(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
	{
		float mu;
		Eigen::Vector3f p;

		if (fabsf(isolevel - valp1) < 0.00001f)
			return(p1);
		if (fabsf(isolevel - valp2) < 0.00001f)
			return(p2);
		if (fabsf(valp1 - valp2) < 0.00001f)
			return(p1);
		mu = (isolevel - valp1) / (valp2 - valp1);
		p.x() = p1.x() + mu * (p2.x() - p1.x());
		p.y() = p1.y() + mu * (p2.y() - p1.y());
		p.z() = p1.z() + mu * (p2.z() - p1.z());

		return p;
	}

	__device__
		Eigen::Vector3f VertexInterp_old(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
		//Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
	{
#pragma region Using Truncation
			float mu;
			Eigen::Vector3f p;

			if (false == FLT_VALID(valp1))
				return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

			if (false == FLT_VALID(valp2))
				return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#define unit_size_0 100.0f
#define unit_size_offset 2500.0f


			if (fabsf(isolevel - valp1) < 0.00001f)
			{
				p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
				return p;
			}

			if (fabsf(isolevel - valp2) < 0.00001f)
			{
				p.x() = floorf(p2.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p2.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p2.z() * unit_size_0) / unit_size_0;
				return p;
			}

			if (fabsf(valp1 - valp2) < 0.00001f)
			{
				p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
				return p;
			}

			mu = (isolevel - valp1) / (valp2 - valp1);
			p.x() = p1.x() + mu * (p2.x() - p1.x());
			p.x() = floorf(p.x() * unit_size_0 + 0.5) / unit_size_0;
			p.y() = p1.y() + mu * (p2.y() - p1.y());
			p.y() = floorf(p.y() * unit_size_0 + 0.5) / unit_size_0;
			p.z() = p1.z() + mu * (p2.z() - p1.z());
			p.z() = floorf(p.z() * unit_size_0 + 0.5) / unit_size_0;

			//p.x() = ((int)(p.x() * unit_size_0 + 0.5)) / unit_size_0;
			//p.y() = ((int)(p.y() * unit_size_0 + 0.5)) / unit_size_0;
			//p.z() = ((int)(p.z() * unit_size_0 + 0.5)) / unit_size_0;

			return p;
#pragma endregion

#pragma region Original
			//float mu = 0.0f;
			//Eigen::Vector3f p = p1;

			//if (fabsf(isolevel - valp1) < 0.00001f)
			//	return(p1);
			//if (fabsf(isolevel - valp2) < 0.00001f)
			//	return(p2);
			//if (fabsf(valp1 - valp2) < 0.00001f)
			//	return(p1);
			//mu = (isolevel - valp1) / (valp2 - valp1);
			//p.x() = p1.x() + mu * (p2.x() - p1.x());
			//p.y() = p1.y() + mu * (p2.y() - p1.y());
			//p.z() = p1.z() + mu * (p2.z() - p1.z());

			//return p;
#pragma endregion
		}
		*/
#pragma endregion

#ifndef BUILD_FOR_CPU
	__device__
		Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
	{
#pragma region Using Truncation
		float mu;
		Eigen::Vector3f p;

		if (false == FLT_VALID(valp1))
			return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

		if (false == FLT_VALID(valp2))
			return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#define unit_size_0 100.0f
#define unit_size_offset 25000.0f


		if (fabsf(isolevel - valp1) < 0.001f)
		{
			p.x() = (floorf(p1.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.y() = (floorf(p1.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.z() = (floorf(p1.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			return p;
		}

		if (fabsf(isolevel - valp2) < 0.001f)
		{
			p.x() = (floorf(p2.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.y() = (floorf(p2.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.z() = (floorf(p2.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			return p;
		}

		if (fabsf(valp1 - valp2) < 0.001f)
		{
			p.x() = (floorf(p1.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.y() = (floorf(p1.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			p.z() = (floorf(p1.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
			return p;
		}

		mu = (isolevel - valp1) / (valp2 - valp1);
		p.x() = p1.x() + mu * (p2.x() - p1.x());
		p.x() = (floorf(p.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.y() = p1.y() + mu * (p2.y() - p1.y());
		p.y() = (floorf(p.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.z() = p1.z() + mu * (p2.z() - p1.z());
		p.z() = (floorf(p.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;

		//p.x() = ((int)(p.x() * unit_size_0 + 0.5)) / unit_size_0;
		//p.y() = ((int)(p.y() * unit_size_0 + 0.5)) / unit_size_0;
		//p.z() = ((int)(p.z() * unit_size_0 + 0.5)) / unit_size_0;

		return p;
#pragma endregion

#pragma region Original
		//float mu = 0.0f;
		//Eigen::Vector3f p = p1;

		//if (fabsf(isolevel - valp1) < 0.00001f)
		// return(p1);
		//if (fabsf(isolevel - valp2) < 0.00001f)
		// return(p2);
		//if (fabsf(valp1 - valp2) < 0.00001f)
		// return(p1);
		//mu = (isolevel - valp1) / (valp2 - valp1);
		//p.x() = p1.x() + mu * (p2.x() - p1.x());
		//p.y() = p1.y() + mu * (p2.y() - p1.y());
		//p.z() = p1.z() + mu * (p2.z() - p1.z());

		//return p;
#pragma endregion
	}

	__device__ Eigen::Vector3f calculateNormalfromVertices(const Eigen::Vector3f* vertex) {
		auto ab = vertex[1] - vertex[0];
		auto ac = vertex[2] - vertex[0];
		auto vertex_normal = ab.cross(ac);
		vertex_normal.normalize();
		return vertex_normal;
	}
	//  Face의 Normal을 계산하고, Camera 방향  Vector와  dot연산을 하면
	//  해당 face가 camera 방향인지, 반대 방향인지를 알수 가 있다
	//  반대방향이라면, Normal 부호를 바꿔주고
	//  Face의  Vertex순서를 1, 2를 바꾸어 주면.. 된다

	 // sc_camera_pos
	__device__ Eigen::Vector3f calculateNormalfromVertices_v2(const Eigen::Vector3f cam_pos, Eigen::Vector3f* vertex, int sc_usetip)
	{
		Eigen::Vector3f cam_dir;
		auto ab = vertex[1] - vertex[0];
		auto ac = vertex[2] - vertex[0];
		//auto face_normal = ab.cross(ac);
		Eigen::Vector3f face_normal = CROSS(ab, ac);
		face_normal.normalize();
		//NORMALIZE(face_normal);

		if (sc_usetip)
		{
			cam_dir.x() = cam_pos.x() - face_normal.x();
			cam_dir.y() = cam_pos.y() - face_normal.y();
			cam_dir.z() = cam_pos.z() - face_normal.z();
		}
		else
		{
			cam_dir.x() = cam_pos.x() - face_normal.x();
			cam_dir.y() = -cam_pos.y() - face_normal.y();
			cam_dir.z() = cam_pos.z() - face_normal.z();
		}
		cam_dir.normalize();
		//NORMALIZE(cam_dir);
		//cam_dir = -cam_dir;

		auto face_dir = face_normal.dot(cam_dir);

		/*if (face_dir < 0.f)
		{
			auto temp_vertex = vertex[1];
			vertex[1] = vertex[2];
			vertex[2] = temp_vertex;

			face_normal = -face_normal;
		}*/

		return (Eigen::Vector3f)face_normal;
	}
};

#endif
#endif

void MarchingCubes::InitGlobalVoxelValues(bool initGlobalValues) {
	if (false == globalVoxelValuesInitialized)
	{
		if (initGlobalValues)
		{
#ifdef USE_GLOBAL_VOXEL
			globalVoxelValues = thrust::device_vector<float>(500 * 500 * 500, FLT_MAX);
			globalVoxelWeights = thrust::device_vector<float>(500 * 500 * 500, 0.0f);
			globalVoxelPositions = thrust::device_vector<Eigen::Vector3f>(500 * 500 * 500, Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
			globalVoxelNormals = thrust::device_vector<Eigen::Vector3f>(500 * 500 * 500, Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
#endif
		}

		//pHashManager->init_cuda_hashtable();
		//pHashManager->set_hashtable_capacities();

		globalHash_info = pHashManager->hashGlobalVoxel_info64;
		globalHash_info_host = pHashManager->hashGlobalVoxel_info64_host;
		//globalHash_value = pHashManager->hashGlobalVoxel64_value;

#ifdef USE_MESH_BASE
		hInfo_global_vtx = pHashManager->hashGlobalVertex_info64;
		hInfo_global_vtx_host = pHashManager->hashGlobalVertex_info64_host;
		hTable_global_vtx = pHashManager->hashGlobalVertex64;
		hTable_global_vtx_value = pHashManager->hashGlobalVertex64_value;
#endif

#ifdef USE_MESH_BASE
		hInfo_global_tri = pHashManager->hashGlobalTriangle_info64;
		hInfo_global_tri_host = pHashManager->hashGlobalTriangle_info64_host;
		hTable_global_tri = pHashManager->hashGlobalTriangle64;
		hTable_global_tri_value = pHashManager->hashGlobalTriangle64_value;
#endif

#ifndef BUILD_FOR_CPU
		//	mscho	@20240805
		cudaMallocHost(&used_cnt_HashVoxel_h, sizeof(uint32_t));
		cudaMallocHost(&used_cnt_Extract_h, sizeof(uint32_t));

		cudaMalloc(&used_cnt_HashVoxel, sizeof(uint32_t));
		cudaMalloc(&used_cnt_localContains, sizeof(uint32_t));
		cudaMalloc(&used_cnt_Extract, sizeof(uint32_t));
		cudaMalloc(&cnt_averageRes, sizeof(uint32_t));

		//	mscho	@20250207
		cudaMallocHost(&host_Count_HashTableUsed, sizeof(uint32_t));
		cudaMallocHost(&host_Count_AvePoints, sizeof(uint32_t));

		//	mscho	@20250228
		cudaMallocHost(&host_used_cnt_HashVoxel, sizeof(uint32_t));
		cudaMallocHost(&host_Count_avgArea, sizeof(uint32_t));
#else
		used_cnt_HashVoxel_h = new uint32_t;
		*used_cnt_HashVoxel_h = 0;
		used_cnt_Extract_h = new uint32_t;
		*used_cnt_Extract_h = 0;

		used_cnt_HashVoxel = new uint32_t;
		used_cnt_localContains = new uint32_t;
		used_cnt_Extract = new uint32_t;
		cnt_averageRes = new uint32_t;

#endif
		globalVoxelValuesInitialized = true;
	}
}

bool MarchingCubes::InitVoxelMemory(size_t hashTableSize_Voxel_max, size_t hashTableSize_Voxel_half) {
	m_MC_voxelValues.resize(hashTableSize_Voxel_max, FLT_MAX);
	m_MC_voxelValueCounts.resize(hashTableSize_Voxel_max, USHRT_MAX);
	m_MC_voxelPositions.resize(hashTableSize_Voxel_max, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
	m_MC_voxelNormals.resize(hashTableSize_Voxel_max, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
	m_MC_voxelColors.resize(hashTableSize_Voxel_max, Eigen::Vector3b(0, 0, 0));
	m_MC_voxelColorScores.resize(hashTableSize_Voxel_max, 0);
	m_MC_voxelSegmentations.resize(hashTableSize_Voxel_max, 0);
	VoxelExtraAttrib initialVoxelExtraAttrib = { 0 };
	m_MC_voxelExtraAttribs.resize(hashTableSize_Voxel_max, initialVoxelExtraAttrib);
#ifdef USE_EXPERIMENTAL_COLOR_OPT2
	m_MC_voxelColorReconData.resize(hashTableSize_Voxel_max, ColorReconData());
#endif//USE_EXPERIMENTAL_COLOR_OPT2

	return true;
}

void MarchingCubes::FreeVoxelMemory() {
	m_MC_voxelValues.clear();
	thrust::device_vector<float>().swap(m_MC_voxelValues);
	m_MC_voxelValueCounts.clear();
	thrust::device_vector<unsigned short>().swap(m_MC_voxelValueCounts);
	m_MC_voxelPositions.clear();
	thrust::device_vector<Eigen::Vector3f>().swap(m_MC_voxelPositions);
	m_MC_voxelNormals.clear();
	thrust::device_vector<Eigen::Vector3f>().swap(m_MC_voxelNormals);
	m_MC_voxelColors.clear();
	thrust::device_vector<Eigen::Vector3b>().swap(m_MC_voxelColors);
	m_MC_voxelColorScores.clear();
	thrust::device_vector<float>().swap(m_MC_voxelColorScores);
	m_MC_voxelSegmentations.clear();
	thrust::device_vector<char>().swap(m_MC_voxelSegmentations);

	VoxelExtraAttrib initialVoxelExtraAttrib = { 0 };
	m_MC_voxelExtraAttribs.clear();
	thrust::device_vector<VoxelExtraAttrib>().swap(m_MC_voxelExtraAttribs);
}

void MarchingCubes::ResetVoxels(cached_allocator & alloc, CUstream_st * stream) {
	m_MC_voxelValueCounts.resize(hashTableSize_Voxel_max);

	m_MC_voxelPositions.resize(hashTableSize_Voxel_max);
	m_MC_voxelColors.resize(hashTableSize_Voxel_max);

	thrust::fill(
		thrust::cuda::par_nosync(alloc).on(stream),
		m_MC_voxelValues.begin(), m_MC_voxelValues.end(), VOXEL_INVALID);// VOXEL_INVALID);
	thrust::fill(
		thrust::cuda::par_nosync(alloc).on(stream),
		m_MC_voxelValueCounts.begin(), m_MC_voxelValueCounts.end(), USHRT_MAX);
	thrust::fill(
		thrust::cuda::par_nosync(alloc).on(stream),
		m_MC_voxelSegmentations.begin(), m_MC_voxelSegmentations.end(), 0);// VOXEL_INVALID);
#ifndef AARON_TEST
	//thrust::fill(
	//	thrust::cuda::par_nosync(alloc).on(m_stream),
	//	m_MC_voxelPositions.begin(), m_MC_voxelPositions.end(), Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX));
#endif
}

#ifndef BUILD_FOR_CPU
// 임의의 데이터를 정렬하는 간단한 퀵소트 커널 함수
// in_data : sorting  된 data 배열
// 중복 제거된 data가 저정될 배열
// tgt_size :  vertex의 갯수... avertage를 위해서 모아놓은 단위
// src_size : average를 수행할 vertex의 갯수
// output_size : 각 vertex에서 average에 사용할, vertex의 중복 제거된 갯수 - 최초에  fill (0)  를 하고서 사용해야 한다.
__global__ void UniqueRemoveKernel(uint32_t * in_data, uint32_t * out_data, uint32_t tgt_size, uint32_t src_size, uint32_t * output_size) {
	uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid > (src_size * tgt_size) - 1)     return;

	uint32_t    vertex_no = threadid / tgt_size;
	uint32_t    tgt_idx = threadid % tgt_size;

	uint32_t* sort_data = (uint32_t*)&in_data[vertex_no * tgt_size];
	uint32_t* uniq_data = (uint32_t*)&out_data[vertex_no * tgt_size];

	if (tgt_idx == 0) {
		uniq_data[0] = sort_data[0]; // 첫 번째 요소는 항상 포함
		return;
	}

	// 중복 요소가 아니면 output 배열에 추가
	if (sort_data[tgt_idx] != sort_data[tgt_idx + 1]) {
		auto new_idx = atomicAdd(&output_size[vertex_no], 1); // output_size를 안전하게 증가시키고 새 인덱스를 얻음
		uniq_data[new_idx] = sort_data[tgt_idx + 1];
	}
}

// 임의의 데이터에서 EmptyValue를 먼저 제거하는 함수
// in_data : sorting  된 data 배열
// 중복 제거된 data가 저정될 배열
// tgt_size :  vertex의 갯수... avertage를 위해서 모아놓은 단위
// src_size : average를 수행할 vertex의 갯수
// output_size : 각 vertex에서 average에 사용할, vertex의 중복 제거된 갯수 - 최초에  fill (0)  를 하고서 사용해야 한다.
__global__ void UniqueRemoveEmptyKernel(uint32_t * in_data, uint32_t * out_data, uint32_t tgt_size, uint32_t src_size, uint32_t * output_size, uint32_t EmptyValue) {
	uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid > (src_size * tgt_size) - 1)     return;

	uint32_t    vertex_no = threadid / tgt_size;
	uint32_t    tgt_idx = threadid % tgt_size;

	uint32_t* sort_data = (uint32_t*)&in_data[vertex_no * tgt_size];
	uint32_t* uniq_data = (uint32_t*)&out_data[vertex_no * tgt_size];

	// 중복 요소가 아니면 output 배열에 추가
	if (sort_data[tgt_idx] != EmptyValue) {
		auto new_idx = atomicAdd(&output_size[vertex_no], 1); // output_size를 안전하게 증가시키고 새 인덱스를 얻음
		uniq_data[new_idx] = sort_data[tgt_idx];
		sort_data[tgt_idx] = EmptyValue;
	}
}

// 임의의 데이터에서 EmptyValue를 먼저 제거하는 함수
// in_data : sorting  된 data 배열
// 중복 제거된 data가 저정될 배열
// tgt_size :  vertex의 갯수... avertage를 위해서 모아놓은 단위
// src_size : average를 수행할 vertex의 갯수
// output_size : 각 vertex에서 average에 사용할, vertex의 중복 제거된 갯수 - 최초에  fill (0)  를 하고서 사용해야 한다.
__global__ void UniqueRemoveDuplicationKernel(uint32_t * in_data, uint32_t * out_data, uint32_t tgt_size, uint32_t src_size, uint32_t * output_size) {
	uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid > (src_size * tgt_size) - 1)     return;

	uint32_t    vertex_no = threadid / tgt_size;
	uint32_t    tgt_idx = threadid % tgt_size;

	//if (tgt_idx > output_size[vertex_no] - 1)     return;

	uint32_t* sort_data = (uint32_t*)&in_data[vertex_no * tgt_size];
	uint32_t* uniq_data = (uint32_t*)&out_data[vertex_no * tgt_size];

	bool    Before_on = false;
	bool    After_on = false;

	if (tgt_idx == 0)
	{
		uniq_data[0] = sort_data[0];
		return;
	}
	else
	{
		for (int i = 0; i < tgt_idx; i++)
		{
			if (sort_data[tgt_idx] == sort_data[i])
			{
				return;
			}
		}
		uniq_data[tgt_idx] = sort_data[tgt_idx];
	}
}

// 입렵버퍼 단일사용함수, 중복 발견 시 현재 데이터를 empty처리
__global__ void UniqueRemoveDuplicationKernel_v2(
	uint32_t * in_data, uint32_t tgt_size, uint32_t src_size, uint32_t EmptyValue)
{
	uint32_t threadid = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadid > (src_size * tgt_size) - 1)     return;

	uint32_t    vertex_no = threadid / tgt_size;
	uint32_t    tgt_idx = threadid % tgt_size;

	uint32_t* sort_data = (uint32_t*)&in_data[vertex_no * tgt_size];

	if (tgt_idx == 0)
	{
		return;
	}
	else
	{
		for (int i = 0; i < tgt_idx; i++)
		{
			if (sort_data[tgt_idx] == sort_data[i])
			{
				sort_data[tgt_idx] = EmptyValue;
				return;
			}
		}
	}
}
#ifdef USE_MESH_BASE
Eigen::Vector2i MarchingCubes::avgRenderData_PtsNm(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<uint32_t>&vtx_replace_idx
	, thrust::device_vector<Eigen::Vector3f>&vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>&vtx_color_repos
	, Eigen::Vector<uint32_t, 3>*view_triangles
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, thrust::device_vector<uint32_t>&hash_usedCheckBuffer
	, size_t	view_triSize
	, size_t view_vtxSize
	, cached_allocator * alloc_, CUstream_st * st)
{
	nvtxRangePushA("avgPtsNm");
	qDebug("avgPtsNm");

	static	int cntFile = 0;
	char szTemp[128];
	sprintf(szTemp, "%04d", cntFile);

#if 0
	//Eigen::Vector3f* host_pts = new Eigen::Vector3f[view_vtxSize];
//Eigen::Vector3f* host_nms = new Eigen::Vector3f[view_vtxSize];
//Eigen::Vector3f* host_clrs = new Eigen::Vector3f[view_vtxSize];
//
//Eigen::Vector3i* host_tri = new Eigen::Vector3i[view_triSize];
	if (0) {


		checkCudaErrors(cudaMemcpyAsync(host_pts, thrust::raw_pointer_cast(view_pos_repos.data()), sizeof(Eigen::Vector3f) * view_vtxSize, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_nms, thrust::raw_pointer_cast(view_nm_repos.data()), sizeof(Eigen::Vector3f) * view_vtxSize, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_clrs, thrust::raw_pointer_cast(view_color_repos.data()), sizeof(Eigen::Vector3b) * view_vtxSize, cudaMemcpyDeviceToHost, st));

		checkCudaErrors(cudaMemcpyAsync(host_tri, thrust::raw_pointer_cast(view_triangles.data()), sizeof(Eigen::Vector3i) * view_triSize, cudaMemcpyDeviceToHost, st));

		std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_1_test" + ".ply";
		H_registration::plyFileWrite_mesh(host_pts, host_nms, host_clrs, host_tri, view_vtxSize, view_triSize, filename, false);
		//pRegistration->plyFileWrite(host_pts, host_nms, host_clrs, h_usedSize_vtx, filename, false);
	}
#endif // 0


	auto _tri_Repos = thrust::raw_pointer_cast(voxel_tri_repos.data());

	auto _vtx_pos_repos = thrust::raw_pointer_cast(vtx_pos_repos.data());
	auto _vtx_nm_repos = thrust::raw_pointer_cast(vtx_nm_repos.data());
	auto _vtx_color_repos = thrust::raw_pointer_cast(vtx_color_repos.data());
	auto _view_pos_repos = view_pos_repos;
	auto _view_nm_repos = view_nm_repos;
	auto _view_color_repos = view_color_repos;
	auto hashinfo_tri = hInfo_global_tri;
	auto hashTable_tri = hTable_global_tri;
	auto hashTable_tri_value = hTable_global_tri_value;

	auto _voxelSize = 10.f;

	auto hashTable_voxel = globalHash;
	auto hashinfo_voxel = globalHash_info;

	thrust::device_vector<uint32_t> tmp_vtxBuff_pts;
	tmp_vtxBuff_pts.resize(view_vtxSize * 96);
	thrust::device_vector<uint32_t> tmp_vtxBuff_pts_2;
	tmp_vtxBuff_pts_2.resize(view_vtxSize * 96);
	thrust::device_vector<uint32_t> tmp_vtxBuff_pts_qnique_size;
	tmp_vtxBuff_pts_qnique_size.resize(view_vtxSize);
	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_qnique_size.begin(),
		tmp_vtxBuff_pts_qnique_size.end(), 0);

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts.begin(),
		tmp_vtxBuff_pts.end(), kEmpty32);
	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_2.begin(),
		tmp_vtxBuff_pts_2.end(), kEmpty32);
	auto _tmpBuff_pts = thrust::raw_pointer_cast(tmp_vtxBuff_pts.data());


	thrust::device_vector<int > tmp_seq;
	tmp_seq.resize(view_vtxSize);

	thrust::sequence(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_seq.begin(), tmp_seq.begin() + view_vtxSize);


	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(view_pos_repos, view_nm_repos, tmp_seq.begin()),
		make_tuple_iterator(view_pos_repos + view_vtxSize, view_nm_repos + view_vtxSize, tmp_seq.end()),
		[
			_tri_Repos
			, _voxelSize
		//, hashTable_voxel, hashinfo_voxel
		, hashinfo_tri, hashTable_tri, hashTable_tri_value
		, _vtx_pos_repos, _vtx_nm_repos
		, _tmpBuff_pts// -------------- 임시 테스트 변수
		]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f avg_vtx;
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		int& idx = thrust::get<2>(tu);

		{

			//100um 기준으로 변경하기위한 voxelSize(10.f )
			//  voxelHash 좌표로 이동을 위한 +2500.f
			//auto xGlobalIndex = (size_t)floorf((curr_vtx.x()) * _voxelSize + 0.5f + 2500.f);
			//auto yGlobalIndex = (size_t)floorf((curr_vtx.y()) * _voxelSize + 0.5f + 2500.f);
			//auto zGlobalIndex = (size_t)floorf((curr_vtx.z()) * _voxelSize + 0.5f + 2500.f);

			//printf("vertex pos = %f, %f, %f    voxelHash pos = %llu, %llu, %llu\n", curr_vtx.x(), curr_vtx.y(), curr_vtx.z(), xGlobalIndex, yGlobalIndex, zGlobalIndex);
		}

		auto xGlobalIndex = (size_t)floorf((curr_vtx.x()) * _voxelSize + 0.5f + 2500.f);
		auto yGlobalIndex = (size_t)floorf((curr_vtx.y()) * _voxelSize + 0.5f + 2500.f);
		auto zGlobalIndex = (size_t)floorf((curr_vtx.z()) * _voxelSize + 0.5f + 2500.f);

		HashKey key[8];
		/*= {
			kEmpty64,kEmpty64,kEmpty64,kEmpty64,
			kEmpty64,kEmpty64,kEmpty64,kEmpty64
		};*/
		uint32_t slot[8];
		/*= {
			kEmpty32,kEmpty32,kEmpty32,kEmpty32,
			kEmpty32,kEmpty32,kEmpty32,kEmpty32
		};*/
		uint32_t triangle[4 * 8];
		Eigen::Vector3f out_normal;
		key[0] = HashKey(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		key[1] = HashKey(xGlobalIndex, yGlobalIndex - 1, zGlobalIndex);
		key[2] = HashKey(xGlobalIndex - 1, yGlobalIndex - 1, zGlobalIndex);
		key[3] = HashKey(xGlobalIndex - 1, yGlobalIndex, zGlobalIndex);

		key[4] = HashKey(xGlobalIndex, yGlobalIndex, zGlobalIndex - 1);
		key[5] = HashKey(xGlobalIndex, yGlobalIndex - 1, zGlobalIndex - 1);
		key[6] = HashKey(xGlobalIndex - 1, yGlobalIndex - 1, zGlobalIndex - 1);
		key[7] = HashKey(xGlobalIndex - 1, yGlobalIndex, zGlobalIndex - 1);

		slot[0] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[0], _tri_Repos);
		slot[1] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[1], _tri_Repos);
		slot[2] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[2], _tri_Repos);
		slot[3] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[3], _tri_Repos);

		slot[4] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[4], _tri_Repos);
		slot[5] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[5], _tri_Repos);
		slot[6] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[6], _tri_Repos);
		slot[7] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[7], _tri_Repos);

		float3 _pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;
		float3 _nm = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;
		Eigen::Vector<uint32_t, 3> idx_triVtx;
		Eigen::Vector3f tmp[3];
		int cnt = 0;

		for (size_t i = 0; i < 8; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				if (slot[i] != kEmpty32)
				{
					idx_triVtx = _tri_Repos[slot[i] * 4 + j];

					if (idx_triVtx.x() != kEmpty32 &&
						idx_triVtx.y() != kEmpty32 &&
						idx_triVtx.z() != kEmpty32
						)
					{
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 0] = idx_triVtx.x();
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 1] = idx_triVtx.y();
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 2] = idx_triVtx.z();

						cnt++;

						//tmp[0] = _view_pos_repos[idx_triVtx.x()];
						//tmp[1] = _view_pos_repos[idx_triVtx.y()];
						//tmp[2] = _view_pos_repos[idx_triVtx.z()];
						////printf("%d, %d, %d\n", idx_triVtx.x(), idx_triVtx.y(), idx_triVtx.z());
						//if ((_pt.x == FLT_MAX) || (_pt.y == FLT_MAX) || (_pt.z == FLT_MAX))
						//{
						//	_pt.x = tmp[0].x();
						//	_pt.x = tmp[1].x();
						//	_pt.x = tmp[2].x();

						//	_pt.y = tmp[0].y();
						//	_pt.y = tmp[1].y();
						//	_pt.y = tmp[2].y();

						//	_pt.z = tmp[0].z();
						//	_pt.z = tmp[1].z();
						//	_pt.z = tmp[2].z();
						//}
						//else {
						//	_pt.x += tmp[0].x();
						//	_pt.x += tmp[1].x();
						//	_pt.x += tmp[2].x();
						//			
						//	_pt.y += tmp[0].y();
						//	_pt.y += tmp[1].y();
						//	_pt.y += tmp[2].y();
						//			
						//	_pt.z += tmp[0].z();
						//	_pt.z += tmp[1].z();
						//	_pt.z += tmp[2].z();
						//}

						//tmp[0] = _view_nm_repos[idx_vtx.x()];
						//tmp[1] = _view_nm_repos[idx_vtx.y()];
						//tmp[2] = _view_nm_repos[idx_vtx.z()];
						//if ((_nm.x == FLT_MAX) || (_nm.y == FLT_MAX) || (_nm.z == FLT_MAX))
						//{
						//	_nm.x = tmp[0].x();
						//	_nm.x = tmp[1].x();
						//	_nm.x = tmp[2].x();

						//	_nm.y = tmp[0].y();
						//	_nm.y = tmp[1].y();
						//	_nm.y = tmp[2].y();

						//	_nm.z = tmp[0].z();
						//	_nm.z = tmp[1].z();
						//	_nm.z = tmp[2].z();
						//}
						//else {
						//	_nm.x += tmp[0].x();
						//	_nm.x += tmp[1].x();
						//	_nm.x += tmp[2].x();

						//	_nm.y += tmp[0].y();
						//	_nm.y += tmp[1].y();
						//	_nm.y += tmp[2].y();

						//	_nm.z += tmp[0].z();
						//	_nm.z += tmp[1].z();
						//	_nm.z += tmp[2].z();
						//}
					}
				}
			}
		}
		//printf("[%d] %f, %f, %f \t%f, %f, %f\n", slot[0], _pt.x, _pt.y, _pt.y, _nm.x, _nm.y, _nm.z);

		/*if (_pt.x != FLT_MAX && _pt.y != FLT_MAX && _pt.z != FLT_MAX)
		{
			if (_nm.x != FLT_MAX && _nm.y != FLT_MAX && _nm.z != FLT_MAX)
				_nm = Normalize(make_float3(_nm.x / cnt, _nm.y / cnt, _nm.z / cnt));

			curr_nm = Eigen::Vector3f(_nm.x, _nm.y, _nm.z);
			avg_vtx = Eigen::Vector3f(_pt.x / cnt, _pt.y / cnt, _pt.z / cnt);

			Eigen::Vector3f diff = curr_vtx - avg_vtx;
			float f = curr_nm.dot(diff);
			curr_vtx = curr_vtx - (curr_nm * f);
		}*/
	}

	);

	int threadblocksize = 0;

#ifdef _D2H_DATACHACK
	checkCudaSync(st);

	uint32_t* host_pts = new uint32_t[view_vtxSize * 96];

	checkCudaErrors(cudaMemcpyAsync(host_pts, _tmpBuff_pts, sizeof(uint32_t) * view_vtxSize * 96, cudaMemcpyDeviceToHost, st));

#endif // _D2H_DATACHACK
	auto _tmpBuff_pts_2 = thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data());
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveEmptyKernel, 0, 0));

	int gridsize = (view_vtxSize * 96 + threadblocksize - 1) / threadblocksize;



	/*UniqueRemoveEmptyKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()), kEmpty32);


	checkCudaSync(st);


	checkCudaErrors(cudaMemcpyAsync(host_pts, _tmpBuff_pts_2, sizeof(uint32_t)* view_vtxSize * 96, cudaMemcpyDeviceToHost, st));*/

	threadblocksize = 0;

	//checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveKernel, 0, 0));
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveDuplicationKernel, 0, 0));
	gridsize = (view_vtxSize * 96 + threadblocksize - 1) / threadblocksize;

	/*UniqueRemoveKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()));*/

	UniqueRemoveDuplicationKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()));
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);

#ifdef _D2H_DATACHACK
	checkCudaErrors(cudaMemcpyAsync(host_pts, _tmpBuff_pts_2, sizeof(uint32_t) * view_vtxSize * 96, cudaMemcpyDeviceToHost, st));
#endif // _D2H_DATACHACK


	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts.begin(),
		tmp_vtxBuff_pts.end(), kEmpty32);

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_qnique_size.begin(),
		tmp_vtxBuff_pts_qnique_size.end(), 0);

	UniqueRemoveEmptyKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()), kEmpty32);
	checkCudaErrors(cudaGetLastError());
	checkCudaSync(st);


#ifdef _D2H_DATACHACK
	checkCudaErrors(cudaMemcpyAsync(host_pts, _tmpBuff_pts, sizeof(uint32_t) * view_vtxSize * 96, cudaMemcpyDeviceToHost, st));

	delete[] host_pts;
#endif // _D2H_DATACHACK


	if (0)for (int i = 0; i < view_vtxSize; ++i)
	{
		nvtxRangePushA("sort");
		// 각 구간의 시작과 끝 포인터를 계산합니다.
		auto start_ptr = tmp_vtxBuff_pts.begin() + i * 96;
		auto end_ptr = start_ptr + 96;

		//// 해당 구간을 정렬합니다.
		//thrust::sort(
		//	thrust::cuda::par_nosync(*alloc_).on(st), 
		//	start_ptr, 
		//	end_ptr);
		//auto data_end = thrust::unique(
		//	thrust::cuda::par_nosync(*alloc_).on(st),
		//	start_ptr,
		//	end_ptr);
		/*thrust::fill(
			thrust::cuda::par_nosync(*alloc_).on(st),
			data_end,
			end_ptr,
			kEmpty32);*/
		checkCudaSync(st);
		nvtxRangePop();
	}
	thrust::transform(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(view_pos_repos, view_nm_repos, tmp_seq.begin()),
		make_tuple_iterator(view_pos_repos + view_vtxSize, view_nm_repos + view_vtxSize, tmp_seq.end()),
		make_tuple_iterator(view_pos_repos, view_nm_repos),
		[
			_tri_Repos
			, _voxelSize
		, hashTable_voxel, hashinfo_voxel
		, _vtx_pos_repos, _vtx_nm_repos
		, _tmpBuff_pts_2// -------------- 임시 테스트 변수
		]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f avg_vtx;
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		int& idx = thrust::get<2>(tu);

		float3 tmp_pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;;
		float3 tmp_nm = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;;
		int cnt = 0;
		for (size_t i = 0; i < 96; i++)
		{
			auto slot = _tmpBuff_pts_2[(idx * 96) + i];
			if (_tmpBuff_pts_2[(idx * 96) + i] == kEmpty32)	break;
			cnt++;
			if (tmp_pt.x == FLT_MAX || tmp_nm.x == FLT_MAX)
			{
				tmp_pt.x = _vtx_pos_repos[slot].x();
				tmp_pt.y = _vtx_pos_repos[slot].y();
				tmp_pt.z = _vtx_pos_repos[slot].z();
				tmp_nm.x = _vtx_nm_repos[slot].x();
				tmp_nm.y = _vtx_nm_repos[slot].y();
				tmp_nm.z = _vtx_nm_repos[slot].z();
			}
			else
			{
				tmp_pt.x += _vtx_pos_repos[slot].x();
				tmp_pt.y += _vtx_pos_repos[slot].y();
				tmp_pt.z += _vtx_pos_repos[slot].z();
				tmp_nm.x += _vtx_nm_repos[slot].x();
				tmp_nm.y += _vtx_nm_repos[slot].y();
				tmp_nm.z += _vtx_nm_repos[slot].z();
			}

			//printf("slot  = %d || %f, %f, %f \n", slot, tmp_pt.x, tmp_pt.y, tmp_pt.z);
		}
		if (cnt == 0) return thrust::make_tuple(curr_vtx, curr_nm);

		if (tmp_pt.x != FLT_MAX && tmp_pt.y != FLT_MAX && tmp_pt.z != FLT_MAX)
		{
			if (tmp_nm.x != FLT_MAX && tmp_nm.y != FLT_MAX && tmp_nm.z != FLT_MAX)
				tmp_nm = Normalize(make_float3(tmp_nm.x / cnt, tmp_nm.y / cnt, tmp_nm.z / cnt));

			curr_nm = Eigen::Vector3f(tmp_nm.x, tmp_nm.y, tmp_nm.z);
			curr_nm = curr_nm.normalized();
			avg_vtx = Eigen::Vector3f(tmp_pt.x / cnt, tmp_pt.y / cnt, tmp_pt.z / cnt);

			Eigen::Vector3f diff = curr_vtx - avg_vtx;

			//printf("curr_vtx, diff  = %f, %f, %f \t %f, %f, %f\n", curr_vtx.x(), curr_vtx.y(), curr_vtx.z(), diff.x(), diff.y(), diff.z());

			float f = curr_nm.dot(diff);
			curr_vtx = curr_vtx - (curr_nm * f);
		}
		return thrust::make_tuple(curr_vtx, curr_nm);
	}
	);

	checkCudaSync(st);

#if 0
	if (0) {

		checkCudaErrors(cudaMemcpyAsync(host_pts, thrust::raw_pointer_cast(view_pos_repos.data()), sizeof(Eigen::Vector3f) * view_vtxSize, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_nms, thrust::raw_pointer_cast(view_nm_repos.data()), sizeof(Eigen::Vector3f) * view_vtxSize, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_clrs, thrust::raw_pointer_cast(view_color_repos.data()), sizeof(Eigen::Vector3f) * view_vtxSize, cudaMemcpyDeviceToHost, st));

		checkCudaErrors(cudaMemcpyAsync(host_tri, thrust::raw_pointer_cast(view_triangles.data()), sizeof(Eigen::Vector3i) * view_triSize, cudaMemcpyDeviceToHost, st));

		sprintf(szTemp, "%04d", cntFile++);
		std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_2_test" + ".ply";
		pRegistration->plyFileWrite_mesh(host_pts, host_nms, host_clrs, host_tri, view_vtxSize, view_triSize, filename, false);
		//pRegistration->plyFileWrite(host_pts, host_nms, host_clrs, h_usedSize_vtx, filename, false);

	}
	delete[] host_pts;
	delete[] host_nms;
	delete[] host_clrs;
	delete[] host_tri;
#endif // 0


	/*	uint32_t* host_pts = new uint32_t[view_vtxSize * 96];

		checkCudaErrors(cudaMemcpyAsync(host_pts, _tmpBuff_pts, sizeof(uint32_t)* view_vtxSize * 96, cudaMemcpyDeviceToHost, st));

		delete[] host_pts;*/
	nvtxRangePop();
	return Eigen::Vector2i(0, 0);
}
#endif

#ifdef USE_MESH_BASE
Eigen::Vector2i MarchingCubes::avgRenderData_PtsNm_v2(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<uint32_t>&vtx_replace_idx
	, thrust::device_vector<Eigen::Vector3f>&vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>&vtx_color_repos
	, Eigen::Vector<uint32_t, 3>*view_triangles
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, thrust::device_vector<uint32_t>&hash_usedCheckBuffer
	, size_t	view_triSize
	, size_t view_vtxSize
	, cached_allocator * alloc_, CUstream_st * st)
{
	nvtxRangePushA("avgPtsNm");
	qDebug("avgPtsNm");

	static	int cntFile = 0;
	char szTemp[128];
	sprintf(szTemp, "%04d", cntFile);

	auto _tri_Repos = thrust::raw_pointer_cast(voxel_tri_repos.data());

	auto _vtx_pos_repos = thrust::raw_pointer_cast(vtx_pos_repos.data());
	auto _vtx_nm_repos = thrust::raw_pointer_cast(vtx_nm_repos.data());
	auto _vtx_color_repos = thrust::raw_pointer_cast(vtx_color_repos.data());

	auto _view_pos_repos = view_pos_repos;
	auto _view_nm_repos = view_nm_repos;
	auto _view_color_repos = view_color_repos;

	auto hashinfo_tri = hInfo_global_tri;
	auto hashTable_tri = hTable_global_tri;
	auto hashTable_tri_value = hTable_global_tri_value;

	auto _voxelSize = 10.f;

	auto hashTable_voxel = globalHash;
	auto hashinfo_voxel = globalHash_info;

	thrust::device_vector<uint32_t> tmp_vtxBuff_pts;
	tmp_vtxBuff_pts.resize(view_vtxSize * 96);
	//thrust::device_vector<uint32_t> tmp_vtxBuff_pts_2;
	//tmp_vtxBuff_pts_2.resize(view_vtxSize * 96);
	//thrust::device_vector<uint32_t> tmp_vtxBuff_pts_qnique_size;
	//tmp_vtxBuff_pts_qnique_size.resize(view_vtxSize);
	/*thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_qnique_size.begin(),
		tmp_vtxBuff_pts_qnique_size.end(), 0);*/

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts.begin(),
		tmp_vtxBuff_pts.end(), kEmpty32);
	/*thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_2.begin(),
		tmp_vtxBuff_pts_2.end(), kEmpty32);*/
	auto _tmpBuff_pts = thrust::raw_pointer_cast(tmp_vtxBuff_pts.data());


	thrust::device_vector<int > tmp_seq;
	tmp_seq.resize(view_vtxSize);

	thrust::sequence(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_seq.begin(), tmp_seq.begin() + view_vtxSize);


	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(view_pos_repos, view_nm_repos, tmp_seq.begin()),
		make_tuple_iterator(view_pos_repos + view_vtxSize, view_nm_repos + view_vtxSize, tmp_seq.end()),
		[
			_tri_Repos
			, _voxelSize
		//, hashTable_voxel, hashinfo_voxel
		, hashinfo_tri, hashTable_tri, hashTable_tri_value
		, _vtx_pos_repos, _vtx_nm_repos
		, _tmpBuff_pts// -------------- 임시 테스트 변수
		]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f avg_vtx;
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		int& idx = thrust::get<2>(tu);

		{

			//100um 기준으로 변경하기위한 voxelSize(10.f )
			//  voxelHash 좌표로 이동을 위한 +2500.f
			//auto xGlobalIndex = (size_t)floorf((curr_vtx.x()) * _voxelSize + 0.5f + 2500.f);
			//auto yGlobalIndex = (size_t)floorf((curr_vtx.y()) * _voxelSize + 0.5f + 2500.f);
			//auto zGlobalIndex = (size_t)floorf((curr_vtx.z()) * _voxelSize + 0.5f + 2500.f);

			//printf("vertex pos = %f, %f, %f    voxelHash pos = %llu, %llu, %llu\n", curr_vtx.x(), curr_vtx.y(), curr_vtx.z(), xGlobalIndex, yGlobalIndex, zGlobalIndex);
		}

		auto xGlobalIndex = (size_t)floorf((curr_vtx.x()) * _voxelSize + 0.5f + 2500.f);
		auto yGlobalIndex = (size_t)floorf((curr_vtx.y()) * _voxelSize + 0.5f + 2500.f);
		auto zGlobalIndex = (size_t)floorf((curr_vtx.z()) * _voxelSize + 0.5f + 2500.f);

		HashKey key[8];
		uint32_t slot[8];
		uint32_t triangle[4 * 8];
		Eigen::Vector3f out_normal;
		key[0] = HashKey(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		key[1] = HashKey(xGlobalIndex, yGlobalIndex - 1, zGlobalIndex);
		key[2] = HashKey(xGlobalIndex - 1, yGlobalIndex - 1, zGlobalIndex);
		key[3] = HashKey(xGlobalIndex - 1, yGlobalIndex, zGlobalIndex);

		key[4] = HaskKey(xGlobalIndex, yGlobalIndex, zGlobalIndex - 1);
		key[5] = HaskKey(xGlobalIndex, yGlobalIndex - 1, zGlobalIndex - 1);
		key[6] = HaskKey(xGlobalIndex - 1, yGlobalIndex - 1, zGlobalIndex - 1);
		key[7] = HaskKey(xGlobalIndex - 1, yGlobalIndex, zGlobalIndex - 1);

		slot[0] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[0], _tri_Repos);
		slot[1] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[1], _tri_Repos);
		slot[2] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[2], _tri_Repos);
		slot[3] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[3], _tri_Repos);

		slot[4] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[4], _tri_Repos);
		slot[5] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[5], _tri_Repos);
		slot[6] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[6], _tri_Repos);
		slot[7] = get_hashtable_lookup_voxelTriange_func64_v4(hashinfo_tri, hashTable_tri, hashTable_tri_value, key[7], _tri_Repos);

		float3 _pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;
		float3 _nm = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;
		Eigen::Vector<uint32_t, 3> idx_triVtx;
		Eigen::Vector3f tmp[3];
		int cnt = 0;

		for (size_t i = 0; i < 8; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				if (slot[i] != kEmpty32)
				{
					idx_triVtx = _tri_Repos[slot[i] * 4 + j];

					if (idx_triVtx.x() != kEmpty32 &&
						idx_triVtx.y() != kEmpty32 &&
						idx_triVtx.z() != kEmpty32
						)
					{
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 0] = idx_triVtx.x();
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 1] = idx_triVtx.y();
						_tmpBuff_pts[(idx * 96) + (cnt * 3) + 2] = idx_triVtx.z();

						cnt++;
					}
				}
			}
		}
	}
	);

	int threadblocksize = 0;

	//auto _tmpBuff_pts_2 = thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data());
	//checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveDuplicationKernel, 0, 0));
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveDuplicationKernel_v2, 0, 0));

	int gridsize = (view_vtxSize * 96 + threadblocksize - 1) / threadblocksize;

	/*UniqueRemoveDuplicationKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()));*/
	UniqueRemoveDuplicationKernel_v2 << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), 96, view_vtxSize, kEmpty32);
	checkCudaErrors(cudaGetLastError());

	/*thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts.begin(),
		tmp_vtxBuff_pts.end(), kEmpty32);

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		tmp_vtxBuff_pts_qnique_size.begin(),
		tmp_vtxBuff_pts_qnique_size.end(), 0);

	threadblocksize = 0;

	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, UniqueRemoveEmptyKernel, 0, 0));
	gridsize = (view_vtxSize * 96 + threadblocksize - 1) / threadblocksize;

	UniqueRemoveEmptyKernel << < gridsize, threadblocksize, 0, st >> >
		(thrust::raw_pointer_cast(tmp_vtxBuff_pts_2.data()), thrust::raw_pointer_cast(tmp_vtxBuff_pts.data()), 96, view_vtxSize, thrust::raw_pointer_cast(tmp_vtxBuff_pts_qnique_size.data()), kEmpty32);
	checkCudaSync(st);*/

	thrust::transform(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(view_pos_repos, view_nm_repos, tmp_seq.begin()),
		make_tuple_iterator(view_pos_repos + view_vtxSize, view_nm_repos + view_vtxSize, tmp_seq.end()),
		make_tuple_iterator(view_pos_repos, view_nm_repos),
		[
			_tri_Repos
			, _voxelSize
		, hashTable_voxel, hashinfo_voxel
		, _vtx_pos_repos, _vtx_nm_repos
		, _tmpBuff_pts// -------------- 임시 테스트 변수
		]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f avg_vtx;
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		int& idx = thrust::get<2>(tu);

		float3 tmp_pt = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;;
		float3 tmp_nm = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);;;
		int cnt = 0;
		for (size_t i = 0; i < 96; i++)
		{
			auto slot = _tmpBuff_pts[(idx * 96) + i];
			//if (_tmpBuff_pts_2[(idx * 96) + i] == kEmpty32)	break;
			if (_tmpBuff_pts[(idx * 96) + i] == kEmpty32)	continue;
			cnt++;
			if (tmp_pt.x == FLT_MAX || tmp_nm.x == FLT_MAX)
			{
				tmp_pt.x = _vtx_pos_repos[slot].x();
				tmp_pt.y = _vtx_pos_repos[slot].y();
				tmp_pt.z = _vtx_pos_repos[slot].z();
				tmp_nm.x = _vtx_nm_repos[slot].x();
				tmp_nm.y = _vtx_nm_repos[slot].y();
				tmp_nm.z = _vtx_nm_repos[slot].z();
			}
			else
			{
				tmp_pt.x += _vtx_pos_repos[slot].x();
				tmp_pt.y += _vtx_pos_repos[slot].y();
				tmp_pt.z += _vtx_pos_repos[slot].z();
				tmp_nm.x += _vtx_nm_repos[slot].x();
				tmp_nm.y += _vtx_nm_repos[slot].y();
				tmp_nm.z += _vtx_nm_repos[slot].z();
			}

			//printf("slot  = %d || %f, %f, %f \n", slot, tmp_pt.x, tmp_pt.y, tmp_pt.z);
		}
		if (cnt == 0) return thrust::make_tuple(curr_vtx, curr_nm);

		if (tmp_pt.x != FLT_MAX && tmp_pt.y != FLT_MAX && tmp_pt.z != FLT_MAX)
		{
			if (tmp_nm.x != FLT_MAX && tmp_nm.y != FLT_MAX && tmp_nm.z != FLT_MAX)
				tmp_nm = Normalize(make_float3(tmp_nm.x / cnt, tmp_nm.y / cnt, tmp_nm.z / cnt));

			curr_nm = Eigen::Vector3f(tmp_nm.x, tmp_nm.y, tmp_nm.z);

			/*if (false == VECTOR3F_VALID_(curr_nm))
			{
				printf("curr_nm : %f\n", curr_nm.x(), curr_nm.y(), curr_nm.z());
			}*/

			avg_vtx = Eigen::Vector3f(tmp_pt.x / cnt, tmp_pt.y / cnt, tmp_pt.z / cnt);

			Eigen::Vector3f diff = curr_vtx - avg_vtx;

			//printf("curr_vtx, diff  = %f, %f, %f \t %f, %f, %f\n", curr_vtx.x(), curr_vtx.y(), curr_vtx.z(), diff.x(), diff.y(), diff.z());

			float f = curr_nm.dot(diff);
			curr_vtx = curr_vtx - (curr_nm * f);
		}
		return thrust::make_tuple(curr_vtx, curr_nm);
	}
	);

	checkCudaSync(st);

	nvtxRangePop();
	return Eigen::Vector2i(0, 0);
}
#endif

#ifdef USE_MESH_BASE
Eigen::Vector2i MarchingCubes::makeViewTriangles_sortVtx(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<uint32_t>&vtx_replace_idx
	, thrust::device_vector<Eigen::Vector3f>&vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>&vtx_color_repos
	, Eigen::Vector<uint32_t, 3>*view_triangles
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, thrust::device_vector<uint32_t>&hash_usedCheckBuffer
	, cached_allocator * alloc_, CUstream_st * st)
{
	nvtxRangePushA("makeTriList");
	qDebug("makeTriList");
	auto _vtx_pos_repos = thrust::raw_pointer_cast(vtx_pos_repos.data());
	auto _vtx_nm_repos = thrust::raw_pointer_cast(vtx_nm_repos.data());
	auto _vtx_color_repos = thrust::raw_pointer_cast(vtx_color_repos.data());
	auto _view_pos_repos = view_pos_repos;
	auto _view_nm_repos = view_nm_repos;
	auto _view_color_repos = view_color_repos;

	std::cout << _view_pos_repos << "\n";

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);

	if (threadblocksize_voxel == 0)
	{
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_voxel, kernel_iterateChkUsed_triangle_hashtable64, 0, 0));
		gridsize_voxel = ((uint32_t)pHashManager->VoxelHash_h->HashTableCapacity + threadblocksize_voxel - 1) / threadblocksize_voxel;
	}

	kernel_iterateChkUsed_triangle_hashtable64 << <gridsize_voxel, threadblocksize_voxel, 0, st >> > (globalHash_info, thrust::raw_pointer_cast(hash_usedCheckBuffer.data()), thrust::raw_pointer_cast(voxel_tri_repos.data()));
	checkCudaErrors(cudaGetLastError());

	if (threadblocksize_vtx == 0)
	{
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_vtx, kernel_iterateChkUsed_vtx_hashtable64, 0, 0));
		gridsize_vtx = ((uint32_t)pHashManager->hashGlobalVertex_info64_host->HashTableCapacity + threadblocksize_vtx - 1) / threadblocksize_vtx;
	}
	kernel_iterateChkUsed_vtx_hashtable64 << <gridsize_vtx, threadblocksize_vtx, 0, st >> > (
		hInfo_global_vtx,
		hTable_global_vtx,
		thrust::raw_pointer_cast(vtx_replace_idx.data())

		, _vtx_pos_repos, _vtx_nm_repos, _vtx_color_repos
		, _view_pos_repos, _view_nm_repos, _view_color_repos
		);
	checkCudaErrors(cudaGetLastError());

	uint32_t h_usedSize;
	uint32_t h_usedSize_vtx;
	cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(&h_usedSize_vtx, &hInfo_global_vtx->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	//qDebug("Kernel grid/block size voxel = %d, %d,   vtx = %d, %d", gridsize_voxel, threadblocksize_voxel, gridsize_vtx, threadblocksize_vtx);
	qDebug("Voxel count : %d", h_usedSize);
	qDebug("Vertex count : %d", h_usedSize_vtx);

	nvtxRangePop();
	auto ptr_triRepos = thrust::raw_pointer_cast(voxel_tri_repos.data());
	auto ptr_viewtriBuff = view_triangles;
	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(hash_usedCheckBuffer.begin(), thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(hash_usedCheckBuffer.begin() + h_usedSize, thrust::make_counting_iterator<size_t>(h_usedSize)),
		[ptr_triRepos, ptr_viewtriBuff] __device__(auto & tu)
	{
		uint32_t repos_num = thrust::get<0>(tu);
		size_t idx = thrust::get<1>(tu);

		for (size_t i = 0; i < 4; i++)
		{
			if ((ptr_triRepos[repos_num * 4 + i].x() != UINT_MAX) && (ptr_triRepos[repos_num * 4 + i].y() != UINT_MAX) && (ptr_triRepos[repos_num * 4 + i].z() != UINT_MAX) &&
				(ptr_triRepos[repos_num * 4 + i].x() != 0) && (ptr_triRepos[repos_num * 4 + i].y() != 0) && (ptr_triRepos[repos_num * 4 + i].z() != 0))
			{
				ptr_viewtriBuff[idx * 4 + i] = ptr_triRepos[repos_num * 4 + i];
				//printf("%d,%d,%d\n", ptr_viewtriBuff[idx * 4 + i].x(), ptr_viewtriBuff[idx * 4 + i].y(), ptr_viewtriBuff[idx * 4 + i].z());
			}
			else
				ptr_viewtriBuff[idx * 4 + i] = Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX);
		}
	}
	);

	thrust::device_ptr<Eigen::Vector<uint32_t, 3>> dev_ptr = thrust::device_pointer_cast(ptr_viewtriBuff);

	auto endPtr = thrust::copy_if(
		thrust::cuda::par_nosync(*alloc_).on(st),
		dev_ptr,
		dev_ptr + (h_usedSize * 4),
		dev_ptr,
		[]__device__(auto & vecTri)
	{
		return (vecTri.x() != UINT_MAX && vecTri.y() != UINT_MAX && vecTri.z() != UINT_MAX);
	}
	);

	auto resSize = thrust::distance(dev_ptr, endPtr);

#pragma region check triangle result
	/*thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		view_triangles.begin(),
		view_triangles.begin() + resSize,
		[]__device__(auto & vecTri)
	{
		if(vecTri.x() != UINT_MAX && vecTri.y() != UINT_MAX && vecTri.z() != UINT_MAX)
			return;
		else
			printf("[copy] %d,%d,%d\n", vecTri.x(), vecTri.y(), vecTri.z());
	}
	);
	checkCudaSync(st);*/
#pragma endregion

	auto _vtx_replace_idx = thrust::raw_pointer_cast(vtx_replace_idx.data());
	thrust::transform(
		thrust::cuda::par_nosync(*alloc_).on(st),
		dev_ptr,
		dev_ptr + resSize,
		dev_ptr,
		[_vtx_replace_idx] __device__(auto & vecTri)
	{

		//printf("[replace] %d,%d,%d -> %d,%d,%d\n", vecTri.x(), vecTri.y(), vecTri.z(), _vtx_replace_idx[vecTri.x()], _vtx_replace_idx[vecTri.y()], _vtx_replace_idx[vecTri.z()]);
		return Eigen::Vector<uint32_t, 3>(_vtx_replace_idx[vecTri.x()], _vtx_replace_idx[vecTri.y()], _vtx_replace_idx[vecTri.z()]);

	}
	);

#pragma region check render data organize

#if 0
	//if (h_usedSize_vtx > 100000 && h_usedSize_vtx < 101000)
	{
		Eigen::Vector3f* host_pts = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3f* host_nms = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3b* host_clrs = new Eigen::Vector3b[h_usedSize_vtx];

		Eigen::Vector3i* host_tri = new Eigen::Vector3i[resSize];


		checkCudaErrors(cudaMemcpyAsync(host_pts, thrust::raw_pointer_cast(view_pos_repos.data()), sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_nms, thrust::raw_pointer_cast(view_nm_repos.data()), sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_clrs, thrust::raw_pointer_cast(view_color_repos.data()), sizeof(Eigen::Vector3b) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));

		checkCudaErrors(cudaMemcpyAsync(host_tri, thrust::raw_pointer_cast(view_triangles.data()), sizeof(Eigen::Vector3i) * resSize, cudaMemcpyDeviceToHost, st));

		static	int cntFile = 0;
		char szTemp[128];
		sprintf(szTemp, "%04d", cntFile++);
		std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_test" + ".ply";
		pRegistration->plyFileWrite_mesh(host_pts, host_nms, host_clrs, host_tri, h_usedSize_vtx, resSize, filename, false);
		//pRegistration->plyFileWrite(host_pts, host_nms, host_clrs, h_usedSize_vtx, filename, false);


		delete[] host_pts;
		delete[] host_nms;
		delete[] host_clrs;
		delete[] host_tri;
	}
#endif

#pragma endregion

	return Eigen::Vector2i(resSize, h_usedSize_vtx);
}
#endif

#ifdef USE_MESH_BASE
uint32_t MarchingCubes::makeUsedBufferList(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<uint32_t>&hash_usedCheckBuffer
	, cached_allocator * alloc_, CUstream_st * st
)
{
	pHashManager->reset_hashKeyCountValue64(hInfo_global_tri, st);

	if (threadblocksize_triangle == 0)
	{
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_triangle, kernel_iterateChkUsed_triangle_hashtable64_v4, 0, 0));
		gridsize_triangle = ((uint32_t)hInfo_global_tri_host->HashTableCapacity + threadblocksize_triangle - 1) / threadblocksize_triangle;
	}


	kernel_iterateChkUsed_triangle_hashtable64_v4 << <gridsize_triangle, threadblocksize_triangle, 0, st >> > (
		hInfo_global_tri
		, hTable_global_tri
		, hTable_global_tri_value
		, thrust::raw_pointer_cast(hash_usedCheckBuffer.data())
		, thrust::raw_pointer_cast(voxel_tri_repos.data())
		);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);
	uint32_t h_usedSize;
	cudaMemcpy(&h_usedSize, &hInfo_global_tri->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	return h_usedSize;
}
#endif

#ifdef USE_MESH_BASE
//	mscho	@20240221
Eigen::Vector2i MarchingCubes::makeViewTriangles_sortVtx_v2(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<uint32_t>&vtx_replace_idx
	, thrust::device_vector<Eigen::Vector3f>&vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>&vtx_color_repos
	, Eigen::Vector<uint32_t, 3>*view_triangles
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, thrust::device_vector<uint32_t>&hash_usedCheckBuffer
	, cached_allocator * alloc_, CUstream_st * st
	, bool mode)
{
	nvtxRangePushA("makeTriList v2");
	qDebug("makeTriList v2");
	auto _vtx_pos_repos = thrust::raw_pointer_cast(vtx_pos_repos.data());
	auto _vtx_nm_repos = thrust::raw_pointer_cast(vtx_nm_repos.data());
	auto _vtx_color_repos = thrust::raw_pointer_cast(vtx_color_repos.data());

	auto _view_pos_repos = view_pos_repos;
	auto _view_nm_repos = view_nm_repos;
	auto _view_color_repos = view_color_repos;

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);

	pHashManager->reset_hashKeyCountValue64(hInfo_global_tri, st);

	if (threadblocksize_triangle == 0)
	{
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_triangle, kernel_iterateChkUsed_triangle_hashtable64_v4, 0, 0));
		gridsize_triangle = ((uint32_t)hInfo_global_tri_host->HashTableCapacity + threadblocksize_triangle - 1) / threadblocksize_triangle;
	}


	kernel_iterateChkUsed_triangle_hashtable64_v4 << <gridsize_triangle, threadblocksize_triangle, 0, st >> > (
		hInfo_global_tri
		, hTable_global_tri
		, hTable_global_tri_value
		, thrust::raw_pointer_cast(hash_usedCheckBuffer.data())
		, thrust::raw_pointer_cast(voxel_tri_repos.data())
		);
	checkCudaErrors(cudaGetLastError());

	//if (threadblocksize_vtx == 0)
	//{
	//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_vtx, kernel_iterateChkUsed_vtxAll_hashtable64_v4, 0, 0));
	//	gridsize_vtx = ((uint32_t)hInfo_global_vtx_host->HashTableCapacity + threadblocksize_vtx - 1) / threadblocksize_vtx;
	//}
	//kernel_iterateChkUsed_vtxAll_hashtable64_v4 << <gridsize_vtx, threadblocksize_vtx, 0, st >> > (
	//	hInfo_global_vtx
	//	, hTable_global_vtx
	//	, hTable_global_vtx_value
	//	);
	//threadblocksize_vtx = 0;
	//uint32_t h_usedSize_vtxAll;
	//cudaMemcpyAsync(&h_usedSize_vtxAll, &hInfo_global_vtx->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	//qDebug("==================================m_View_VTXALLSize = %zd", h_usedSize_vtxAll);

	//pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);

	if (threadblocksize_vtx == 0)
	{
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize_vtx, kernel_iterateChkUsed_vtx_hashtable64_v4, 0, 0));
		gridsize_vtx = ((uint32_t)hInfo_global_vtx_host->HashTableCapacity + threadblocksize_vtx - 1) / threadblocksize_vtx;
	}
	kernel_iterateChkUsed_vtx_hashtable64_v4 << <gridsize_vtx, threadblocksize_vtx, 0, st >> > (
		hInfo_global_vtx
		, hTable_global_vtx
		, hTable_global_vtx_value
		, thrust::raw_pointer_cast(vtx_replace_idx.data())

		, _vtx_pos_repos, _vtx_nm_repos, _vtx_color_repos
		, _view_pos_repos, _view_nm_repos, _view_color_repos
		);
	checkCudaErrors(cudaGetLastError());

	uint32_t h_usedSize;
	uint32_t h_usedSize_vtx;
	cudaMemcpyAsync(&h_usedSize, &hInfo_global_tri->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(&h_usedSize_vtx, &hInfo_global_vtx->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	//qDebug("Kernel grid/block size voxel = %d, %d,   vtx = %d, %d", gridsize_voxel, threadblocksize_voxel, gridsize_vtx, threadblocksize_vtx);
	//qDebug(" v2 Triangle(Voxel) count : %d", h_usedSize);
	//qDebug(" v2 Vertex count : %d", h_usedSize_vtx);

	nvtxRangePop();
	//#define TEST_CHANGE_ROUTINE



#ifdef TEST_CHANGE_ROUTINE
	auto endPtr = thrust::copy_if(
		thrust::cuda::par_nosync(*alloc_).on(st),
		view_triangles,
		view_triangles + (h_usedSize * 4),
		view_triangles,
		[]__device__(auto & vecTri)
	{
		return (vecTri.x() != UINT_MAX && vecTri.y() != UINT_MAX && vecTri.z() != UINT_MAX);
	}
	);


	auto ptr_triRepos = thrust::raw_pointer_cast(voxel_tri_repos.data());
	auto ptr_viewtriBuff = view_triangles;
	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(hash_usedCheckBuffer.begin(), thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(hash_usedCheckBuffer.begin() + h_usedSize, thrust::make_counting_iterator<size_t>(h_usedSize)),
		[ptr_triRepos, ptr_viewtriBuff] __device__(auto & tu)
	{
		uint32_t repos_num = thrust::get<0>(tu);
		size_t idx = thrust::get<1>(tu);

		for (size_t i = 0; i < 4; i++)
		{
			if ((ptr_triRepos[repos_num * 4 + i].x() != UINT_MAX) && (ptr_triRepos[repos_num * 4 + i].y() != UINT_MAX) && (ptr_triRepos[repos_num * 4 + i].z() != UINT_MAX) &&
				(ptr_triRepos[repos_num * 4 + i].x() != 0) && (ptr_triRepos[repos_num * 4 + i].y() != 0) && (ptr_triRepos[repos_num * 4 + i].z() != 0))
			{
				ptr_viewtriBuff[idx * 4 + i] = ptr_triRepos[repos_num * 4 + i];
				//printf("%d,%d,%d\n", ptr_viewtriBuff[idx * 4 + i].x(), ptr_viewtriBuff[idx * 4 + i].y(), ptr_viewtriBuff[idx * 4 + i].z());
			}
			else
				ptr_viewtriBuff[idx * 4 + i] = Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX);
		}
	}
	);
	qDebug("view Tirangle Size Shcek %d ", h_usedSize);
#else

	auto ptr_triRepos = thrust::raw_pointer_cast(voxel_tri_repos.data());
	auto ptr_viewtriBuff = view_triangles;
	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(hash_usedCheckBuffer.begin(), thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(hash_usedCheckBuffer.begin() + h_usedSize, thrust::make_counting_iterator<size_t>(h_usedSize)),
		[ptr_triRepos, ptr_viewtriBuff, _vtx_pos_repos] __device__(auto & tu)
	{
		uint32_t repos_num = thrust::get<0>(tu);
		size_t idx = thrust::get<1>(tu);

		for (size_t i = 0; i < 4; i++)
		{
			if ((ptr_triRepos[repos_num * 4 + i].x() != kEmpty32) && (ptr_triRepos[repos_num * 4 + i].y() != kEmpty32) && (ptr_triRepos[repos_num * 4 + i].z() != kEmpty32))
			{
				//printf("%d,%d,%d\n", ptr_viewtriBuff[idx * 4 + i].x(), ptr_viewtriBuff[idx * 4 + i].y(), ptr_viewtriBuff[idx * 4 + i].z());
				Eigen::Vector3f triangle_vtx[3];

				triangle_vtx[0] = _vtx_pos_repos[ptr_triRepos[repos_num * 4 + i].x()];
				triangle_vtx[1] = _vtx_pos_repos[ptr_triRepos[repos_num * 4 + i].y()];
				triangle_vtx[2] = _vtx_pos_repos[ptr_triRepos[repos_num * 4 + i].z()];

				float distance[3];


				distance[0] = norm3df(triangle_vtx[0].x() - triangle_vtx[1].x(), triangle_vtx[0].y() - triangle_vtx[1].y(), triangle_vtx[0].z() - triangle_vtx[1].z());
				distance[1] = norm3df(triangle_vtx[1].x() - triangle_vtx[2].x(), triangle_vtx[1].y() - triangle_vtx[2].y(), triangle_vtx[1].z() - triangle_vtx[2].z());
				distance[2] = norm3df(triangle_vtx[2].x() - triangle_vtx[0].x(), triangle_vtx[2].y() - triangle_vtx[0].y(), triangle_vtx[2].z() - triangle_vtx[0].z());

				const float distance_limit = 0.2;
				if (distance[0] > distance_limit || distance[1] > distance_limit || distance[2] > distance_limit)
				{
					printf("make triangle List , too far : %llu\n", idx);
					ptr_viewtriBuff[idx * 4 + i] = Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX);
				}
				else
				{
					ptr_viewtriBuff[idx * 4 + i] = ptr_triRepos[repos_num * 4 + i];

				}
			}
			else
				ptr_viewtriBuff[idx * 4 + i] = Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX);
		}
	}
	);
	qDebug("view Tirangle Size Shcek %d ", h_usedSize);

	auto endPtr = thrust::copy_if(
		thrust::cuda::par_nosync(*alloc_).on(st),
		view_triangles,
		view_triangles + (h_usedSize * 4),
		view_triangles,
		[]__device__(auto & vecTri)
	{
		return (vecTri.x() != UINT_MAX && vecTri.y() != UINT_MAX && vecTri.z() != UINT_MAX);
	}
	);
#endif // TEST_CHANGE_ROUTINE

	auto resSize = thrust::distance(view_triangles, endPtr);

#pragma region check triangle result
	/*thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		view_triangles.begin(),
		view_triangles.begin() + resSize,
		[]__device__(auto & vecTri)
	{
		if(vecTri.x() != UINT_MAX && vecTri.y() != UINT_MAX && vecTri.z() != UINT_MAX)
			return;
		else
			printf("[copy] %d,%d,%d\n", vecTri.x(), vecTri.y(), vecTri.z());
	}
	);
	checkCudaSync(st);*/
#pragma endregion

	auto _vtx_replace_idx = thrust::raw_pointer_cast(vtx_replace_idx.data());
	thrust::transform(
		thrust::cuda::par_nosync(*alloc_).on(st),
		view_triangles,
		view_triangles + resSize,
		view_triangles,
		[_vtx_replace_idx] __device__(auto & vecTri)
	{

		//printf("[replace] %d,%d,%d -> %d,%d,%d\n", vecTri.x(), vecTri.y(), vecTri.z(), _vtx_replace_idx[vecTri.x()], _vtx_replace_idx[vecTri.y()], _vtx_replace_idx[vecTri.z()]);
		return Eigen::Vector<uint32_t, 3>(_vtx_replace_idx[vecTri.x()], _vtx_replace_idx[vecTri.y()], _vtx_replace_idx[vecTri.z()]);

	}
	);




	if (mode) {
		std::shared_ptr<thrust::device_vector<int>> vtx_idx_origin = pRegistration->m_ds_keys_x;
		std::shared_ptr<thrust::device_vector<int>> vtx_replacePos = pRegistration->m_ds_keys_y;
		std::shared_ptr<thrust::device_vector<int>> vtx_idx_replace = pRegistration->m_ds_keys_z;

		auto devptr_view_pos = thrust::device_pointer_cast(view_pos_repos);
		auto devptr_view_nm = thrust::device_pointer_cast(view_nm_repos);
		auto devptr_view_color = thrust::device_pointer_cast(view_color_repos);

		if (vtx_idx_origin->size() < h_usedSize_vtx) vtx_idx_origin->resize(h_usedSize_vtx);
		if (vtx_replacePos->size() < h_usedSize_vtx) vtx_replacePos->resize(h_usedSize_vtx);
		if (vtx_idx_replace->size() < h_usedSize_vtx) vtx_idx_replace->resize(h_usedSize_vtx);

		///unique index
		thrust::sequence(
			thrust::cuda::par_nosync(*alloc_).on(st),
			vtx_idx_origin->begin(), vtx_idx_origin->begin() + h_usedSize_vtx, 0, 1);
		thrust::sequence(
			thrust::cuda::par_nosync(*alloc_).on(st),
			vtx_replacePos->begin(), vtx_replacePos->begin() + h_usedSize_vtx, 0, 1);

		// 포인트와 인덱스를 튜플로 묶어 정렬
		auto begin_zip = make_tuple_iterator(devptr_view_pos, devptr_view_nm, devptr_view_color, vtx_idx_origin->begin());
		auto end_zip = make_tuple_iterator(devptr_view_pos + h_usedSize_vtx, devptr_view_nm + h_usedSize_vtx, devptr_view_color + h_usedSize_vtx, vtx_idx_origin->begin() + h_usedSize_vtx);

		thrust::sort(
			thrust::cuda::par_nosync(*alloc_).on(st),
			begin_zip, end_zip);

		auto begin_zip_2 = make_tuple_iterator(devptr_view_pos, devptr_view_nm, devptr_view_color, vtx_replacePos->begin());
		auto end_zip_2 = make_tuple_iterator(devptr_view_pos + h_usedSize_vtx, devptr_view_nm + h_usedSize_vtx, devptr_view_color + h_usedSize_vtx, vtx_replacePos->begin() + h_usedSize_vtx);

		// 중복 제거
		auto end_unique_zip = thrust::unique(
			thrust::cuda::par_nosync(*alloc_).on(st),
			begin_zip_2, end_zip_2);
		size_t unique_Size = thrust::distance(begin_zip_2, end_unique_zip);

		auto vtx_replacePos_rawPtr = thrust::raw_pointer_cast(vtx_replacePos->data());
		auto vtx_idx_replace_rawPtr = thrust::raw_pointer_cast(vtx_idx_replace->data());

		thrust::for_each(
			thrust::cuda::par_nosync(*alloc_).on(st),
			thrust::make_counting_iterator<int>(0),
			thrust::make_counting_iterator<int>(unique_Size),
			[vtx_replacePos_rawPtr, vtx_idx_replace_rawPtr, h_usedSize_vtx, unique_Size] __device__(const int& idx)
		{
			size_t first = vtx_replacePos_rawPtr[idx];
			size_t last = (idx < unique_Size - 1) ? vtx_replacePos_rawPtr[idx + 1] : unique_Size;

			for (size_t i = first; i < last; i++)
				vtx_idx_replace_rawPtr[i] = idx;
		}
		);


		//thrust::host_vector<int> host_vec(h_usedSize_vtx);

		//// 디바이스 벡터에서 호스트 벡터로 복사
		//thrust::copy(vtx_idx_replace->begin(), vtx_idx_replace->begin() + h_usedSize_vtx, host_vec.begin());

		//// 호스트 벡터의 값을 출력
		//for (int value : host_vec) {
		//	std::cout << value << " ";
		//}
		//std::cout << std::endl;

		auto begin_idx = make_tuple_iterator(vtx_idx_origin->begin(), vtx_idx_replace->begin());
		auto end_idx = make_tuple_iterator(vtx_idx_origin->begin() + h_usedSize_vtx, vtx_idx_replace->begin() + h_usedSize_vtx);

		thrust::sort(
			thrust::cuda::par_nosync(*alloc_).on(st),
			begin_idx, end_idx);

		// triangles 업데이트
		thrust::for_each(
			thrust::cuda::par_nosync(*alloc_).on(st),
			view_triangles,
			view_triangles + resSize,
			[vtx_idx_replace_rawPtr] __device__(Eigen::Vector<uint32_t, 3>&tri) {
			tri(0, 0) = vtx_idx_replace_rawPtr[tri(0, 0)];
			tri(1, 0) = vtx_idx_replace_rawPtr[tri(1, 0)];
			tri(2, 0) = vtx_idx_replace_rawPtr[tri(2, 0)];
		});
	}











#pragma region check render data organize
	if (0) {
		Eigen::Vector3f* host_pts = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3f* host_nms = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3b* host_clrs = new Eigen::Vector3b[h_usedSize_vtx];

		checkCudaErrors(cudaMemcpyAsync(host_pts, view_pos_repos, sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_nms, view_nm_repos, sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_clrs, view_color_repos, sizeof(Eigen::Vector3b) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));

		static	int cntFile = 0;
		char szTemp[128];
		sprintf(szTemp, "%04d", cntFile++);
		std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_test" + ".ply";
		pRegistration->plyFileWrite_Ex(
			filename,
			h_usedSize_vtx,
			host_pts,
			host_nms,
			host_clrs
		);

		delete[] host_pts;
		delete[] host_nms;
		delete[] host_clrs;
	}
#if 0
	//if (h_usedSize_vtx > 100000 && h_usedSize_vtx < 101000)
	{
		Eigen::Vector3f* host_pts = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3f* host_nms = new Eigen::Vector3f[h_usedSize_vtx];
		Eigen::Vector3b* host_clrs = new Eigen::Vector3b[h_usedSize_vtx];

		Eigen::Vector3i* host_tri = new Eigen::Vector3i[resSize];


		checkCudaErrors(cudaMemcpyAsync(host_pts, thrust::raw_pointer_cast(view_pos_repos.data()), sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_nms, thrust::raw_pointer_cast(view_nm_repos.data()), sizeof(Eigen::Vector3f) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));
		checkCudaErrors(cudaMemcpyAsync(host_clrs, thrust::raw_pointer_cast(view_color_repos.data()), sizeof(Eigen::Vector3b) * h_usedSize_vtx, cudaMemcpyDeviceToHost, st));

		checkCudaErrors(cudaMemcpyAsync(host_tri, thrust::raw_pointer_cast(view_triangles.data()), sizeof(Eigen::Vector3i) * resSize, cudaMemcpyDeviceToHost, st));

		static	int cntFile = 0;
		char szTemp[128];
		sprintf(szTemp, "%04d", cntFile++);
		std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_test" + ".ply";
		pRegistration->plyFileWrite_mesh(host_pts, host_nms, host_clrs, host_tri, h_usedSize_vtx, resSize, filename, false);
		//pRegistration->plyFileWrite(host_pts, host_nms, host_clrs, h_usedSize_vtx, filename, false);


		delete[] host_pts;
		delete[] host_nms;
		delete[] host_clrs;
		delete[] host_tri;
	}
#endif

#pragma endregion

	return Eigen::Vector2i(resSize, h_usedSize_vtx);
}
#endif
/* 사용되지 않음
void saveOffFile(
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>& voxel_tri_repos
	, thrust::device_vector<Eigen::Vector3f>& vtx_pos_repos
	, size_t view_triSize
)
{
	cudaDeviceSynchronize();

	thrust::host_vector < Eigen::Vector<uint32_t, 3>>host_triangles(view_triSize);//voxel_tri_repos.size());

	thrust::copy(
		voxel_tri_repos.begin(), voxel_tri_repos.begin() + view_triSize,
		host_triangles.begin()
	);

	//();
	thrust::host_vector < Eigen::Vector3f>host_pos(vtx_pos_repos.begin(), vtx_pos_repos.end());
	cudaDeviceSynchronize();

	OFFFormat off;

	for (auto& t : host_triangles)
	{
		if (false == VECTOR3F_VALID_(t))
			continue;

		auto idx_x = t.x();
		auto idx_y = t.y();
		auto idx_z = t.z();

		if (
			(host_pos[idx_x].x() != 0.0f && host_pos[idx_x].y() != 0.0f && host_pos[idx_x].z() != 0.0f) &&
			(host_pos[idx_y].x() != 0.0f && host_pos[idx_y].y() != 0.0f && host_pos[idx_y].z() != 0.0f) &&
			(host_pos[idx_z].x() != 0.0f && host_pos[idx_z].y() != 0.0f && host_pos[idx_z].z() != 0.0f)
			)
		{

			off.AddPointFloat3(host_pos[idx_x].data());
			off.AddIndex((off.GetPoints().size() - 1) / 3);

			off.AddPointFloat3(host_pos[idx_y].data());
			off.AddIndex((off.GetPoints().size() - 1) / 3);

			off.AddPointFloat3(host_pos[idx_z].data());
			off.AddIndex((off.GetPoints().size() - 1) / 3);

			{
				off.AddColor(255, 255, 255);
				off.AddColor(255, 255, 255);
				off.AddColor(255, 255, 255);
			}
		}
	}
	static	int cntFile = 0;
	char szTemp[128];
	sprintf(szTemp, "%04d", cntFile++);
	std::string filename = GetSaveDataFolderPath() + "\\MC_" + std::string(szTemp) + ".off";
	off.Serialize(filename);
	//off.Serialize(GetSaveDataFolderPath() + "\\MCResult.off");

	qDebug("Saved...");
}
*/
/*
__global__ void kernel_lookup_icp_taget_pcd(
	HashKey64* hashinfo,
	Eigen::Vector<uint32_t, 3>* triangle_buf,
	const Eigen::Vector3f* vertex,
	const Eigen::Vector3f* normal,
	//const uint32_t* vertex_used,
	const Eigen::Vector3f Bounding_base,	// Target AABB
	const Eigen::Vector3i BoundingBox,		// Target 에서 가져오는 Voxel 개수
	const int voxel_step,					// 100 ~ 200 um 선택
	Eigen::Vector3f* tgt_pos,				// ICP Target(적당한 길이를 파악하여 지정)
	Eigen::Vector3f* tgt_normal,			// ICP Target(적당한 길이를 파악하여 지정)
	uint32_t* tgt_cnt,						// 함수 내부에서 atomic 하게 증가
	size_t max_voxel						// = BoundingBox.x * BoundingBox.y * BoundingBox.z
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid < max_voxel)
	{
		Eigen::Vector3i base_voxel;
		Eigen::Vector3i voxel_idx;

		voxel_idx.z() = threadid / (BoundingBox.x() * BoundingBox.y());
		voxel_idx.y() = (threadid % (BoundingBox.x() * BoundingBox.y())) / BoundingBox.x();
		voxel_idx.x() = (threadid % (BoundingBox.x() * BoundingBox.y())) % BoundingBox.x();

		voxel_idx *= voxel_step;



		base_voxel.x() = (int)floorf(Bounding_base.x() * 10.f + 2500.f);
		base_voxel.y() = (int)floorf(Bounding_base.y() * 10.f + 2500.f);
		base_voxel.z() = (int)floorf(Bounding_base.z() * 10.f + 2500.f);

		if (base_voxel.x() % 2) base_voxel.x()--; // 200 um 간격
		if (base_voxel.y() % 2) base_voxel.y()--; // 200 um 간격
		if (base_voxel.z() % 2) base_voxel.z()--; // 200 um 간격

		voxel_idx += base_voxel;

		HashKey voxel_key(voxel_idx.x(), voxel_idx.y(), voxel_idx.z());

		//printf("voxel_idx : %d, %d, %d, voxel_key : %llu\n", voxel_idx.x(), voxel_idx.y(), voxel_idx.z(), voxel_key);

		uint32_t temp_triangle_vertex[4 * 3] = { kEmpty32, };
		//	mscho	@20240221
		for (int i = 0; i < 12; i++)temp_triangle_vertex[i] = kEmpty32;

		auto _voxelTriangles_array = triangle_buf->data();
		// 해당 voxel로 지정된 triangle list에서 vertex index를 받아온다.
		auto voxel_slot = _hashtable_lookup_triangle_list_func64
		(
			hashinfo, hashtable,
			_voxelTriangles_array,
			temp_triangle_vertex,
			voxel_key
		);

		if (voxel_slot == kEmpty32) return;

		// 아래의  Loop가 다 돌고나면,
		// tgt_pos[prev_n] / tgt_normal[prev_n] : 여기에 ICP target 용 정보가 다 모이게 된다.
		int cntPts = 0;
		Eigen::Vector3f pts[12];
		Eigen::Vector3f nms[12];
		for (int i = 0; i < 4; i++)
		{
			Eigen::Vector<uint32_t, 3> triangle_idx = Eigen::Vector<uint32_t, 3>(temp_triangle_vertex[i * 3], temp_triangle_vertex[i * 3 + 1], temp_triangle_vertex[i * 3 + 2]);
			//printf("(0)triangle_idx : %d, %d, %d\n", temp_triangle_vertex[i * 3], temp_triangle_vertex[i * 3+1], temp_triangle_vertex[i * 3+2]);
			//printf("(1)triangle_idx : %d, %d, %d\n\n", triangle_idx.x(), triangle_idx.y(), triangle_idx.z());

			if (triangle_idx.x() != kEmpty32 && triangle_idx.y() != kEmpty32 && triangle_idx.z() != kEmpty32)
			{
				///printf("triangle_idx : %d, %d, %d\n", triangle_idx.x(), triangle_idx.y(), triangle_idx.z());

				// Todo
				// 삼각형의 면적을 구하고
				// 무게중심을 구해서, 좌표로 지정
				//  Vertex 의  Normal의 평균을 구한다.
				// 이것들을 배열로 저장해 놓는다.
				auto& v0 = vertex[triangle_idx.x()];
				auto& v1 = vertex[triangle_idx.y()];
				auto& v2 = vertex[triangle_idx.z()];

				//printf("%f, %f, %f / %f, %f, %f / %f, %f, %f\n", v0.x(), v0.y(), v0.z(), v1.x(), v1.y(), v1.z(), v2.x(), v2.y(), v2.z());

				float3 p0 = { v0.x(), v0.y(), v0.z() };
				float3 p1 = { v1.x(), v1.y(), v1.z() };
				float3 p2 = { v2.x(), v2.y(), v2.z() };

				float3 normal;
				float area;
				GetTriangleNormalAndArea(p0, p1, p2, normal, area); // 삼각형의 법선과 면적을 한번에

				//printf("%f, %f, %f / %f, %f, %f / %f, %f, %f\n", p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);

				if (0.0015f < area)
				{
					float3 centroid = GetCentroid(p0, p1, p2);		// 삼각형 무게중심
#if 0
						uint32_t prev_n = atomicAdd(tgt_cnt, 1);
						tgt_pos[prev_n] = Eigen::Vector3f(centroid.x, centroid.y, centroid.z);
						tgt_normal[prev_n] = Eigen::Vector3f(normal.x, normal.y, normal.z);
#else
						pts[cntPts] = Eigen::Vector3f(centroid.x, centroid.y, centroid.z);
						nms[cntPts] = Eigen::Vector3f(normal.x, normal.y, normal.z);
						cntPts++;
#endif
					}
				}
				else
					continue;
			}
			if (0 < cntPts) {
				Eigen::Vector3f avgPt(0., 0., 0.);
				Eigen::Vector3f avgNm(0., 0., 0.);
				for (size_t i = 0; i < cntPts; i++)
				{
					avgPt += pts[i];
					avgNm += nms[i];
				}
				avgPt /= cntPts;
				avgNm /= cntPts;
				uint32_t prev_n = atomicAdd(tgt_cnt, 1);
				tgt_pos[prev_n] = avgPt + Eigen::Vector3f(0.05, 0.05, 0.05);
				tgt_normal[prev_n] = avgNm;
			}
		}
	}
*/
#endif

#ifndef BUILD_FOR_CPU
__global__ void Kernel_InterpolateDepthmap_v3(
	int depthmapInterpolateLevel,
	Eigen::Vector3f * depthMap
	, unsigned int* depthMapCnt
	, Eigen::Vector3f * _normalMap
	, unsigned int* colorMap
	, size_t hLength
	, size_t vLength
	, float pointMag
	, unsigned char* current_img_0
	, unsigned char* current_img_45
	, const Eigen::Matrix4f dev_camRT
	, const Eigen::Matrix3f dev_cam_tilt
);
#endif

#ifndef BUILD_FOR_CPU
__global__ void MarchingCubesKernel::Kernel_Filter(
	Eigen::Vector3f * depthMap,
	unsigned short* depthmap_mask,
	Eigen::Vector3f * normalMap,
	unsigned int* colorMap,
	Eigen::Matrix4f transform,
	Eigen::Vector3f filterMin,
	Eigen::Vector3f filterMax,
	float zBeginOffset, float zEnd,
	HashKey64 * globalHash_info,
	size_t hLength, size_t vLength, size_t dLength, float voxelSize,
	Eigen::AlignedBox3f globalScanAreaAABB,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * voxelNormals,
	Eigen::Vector3b * voxelColors,
	int meshColorMode,
	Eigen::Vector3f * filteredPoints,
	Eigen::Vector3f * filteredPointNormals,
	Eigen::Vector4f * filteredPointColors,
	unsigned int* filteredPointsCount)
{
	printf("Kernel filter\n");

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;

	auto xIndex = (size_t)((threadid) % hLength);
	auto yIndex = (size_t)((threadid) / hLength);

	auto v = depthMap[yIndex * hLength + xIndex];
	auto depthMapNormal = normalMap[yIndex * hLength + xIndex];
	unsigned short surfaceVoxelValueCount = SHRT_MAX;

	//if (false == FLT_VALID(v.z()))
	//	return;

	if (0 == depthmap_mask[yIndex * hLength + xIndex])
	{
		if (nullptr != filteredPoints)
		{
			auto lx = (float)xIndex * voxelSize - (hLength * voxelSize * 0.5f) + voxelSize * 0.5f;
			auto ly = (float)yIndex * voxelSize - (vLength * voxelSize * 0.5f) + voxelSize * 0.5f;
			Eigen::Vector4f gp4 = transform * Eigen::Vector4f(lx, ly, v.z(), 1.0f);
			auto gp = Eigen::Vector3f(gp4.x(), gp4.y(), gp4.z());

			auto old = atomicAdd(filteredPointsCount, 1);
			filteredPoints[old] = Eigen::Vector3f(gp.x(), gp.y(), gp.z());
			Eigen::Vector3f scanDir = transform.block<3, 1>(0, 2).normalized();
			filteredPointNormals[old] = scanDir;
			filteredPointColors[old] = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.3f);
		}
		return;
	}

	auto lx = (float)xIndex * voxelSize - (hLength * voxelSize * 0.5f) + voxelSize * 0.5f;
	auto ly = (float)yIndex * voxelSize - (vLength * voxelSize * 0.5f) + voxelSize * 0.5f;

	if ((filterMin.x() <= lx && lx <= filterMax.x()) &&
		(filterMin.y() <= ly && ly <= filterMax.y()))
	{
		for (float zPos = filterMax.z() + zBeginOffset; zPos <= zEnd; zPos += voxelSize)
		{
			auto lz = zPos;
			Eigen::Vector4f gp4 = transform * Eigen::Vector4f(lx, ly, lz, 1.0f);
			auto gp = Eigen::Vector3f(gp4.x(), gp4.y(), gp4.z());

			auto xGlobalIndex = (size_t)floorf(gp.x() / voxelSize - globalScanAreaAABB.min().x() / voxelSize);
			auto yGlobalIndex = (size_t)floorf(gp.y() / voxelSize - globalScanAreaAABB.min().y() / voxelSize);
			auto zGlobalIndex = (size_t)floorf(gp.z() / voxelSize - globalScanAreaAABB.min().z() / voxelSize);

			if (nullptr != filteredPoints)
			{
				if ((filterMax.z() + zBeginOffset - voxelSize <= zPos || zPos <= zEnd + voxelSize) &&
					(filterMin.x() - voxelSize <= lx || lx <= filterMax.x() + voxelSize) &&
					(filterMin.y() - voxelSize <= ly || ly <= filterMax.y() + voxelSize))
				{
					auto old = atomicAdd(filteredPointsCount, 1);
					filteredPoints[old] = Eigen::Vector3f(gp.x(), gp.y(), gp.z());
					Eigen::Vector3f scanDir = transform.block<3, 1>(0, 2).normalized();
					filteredPointNormals[old] = scanDir;
					filteredPointColors[old] = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.3f);
				}
			}

			HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
			uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(key);
			if (kEmpty32 != hashSlot_idx)
			{
				float vv = VV2D(voxelValues[hashSlot_idx]);
				unsigned short vc = VOXELCNT_VALUE(voxelValueCounts[hashSlot_idx]);

				if (VOXEL_INVALID != voxelValues[hashSlot_idx])
				{
					voxelValues[hashSlot_idx] = VOXEL_INVALID;
					voxelValueCounts[hashSlot_idx] = USHRT_MAX;
				}
			}
		}
	}

	//unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//if (threadid > (hLength * vLength) - 1) return;

	//auto xIndex = (size_t)((threadid) % hLength);
	//auto yIndex = (size_t)((threadid) / hLength);

	//auto v = depthMap[yIndex * hLength + xIndex];
	//auto depthMapNormal = normalMap[yIndex * hLength + xIndex];
	//unsigned short surfaceVoxelValueCount = SHRT_MAX;

	////if (false == FLT_VALID(v.z()))
	////	return;

	//if (0 == depthmap_mask[yIndex * hLength + xIndex])
	//{
	//	if (nullptr != filteredPoints)
	//	{
	//		auto lx = (float)xIndex * voxelSize - (hLength * voxelSize * 0.5f) + voxelSize * 0.5f;
	//		auto ly = (float)yIndex * voxelSize - (vLength * voxelSize * 0.5f) + voxelSize * 0.5f;
	//		Eigen::Vector4f gp4 = transform * Eigen::Vector4f(lx, ly, v.z(), 1.0f);
	//		auto gp = Eigen::Vector3f(gp4.x(), gp4.y(), gp4.z());

	//		auto old = atomicAdd(filteredPointsCount, 1);
	//		filteredPoints[old] = Eigen::Vector3f(gp.x(), gp.y(), gp.z());
	//		Eigen::Vector3f scanDir = transform.block<3, 1>(0, 2).normalized();
	//		filteredPointNormals[old] = scanDir;
	//		filteredPointColors[old] = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.3f);
	//	}
	//	return;
	//}

	//auto lx = (float)xIndex * voxelSize - (hLength * voxelSize * 0.5f) + voxelSize * 0.5f;
	//auto ly = (float)yIndex * voxelSize - (vLength * voxelSize * 0.5f) + voxelSize * 0.5f;

	//if ((filterMin.x() <= lx && lx <= filterMax.x()) &&
	//	(filterMin.y() <= ly && ly <= filterMax.y()))
	//{
	//	for (float zPos = filterMax.z() + zBeginOffset; zPos <= zEnd; zPos += voxelSize)
	//	{
	//		auto lz = zPos;
	//		Eigen::Vector4f gp4 = transform * Eigen::Vector4f(lx, ly, lz, 1.0f);
	//		auto gp = Eigen::Vector3f(gp4.x(), gp4.y(), gp4.z());

	//		auto xGlobalIndex = (size_t)floorf(gp.x() / voxelSize - globalScanAreaAABB.min().x() / voxelSize);
	//		auto yGlobalIndex = (size_t)floorf(gp.y() / voxelSize - globalScanAreaAABB.min().y() / voxelSize);
	//		auto zGlobalIndex = (size_t)floorf(gp.z() / voxelSize - globalScanAreaAABB.min().z() / voxelSize);

	//		if (nullptr != filteredPoints)
	//		{
	//			if ((filterMax.z() + zBeginOffset - voxelSize <= zPos || zPos <= zEnd + voxelSize) &&
	//				(filterMin.x() - voxelSize <= lx || lx <= filterMax.x() + voxelSize) &&
	//				(filterMin.y() - voxelSize <= ly || ly <= filterMax.y() + voxelSize))
	//			{
	//				auto old = atomicAdd(filteredPointsCount, 1);
	//				filteredPoints[old] = Eigen::Vector3f(gp.x(), gp.y(), gp.z());
	//				Eigen::Vector3f scanDir = transform.block<3, 1>(0, 2).normalized();
	//				filteredPointNormals[old] = scanDir;
	//				filteredPointColors[old] = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 0.3f);
	//			}
	//		}

	//		HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
	//		uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(key);
	//		if (kEmpty32 != hashSlot_idx)
	//		{
	//			float vv = VV2D(voxelValues[hashSlot_idx]);
	//			unsigned short vc = voxelValueCounts[hashSlot_idx];

	//			if (VOXEL_INVALID != voxelValues[hashSlot_idx])
	//			{
	//				voxelValues[hashSlot_idx] = VOXEL_INVALID;
	//				voxelValueCounts[hashSlot_idx] = 0;
	//			}
	//		}
	//	}
	//}
}
#endif

#ifndef BUILD_FOR_CPU
//	mscho	@20240521
//	Depthmap의 boundary를 미리 계산해서, 
//	Integrate 함수의 kernel 실행갯수를 줄여준다.
__global__ void Kernel_DepthMap_Boundary(
	Eigen::Vector3f * depthMap
	, int32_t * DepthMap_Bound	//	mscho	@20240523
	, size_t hLength
	, size_t vLength
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;

	size_t index = threadid;

	auto yIndex = (size_t)(index / hLength);
	auto xIndex = (size_t)(index % hLength);

	auto v = depthMap[yIndex * hLength + xIndex];
	auto z_map = v.z();

	if (FLT_VALID(z_map))
	{
		atomicMin(&DepthMap_Bound[0], xIndex);
		atomicMax(&DepthMap_Bound[1], xIndex);
		atomicMin(&DepthMap_Bound[2], yIndex);
		atomicMax(&DepthMap_Bound[3], yIndex);
	}
}
#endif

//	mscho	@20240527
//	Depthmap의 boundary를 미리 계산해서, 
//	Integrate 함수의 kernel 실행갯수를 줄여준다.
__global__ void MarchingCubesKernel::Kernel_DepthMap_Boundary_v3(
	Eigen::Vector3f * depthMap
	, int* DepthMap_Bound
	, float zDepthUnit
	, size_t hLength
	, size_t vLength
	, size_t voxelx
	, size_t voxely
	, size_t scalex
	, size_t scaley
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;

	size_t index = threadid;

	auto yIndex = (size_t)(index / hLength);
	auto xIndex = (size_t)(index % hLength);

	auto voxel_xidx = xIndex / scalex;
	auto voxel_yidx = yIndex / scaley;

	auto v = depthMap[yIndex * hLength + xIndex];
	auto z_map = v.z();


	if (FLT_VALID(z_map))
	{
		atomicMin(&DepthMap_Bound[0], int(xIndex));
		atomicMax(&DepthMap_Bound[1], int(xIndex));
		atomicMin(&DepthMap_Bound[2], int(yIndex));
		atomicMax(&DepthMap_Bound[3], int(yIndex));

		int32_t z_value = floorf(z_map / zDepthUnit + 10000.0);
		atomicMin(&DepthMap_Bound[7], z_value);	//	Minimum
		atomicMax(&DepthMap_Bound[9], z_value);	//	Maximum

		atomicMin(&DepthMap_Bound[10], int(voxel_xidx));
		atomicMax(&DepthMap_Bound[11], int(voxel_xidx));
		atomicMin(&DepthMap_Bound[12], int(voxel_yidx));
		atomicMax(&DepthMap_Bound[13], int(voxel_yidx));

		z_value = floorf(z_map / zDepthUnit + 10000.0);
		atomicMin(&DepthMap_Bound[17], z_value);	//	Minimum
		atomicMax(&DepthMap_Bound[19], z_value);	//	Maximum
	}
#else
	int threadCount = omp_get_max_threads();// omp_get_num_threads();
	std::vector<std::array<int, 20>> depthMapBoundsPerThread;
	depthMapBoundsPerThread.resize(size_t(threadCount));

	for (int iThread = 0; iThread < threadCount; iThread++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];
		memcpy(depthMapBoundPerThread.data(), DepthMap_Bound, 20 * sizeof(int));
	}

	size_t pointCount = hLength * vLength;

#pragma omp parallel for schedule(dynamic, 256)
	for (int index = 0; index < pointCount; index++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[omp_get_thread_num()];

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto v = depthMap[yIndex * hLength + xIndex];
		auto z_map = v.z();

		if (FLT_VALID(z_map))
		{
			auto voxel_xidx = xIndex / scalex;
			auto voxel_yidx = yIndex / scaley;

			depthMapBoundPerThread[0] = min(depthMapBoundPerThread[0], int(xIndex));
			depthMapBoundPerThread[1] = max(depthMapBoundPerThread[1], int(xIndex));
			depthMapBoundPerThread[2] = min(depthMapBoundPerThread[2], int(yIndex));
			depthMapBoundPerThread[3] = max(depthMapBoundPerThread[3], int(yIndex));

			int32_t z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[7] = min(depthMapBoundPerThread[7], z_value);	//	Minimum
			depthMapBoundPerThread[9] = max(depthMapBoundPerThread[9], z_value);	//	Maximum

			depthMapBoundPerThread[10] = min(depthMapBoundPerThread[10], int(voxel_xidx));
			depthMapBoundPerThread[11] = max(depthMapBoundPerThread[11], int(voxel_xidx));
			depthMapBoundPerThread[12] = min(depthMapBoundPerThread[12], int(voxel_yidx));
			depthMapBoundPerThread[13] = max(depthMapBoundPerThread[13], int(voxel_yidx));

			z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[17] = min(depthMapBoundPerThread[17], z_value);	//	Minimum
			depthMapBoundPerThread[19] = max(depthMapBoundPerThread[19], z_value);	//	Maximum
		}
	}

	// 개별 쓰레드에서 작업한 내용을 합쳐준다.
	for (int iThread = 0; iThread < threadCount; iThread++) {
		const std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];

		DepthMap_Bound[0] = min(DepthMap_Bound[0], depthMapBoundPerThread[0]);
		DepthMap_Bound[1] = max(DepthMap_Bound[1], depthMapBoundPerThread[1]);
		DepthMap_Bound[2] = min(DepthMap_Bound[2], depthMapBoundPerThread[2]);
		DepthMap_Bound[3] = max(DepthMap_Bound[3], depthMapBoundPerThread[3]);

		DepthMap_Bound[7] = min(DepthMap_Bound[7], depthMapBoundPerThread[7]);
		DepthMap_Bound[9] = max(DepthMap_Bound[9], depthMapBoundPerThread[9]);

		DepthMap_Bound[10] = min(DepthMap_Bound[10], depthMapBoundPerThread[10]);
		DepthMap_Bound[11] = max(DepthMap_Bound[11], depthMapBoundPerThread[11]);
		DepthMap_Bound[12] = min(DepthMap_Bound[12], depthMapBoundPerThread[12]);
		DepthMap_Bound[13] = max(DepthMap_Bound[13], depthMapBoundPerThread[13]);

		DepthMap_Bound[17] = min(DepthMap_Bound[17], depthMapBoundPerThread[17]);
		DepthMap_Bound[19] = max(DepthMap_Bound[19], depthMapBoundPerThread[19]);

	}
#endif
}


//	mscho	@20250313
//	Depthmap의 boundary를 미리 계산해서, 
//	Integrate 함수의 kernel 실행갯수를 줄여준다.
__global__ void MarchingCubesKernel::Kernel_DepthMap_Boundary_Map1(
	Eigen::Vector3f * depthMap
	, int* DepthMap_Bound
	, float zDepthUnit
	, size_t hLength
	, size_t vLength
)
{


#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;

	size_t index = threadid;

	auto yIndex = (size_t)(index / hLength);
	auto xIndex = (size_t)(index % hLength);

	auto v = depthMap[yIndex * hLength + xIndex];
	auto z_map = v.z();


	if (FLT_VALID(z_map))
	{
		atomicMin(&DepthMap_Bound[0], int(xIndex));
		atomicMax(&DepthMap_Bound[1], int(xIndex));
		atomicMin(&DepthMap_Bound[2], int(yIndex));
		atomicMax(&DepthMap_Bound[3], int(yIndex));

		int32_t z_value = (int32_t)floorf(z_map / zDepthUnit + 10000.0);
		atomicMin(&DepthMap_Bound[7], z_value);	//	Minimum
		atomicMax(&DepthMap_Bound[9], z_value);	//	Maximum
	}
#else
	int threadCount = omp_get_max_threads();// omp_get_num_threads();
	std::vector<std::array<int, 20>> depthMapBoundsPerThread;
	depthMapBoundsPerThread.resize(size_t(threadCount));

	for (int iThread = 0; iThread < threadCount; iThread++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];
		memcpy(depthMapBoundPerThread.data(), DepthMap_Bound, 20 * sizeof(int));
	}

	size_t pointCount = hLength * vLength;

#pragma omp parallel for schedule(dynamic, 256)
	for (int index = 0; index < pointCount; index++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[omp_get_thread_num()];

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto v = depthMap[yIndex * hLength + xIndex];
		auto z_map = v.z();

		if (FLT_VALID(z_map))
		{
			auto voxel_xidx = xIndex / scalex;
			auto voxel_yidx = yIndex / scaley;

			depthMapBoundPerThread[0] = min(depthMapBoundPerThread[0], int(xIndex));
			depthMapBoundPerThread[1] = max(depthMapBoundPerThread[1], int(xIndex));
			depthMapBoundPerThread[2] = min(depthMapBoundPerThread[2], int(yIndex));
			depthMapBoundPerThread[3] = max(depthMapBoundPerThread[3], int(yIndex));

			int32_t z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[7] = min(depthMapBoundPerThread[7], z_value);	//	Minimum
			depthMapBoundPerThread[9] = max(depthMapBoundPerThread[9], z_value);	//	Maximum

			depthMapBoundPerThread[10] = min(depthMapBoundPerThread[10], int(voxel_xidx));
			depthMapBoundPerThread[11] = max(depthMapBoundPerThread[11], int(voxel_xidx));
			depthMapBoundPerThread[12] = min(depthMapBoundPerThread[12], int(voxel_yidx));
			depthMapBoundPerThread[13] = max(depthMapBoundPerThread[13], int(voxel_yidx));

			z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[17] = min(depthMapBoundPerThread[17], z_value);	//	Minimum
			depthMapBoundPerThread[19] = max(depthMapBoundPerThread[19], z_value);	//	Maximum
		}
	}

	// 개별 쓰레드에서 작업한 내용을 합쳐준다.
	for (int iThread = 0; iThread < threadCount; iThread++) {
		const std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];

		DepthMap_Bound[0] = min(DepthMap_Bound[0], depthMapBoundPerThread[0]);
		DepthMap_Bound[1] = max(DepthMap_Bound[1], depthMapBoundPerThread[1]);
		DepthMap_Bound[2] = min(DepthMap_Bound[2], depthMapBoundPerThread[2]);
		DepthMap_Bound[3] = max(DepthMap_Bound[3], depthMapBoundPerThread[3]);

		DepthMap_Bound[7] = min(DepthMap_Bound[7], depthMapBoundPerThread[7]);
		DepthMap_Bound[9] = max(DepthMap_Bound[9], depthMapBoundPerThread[9]);

		DepthMap_Bound[10] = min(DepthMap_Bound[10], depthMapBoundPerThread[10]);
		DepthMap_Bound[11] = max(DepthMap_Bound[11], depthMapBoundPerThread[11]);
		DepthMap_Bound[12] = min(DepthMap_Bound[12], depthMapBoundPerThread[12]);
		DepthMap_Bound[13] = max(DepthMap_Bound[13], depthMapBoundPerThread[13]);

		DepthMap_Bound[17] = min(DepthMap_Bound[17], depthMapBoundPerThread[17]);
		DepthMap_Bound[19] = max(DepthMap_Bound[19], depthMapBoundPerThread[19]);

	}
#endif
}


//	mscho	@20250313
//	Depthmap의 boundary를 미리 계산해서, 
//	Integrate 함수의 kernel 실행갯수를 줄여준다.
__global__ void MarchingCubesKernel::Kernel_DepthMap_Boundary_Map3(
	Eigen::Vector3f * depthMap
	, int* DepthMap_Bound
	, float zDepthUnit
	, size_t hLength
	, size_t vLength
)
{


#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;

	size_t index = threadid;

	auto yIndex = (size_t)(index / hLength);
	auto xIndex = (size_t)(index % hLength);

	auto v = depthMap[yIndex * hLength + xIndex];
	auto z_map = v.z();


	if (FLT_VALID(z_map))
	{
		atomicMin(&DepthMap_Bound[10], int(xIndex));
		atomicMax(&DepthMap_Bound[11], int(xIndex));
		atomicMin(&DepthMap_Bound[12], int(yIndex));
		atomicMax(&DepthMap_Bound[13], int(yIndex));

		int32_t z_value = (int32_t)floorf(z_map / zDepthUnit + 10000.0);
		atomicMin(&DepthMap_Bound[17], z_value);	//	Minimum
		atomicMax(&DepthMap_Bound[19], z_value);	//	Maximum
	}
#else
	int threadCount = omp_get_max_threads();// omp_get_num_threads();
	std::vector<std::array<int, 20>> depthMapBoundsPerThread;
	depthMapBoundsPerThread.resize(size_t(threadCount));

	for (int iThread = 0; iThread < threadCount; iThread++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];
		memcpy(depthMapBoundPerThread.data(), DepthMap_Bound, 20 * sizeof(int));
	}

	size_t pointCount = hLength * vLength;

#pragma omp parallel for schedule(dynamic, 256)
	for (int index = 0; index < pointCount; index++) {
		std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[omp_get_thread_num()];

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto v = depthMap[yIndex * hLength + xIndex];
		auto z_map = v.z();

		if (FLT_VALID(z_map))
		{
			auto voxel_xidx = xIndex / scalex;
			auto voxel_yidx = yIndex / scaley;

			depthMapBoundPerThread[0] = min(depthMapBoundPerThread[0], int(xIndex));
			depthMapBoundPerThread[1] = max(depthMapBoundPerThread[1], int(xIndex));
			depthMapBoundPerThread[2] = min(depthMapBoundPerThread[2], int(yIndex));
			depthMapBoundPerThread[3] = max(depthMapBoundPerThread[3], int(yIndex));

			int32_t z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[7] = min(depthMapBoundPerThread[7], z_value);	//	Minimum
			depthMapBoundPerThread[9] = max(depthMapBoundPerThread[9], z_value);	//	Maximum

			depthMapBoundPerThread[10] = min(depthMapBoundPerThread[10], int(voxel_xidx));
			depthMapBoundPerThread[11] = max(depthMapBoundPerThread[11], int(voxel_xidx));
			depthMapBoundPerThread[12] = min(depthMapBoundPerThread[12], int(voxel_yidx));
			depthMapBoundPerThread[13] = max(depthMapBoundPerThread[13], int(voxel_yidx));

			z_value = floorf(z_map / zDepthUnit + 10000.0);
			depthMapBoundPerThread[17] = min(depthMapBoundPerThread[17], z_value);	//	Minimum
			depthMapBoundPerThread[19] = max(depthMapBoundPerThread[19], z_value);	//	Maximum
		}
	}

	// 개별 쓰레드에서 작업한 내용을 합쳐준다.
	for (int iThread = 0; iThread < threadCount; iThread++) {
		const std::array<int, 20>& depthMapBoundPerThread = depthMapBoundsPerThread[iThread];

		DepthMap_Bound[0] = min(DepthMap_Bound[0], depthMapBoundPerThread[0]);
		DepthMap_Bound[1] = max(DepthMap_Bound[1], depthMapBoundPerThread[1]);
		DepthMap_Bound[2] = min(DepthMap_Bound[2], depthMapBoundPerThread[2]);
		DepthMap_Bound[3] = max(DepthMap_Bound[3], depthMapBoundPerThread[3]);

		DepthMap_Bound[7] = min(DepthMap_Bound[7], depthMapBoundPerThread[7]);
		DepthMap_Bound[9] = max(DepthMap_Bound[9], depthMapBoundPerThread[9]);

		DepthMap_Bound[10] = min(DepthMap_Bound[10], depthMapBoundPerThread[10]);
		DepthMap_Bound[11] = max(DepthMap_Bound[11], depthMapBoundPerThread[11]);
		DepthMap_Bound[12] = min(DepthMap_Bound[12], depthMapBoundPerThread[12]);
		DepthMap_Bound[13] = max(DepthMap_Bound[13], depthMapBoundPerThread[13]);

		DepthMap_Bound[17] = min(DepthMap_Bound[17], depthMapBoundPerThread[17]);
		DepthMap_Bound[19] = max(DepthMap_Bound[19], depthMapBoundPerThread[19]);

	}
#endif
}


//	mscho	@20240527 .....
__global__ void MarchingCubesKernel::Kernel_InterpolateDepthmap_v5(
	int depthmapInterpolateLevel,
	Eigen::Vector3f * depthMap
	, unsigned int* depthMapCnt
	, Eigen::Vector3f * _normalMap
	, unsigned int* colorMap
	, unsigned short* materialMap
	, float* alphaMap
	, float* specularMap
	, size_t hLength
	, size_t vLength
	, float xUnit
	, float yUnit
	, float pointMag
	, bool bDeepLearningEnable				// true : DeepLearning enable & initial OK
	, unsigned short* deeplearning_inference	//	deep learning 추론의 결과가 저장되어 있다.. 400x480
	, unsigned char* current_img_0
	, unsigned char* current_img_45
	, const Eigen::Matrix4f transform_45
	, const Eigen::Matrix4f dev_camRT
	, const Eigen::Matrix3f dev_cam_tilt
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;
	{
#else
	const int threadCount = hLength * vLength;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif

		size_t index = threadid;

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto v = depthMap[yIndex * hLength + xIndex];
		auto z_map = v.z();
		Eigen::Vector3f ave_pos = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3f ave_normal = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3i ave_color = Eigen::Vector3i(0, 0, 0);

		float	min_z = +1000.0;
		float	max_z = -1000.0;

		float	alpha_sum = 0.f;
		float	alpha_ave = 0.f;

		float	specular_sum = 0.f;
		float	specular_ave = 0.f;

		if (!FLT_VALID(z_map))
		{
			//	mscho	@20240527
			auto lx = (float)xIndex * xUnit - ((int)hLength * xUnit * 0.5f) + xUnit * 0.5f;
			auto ly = (float)yIndex * yUnit - ((int)vLength * yUnit * 0.5f) + yUnit * 0.5f;
			int max_gap = depthmapInterpolateLevel;
			int cnt = 0;
			for (int gap = 1; gap <= max_gap; gap++)
			{
				int sx = xIndex >= gap ? xIndex - gap : 0;
				int sy = yIndex >= gap ? yIndex - gap : 0;
				int ex = xIndex < hLength - gap ? xIndex + gap : hLength - 1;
				int ey = yIndex < vLength - gap ? yIndex + gap : vLength - 1;
				cnt = 0;
				float ave_z = 0.f;
				bool ver_edge = false;

				for (int y = sy; y <= ey; y++)
				{
					if (y != sy && y != ey)	// 세로로 가장자리가 아니라면
						ver_edge = false;
					else
						ver_edge = true;

					for (int x = sx; x <= ex; x++)
					{
						if (!ver_edge && (x != sx && x != ex))	// 가로로 가장자리가 아니라면, 세로도 
							continue;
						auto map_idx = y * hLength + x;
						auto _v = depthMap[map_idx];
						if (depthMapCnt[map_idx] > 0 && UINT_VALID(depthMapCnt[map_idx]))
						{
							ave_z += _v.z();
							if (min_z > _v.z())	min_z = _v.z();
							if (max_z < _v.z())	max_z = _v.z();

							ave_pos += _v;
							ave_normal += _normalMap[map_idx];
							ave_color.x() += (int)colorMap[map_idx * 3 + 0];
							ave_color.y() += (int)colorMap[map_idx * 3 + 1];
							ave_color.z() += (int)colorMap[map_idx * 3 + 2];
							cnt++;

							alpha_sum += alphaMap[map_idx];
							specular_sum += specularMap[map_idx];
						}
					}
				}

				if (cnt)
				{
					if (fabsf(max_z - min_z) < 2.0f)
					{
						z_map = ave_z / (float)cnt;
						ave_pos /= (float)cnt;
						ave_color /= (float)cnt;
						float dist = norm3df(ave_normal.x(), ave_normal.y(), ave_normal.z());
						ave_normal /= dist;

						alpha_ave = alpha_sum / (float)cnt;
						specular_ave = specular_sum / (float)cnt;
						break; // 내부 루프 탈출
					}
					else
					{
						cnt = 0;
					}

				}
			}
			//	mscho	@20240527
			auto thread_idx = yIndex * hLength + xIndex;
			if (cnt > 0) {

				v = ave_pos;
				auto newPts = Eigen::Vector3f(lx, ly, v.z());
				Eigen::Vector3b color;
				auto p45 = Eigen::Vector3f((transform_45 * Eigen::Vector4f(newPts.x(), newPts.y(), newPts.z(), 1.0f)).head(3));
				bool bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, newPts / pointMag, p45 / pointMag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt);
				if (!bImg)
					kernel_return;

				depthMap[thread_idx] = newPts;
				alphaMap[thread_idx] = alpha_ave;
				specularMap[thread_idx] = specular_ave;

				Eigen::Vector3f hsvTex(0, 0, 0);
				Eigen::Vector3f hsvAvg(0, 0, 0);
				ColorUtil::rgb_to_hsv(Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f, hsvTex);
				ColorUtil::rgb_to_hsv(Eigen::Vector3f(ave_color.x(), ave_color.y(), ave_color.z()) / 255.0f, hsvAvg);

				float hueDiff = fabsf(hsvTex.x() - hsvAvg.x());
				float satDiff = fabsf(hsvTex.y() - hsvAvg.y());
				float valueDiff = fabsf(hsvTex.z() - hsvAvg.z());
				hueDiff = hueDiff < 180.0f ? hueDiff : 360 - hueDiff;
				if (hueDiff < 12.0f && satDiff < 0.1f && valueDiff < 0.1f) {

					depthMapCnt[thread_idx] = UINT_MAX;
					_normalMap[thread_idx] = ave_normal;
					colorMap[(thread_idx) * 3 + 0] = ave_color.x();
					colorMap[(thread_idx) * 3 + 1] = ave_color.y();
					colorMap[(thread_idx) * 3 + 2] = ave_color.z();

					if (bDeepLearningEnable) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(newPts / pointMag, imgXY, dev_camRT, dev_cam_tilt)) {
							materialMap[thread_idx] = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
						}
					}
				}
			}
			else {
				depthMapCnt[thread_idx] = 0;
			}
		}
	}
	}


//	mscho	@20240527
__global__ void MarchingCubesKernel::Kernel_SmartInterpolateDepthmap_v3(
	Eigen::Vector3f * depthMap
	, unsigned int* depthMapCnt
	, Eigen::Vector3f * _normalMap
	, unsigned int* colorMap
	, unsigned short* materialMap
	, float* alphaMap
	, float* specularMap
	, size_t hLength
	, size_t vLength
	, float	xUnit
	, float yUnit
	, float pointMag
	, bool bDeepLearningEnable				// true : DeepLearning enable & initial OK
	, unsigned short* deeplearning_inference	//	deep learning 추론의 결과가 저장되어 있다.. 400x480
	, unsigned char* current_img_0
	, unsigned char* current_img_45
	, const Eigen::Matrix4f transform_45
	, const Eigen::Matrix4f dev_camRT
	, const Eigen::Matrix3f dev_cam_tilt
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;
	{
#else
	const int threadCount = hLength * vLength;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif

		size_t index = threadid;

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto map_idx = yIndex * hLength + xIndex;
		auto v = depthMap[map_idx];
		auto z_map = v.z();
		Eigen::Vector3f ave_pos = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3f ave_normal = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3i ave_color = Eigen::Vector3i(0, 0, 0);

		float	min_z = +1000.0;
		float	max_z = -1000.0;

		float	alpha_sum = 0.f;
		float	alpha_ave = 0.f;

		float	specular_sum = 0.f;
		float	specular_ave = 0.f;

		if (!FLT_VALID(z_map))
		{
			//	mscho	@20240527
			auto lx = (float)xIndex * xUnit - ((int)hLength * xUnit * 0.5f) + xUnit * 0.5f;
			auto ly = (float)yIndex * yUnit - ((int)vLength * yUnit * 0.5f) + yUnit * 0.5f;

			int max_gap = devAIInterpolation.maxValue;
			int cnt = 0;
			int gap = 1;
			for (; gap <= max_gap; gap++)
			{
				alpha_sum = 0.f;
				alpha_ave = 0.f;

				specular_sum = 0.f;
				specular_ave = 0.f;
				int sx = xIndex >= gap ? xIndex - gap : 0;
				int sy = yIndex >= gap ? yIndex - gap : 0;
				int ex = xIndex < hLength - gap ? xIndex + gap : hLength - 1;
				int ey = yIndex < vLength - gap ? yIndex + gap : vLength - 1;
				cnt = 0;
				float ave_z = 0.f;
				bool ver_edge = false;

				for (int y = sy; y <= ey; y++)
				{
					if (y != sy && y != ey)	// 세로로 가장자리가 아니라면
						ver_edge = false;
					else
						ver_edge = true;

					for (int x = sx; x <= ex; x++)
					{
						if (!ver_edge && (x != sx && x != ex))	// 가로로 가장자리가 아니라면, 세로도 
							continue;
						auto map_idx = y * hLength + x;
						auto _v = depthMap[map_idx];
						if (depthMapCnt[map_idx] > 0 && UINT_VALID(depthMapCnt[map_idx]))
						{
							ave_z += _v.z();
							if (min_z > _v.z())	min_z = _v.z();
							if (max_z < _v.z())	max_z = _v.z();

							ave_pos += _v;
							ave_normal += _normalMap[map_idx];
							ave_color.x() += (int)colorMap[map_idx * 3 + 0];
							ave_color.y() += (int)colorMap[map_idx * 3 + 1];
							ave_color.z() += (int)colorMap[map_idx * 3 + 2];
							cnt++;

							alpha_sum += alphaMap[map_idx];
							specular_sum += specularMap[map_idx];
						}
					}
				}

				if (cnt)
				{
					if (fabsf(max_z - min_z) < 2.0f)
					{
						z_map = ave_z / (float)cnt;
						ave_pos /= (float)cnt;
						ave_color = ave_color / cnt;
						float dist = norm3df(ave_normal.x(), ave_normal.y(), ave_normal.z());
						ave_normal /= dist;

						alpha_ave = alpha_sum / (float)cnt;
						specular_ave = specular_sum / (float)cnt;
						break; // 내부 루프 탈출
					}
					else
					{
						cnt = 0;
					}

				}
			}

			if (cnt > 0) {
				v = ave_pos;
				auto newPts = Eigen::Vector3f(lx, ly, v.z());
				Eigen::Vector3b color;
				auto p45 = Eigen::Vector3f((transform_45 * Eigen::Vector4f(newPts.x(), newPts.y(), newPts.z(), 1.0f)).head(3));
				bool bImg = ColorUtil::getPixelCoord_pos_Mix_v2(color, newPts / pointMag, p45 / pointMag, current_img_0, current_img_45, dev_camRT, dev_cam_tilt);
				if (!bImg)
					kernel_return;
				depthMap[map_idx] = newPts;
				alphaMap[map_idx] = alpha_ave;
				specularMap[map_idx] = specular_ave;

				Eigen::Vector3f hsvTex(0, 0, 0);
				Eigen::Vector3f hsvAvg(0, 0, 0);
				ColorUtil::rgb_to_hsv(Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f, hsvTex);
				ColorUtil::rgb_to_hsv(Eigen::Vector3f(ave_color.x(), ave_color.y(), ave_color.z()) / 255.0f, hsvAvg);

				float hueDiff = fabsf(hsvTex.x() - hsvAvg.x());
				float satDiff = fabsf(hsvTex.y() - hsvAvg.y());
				float valueDiff = fabsf(hsvTex.z() - hsvAvg.z());
				hueDiff = hueDiff < 180.0f ? hueDiff : 360 - hueDiff;
				if (hueDiff < 12.0f && satDiff < 0.1f && valueDiff < 0.1f) {

					if (bDeepLearningEnable) {
						Eigen::Vector2i imgXY(0, 0);
						if (ColorUtil::getPixelCoord_pos(newPts / pointMag, imgXY, dev_camRT, dev_cam_tilt)) {
							auto materialID = deeplearning_inference[imgXY.y() * 400 + imgXY.x()];
							if (materialID < 16) {
								if (devAIInterpolation.values[materialID] >= gap) {
									depthMapCnt[map_idx] = UINT_MAX;
									materialMap[map_idx] = materialID;
									_normalMap[map_idx] = ave_normal;
									colorMap[map_idx * 3 + 0] = color.x();
									colorMap[map_idx * 3 + 1] = color.y();
									colorMap[map_idx * 3 + 2] = color.z();
								}
							}
						}
					}
					else {
						depthMapCnt[map_idx] = UINT_MAX;
						_normalMap[map_idx] = ave_normal;
						colorMap[map_idx * 3 + 0] = color.x();
						colorMap[map_idx * 3 + 1] = color.y();
						colorMap[map_idx * 3 + 2] = color.z();
					}
				}
			}
			else {
				depthMapCnt[map_idx] = 0;
			}
		}
	}
	}

//	mscho	@20240415
//	Directional TSDF
//  Normal ¹æCaA¸·I SDF¸| °e≫eCI¿ⓒ update CI¿ⓒ AØ´U.
__global__ void MarchingCubesKernel::Kernel_Integrate_v7(
	Eigen::Vector3f * depthMap
	, Eigen::Vector3f * _normalMap
	, unsigned int* colorMap
	, Eigen::Matrix4f transform
	, Eigen::Matrix4f transform_normal
	, HashKey64 * globalHash_info
	, size_t hLength
	, size_t vLength
	, size_t dLength
	, float voxelSize
	, Eigen::AlignedBox3f globalScanAreaAABB
	, voxel_value_t * voxelValues
	, unsigned short* voxelValueCounts
	, Eigen::Vector3f * voxelNormals
	, Eigen::Vector3b * voxelColors
	, int meshColorMode
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > (hLength * vLength) - 1) return;
	{
#else
	const int threadCount = hLength * vLength;
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < threadCount; threadid++) {
#endif

		auto voxel_ = 10.f;
		size_t index = threadid;

		auto yIndex = (size_t)(index / hLength);
		auto xIndex = (size_t)(index % hLength);

		auto v = depthMap[yIndex * hLength + xIndex];
		auto z_map = v.z();
		Eigen::Vector3f ave_pos = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3f ave_normal = Eigen::Vector3f(0.f, 0.f, 0.f);
		Eigen::Vector3i ave_color = Eigen::Vector3i(0, 0, 0);
		Eigen::Vector3i map_color(0, 0, 0);

		Eigen::Vector3f _normal;
		float z_value;
		if (FLT_VALID(z_map))

		{

			auto map_idx = yIndex * hLength + xIndex;

			_normal = _normalMap[map_idx];

			z_value = z_map;

			map_color[0] = colorMap[map_idx * 3 + 0];

			map_color[1] = colorMap[map_idx * 3 + 1];

			map_color[2] = colorMap[map_idx * 3 + 2];

		}

		else if (0)

		{

			int     max_gap = 3;

			for (int gap = 1; gap <= max_gap; gap++)

			{

				int     sx = xIndex >= gap ? xIndex - gap : 0;

				int     sy = yIndex >= gap ? yIndex - gap : 0;

				int     ex = xIndex < hLength - gap ? xIndex + gap : hLength - 1;

				int     ey = yIndex < vLength - gap ? yIndex + gap : vLength - 1;



				int     cnt = 0;

				//Eigen::Vector3f ave_map = Eigen::Vector3f(0.f, 0.f, 0.f);

				float ave_z = 0.f;

				bool    ver_edge = false;

				for (int y = sy; y <= ey; y++)

				{

					if (y != sy && y != ey)

						ver_edge = false;

					else

						ver_edge = true;



					for (int x = sx; x <= ex; x++)

					{

						if (!ver_edge && (x != sx && x != ex))

							continue;



						auto map_idx = y * hLength + x;

						auto _v = depthMap[map_idx];



						if (FLT_VALID(_v.z()))

						{

							ave_z += _v.z();

							ave_pos += v;

							ave_normal += _normalMap[map_idx];

							ave_color.x() += (int)colorMap[map_idx * 3 + 0];

							ave_color.y() += (int)colorMap[map_idx * 3 + 1];

							ave_color.z() += (int)colorMap[map_idx * 3 + 2];

							cnt++;

						}

					}

				}

				if (cnt)

				{

					z_map = ave_z / (float)cnt;

					ave_pos /= (float)cnt;

					ave_color /= (float)cnt;

					float dist = norm3df(ave_normal.x(), ave_normal.y(), ave_normal.z());

					ave_normal /= dist;

					break;

				}



			}

			v = ave_pos;

			_normal = ave_normal;

			z_value = z_map;

			map_color = ave_color;

		}


		if (FLT_VALID(z_value))
		{
			//  CØ´c Æ÷AIÆ®°¡ μe¾i°￥ ±U·I¹u voxel z value 
			auto zIndex = (size_t)floorf(z_value * voxel_ + 2500.f);
			auto z_pos = (float)((int64_t)zIndex - 2500) / voxel_ + voxelSize * 0.5f;
			float value = z_pos - v.z();

			auto gp = transform * Eigen::Vector4f(v.x(), v.y(), v.z(), 1.0f);

			auto normal_local_z = Eigen::Vector3f(0.f, 0.f, 1.f);
			auto normal_global_z = transform_normal * Eigen::Vector4f(normal_local_z.x(), normal_local_z.y(), normal_local_z.z(), 0.f);
			auto plus_normal = Eigen::Vector3f(normal_global_z.x(), normal_global_z.y(), normal_global_z.z());
			plus_normal.normalize();
			//Eigen::Vector3f local_normal = Eigen::Vector3f(0,0,0);
			auto local_normal = _normal;

			//for (size_t y = 0; y < 3; y++)
			//{
			//	for (size_t x = 0; x < 3; x++)
			//	{
			//		if ((x > 0 && x < hLength - 1) && (y > 0 && y < vLength - 1))
			//		{
			//			local_normal += _normalMap[(yIndex - 1 + y) * hLength + (xIndex - 1 + x)];
			//		}
			//	}
			//}
			local_normal.normalize();

			auto global_n = transform_normal * Eigen::Vector4f(local_normal.x(), local_normal.y(), local_normal.z(), 0.f);
			auto global_normal = Eigen::Vector3f(global_n.x(), global_n.y(), global_n.z());
			global_normal.normalize();


			auto normal_dot = local_normal.dot(normal_local_z);
			value *= normal_dot;

			//printf("x:%llu, y:%llu, z:%llu (%f, %f, %f) , normal(%f, %f, %f), g normal(%f, %f, %f)\n", xIndex, yIndex, zIndex, v.x(), v.y(), v.z(), local_normal.x(), local_normal.y(), local_normal.z(), global_n.x(), global_n.y(), global_n.z());



			auto xGlobalIndex = (size_t)floorf(gp.x() / voxelSize - globalScanAreaAABB.min().x() / voxelSize);
			auto yGlobalIndex = (size_t)floorf(gp.y() / voxelSize - globalScanAreaAABB.min().y() / voxelSize);
			auto zGlobalIndex = (size_t)floorf(gp.z() / voxelSize - globalScanAreaAABB.min().z() / voxelSize);

			HashKey voxel_key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
			//uint32_t hashSlot_idx = globalHash_info->get_insert_idx_func64_v4(voxel_key);
			//	mscho	@20240403
			uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(voxel_key);
			if (hashSlot_idx == kEmpty32)
			{
				hashSlot_idx = globalHash_info->get_insert_idx_func64_v4(voxel_key);
			}

			//printf("** pos [%llu, %llu] (%f, %f, %f) - %f\n", xIndex, yIndex, gp.x(), gp.y(), gp.z(), value);


			if (VOXEL_INVALID == voxelValues[hashSlot_idx])
			{
				voxelValues[hashSlot_idx] = D2VV(value);
				voxelValueCounts[hashSlot_idx] = 1;
				voxelNormals[hashSlot_idx] = global_normal;
			}
			else
			{
				if (-30000 < voxelValues[hashSlot_idx] + D2VV(value) && 30000 > voxelValues[hashSlot_idx] + D2VV(value))
				{
					voxelValues[hashSlot_idx] += D2VV(value);
					voxelValueCounts[hashSlot_idx] = voxelValueCounts[hashSlot_idx] + 1;
					voxelNormals[hashSlot_idx] += global_normal;
				}
			}

			auto NormalSum = voxelNormals[hashSlot_idx];

			//if (false == FLT_VALID(voxelValues[hashSlot_idx]))
			//{
			//	voxelValues[hashSlot_idx] = D2VV(value);
			//	//printf("\t< New >Center Voxel (%f, %f, %f): slot = %d value = %f, value(add) = %f \n", gp.x(), gp.y(), gp.z(), hashSlot_idx, value, VV2D(voxelValues[hashSlot_idx]) / (float)voxelValueCounts[hashSlot_idx]);
			//}
			//else
			//{
			//	voxelValues[hashSlot_idx] += D2VV(value);
			//	//printf("\t< Update >Center Voxel (%f, %f, %f): slot = %d value = %f, value(add) = %f \n", gp.x(), gp.y(), gp.z(), hashSlot_idx, value, VV2D(voxelValues[hashSlot_idx]) / (float)voxelValueCounts[hashSlot_idx]);
			//}

			//if (USHORT_VALID(voxelValueCounts[hashSlot_idx]))
			//{
			//	//atomicAdd(&voxelValueCounts[hashSlot_idx], 1);
			//	voxelValueCounts[hashSlot_idx] = voxelValueCounts[hashSlot_idx] + 1;
			//	if (255 < voxelValueCounts[hashSlot_idx])
			//	{
			//		voxelValueCounts[hashSlot_idx] = 255;
			//	}
			//}
			//else
			//{
			//	voxelValueCounts[hashSlot_idx] = 1;
			//}

			auto cr = colorMap[yIndex * hLength * 3 + xIndex * 3 + 0];
			auto cg = colorMap[yIndex * hLength * 3 + xIndex * 3 + 1];
			auto cb = colorMap[yIndex * hLength * 3 + xIndex * 3 + 2];
			voxelColors[hashSlot_idx] = Eigen::Vector3b(cr, cg, cb);

			if (meshColorMode == 1)
			{
				unsigned char r = 0;
				unsigned char g = 0;
				unsigned char b = 0;

				unsigned short count = voxelValueCounts[hashSlot_idx];
				if (count < 64)
				{
					r = 255;
					g = count * 4;
					b = 0;
				}
				else if (64 <= count && count < 128)
				{
					r = 255 - (count - 64) * 4;
					g = 255;
					b = 0;
				}
				else if (128 <= count && count < 192)
				{
					r = 0;
					g = 255;
					b = (count - 128) * 4;
				}
				else if (192 <= count && count < 255)
				{
					r = 0;
					g = 255 - (count - 192) * 4;
					b = 255;
				}
				else
				{
					r = 0;
					g = 0;
					b = 255;
				}
				voxelColors[hashSlot_idx] = Eigen::Vector3b(r, g, b);
			}

			//for (int i = 0; i < 21; i++)
			//{
			//	if (i == 10)	continue;
			for (int i = 0; i < 21; i++)
			{
				if (i == 10)	continue;
				int		offset = i - 10;
				float	distance = (float)offset / 10.0;

				if (fabs(distance + value) > 1.f)	continue;

				Eigen::Vector3f voxel_normal = NormalSum;
				voxel_normal.normalize();
				Eigen::Vector3f tsdf_pos;

				//if (offset > 0)
				//	tsdf_pos = Eigen::Vector3f(gp.x(), gp.y(), gp.z())
				//	+ plus_normal * distance;
				//else
				tsdf_pos = Eigen::Vector3f(gp.x(), gp.y(), gp.z())
					+ global_normal * distance;

				distance += value;

				auto xIndex = (size_t)floorf(tsdf_pos.x() * voxel_ + 2500.f);
				auto yIndex = (size_t)floorf(tsdf_pos.y() * voxel_ + 2500.f);
				auto zIndex = (size_t)floorf(tsdf_pos.z() * voxel_ + 2500.f);
				HashKey tsdf_key(xIndex, yIndex, zIndex);

				uint32_t tsdf_hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(tsdf_key);
				if (tsdf_hashSlot_idx == kEmpty32)
				{
					tsdf_hashSlot_idx = globalHash_info->get_insert_idx_func64_v4(tsdf_key);
				}

				if (VOXEL_INVALID == voxelValues[tsdf_hashSlot_idx])
				{
					voxelValues[tsdf_hashSlot_idx] = D2VV(distance);
					voxelValueCounts[tsdf_hashSlot_idx] = 1;
				}
				else
				{
					if (-30000 < voxelValues[tsdf_hashSlot_idx] + D2VV(value) && 30000 > voxelValues[tsdf_hashSlot_idx] + D2VV(distance))
					{
						voxelValues[tsdf_hashSlot_idx] += D2VV(distance);
						voxelValueCounts[tsdf_hashSlot_idx] = voxelValueCounts[hashSlot_idx] + 1;
					}
				}
				voxelColors[tsdf_hashSlot_idx] = Eigen::Vector3b(cr, cg, cb);
			}
		}
	}
	}

void MarchingCubes::setGlobalAABB(const Eigen::Vector3f & aabb_min,
	const Eigen::Vector3f & aabb_max, float VoxelSize) {

	m_globalScanAreaAABB.extend(aabb_min);
	m_globalScanAreaAABB.extend(aabb_max);
	m_voxelSize = VoxelSize;

	m_globalVoxelCountX = (size_t)((m_globalScanAreaAABB.max() - m_globalScanAreaAABB.min()).x() / m_voxelSize);
	m_globalVoxelCountY = (size_t)((m_globalScanAreaAABB.max() - m_globalScanAreaAABB.min()).y() / m_voxelSize);
	m_globalVoxelCountZ = (size_t)((m_globalScanAreaAABB.max() - m_globalScanAreaAABB.min()).z() / m_voxelSize);
	m_globalVoxelCount = m_globalVoxelCountX * m_globalVoxelCountY * m_globalVoxelCountZ;
}

void MarchingCubes::setLocalAABB(const Eigen::Vector3f & aabb_min,
	const Eigen::Vector3f & aabb_max) {

	float xmin = (floorf(aabb_min.x() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;
	float ymin = (floorf(aabb_min.y() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;
	float zmin = (floorf(aabb_min.z() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;

	float xmax = (floorf(aabb_max.x() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;
	float ymax = (floorf(aabb_max.y() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;
	float zmax = (floorf(aabb_max.z() / m_voxelSize + 2500.f) - 2500.f) * m_voxelSize;

	m_localScanAreaAABB = Eigen::AlignedBox3f(Eigen::Vector3f(xmin, ymin, zmin), Eigen::Vector3f(xmax, ymax, zmax));

	m_localVoxelCountX = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).x() / m_voxelSize);
	m_localVoxelCountY = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).y() / m_voxelSize);
	m_localVoxelCountZ = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).z() / m_voxelSize);
	m_localVoxelCount = m_localVoxelCountX * m_localVoxelCountY * m_localVoxelCountZ;

}

void MarchingCubes::setLocalAABB_expand(int size) {

	if (size == -1)
		size = 1;

	Eigen::Vector3f newMin = m_localScanAreaAABB.min();
	Eigen::Vector3f newMax = m_localScanAreaAABB.max();

	newMin.x() -= (m_voxelSize * (float)size);
	newMin.y() -= (m_voxelSize * (float)size);
	newMin.z() -= (m_voxelSize * (float)size);

	newMax.x() += (m_voxelSize * (float)size);
	newMax.y() += (m_voxelSize * (float)size);
	newMax.z() += (m_voxelSize * (float)size);

	// AABB 업데이트
	m_localScanAreaAABB = Eigen::AlignedBox3f(newMin, newMax);

	m_localVoxelCountX = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).x() / m_voxelSize);
	m_localVoxelCountY = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).y() / m_voxelSize);
	m_localVoxelCountZ = (size_t)((m_localScanAreaAABB.max() - m_localScanAreaAABB.min()).z() / m_voxelSize);
	m_localVoxelCount = m_localVoxelCountX * m_localVoxelCountY * m_localVoxelCountZ;
};


#ifndef BUILD_FOR_CPU
__device__ bool getPixelCoord_pos(const Eigen::Matrix4f icp_transform_, const Eigen::Vector3f vertex, Eigen::Vector<int, 2>&cam_pos, const Eigen::Matrix4f dev_camRT, const Eigen::Matrix3f dev_cam_tilt)
{
	//  global 좌표계의 position을 읽어오고             
	auto p_global = Eigen::Vector4f(vertex.x(), vertex.y(), vertex.z(), 1.0f);
	//  local 좌표계로 이동하도록 inverse transform matrix를 곱해주고
	auto tr_local = icp_transform_.inverse() * p_global;
	//  변환된 자표를 이용해서, Local position을 만든다음
	auto local_pts = Eigen::Vector3f(tr_local.x(), tr_local.y(), tr_local.z());
	auto inPos = Eigen::Vector4f(local_pts.x(), local_pts.y(), local_pts.z(), 1.0f);
	auto camPos = dev_camRT * inPos;

	float rx = camPos[0] / camPos[2];
	float ry = camPos[1] / camPos[2];

	Eigen::Vector3f CamPos3f(rx, ry, 1);

	const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
	float tx = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
	float ty = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();

	float u = (tx * dev_cam_cfx + dev_cam_ccx) - 0.5;
	float v = (ty * dev_cam_cfy + dev_cam_ccy) - 0.5;

	cam_pos.x() = (int)((tx * dev_cam_cfx + dev_cam_ccx) - 0.5);
	cam_pos.y() = (int)((ty * dev_cam_cfy + dev_cam_ccy) - 0.5);

	if (cam_pos.x() < 0 || cam_pos.x() > dev_cam_w - 1 || cam_pos.y() < 0 || cam_pos.y() > dev_cam_h - 1)
		return false;
	else
		return true;
}

// 0도 , 45도 모두 읽어서, 평균값으로 texture color를 만들어 내는 방법
__device__ Eigen::Vector3b getPixelCoord_pos_Mix(
	const Eigen::Matrix4f icp_transform_0_,
	const Eigen::Matrix4f icp_transform_45_,
	const Eigen::Vector3f vertex,
	const unsigned char* img0_,
	const unsigned char* img45_,
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt
)
{
	Eigen::Vector<int, 2> img_pixel_pos_0;
	Eigen::Vector<int, 2> img_pixel_pos_45;
	Eigen::Vector3b cam_pixel;

	// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
	bool bimg_area_0 = getPixelCoord_pos(icp_transform_0_, vertex, img_pixel_pos_0, dev_camRT, dev_cam_tilt);
	bool bimg_area_45 = getPixelCoord_pos(icp_transform_45_, vertex, img_pixel_pos_45, dev_camRT, dev_cam_tilt);
	//qDebug("img_pixel_pos_0 = %d, %d, %s / img_pixel_pos_45 = %d, %d , %s, ",
	//	img_pixel_pos_0.x(), img_pixel_pos_0.y(), bimg_area_0, img_pixel_pos_45.x(), img_pixel_pos_45.y(), bimg_area_45);

	if (bimg_area_0 && bimg_area_45)
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		cam_pixel.x() = (unsigned char)(((int)img0_[img_index_offset_0] + (int)img45_[img_index_offset_45]) / 2);
		cam_pixel.y() = (unsigned char)(((int)img0_[img_index_offset_0 + 1] + (int)img45_[img_index_offset_45 + 1]) / 2);
		cam_pixel.z() = (unsigned char)(((int)img0_[img_index_offset_0 + 2] + (int)img45_[img_index_offset_45 + 2]) / 2);
	}
	else
	{
		size_t  img_index_offset_0 = (img_pixel_pos_0.y() * dev_cam_w + img_pixel_pos_0.x()) * 4;
		size_t  img_index_offset_45 = (img_pixel_pos_45.y() * dev_cam_w + img_pixel_pos_45.x()) * 4;

		if (bimg_area_0)
		{
			cam_pixel.x() = img0_[img_index_offset_0];
			cam_pixel.y() = img0_[img_index_offset_0 + 1];
			cam_pixel.z() = img0_[img_index_offset_0 + 2];
		}
		else 	if (bimg_area_45)
		{
			cam_pixel.x() = img45_[img_index_offset_45];
			cam_pixel.y() = img45_[img_index_offset_45 + 1];
			cam_pixel.z() = img45_[img_index_offset_45 + 2];
		}
		else
		{
			// 45도에서도 벗어난 다면
// Error 처리
			cam_pixel.x() = 0;
			cam_pixel.y() = 0;
			cam_pixel.z() = 0;
		}
	}

	return cam_pixel;
}

// 여기에서 Vertex정보는 global 좌표계이어야 한다..
// 만약에 Local 좌표계라고 한다면..
// RT 를 Identity 로 보내면 된다.
__device__ Eigen::Vector3b getPixelCoord_pos_full(
	const Eigen::Matrix4f icp_transform_0_,
	const Eigen::Matrix4f icp_transform_45_,
	const Eigen::Vector3f vertex,
	const unsigned char* img0_,
	const unsigned char* img45_,
	Eigen::Vector<int, 2>&cam_pos,
	const Eigen::Matrix4f dev_camRT,
	const Eigen::Matrix3f dev_cam_tilt
)
{

	Eigen::Vector<int, 2> img_pixel_pos;

	// 0도 Image를 벗어나지 않는 좌표라면, 0도에서 이미지를 가져가도록 한다
	if (getPixelCoord_pos(icp_transform_0_, vertex, img_pixel_pos, dev_camRT, dev_cam_tilt))
	{
		Eigen::Vector3b cam_pixel;
		size_t  img_index_offset = (img_pixel_pos.x() * dev_cam_w + img_pixel_pos.x()) * 4;

		cam_pixel.x() = img0_[img_index_offset];
		cam_pixel.y() = img0_[img_index_offset + 1];
		cam_pixel.z() = img0_[img_index_offset + 2];

		return cam_pixel;
	}
	else
	{
		// 0도 이미지의 좌표를 벗어난 다면, 45도 만으로 만들 수 있는 영역이므로
		// 45도 RT와 Image를 이용해서 다시 한번 수행하도록 한다.
		Eigen::Vector3b cam_pixel;
		if (getPixelCoord_pos(icp_transform_45_, vertex, img_pixel_pos, dev_camRT, dev_cam_tilt))
		{
			size_t  img_index_offset = (img_pixel_pos.x() * dev_cam_w + img_pixel_pos.x()) * 4;
			cam_pixel.x() = img45_[img_index_offset];
			cam_pixel.y() = img45_[img_index_offset + 1];
			cam_pixel.z() = img45_[img_index_offset + 2];
		}
		else
		{
			// 45도에서도 벗어난 다면
			// Error 처리
			cam_pixel.x() = 0;
			cam_pixel.y() = 0;
			cam_pixel.z() = 0;
		}
		return cam_pixel;
	}
}
#endif

#pragma region OLD
/*
__device__
	Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
	//Eigen::Vector3f VertexInterp_old(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
{
	float mu;
	Eigen::Vector3f p;

	if (fabsf(isolevel - valp1) < 0.00001f)
		return(p1);
	if (fabsf(isolevel - valp2) < 0.00001f)
		return(p2);
	if (fabsf(valp1 - valp2) < 0.00001f)
		return(p1);
	mu = (isolevel - valp1) / (valp2 - valp1);
	p.x() = p1.x() + mu * (p2.x() - p1.x());
	p.y() = p1.y() + mu * (p2.y() - p1.y());
	p.z() = p1.z() + mu * (p2.z() - p1.z());

	return p;
}

__device__
	Eigen::Vector3f VertexInterp_old(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
	//Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
{
#pragma region Using Truncation
			float mu;
			Eigen::Vector3f p;

			if (false == FLT_VALID(valp1))
				return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

			if (false == FLT_VALID(valp2))
				return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#define unit_size_0 100.0f
#define unit_size_offset 2500.0f


			if (fabsf(isolevel - valp1) < 0.00001f)
			{
				p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
				return p;
			}

			if (fabsf(isolevel - valp2) < 0.00001f)
			{
				p.x() = floorf(p2.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p2.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p2.z() * unit_size_0) / unit_size_0;
				return p;
			}

			if (fabsf(valp1 - valp2) < 0.00001f)
			{
				p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
				p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
				p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
				return p;
			}

			mu = (isolevel - valp1) / (valp2 - valp1);
			p.x() = p1.x() + mu * (p2.x() - p1.x());
			p.x() = floorf(p.x() * unit_size_0 + 0.5) / unit_size_0;
			p.y() = p1.y() + mu * (p2.y() - p1.y());
			p.y() = floorf(p.y() * unit_size_0 + 0.5) / unit_size_0;
			p.z() = p1.z() + mu * (p2.z() - p1.z());
			p.z() = floorf(p.z() * unit_size_0 + 0.5) / unit_size_0;

			//p.x() = ((int)(p.x() * unit_size_0 + 0.5)) / unit_size_0;
			//p.y() = ((int)(p.y() * unit_size_0 + 0.5)) / unit_size_0;
			//p.z() = ((int)(p.z() * unit_size_0 + 0.5)) / unit_size_0;

			return p;
#pragma endregion

#pragma region Original
			//float mu = 0.0f;
			//Eigen::Vector3f p = p1;

			//if (fabsf(isolevel - valp1) < 0.00001f)
			//	return(p1);
			//if (fabsf(isolevel - valp2) < 0.00001f)
			//	return(p2);
			//if (fabsf(valp1 - valp2) < 0.00001f)
			//	return(p1);
			//mu = (isolevel - valp1) / (valp2 - valp1);
			//p.x() = p1.x() + mu * (p2.x() - p1.x());
			//p.y() = p1.y() + mu * (p2.y() - p1.y());
			//p.z() = p1.z() + mu * (p2.z() - p1.z());

			//return p;
#pragma endregion
		}
		*/
#pragma endregion

#ifndef BUILD_FOR_CPU
__device__
Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f & p1, const Eigen::Vector3f & p2, float valp1, float valp2)
{
#pragma region Using Truncation
	float mu;
	Eigen::Vector3f p;

	if (false == FLT_VALID(valp1))
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

	if (false == FLT_VALID(valp2))
		return Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#define unit_size_0 100.0f
#define unit_size_offset 25000.0f


	if (fabsf(isolevel - valp1) < 0.001f)
	{
		p.x() = (floorf(p1.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.y() = (floorf(p1.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.z() = (floorf(p1.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		return p;
	}

	if (fabsf(isolevel - valp2) < 0.001f)
	{
		p.x() = (floorf(p2.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.y() = (floorf(p2.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.z() = (floorf(p2.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		return p;
	}

	if (fabsf(valp1 - valp2) < 0.001f)
	{
		p.x() = (floorf(p1.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.y() = (floorf(p1.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		p.z() = (floorf(p1.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
		return p;
	}

	mu = (isolevel - valp1) / (valp2 - valp1);
	p.x() = p1.x() + mu * (p2.x() - p1.x());
	p.x() = (floorf(p.x() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
	p.y() = p1.y() + mu * (p2.y() - p1.y());
	p.y() = (floorf(p.y() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;
	p.z() = p1.z() + mu * (p2.z() - p1.z());
	p.z() = (floorf(p.z() * unit_size_0 + unit_size_offset) - unit_size_offset) / unit_size_0;

	//p.x() = ((int)(p.x() * unit_size_0 + 0.5)) / unit_size_0;
	//p.y() = ((int)(p.y() * unit_size_0 + 0.5)) / unit_size_0;
	//p.z() = ((int)(p.z() * unit_size_0 + 0.5)) / unit_size_0;

	return p;
#pragma endregion

#pragma region Original
	//float mu = 0.0f;
	//Eigen::Vector3f p = p1;

	//if (fabsf(isolevel - valp1) < 0.00001f)
	// return(p1);
	//if (fabsf(isolevel - valp2) < 0.00001f)
	// return(p2);
	//if (fabsf(valp1 - valp2) < 0.00001f)
	// return(p1);
	//mu = (isolevel - valp1) / (valp2 - valp1);
	//p.x() = p1.x() + mu * (p2.x() - p1.x());
	//p.y() = p1.y() + mu * (p2.y() - p1.y());
	//p.z() = p1.z() + mu * (p2.z() - p1.z());

	//return p;
#pragma endregion
}

__device__ Eigen::Vector3f calculateNormalfromVertices(const Eigen::Vector3f * vertex) {
	auto ab = vertex[1] - vertex[0];
	auto ac = vertex[2] - vertex[0];
	auto vertex_normal = ab.cross(ac);
	vertex_normal.normalize();
	return vertex_normal;
}
//  Face의 Normal을 계산하고, Camera 방향  Vector와  dot연산을 하면
//  해당 face가 camera 방향인지, 반대 방향인지를 알수 가 있다
//  반대방향이라면, Normal 부호를 바꿔주고
//  Face의  Vertex순서를 1, 2를 바꾸어 주면.. 된다

 // sc_camera_pos
__device__ Eigen::Vector3f calculateNormalfromVertices_v2(const Eigen::Vector3f cam_pos, Eigen::Vector3f * vertex, int sc_usetip)
{
	Eigen::Vector3f cam_dir;
	auto ab = vertex[1] - vertex[0];
	auto ac = vertex[2] - vertex[0];
	//auto face_normal = ab.cross(ac);
	Eigen::Vector3f face_normal = CROSS(ab, ac);
	face_normal.normalize();
	//NORMALIZE(face_normal);

	if (sc_usetip)
	{
		cam_dir.x() = cam_pos.x() - face_normal.x();
		cam_dir.y() = cam_pos.y() - face_normal.y();
		cam_dir.z() = cam_pos.z() - face_normal.z();
	}
	else
	{
		cam_dir.x() = cam_pos.x() - face_normal.x();
		cam_dir.y() = -cam_pos.y() - face_normal.y();
		cam_dir.z() = cam_pos.z() - face_normal.z();
	}
	cam_dir.normalize();
	//NORMALIZE(cam_dir);
	//cam_dir = -cam_dir;

	auto face_dir = face_normal.dot(cam_dir);

	/*if (face_dir < 0.f)
	{
		auto temp_vertex = vertex[1];
		vertex[1] = vertex[2];
		vertex[2] = temp_vertex;

		face_normal = -face_normal;
	}*/

	return (Eigen::Vector3f)face_normal;
}
#endif

#ifndef BUILD_FOR_CPU
#ifdef USE_MESH_BASE
//	mscho	@20240221
int MarchingCubes::Extract_v4(
	unsigned char* current_img_0,
	unsigned char* current_img_45,
	const Eigen::Matrix4f & transform_0,
	const Eigen::Matrix4f & transform_45,
	thrust::device_vector<voxel_value_t>&voxelValues,
	thrust::device_vector<unsigned short>&voxelValueCounts,
	thrust::device_vector<Eigen::Vector3b>&voxelColors,
	Eigen::Vector3f cam_pos,
	cached_allocator * alloc_, CUstream_st * st
	, thrust::device_vector< Eigen::Vector<uint32_t, 3>>&voxel_tri_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>&vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>&vtx_color_repos
	, thrust::device_vector<uint32_t>&vtx_dupCnt_repos
)
{
#pragma region triangle, vtx 초기화 테스트
	//{
	//	pHashManager->delete_lookuptable64(
	//		hTable_global_vtx,
	//		(uint32_t)pHashManager->hashGlobalVertex_info64_host->HashTableCapacity,
	//		st);
	//	thrust::fill(
	//		thrust::cuda::par_nosync(*alloc_).on(st),
	//		voxel_tri_repos.begin(), voxel_tri_repos.end(), Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX));
	//}  
#pragma endregion

	//Kernel_Extract<<<BLOCKS_PER_GRID_THREAD_N(m_localVoxelCount, THRED64_PER_BLOCK), THRED64_PER_BLOCK, 0, st>>>(
	//	m_localScanAreaAABB, m_localVoxelCountX, m_localVoxelCountY, m_localVoxelCountZ, m_localVoxelCount, m_voxelSize,
	//	m_globalScanAreaAABB, m_globalVoxelCountX, m_globalVoxelCountY, m_globalVoxelCountZ, m_globalVoxelCount,
	//	thrust::raw_pointer_cast(voxelValues.data()), thrust::raw_pointer_cast(voxelColors.data()), globalHash_info, hInfo_global_vtx, hTable_global_vtx,
	//	cam_pos, _dev_camRT, _dev_cam_tilt, _dev_cam_tilt_inv,
	//	thrust::raw_pointer_cast(voxel_tri_repos.data()), thrust::raw_pointer_cast(vtx_pos_repos.data()), thrust::raw_pointer_cast(vtx_nm_repos.data()),
	//	thrust::raw_pointer_cast(vtx_color_repos.data()), thrust::raw_pointer_cast(vtx_dupCnt_repos.data()), 0.0f);


	{
		float isoValue = 0.0f;
		//	mscho	@20240221
		BuildGridFunctor_v7 buildGridFunctor;
		buildGridFunctor.hashTable_voxel = globalHash;
		buildGridFunctor.hashinfo_voxel = globalHash_info;
		buildGridFunctor.hashTable_voxel_value = globalHash_value;

		buildGridFunctor.hashTable_triangle = hTable_global_tri;
		buildGridFunctor.hashinfo_triangle = hInfo_global_tri;
		buildGridFunctor.hashTable_triangle_value = hTable_global_tri_value;

		buildGridFunctor.hashTable_vtx = hTable_global_vtx;
		buildGridFunctor.hashinfo_vtx = hInfo_global_vtx;
		buildGridFunctor.hashTable_vtx_value = hTable_global_vtx_value;

		buildGridFunctor.voxelColors = thrust::raw_pointer_cast(voxelColors.data());
		buildGridFunctor.values = thrust::raw_pointer_cast(voxelValues.data());
		buildGridFunctor.voxelValueCounts = thrust::raw_pointer_cast(voxelValueCounts.data());

		buildGridFunctor.localMinX = m_localScanAreaAABB.min().x();
		buildGridFunctor.localMinY = m_localScanAreaAABB.min().y();
		buildGridFunctor.localMinZ = m_localScanAreaAABB.min().z();
		buildGridFunctor.globalMinX = m_globalScanAreaAABB.min().x();
		buildGridFunctor.globalMinY = m_globalScanAreaAABB.min().y();
		buildGridFunctor.globalMinZ = m_globalScanAreaAABB.min().z();
		buildGridFunctor.localVoxelCountX = m_localVoxelCountX;
		buildGridFunctor.localVoxelCountY = m_localVoxelCountY;
		buildGridFunctor.localVoxelCountZ = m_localVoxelCountZ;
		buildGridFunctor.globalVoxelCountX = m_globalVoxelCountX;
		buildGridFunctor.globalVoxelCountY = m_globalVoxelCountY;
		buildGridFunctor.globalVoxelCountZ = m_globalVoxelCountZ;

		buildGridFunctor.voxelSize = m_voxelSize;
		buildGridFunctor.isoValue = isoValue;

		buildGridFunctor.transform_0 = transform_0;
		buildGridFunctor.transform_45 = transform_45;

		buildGridFunctor.img_0 = current_img_0;
		buildGridFunctor.img_45 = current_img_45;

		buildGridFunctor._cam_pos = cam_pos;
		buildGridFunctor._camRT = _dev_camRT;
		buildGridFunctor._cam_tilt = _dev_cam_tilt;
		buildGridFunctor._cam_tilt_inv = _dev_cam_tilt_inv;

		buildGridFunctor.repos_triangle = thrust::raw_pointer_cast(voxel_tri_repos.data());
		buildGridFunctor.repos_vtx_pos = thrust::raw_pointer_cast(vtx_pos_repos.data());
		buildGridFunctor.repos_vtx_nm = thrust::raw_pointer_cast(vtx_nm_repos.data());
		buildGridFunctor.repos_vtx_color = thrust::raw_pointer_cast(vtx_color_repos.data());
		buildGridFunctor.repos_vtx_dupCnt = thrust::raw_pointer_cast(vtx_dupCnt_repos.data());

		//qDebug("Start");
		nvtxRangePushA("@Arron/BuildGridCells");
		thrust::for_each(
			thrust::cuda::par_nosync(*alloc_).on(st),
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(m_localVoxelCount),
			buildGridFunctor);
		checkCudaSync(st);
		nvtxRangePop();
		//qDebug("End");

	}
	return true;
}
#endif
#endif

#ifndef BUILD_FOR_CPU
/* no longer used
//	 Extract Average 통합화
// 240320 shshin
int MarchingCubes::Extract_v7(
	unsigned char* current_img_0,
	unsigned char* current_img_45,
	const Eigen::Matrix4f& transform_0,
	const Eigen::Matrix4f& transform_45,
	thrust::device_vector<voxel_value_t>& voxelValues,
	thrust::device_vector<unsigned short>& voxelValueCounts,
	thrust::device_vector<Eigen::Vector3b>& voxelColors,
	Eigen::Vector3f cam_pos,
	cached_allocator* alloc_, CUstream_st* st
	, thrust::device_vector< Eigen::Vector<uint32_t, 3>>& voxel_tri_repos
	, thrust::device_vector<Eigen::Vector3f>& vtx_pos_repos
	, thrust::device_vector<Eigen::Vector3f>& vtx_nm_repos
	, thrust::device_vector<Eigen::Vector3b>& vtx_color_repos
	, thrust::device_vector<uint32_t>& vtx_dupCnt_repos
)
{

	{
		float isoValue = 0.0f;
		//	mscho	@20240221
		BuildGridFunctor_v7 buildGridFunctor;
		buildGridFunctor.hashTable_voxel = globalHash;
		buildGridFunctor.hashinfo_voxel = globalHash_info;
		buildGridFunctor.hashTable_voxel_value = globalHash_value;

		buildGridFunctor.hashTable_triangle = hTable_global_tri;
		buildGridFunctor.hashinfo_triangle = hInfo_global_tri;
		buildGridFunctor.hashTable_triangle_value = hTable_global_tri_value;

		buildGridFunctor.hashTable_vtx = hTable_global_vtx;
		buildGridFunctor.hashinfo_vtx = hInfo_global_vtx;
		buildGridFunctor.hashTable_vtx_value = hTable_global_vtx_value;

		buildGridFunctor.voxelColors = thrust::raw_pointer_cast(voxelColors.data());
		buildGridFunctor.values = thrust::raw_pointer_cast(voxelValues.data());
		buildGridFunctor.voxelValueCounts = thrust::raw_pointer_cast(voxelValueCounts.data());

		buildGridFunctor.localMinX = m_localScanAreaAABB.min().x();
		buildGridFunctor.localMinY = m_localScanAreaAABB.min().y();
		buildGridFunctor.localMinZ = m_localScanAreaAABB.min().z();
		buildGridFunctor.globalMinX = m_globalScanAreaAABB.min().x();
		buildGridFunctor.globalMinY = m_globalScanAreaAABB.min().y();
		buildGridFunctor.globalMinZ = m_globalScanAreaAABB.min().z();
		buildGridFunctor.localVoxelCountX = m_localVoxelCountX;
		buildGridFunctor.localVoxelCountY = m_localVoxelCountY;
		buildGridFunctor.localVoxelCountZ = m_localVoxelCountZ;
		buildGridFunctor.globalVoxelCountX = m_globalVoxelCountX;
		buildGridFunctor.globalVoxelCountY = m_globalVoxelCountY;
		buildGridFunctor.globalVoxelCountZ = m_globalVoxelCountZ;

		buildGridFunctor.voxelSize = m_voxelSize;
		buildGridFunctor.isoValue = isoValue;

		buildGridFunctor.transform_0 = transform_0;
		buildGridFunctor.transform_45 = transform_45;

		buildGridFunctor.img_0 = current_img_0;
		buildGridFunctor.img_45 = current_img_45;

		buildGridFunctor._cam_pos = cam_pos;
		buildGridFunctor._camRT = _dev_camRT;
		buildGridFunctor._cam_tilt = _dev_cam_tilt;
		buildGridFunctor._cam_tilt_inv = _dev_cam_tilt_inv;

		buildGridFunctor.repos_triangle = thrust::raw_pointer_cast(voxel_tri_repos.data());
		buildGridFunctor.repos_vtx_pos = thrust::raw_pointer_cast(vtx_pos_repos.data());
		buildGridFunctor.repos_vtx_nm = thrust::raw_pointer_cast(vtx_nm_repos.data());
		buildGridFunctor.repos_vtx_color = thrust::raw_pointer_cast(vtx_color_repos.data());
		buildGridFunctor.repos_vtx_dupCnt = thrust::raw_pointer_cast(vtx_dupCnt_repos.data());

		//qDebug("Start");
		nvtxRangePushA("@Arron/BuildGridCells");
		thrust::for_each(
			thrust::cuda::par_nosync(*alloc_).on(st),
			thrust::make_counting_iterator<size_t>(0),
			thrust::make_counting_iterator<size_t>(m_localVoxelCount),
			buildGridFunctor);
		checkCudaSync(st);
		nvtxRangePop();
		//qDebug("End");

	}
	return true;
}
*/
#endif

#ifndef BUILD_FOR_CPU
/*
__global__
	void Kernel_GetCorrespondPointAndNormal(Eigen::Vector3f* positions, size_t positionsSize, float range,
		Eigen::Vector3f* correspondPoints, Eigen::Vector3f* correspondNormals,
		float minValidArea, float voxelSize, float3 globalScanAreaMin, float3 globalScanAreaMax,
		HashKey64* voxelHashInfo, HashEntry* voxelHash,
		Eigen::Vector<uint32_t, 3>* triangles, Eigen::Vector3f* vertices)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > positionsSize - 1) return;

	int count = 0;

	auto& position = positions[threadid];

	auto& correspondPoint = correspondPoints[threadid];

	correspondPoint.x() = FLT_MAX;
	correspondPoint.y() = FLT_MAX;
	correspondPoint.z() = FLT_MAX;

	auto& correspondNormal = correspondNormals[threadid];

	correspondNormal.x() = FLT_MAX;
	correspondNormal.y() = FLT_MAX;
	correspondNormal.z() = FLT_MAX;

	float3 p = { position.x(), position.y(), position.z() };

	auto aabbMin = float3{ __fsub_rn(p.x, range), __fsub_rn(p.y, range), __fsub_rn(p.z, range) };
	auto aabbMax = float3{ __fadd_rn(p.x, range), __fadd_rn(p.y, range), __fadd_rn(p.z, range) };

	//printf("%f, %f, %f     %f, %f, %f\n", aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z);

	auto minKey = GetHashKey(aabbMin, globalScanAreaMin, voxelSize);
	auto maxKey = GetHashKey(aabbMax, globalScanAreaMin, voxelSize);
	auto centerKey = GetHashKey(p, globalScanAreaMin, voxelSize);



	auto maxStepX = max((maxKey.x - centerKey.x), (centerKey.x - minKey.x));
	auto maxStepY = max((maxKey.y - centerKey.y), (centerKey.y - minKey.y));
	auto maxStepZ = max((maxKey.z - centerKey.z), (centerKey.z - minKey.z));
	auto maxStep = max(max(maxStepX, maxStepY), maxStepZ);

	//printf("minKey : %llu, %llu, %llu     maxKey : %llu, %llu, %llu\n", minKey.x, minKey.y, minKey.z, maxKey.x, maxKey.y, maxKey.z);
	//printf("maxStep: %llu\n", maxStep);

	auto voxelCountX = __float2ull_rn(__fdiv_rn(__fsub_rn(globalScanAreaMax.x, globalScanAreaMin.x), voxelSize));
	auto voxelCountY = __float2ull_rn(__fdiv_rn(__fsub_rn(globalScanAreaMax.y, globalScanAreaMin.y), voxelSize));
	auto voxelCountZ = __float2ull_rn(__fdiv_rn(__fsub_rn(globalScanAreaMax.z, globalScanAreaMin.z), voxelSize));
	//auto voxelCount = globalVoxelCountX * globalVoxelCountY * globalVoxelCountZ;

	// 주석처리 사유: warning #186-D: pointless comparison of unsigned integer with zero
	// if (minKey.x < 0) minKey.x = 0;
	// if (minKey.y < 0) minKey.y = 0;
	// if (minKey.z < 0) minKey.z = 0;

	if (maxKey.x > voxelCountX - 1) maxKey.x = voxelCountX - 1;
	if (maxKey.y > voxelCountY - 1) maxKey.y = voxelCountY - 1;
	if (maxKey.z > voxelCountZ - 1) maxKey.z = voxelCountZ - 1;

	//return; // 35.497 us

	for (size_t step = 0; step < maxStep; step++)
	{
		for (auto z = centerKey.z - step; z <= centerKey.z + step; z++)
		{
			for (auto y = centerKey.y - step; y <= centerKey.y + step; y++)
			{
				for (auto x = centerKey.x - step; x <= centerKey.x + step; x++)
				{
					if (x < minKey.x) x = minKey.x;
					if (y < minKey.y) y = minKey.y;
					if (z < minKey.z) z = minKey.z;

					if (x > maxKey.x) x = maxKey.x;
					if (y > maxKey.y) y = maxKey.y;
					if (z > maxKey.z) z = maxKey.z;

					count++;

					//printf("check\n");

					if ((x == centerKey.x - step || x == centerKey.x + step) ||
						(y == centerKey.y - step || y == centerKey.y + step) ||
						(z == centerKey.z - step || z == centerKey.z + step))
					{
						// return; // 36.861 us

						HashKey key(x, y, z);
						auto triangleIndex = get_hashtable_lookup_idx_func64(voxelHashInfo, voxelHash, key);

						//return; // 51.548 us
						//Eigen::Vector<uint32_t, 3> triangle0;
						//Eigen::Vector<uint32_t, 3> triangle1;
						//Eigen::Vector<uint32_t, 3> triangle2;
						//Eigen::Vector<uint32_t, 3> triangle3;

						if (triangleIndex == kEmpty)
						{
							//printf("continue\n");
							continue;
						}
						else
						{
							auto triangle0 = triangles[triangleIndex * 4];
							auto triangle1 = triangles[triangleIndex * 4 + 1];
							auto triangle2 = triangles[triangleIndex * 4 + 2];
							auto triangle3 = triangles[triangleIndex * 4 + 3];

							if ((false == UINT_VALID(triangle0.x())) &&
								(false == UINT_VALID(triangle1.x())) &&
								(false == UINT_VALID(triangle2.x())) &&
								(false == UINT_VALID(triangle3.x())))
							{
								//printf("[%6d] continue\n", threadid);
								continue;
							}
						}

						//printf("threadid: %d\n", threadid);

						// return; // 2.934 ms

						float3 normal{ 0.0f, 0.0f, 0.0f };
						float count = 0.0f;

						float currentDistance = FLT_MAX;
						float3 nearestPoint = { FLT_MAX, FLT_MAX, FLT_MAX };

						//printf("triangleIndex : %d\n", triangleIndex);

						auto triangle0 = triangles[triangleIndex * 4];
						auto t0 = uint3{ triangle0.x(), triangle0.y(), triangle0.z() };
						if (VECTOR3U_VALID(t0))
						{
							//printf("t0 - index\n");

							auto v0 = vertices[t0.x];
							auto v1 = vertices[t0.y];
							auto v2 = vertices[t0.z];

							float3 p0{ v0.x(), v0.y(), v0.z() };
							float3 p1{ v1.x(), v1.y(), v1.z() };
							float3 p2{ v2.x(), v2.y(), v2.z() };

							if (VECTOR3F_VALID_(v0) && VECTOR3F_VALID_(v1) && VECTOR3F_VALID_(v2))
							{
								//printf("t0 - vertex\n");

								auto centroid = GetCentroid(p0, p1, p2);
								float distance = GetDistance(p, centroid);
								if (range >= distance)
								{
									float3 n;
									float area;
									GetTriangleNormalAndArea(p0, p1, p2, n, area);

									if (area >= minValidArea)
									{
										if (distance < currentDistance)
										{
											currentDistance = distance;
											nearestPoint = centroid;
										}

										normal.x = __fadd_rn(normal.x, n.x);
										normal.y = __fadd_rn(normal.y, n.y);
										normal.z = __fadd_rn(normal.z, n.z);

										count = __fadd_rn(count, 1.0f);
									}
									//else
									//{
									//	printf("Area : %f\n", area);
									//}
								}
							}
						}

						auto triangle1 = triangles[triangleIndex * 4 + 1];
						auto t1 = uint3{ triangle1.x(), triangle1.y(), triangle1.z() };
						if (VECTOR3U_VALID(t1))
						{
							//printf("t1 - index\n");

							auto v0 = vertices[t1.x];
							auto v1 = vertices[t1.y];
							auto v2 = vertices[t1.z];

							float3 p0{ v0.x(), v0.y(), v0.z() };
							float3 p1{ v1.x(), v1.y(), v1.z() };
							float3 p2{ v2.x(), v2.y(), v2.z() };

							if (VECTOR3F_VALID_(v0) && VECTOR3F_VALID_(v1) && VECTOR3F_VALID_(v2))
							{
								//printf("t1 - vertex\n");

								auto centroid = GetCentroid(p0, p1, p2);
								float distance = GetDistance(p, centroid);
								if (range >= distance)
								{
									float3 n;
									float area;
									GetTriangleNormalAndArea(p0, p1, p2, n, area);

									if (area >= minValidArea)
									{
										if (distance < currentDistance)
										{
											currentDistance = distance;
											nearestPoint = centroid;
										}

										normal.x = __fadd_rn(normal.x, n.x);
										normal.y = __fadd_rn(normal.y, n.y);
										normal.z = __fadd_rn(normal.z, n.z);

										count = __fadd_rn(count, 1.0f);
									}
									//else
									//{
									//	printf("Area : %f\n", area);
									//}
								}
							}
						}

						auto triangle2 = triangles[triangleIndex * 4 + 2];
						auto t2 = uint3{ triangle2.x(), triangle2.y(), triangle2.z() };
						if (VECTOR3U_VALID(t2))
						{
							//printf("t2 - index\n");

							auto v0 = vertices[t2.x];
							auto v1 = vertices[t2.y];
							auto v2 = vertices[t2.z];

							float3 p0{ v0.x(), v0.y(), v0.z() };
							float3 p1{ v1.x(), v1.y(), v1.z() };
							float3 p2{ v2.x(), v2.y(), v2.z() };

							if (VECTOR3F_VALID_(v0) && VECTOR3F_VALID_(v1) && VECTOR3F_VALID_(v2))
							{
								//printf("t2 - vertex\n");

								auto centroid = GetCentroid(p0, p1, p2);
								float distance = GetDistance(p, centroid);
								if (range >= distance)
								{
									float3 n;
									float area;
									GetTriangleNormalAndArea(p0, p1, p2, n, area);

									if (area >= minValidArea)
									{
										if (distance < currentDistance)
										{
											currentDistance = distance;
											nearestPoint = centroid;
										}

										normal.x = __fadd_rn(normal.x, n.x);
										normal.y = __fadd_rn(normal.y, n.y);
										normal.z = __fadd_rn(normal.z, n.z);

										count = __fadd_rn(count, 1.0f);
									}
									//else
									//{
									//	printf("Area : %f\n", area);
									//}
								}
							}
						}

						auto triangle3 = triangles[triangleIndex * 4 + 3];
						auto t3 = uint3{ triangle3.x(), triangle3.y(), triangle3.z() };
						if (VECTOR3U_VALID(t3))
						{
							//printf("t3 - index\n");

							auto v0 = vertices[t3.x];
							auto v1 = vertices[t3.y];
							auto v2 = vertices[t3.z];

							float3 p0{ v0.x(), v0.y(), v0.z() };
							float3 p1{ v1.x(), v1.y(), v1.z() };
							float3 p2{ v2.x(), v2.y(), v2.z() };

							if (VECTOR3F_VALID_(v0) && VECTOR3F_VALID_(v1) && VECTOR3F_VALID_(v2))
							{
								//printf("t3 - vertex\n");

								auto centroid = GetCentroid(p0, p1, p2);
								float distance = GetDistance(p, centroid);
								if (range >= distance)
								{
									float3 n;
									float area;
									GetTriangleNormalAndArea(p0, p1, p2, n, area);

									if (area >= minValidArea)
									{
										if (distance < currentDistance)
										{
											currentDistance = distance;
											nearestPoint = centroid;
										}

										normal.x = __fadd_rn(normal.x, n.x);
										normal.y = __fadd_rn(normal.y, n.y);
										normal.z = __fadd_rn(normal.z, n.z);

										count = __fadd_rn(count, 1.0f);
									}
									//else
									//{
									//	printf("Area : %f\n", area);
									//}
								}
							}
						}

						if (count > 0.0f)
						{
							normal.x = __fdiv_rn(normal.x, count);
							normal.y = __fdiv_rn(normal.y, count);
							normal.z = __fdiv_rn(normal.z, count);

							correspondPoint.x() = nearestPoint.x;
							correspondPoint.y() = nearestPoint.y;
							correspondPoint.z() = nearestPoint.z;
							correspondNormal.x() = normal.x;
							correspondNormal.y() = normal.y;
							correspondNormal.z() = normal.z;

							//printf("OK\n");

							printf("loop count : %f\n", count);

							return;
						}

						// 3.248 ms
					}
				}
			}
		}
	}

	printf("loop count : %d\n", count);
}
*/
#endif

//	__device__
//		Eigen::Vector3f VertexInterp(float isolevel, const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, float valp1, float valp2)
//	{
//#pragma region Using Truncation
//		float mu;
//		Eigen::Vector3f p;
//
//		if (fabsf(isolevel - valp1) < 0.00001f)
//			return(p1);
//		if (fabsf(isolevel - valp2) < 0.00001f)
//			return(p2);
//		if (fabsf(valp1 - valp2) < 0.00001f)
//			return(p1);
//		mu = (isolevel - valp1) / (valp2 - valp1);
//		p.x() = p1.x() + mu * (p2.x() - p1.x());
//		p.y() = p1.y() + mu * (p2.y() - p1.y());
//		p.z() = p1.z() + mu * (p2.z() - p1.z());
//		//float unit_size_0 = 100.0f;
//		float unit_size_1 = 0.01f;
//#define unit_size_0 100.0f
//
//		if (fabsf(isolevel - valp1) < 0.00001f)
//		{
//			p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
//			p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
//			p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
//			return p;
//		}
//
//		if (fabsf(isolevel - valp2) < 0.00001f)
//		{
//			p.x() = floorf(p2.x() * unit_size_0) / unit_size_0;
//			p.y() = floorf(p2.y() * unit_size_0) / unit_size_0;
//			p.z() = floorf(p2.z() * unit_size_0) / unit_size_0;
//			return p;
//		}
//
//		if (fabsf(valp1 - valp2) < 0.00001f)
//		{
//			p.x() = floorf(p1.x() * unit_size_0) / unit_size_0;
//			p.y() = floorf(p1.y() * unit_size_0) / unit_size_0;
//			p.z() = floorf(p1.z() * unit_size_0) / unit_size_0;
//			return p;
//		}
//
//		mu = (isolevel - valp1) / (valp2 - valp1);
//		p.x() = p1.x() + mu * (p2.x() - p1.x());
//		p.x() = floorf(p.x() * unit_size_0) / unit_size_0;
//		p.y() = p1.y() + mu * (p2.y() - p1.y());
//		p.y() = floorf(p.y() * unit_size_0) / unit_size_0;
//		p.z() = p1.z() + mu * (p2.z() - p1.z());
//		p.z() = floorf(p.z() * unit_size_0) / unit_size_0;
//
//		p.x() = ((int)(p.x() * unit_size_0 + 0.5)) / unit_size_0;
//		p.y() = ((int)(p.y() * unit_size_0 + 0.5)) / unit_size_0;
//		p.z() = ((int)(p.z() * unit_size_0 + 0.5)) / unit_size_0;
//
//		return p;
//#pragma endregion
//
//#pragma region Original
//		//float mu = 0.0f;
//		//Eigen::Vector3f p = p1;
//
//		//if (fabsf(isolevel - valp1) < 0.00001f)
//		//	return(p1);
//		//if (fabsf(isolevel - valp2) < 0.00001f)
//		//	return(p2);
//		//if (fabsf(valp1 - valp2) < 0.00001f)
//		//	return(p1);
//		//mu = (isolevel - valp1) / (valp2 - valp1);
//		//p.x() = p1.x() + mu * (p2.x() - p1.x());
//		//p.y() = p1.y() + mu * (p2.y() - p1.y());
//		//p.z() = p1.z() + mu * (p2.z() - p1.z());
//
//		//return p;
//#pragma endregion
//	}

#ifndef BUILD_FOR_CPU
	//view buffer to ICP target PC
void MarchingCubes::get_IcpTarget_v3(
	cached_allocator * alloc_, CUstream_st * st
	, Eigen::Vector3f * view_vtx_pos
	, Eigen::Vector3f * view_vtx_nm
	, std::shared_ptr<pointcloud_Hios> out_target
	, size_t vtxSize
)
{
	out_target->resizePointcloudPNC(vtxSize);
	auto vtx_pos = thrust::device_pointer_cast(view_vtx_pos);
	auto vtx_nm = thrust::device_pointer_cast(view_vtx_nm);

	auto out_pos = out_target->points_.data();
	auto out_nm = out_target->normals_.data();

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(vtx_pos, vtx_nm, thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(vtx_pos + vtxSize, vtx_nm + vtxSize, thrust::make_counting_iterator<size_t>(vtxSize)),
		[out_pos, out_nm]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		size_t idx = thrust::get<2>(tu);
		if (VECTOR3F_VALID_(curr_vtx))
		{
			out_pos[idx] = curr_vtx;
			out_nm[idx] = curr_nm;
		}
		else
			printf("makeICP Invalid!\n");
	}
	);
	out_target->m_nowSize = vtxSize;
}

//Copy PC  to view buffer
void MarchingCubes::set_renderBuffer_v1(
	cached_allocator * alloc_, CUstream_st * st
	, Eigen::Vector3f * view_vtx_pos
	, Eigen::Vector3f * view_vtx_nm
	, Eigen::Vector3f * view_vtx_clr
	, std::shared_ptr<pointcloud_Hios> in_PC
	, size_t & out_vtxSize
)
{
	auto vtx_pos = thrust::device_pointer_cast(view_vtx_pos);
	auto vtx_nm = thrust::device_pointer_cast(view_vtx_nm);
	auto vtx_clr = thrust::device_pointer_cast(view_vtx_clr);

	auto in_pos = in_PC->points_.data();
	auto in_nm = in_PC->normals_.data();
	auto in_clr = in_PC->colors_.data();

	out_vtxSize = in_PC->m_nowSize;

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(in_PC->points_.begin(), in_PC->normals_.begin(), in_PC->colors_.begin(), thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(in_PC->points_.begin() + out_vtxSize, in_PC->normals_.begin() + out_vtxSize, in_PC->colors_.begin() + out_vtxSize, thrust::make_counting_iterator<size_t>(out_vtxSize)),
		[vtx_pos, vtx_nm, vtx_clr]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		Eigen::Vector3b& curr_clor = thrust::get<2>(tu);
		size_t idx = thrust::get<3>(tu);
		if (VECTOR3F_VALID_(curr_vtx))
		{
			vtx_pos[idx] = curr_vtx;
			vtx_nm[idx] = curr_nm.normalized();
			vtx_clr[idx] = Eigen::Vector3f(
				static_cast<float>(curr_clor.x()) / 255.0f,
				static_cast<float>(curr_clor.y()) / 255.0f,
				static_cast<float>(curr_clor.z()) / 255.0f
			);
		}
		else
			printf("makeICP Invalid!\n");
	}
	);
}

//Add Copy mID
void MarchingCubes::set_renderBuffer_v2(
	cached_allocator * alloc_, CUstream_st * st
	, Eigen::Vector3f * view_vtx_pos
	, Eigen::Vector3f * view_vtx_nm
	, Eigen::Vector3b * view_vtx_clr
	, VoxelExtraAttrib * view_extra_attrib
	, const std::shared_ptr<pointcloud_Hios> in_PC
	, size_t & out_vtxSize
)
{
	auto vtx_pos = thrust::device_pointer_cast(view_vtx_pos);
	auto vtx_nm = thrust::device_pointer_cast(view_vtx_nm);
	auto vtx_clr = thrust::device_pointer_cast(view_vtx_clr);
	auto vtx_extra_attrib = thrust::device_pointer_cast(view_extra_attrib);

	auto in_pos = in_PC->points_.data();
	auto in_nm = in_PC->normals_.data();
	auto in_clr = in_PC->colors_.data();
	auto in_extraAttrib = in_PC->extraAttribs_.data();

	out_vtxSize = in_PC->m_nowSize;

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(in_PC->points_.begin(), in_PC->normals_.begin(), in_PC->colors_.begin(), in_PC->extraAttribs_.begin(), thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(in_PC->points_.begin() + out_vtxSize, in_PC->normals_.begin() + out_vtxSize, in_PC->colors_.begin() + out_vtxSize,
			in_PC->extraAttribs_.begin() + out_vtxSize, thrust::make_counting_iterator<size_t>(out_vtxSize)),
		[vtx_pos, vtx_nm, vtx_clr, vtx_extra_attrib]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		Eigen::Vector3b& curr_clor = thrust::get<2>(tu);
		VoxelExtraAttrib& curr_extraAttrib = thrust::get<3>(tu);
		size_t idx = thrust::get<4>(tu);

		if (VECTOR3F_VALID_(curr_vtx))
		{
			vtx_pos[idx] = curr_vtx;
			vtx_nm[idx] = curr_nm.normalized();
			vtx_clr[idx] = curr_clor;
			vtx_extra_attrib[idx] = curr_extraAttrib;
		}
		else
			printf("makeICP Invalid!\n");
	}
	);
}
#endif

//view buffer to ICP target PC
void MarchingCubes::get_IcpTarget_v3_color(
	cached_allocator * alloc_, CUstream_st * st,
	Eigen::Vector3f * view_vtx_pos
	, Eigen::Vector3f * view_vtx_nm
	, Eigen::Vector3b * view_vtx_clr
	, VoxelExtraAttrib * view_vtx_extraAttrib
	, std::shared_ptr<pointcloud_Hios> out_target
	, size_t vtxSize
)
{
	if (out_target->points_.size() < vtxSize)
		out_target->points_.resize(vtxSize * 1.2);
	if (out_target->colors_.size() < vtxSize)
		out_target->colors_.resize(vtxSize * 1.2);
	if (out_target->normals_.size() < vtxSize)
		out_target->normals_.resize(vtxSize * 1.2);
	if (out_target->extraAttribs_.size() < vtxSize)
		out_target->extraAttribs_.resize(vtxSize * 1.2);

#ifndef BUILD_FOR_CPU
	auto vtx_pos = thrust::device_pointer_cast(view_vtx_pos);
	auto vtx_nm = thrust::device_pointer_cast(view_vtx_nm);
	auto vtx_clr = thrust::device_pointer_cast(view_vtx_clr);
	auto vtx_extraAttrib = thrust::device_pointer_cast(view_vtx_extraAttrib);
#else
	auto vtx_pos = view_vtx_pos;
	auto vtx_nm = view_vtx_nm;
	auto vtx_clr = view_vtx_clr;
	auto vtx_extraAttrib = view_vtx_extraAttrib;
#endif

	auto out_pos = out_target->points_.data();
	auto out_nm = out_target->normals_.data();
	auto out_clr = out_target->colors_.data();
	auto out_extraAttrib = out_target->extraAttribs_.data();

	thrust::for_each(
#ifndef BUILD_FOR_CPU
		thrust::cuda::par_nosync(*alloc_).on(st),
#else
		thrust::omp::par,
#endif
		make_tuple_iterator(vtx_pos, vtx_nm, vtx_clr, vtx_extraAttrib, thrust::make_counting_iterator<size_t>(0)),
		make_tuple_iterator(vtx_pos + vtxSize, vtx_nm + vtxSize, vtx_clr + vtxSize, vtx_extraAttrib + vtxSize, thrust::make_counting_iterator<size_t>(vtxSize)),
		[out_pos, out_nm, out_clr, out_extraAttrib]__device__(auto & tu)
	{
		Eigen::Vector3f& curr_vtx = thrust::get<0>(tu);
		Eigen::Vector3f& curr_nm = thrust::get<1>(tu);
		Eigen::Vector3b& curr_clor = thrust::get<2>(tu);
		VoxelExtraAttrib& curr_extraAttrib = thrust::get<3>(tu);
		size_t idx = thrust::get<4>(tu);
		if (VECTOR3F_VALID_(curr_vtx))
		{
			out_pos[idx] = curr_vtx;
			out_nm[idx] = curr_nm.normalized();
			out_clr[idx] = curr_clor;
			out_extraAttrib[idx] = curr_extraAttrib;
			/*printf("VP = %f, %f, %f, VN = %f, %f, %f, VC = %f, %f, %f, MID = %d\n",
				curr_vtx.x(), curr_vtx.y(), curr_vtx.z(),
				curr_nm.x(), curr_nm.y(), curr_nm.z(),
				curr_clor.x(), curr_clor.y(), curr_clor.z(), curr_mId);*/
		}
		else
			printf("makeICP Invalid!\n");
	}
	);
	out_target->m_nowSize = vtxSize;
}

#ifndef BUILD_FOR_CPU
void MarchingCubes::get_IcpTarget_v5(
	cached_allocator * alloc_, CUstream_st * st
	, Eigen::Vector3f * view_vtx_pos
	, Eigen::Vector3f * view_vtx_nm
	, std::shared_ptr<pointcloud_Hios> out_target
	, size_t Size
)
{
	NvtxRangeCuda nvtxPrint("get_IcpTarget_v5");
	auto vtx_pos = thrust::device_pointer_cast(view_vtx_pos);
	auto vtx_nm = thrust::device_pointer_cast(view_vtx_nm);

	auto out_pos = out_target->points_.data();
	auto out_nm = out_target->normals_.data();

	thrust::copy(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(vtx_pos, vtx_nm),
		make_tuple_iterator(vtx_pos + Size, vtx_nm + Size),
		make_tuple_iterator(out_pos, out_nm)
	);
	out_target->m_nowSize = Size;
}
#endif 

size_t MarchingCubes::GetCacheLength() {
	return exeInfo.cache_length;
}
#pragma region GatherOccupiedVoxelIndices

void MarchingCubes::initialize_ExecutionInfo(cached_allocator * alloc_, CUstream_st * st)
{
	if (exeInfoInitialized == true) return;
	InitGlobalVoxelValues();

	exeInfo.global.SetGlobalMinMax(Eigen::Vector3f(-250.0f, -250.0f, -250.0f), Eigen::Vector3f(250.0f, 250.0f, 250.0f));
	exeInfo.global.SetLocalMinMax(Eigen::Vector3f(-250.0f, -250.0f, -250.0f), Eigen::Vector3f(250.0f, 250.0f, 250.0f));

	exeInfo.local.SetGlobalMinMax(Eigen::Vector3f(-250.0f, -250.0f, -250.0f), Eigen::Vector3f(250.0f, 250.0f, 250.0f));
	exeInfo.local.SetLocalMinMax(Eigen::Vector3f(-10.0f, -15.0f, -11.0f), Eigen::Vector3f(14.0f, 17.0f, 11.0f));

	exeInfo.cache_length = 41.f;
	Eigen::Vector3f localMin{ -10.0f, -12.5f, -12.5f };
	Eigen::Vector3f cacheMin{ localMin.x(), localMin.y(), localMin.z() };
	Eigen::Vector3f cacheMax{ localMin.x() + exeInfo.cache_length, localMin.y() + exeInfo.cache_length, localMin.z() + exeInfo.cache_length };

	exeInfo.cache.SetGlobalMinMax(exeInfo.global.globalMin, exeInfo.global.globalMax);
	exeInfo.cache.SetLocalMinMax(cacheMin, cacheMax);

	exeInfo.globalHashInfo = globalHash_info;
	exeInfo.globalHash = globalHash_info_host->hashtable;
	exeInfo.globalHashInfo_host = globalHash_info_host;

	exeInfo.voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	exeInfo.voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	exeInfo.voxelPositions = thrust::raw_pointer_cast(m_MC_voxelPositions.data());
	exeInfo.voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	exeInfo.voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	exeInfo.voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	exeInfo.voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	exeInfo.voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	uint32_t* gridSlotIndexCache = nullptr;
	uint32_t* gridSlotIndexCache_2 = nullptr;
#ifndef BUILD_FOR_CPU
	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	//cudaMallocAsync(&gridSlotIndexCache, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);
	//cudaMallocAsync(&gridSlotIndexCache_2, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);
	exeInfo.maxSize_cache = exeInfo.cache.voxelCount;
	cudaMalloc(&gridSlotIndexCache, sizeof(uint32_t) * exeInfo.maxSize_cache);
	cudaMalloc(&gridSlotIndexCache_2, sizeof(uint32_t) * exeInfo.maxSize_cache);
	cudaMemsetAsync(gridSlotIndexCache, 0xFF, sizeof(uint32_t) * exeInfo.maxSize_cache, st);
	cudaMemsetAsync(gridSlotIndexCache_2, 0xFF, sizeof(uint32_t) * exeInfo.maxSize_cache, st);
#else
	gridSlotIndexCache = new uint32_t[exeInfo.cache.voxelCount];
	gridSlotIndexCache_2 = new uint32_t[exeInfo.cache.voxelCount];
	memset(gridSlotIndexCache, 0xff, sizeof(uint32_t) * exeInfo.cache.voxelCount);
	memset(gridSlotIndexCache_2, 0xff, sizeof(uint32_t) * exeInfo.cache.voxelCount);
#endif
	exeInfo.gridSlotIndexCache = gridSlotIndexCache;
	exeInfo.gridSlotIndexCache_pts = gridSlotIndexCache_2;

	//======================================================================
	//	mscho	@20240717
	//exeInfo.blockSize = 10;
	//exeInfo.blockIndex = (uint32_t)exeInfo.globalHashInfo_host->HashTableCapacity / exeInfo.blockSize;
	//exeInfo.blockRemainder = (uint32_t)exeInfo.globalHashInfo_host->HashTableCapacity % exeInfo.blockSize;
	//if (exeInfo.blockRemainder != 0)
	//	exeInfo.blockIndex++;

	exeInfo.blockSize = 1;
	exeInfo.blockIndex = (uint32_t)exeInfo.globalHashInfo_host->HashTableCapacity / exeInfo.blockSize;
	exeInfo.blockRemainder = (uint32_t)exeInfo.globalHashInfo_host->HashTableCapacity % exeInfo.blockSize;
	if (exeInfo.blockRemainder != 0)
		exeInfo.blockIndex++;
	//======================================================================

	setGlobalAABB(Eigen::Vector3f(-250.0f, -250.0f, -250.0f), Eigen::Vector3f(250.0f, 250.0f, 250.0f), 0.1f);

	if (nullptr != noiseFilter)
	{
		noiseFilter->Terminate();
		delete noiseFilter;
	}
	noiseFilter = new Filtering::NoiseFilter;
	noiseFilter->Initialize();

	exeInfoInitialized = true;

	exeInfo.Dump();
}

void MarchingCubes::exeInfo_update_LocalMinMax(const Eigen::Vector3f & aabb_min,
	const Eigen::Vector3f & aabb_max) {
	exeInfo.local.SetLocalMinMax(aabb_min, aabb_max);
}

void MarchingCubes::exeInfo_update_LocalMinMax(const Eigen::Vector3f & aabb_min,
	const Eigen::Vector3f & aabb_max, int marginSize_occ, int marginSize_update) {
	if (marginSize_occ < 0)
		marginSize_occ = 0;
	if (marginSize_update < 0)
		marginSize_update = 0;

	Eigen::Vector3f newMin = aabb_min;
	Eigen::Vector3f newMax = aabb_max;
	Eigen::Vector3f newCacheMin = aabb_min;

	newMin.x() -= (m_voxelSize * (float)marginSize_occ);
	newMin.y() -= (m_voxelSize * (float)marginSize_occ);
	newMin.z() -= (m_voxelSize * (float)marginSize_occ);

	newMax.x() += (m_voxelSize * (float)marginSize_occ);
	newMax.y() += (m_voxelSize * (float)marginSize_occ);
	newMax.z() += (m_voxelSize * (float)marginSize_occ);

	newCacheMin.x() -= (m_voxelSize * (float)marginSize_update);
	newCacheMin.y() -= (m_voxelSize * (float)marginSize_update);
	newCacheMin.z() -= (m_voxelSize * (float)marginSize_update);

	Eigen::Vector3f newCacheMax{ newMin.x() + exeInfo.cache_length, newMin.y() + exeInfo.cache_length, newMin.z() + exeInfo.cache_length };

	exeInfo.local.SetLocalMinMax(newMin, newMax);

	exeInfo.cache.SetLocalMinMax(newCacheMin, newCacheMax);
}

#ifndef BUILD_FOR_CPU
int	MarchingCubes::resetHashTable(HashKey64 * host_HashInfo, CUstream_st * st)
{
	pHashManager->reset_hash(host_HashInfo, st);
	return 0;
}

// 
uint32_t    MarchingCubes::refreshHashTable(bool DoAlive, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return 0;

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize;
	uint32_t* ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff.data());

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	if (DoAlive) {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1 << <gridsize, threadblocksize, 0, st >> > (
			globalHash_info
			, ext_used_Key
			);
		checkCudaErrors(cudaGetLastError());
	}
	else {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (
			globalHash_info
			, ext_used_Key
			);
		checkCudaErrors(cudaGetLastError());
	}
	cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	if (FOR_TEST_PRINT)qDebug("\n [1] refreshHashTable usedCheck Size = %d", h_usedSize);

	if (h_usedSize == 0)return 0;

	auto _globalHash_info = globalHash_info;
	auto _globalHash = globalHash_info_host->hashtable;
	//auto _globalHash_value = globalHash_value;

	//size_t													back_usedSize;
	//thrust::device_ptr<HashEntry>							backashtable = thrust::device_malloc<uint64_t>(h_usedSize);
	//thrust::device_ptr<voxel_value_t>						back_voxelValues = thrust::device_malloc<voxel_value_t>(h_usedSize);
	//thrust::device_ptr<unsigned short>					back_voxelValueCounts = thrust::device_malloc<unsigned short>(h_usedSize);
	//thrust::device_ptr<Eigen::Vector3f>					back_voxelNormals = thrust::device_malloc<Eigen::Vector3f>(h_usedSize);
	//thrust::device_ptr<Eigen::Vector3b>	back_voxelColors = thrust::device_malloc<Eigen::Vector3b>(h_usedSize);
	//thrust::device_ptr<float>								back_voxelColorScores = thrust::device_malloc<float>(h_usedSize);

	thrust::device_vector<HashEntry>						backashtable(h_usedSize);
	thrust::device_vector<voxel_value_t>					back_voxelValues(h_usedSize);
	thrust::device_vector<unsigned short>					back_voxelValueCounts(h_usedSize);
	thrust::device_vector<Eigen::Vector3f>					back_voxelNormals(h_usedSize);
	thrust::device_vector<Eigen::Vector3b>	back_voxelColors(h_usedSize);
	thrust::device_vector<float>							back_voxelColorScores(h_usedSize);
	thrust::device_vector<char>								back_voxelSegmentations(h_usedSize);
	thrust::device_vector<VoxelExtraAttrib>					back_voxelExtraAttribs(h_usedSize);

	thrust::device_vector<voxel_value_t>& voxel_value_repos = m_MC_voxelValues;
	thrust::device_vector<unsigned short>& voxel_valueCnt_repos = m_MC_voxelValueCounts;
	thrust::device_vector<Eigen::Vector3f>& voxel_nms_repos = m_MC_voxelNormals;
	thrust::device_vector<Eigen::Vector3b>& voxel_clrs_repos = m_MC_voxelColors;
	thrust::device_vector<float>& voxel_clrScores_repos = m_MC_voxelColorScores;
	thrust::device_vector<char>& voxel_segmentations_repos = m_MC_voxelSegmentations;
	thrust::device_vector<VoxelExtraAttrib>& voxel_extraAttribs_repos = m_MC_voxelExtraAttribs;

	const HashEntry* oTable = globalHash_info_host->hashtable;
	const auto oVV = thrust::raw_pointer_cast(voxel_value_repos.data());
	const auto oVVC = thrust::raw_pointer_cast(voxel_valueCnt_repos.data());
	const auto oVN = thrust::raw_pointer_cast(voxel_nms_repos.data());
	const auto oVC = thrust::raw_pointer_cast(voxel_clrs_repos.data());
	const auto oVCS = thrust::raw_pointer_cast(voxel_clrScores_repos.data());
	const auto oVSG = thrust::raw_pointer_cast(voxel_segmentations_repos.data());
	const auto oVEA = thrust::raw_pointer_cast(voxel_extraAttribs_repos.data());

	auto tTable = thrust::raw_pointer_cast(backashtable.data());
	auto tVV = thrust::raw_pointer_cast(back_voxelValues.data());
	auto tVVC = thrust::raw_pointer_cast(back_voxelValueCounts.data());
	auto tVN = thrust::raw_pointer_cast(back_voxelNormals.data());
	auto tVC = thrust::raw_pointer_cast(back_voxelColors.data());
	auto tVCS = thrust::raw_pointer_cast(back_voxelColorScores.data());
	auto tVSG = thrust::raw_pointer_cast(back_voxelSegmentations.data());
	auto tVEA = thrust::raw_pointer_cast(back_voxelExtraAttribs.data());

	uint32_t* d_count;
	uint32_t h_count = 0; // 호스트 카운트 값
	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	//cudaMallocAsync(&d_count, sizeof(uint32_t), st);
	cudaMalloc(&d_count, sizeof(uint32_t));
	cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st);


	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_used_buff.begin(),
		m_MC_used_buff.begin() + h_usedSize,
		[d_count,
		oTable, oVV, oVVC, oVN, oVC, oVCS, oVEA, oVSG,
		tTable, tVV, tVVC, tVN, tVC, tVCS, tVEA, tVSG
		]__device__(unsigned int& index) {
		if (
			FLT_VALID(oVV[index]) &&
			USHORT_VALID(oVVC[index]) &&
			FLT_VALID(oVN[index].x()) &&
			FLT_VALID(oVN[index].y()) &&
			FLT_VALID(oVN[index].z())
			)
		{
			if (oVVC[index] == 0) return;
			if (-1 < oVV[index] / (float)oVVC[index] &&
				1 > oVV[index] / (float)oVVC[index])
			{
				uint32_t _prev_table_ = atomicAdd(d_count, 1);

				tTable[_prev_table_] = oTable[index];
				tVV[_prev_table_] = oVV[index];
				tVVC[_prev_table_] = oVVC[index];
				tVN[_prev_table_] = oVN[index];
				tVC[_prev_table_] = oVC[index];
				tVCS[_prev_table_] = oVCS[index];
				tVSG[_prev_table_] = oVSG[index];
				tVEA[_prev_table_] = oVEA[index];
			}
		}
		/*else
		{
			printf("VV = %llu, VVC = %f, VN = %f, %f, %f, VC = %d, %d, %d, VCS = %f\n",
				oVV[index],
				oVVC[index],
				oVN[index].x(), oVN[index].y(), oVN[index].z(),
				oVC[index].x(), oVC[index].y(), oVC[index].z(),
				oVCS[index]);
		}*/
	});
	cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	if (1) //hash 珥湲고 
	{
		resetHashTable(globalHash_info_host, st);

		auto executionPolicy = thrust::cuda::par_nosync.on(st);
		thrust::fill(executionPolicy, voxel_value_repos.begin(), voxel_value_repos.end(), FLT_MAX);// VOXEL_INVALID);
		thrust::fill(executionPolicy, voxel_valueCnt_repos.begin(), voxel_valueCnt_repos.end(), 0);
		thrust::fill(executionPolicy, voxel_nms_repos.begin(), voxel_nms_repos.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		thrust::fill(executionPolicy, voxel_clrs_repos.begin(), voxel_clrs_repos.end(), Eigen::Vector3b(0, 0, 0));
		thrust::fill(executionPolicy, voxel_clrScores_repos.begin(), voxel_clrScores_repos.end(), 0.0f);
		thrust::fill(executionPolicy, voxel_segmentations_repos.begin(), voxel_segmentations_repos.end(), 0.0f);
		VoxelExtraAttrib emptyExtraAttrib = { 0 };
		thrust::fill(executionPolicy, voxel_extraAttribs_repos.begin(), voxel_extraAttribs_repos.end(), emptyExtraAttrib);
	}

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		thrust::make_counting_iterator<size_t>(0),
		thrust::make_counting_iterator<size_t>(h_count),
		[_globalHash, _globalHash_info,
		tTable,
		tVV, tVVC, tVN, tVC, tVCS, tVEA, tVSG,
		oVV, oVVC, oVN, oVC, oVCS, oVEA, oVSG
		]__device__(size_t index) {
		//printf("%llu\n", tTable[index]);
		auto hashSlot_idx = _globalHash_info->get_insert_idx_func64_v4(tTable[index]);
		oVV[hashSlot_idx] = tVV[index];
		oVVC[hashSlot_idx] = tVVC[index];
		oVN[hashSlot_idx] = tVN[index];
		oVC[hashSlot_idx] = tVC[index];
		oVCS[hashSlot_idx] = tVCS[index];
		oVSG[hashSlot_idx] = tVSG[index];
		oVEA[hashSlot_idx] = tVEA[index];
	}
	);


	h_usedSize = 0;

	{ /// hash 복원 결과 사이즈 확인
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;


		pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

		kernel_iterateChkUsed_voxelCnt_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (globalHash_info);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

		checkCudaSync(st);
	}

	//thrust::device_free(backashtable);
	//thrust::device_free(back_voxelValues);
	//thrust::device_free(back_voxelValueCounts);
	//thrust::device_free(back_voxelNormals);
	//thrust::device_free(back_voxelColors);
	//thrust::device_free(back_voxelColorScores);

	checkCudaSync(st);

	qDebug(" \n[2] refreshHashTable usedCheck Size = %d", h_usedSize);

	return h_count;
}

// Voxel들 일부가 제거 되었을 때 Index의 2차 탐색을 줄이기 위해 HashTable을 정리한다.
uint32_t    MarchingCubes::refreshHashTable_v2(bool DoAlive, HashEntry * *KeyRepos, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return 0;

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize;
	uint32_t* ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff.data());

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	// 해시 테이블에서 사용된 Index를 ext_used_key 모은다.
	if (DoAlive) {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1 << <gridsize, threadblocksize, 0, st >> > (
			globalHash_info
			, ext_used_Key
			);
		checkCudaErrors(cudaGetLastError());
	}
	else {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (
			globalHash_info
			, ext_used_Key
			);
		checkCudaErrors(cudaGetLastError());
	}
	cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	qDebug("\tsh) [1] refreshHashTable usedCheck Size = %d", h_usedSize);

	if (h_usedSize == 0)return 0;

	auto _globalHash_info = globalHash_info;
	auto _globalHash = globalHash_info_host->hashtable;
	//auto _globalHash_value = globalHash_value;

	//size_t									back_usedSize;
	//thrust::device_ptr<uint64_t>				backashtable = thrust::device_malloc<uint64_t>(h_usedSize);
	//thrust::device_ptr<voxel_value_t>			back_voxelValues = thrust::device_malloc<voxel_value_t>(h_usedSize);
	//thrust::device_ptr<unsigned short>		back_voxelValueCounts = thrust::device_malloc<unsigned short>(h_usedSize);
	//thrust::device_ptr<Eigen::Vector3f>		back_voxelNormals = thrust::device_malloc<Eigen::Vector3f>(h_usedSize);
	//thrust::device_ptr<Eigen::Vector3b>		back_voxelColors = thrust::device_malloc<Eigen::Vector3b>(h_usedSize);
	//thrust::device_ptr<float>					back_voxelColorScores = thrust::device_malloc<float>(h_usedSize);

	thrust::device_vector<HashEntry>			backashtable(h_usedSize);
	thrust::device_vector<voxel_value_t>		back_voxelValues(h_usedSize);
	thrust::device_vector<unsigned short>		back_voxelValueCounts(h_usedSize);
	thrust::device_vector<Eigen::Vector3f>		back_voxelNormals(h_usedSize);
	thrust::device_vector<Eigen::Vector3b>		back_voxelColors(h_usedSize);
	thrust::device_vector<float>				back_voxelColorScores(h_usedSize);
	thrust::device_vector<VoxelExtraAttrib>		back_voxelExtraAttribs(h_usedSize);

	thrust::device_vector<voxel_value_t>& voxel_value_repos = m_MC_voxelValues;
	thrust::device_vector<unsigned short>& voxel_valueCnt_repos = m_MC_voxelValueCounts;
	thrust::device_vector<Eigen::Vector3f>& voxel_nms_repos = m_MC_voxelNormals;
	thrust::device_vector<Eigen::Vector3b>& voxel_clrs_repos = m_MC_voxelColors;
	thrust::device_vector<float>& voxel_clrScores_repos = m_MC_voxelColorScores;
	thrust::device_vector<VoxelExtraAttrib>& voxel_extraAttribs_repos = m_MC_voxelExtraAttribs;

	HashEntry* oTable = globalHash_info_host->hashtable;
	auto oVV = thrust::raw_pointer_cast(voxel_value_repos.data());
	auto oVVC = thrust::raw_pointer_cast(voxel_valueCnt_repos.data());
	auto oVN = thrust::raw_pointer_cast(voxel_nms_repos.data());
	auto oVC = thrust::raw_pointer_cast(voxel_clrs_repos.data());
	auto oVCS = thrust::raw_pointer_cast(voxel_clrScores_repos.data());
	auto oVEA = thrust::raw_pointer_cast(voxel_extraAttribs_repos.data());

	HashEntry* tTable = thrust::raw_pointer_cast(backashtable.data());
	auto tVV = thrust::raw_pointer_cast(back_voxelValues.data());
	auto tVVC = thrust::raw_pointer_cast(back_voxelValueCounts.data());
	auto tVN = thrust::raw_pointer_cast(back_voxelNormals.data());
	auto tVC = thrust::raw_pointer_cast(back_voxelColors.data());
	auto tVCS = thrust::raw_pointer_cast(back_voxelColorScores.data());
	auto tVEA = thrust::raw_pointer_cast(back_voxelExtraAttribs.data());

	uint32_t* d_count;
	uint32_t h_count = 0; // 호스트 카운트 값
	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	//cudaMallocAsync(&d_count, sizeof(uint32_t), st);
	cudaMalloc(&d_count, sizeof(uint32_t));
	cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st);


	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_used_buff.begin(),
		m_MC_used_buff.begin() + h_usedSize,
		[d_count,
		oTable, oVV, oVVC, oVN, oVC, oVCS, oVEA,
		tTable, tVV, tVVC, tVN, tVC, tVCS, tVEA
		]__device__(unsigned int& index) {
		if (
			FLT_VALID(oVV[index]) &&
			USHORT_VALID(oVVC[index]) &&
			FLT_VALID(oVN[index].x()) &&
			FLT_VALID(oVN[index].y()) &&
			FLT_VALID(oVN[index].z())
			)
		{
			if (oVVC[index] == 0) return;
			if (-1 < oVV[index] / (float)oVVC[index] &&
				1 > oVV[index] / (float)oVVC[index])
			{
				uint32_t _prev_table_ = atomicAdd(d_count, 1);

				tTable[_prev_table_] = oTable[index];
				tVV[_prev_table_] = oVV[index];
				tVVC[_prev_table_] = oVVC[index];
				tVN[_prev_table_] = oVN[index];
				tVC[_prev_table_] = oVC[index];
				tVCS[_prev_table_] = oVCS[index];
				tVEA[_prev_table_] = oVEA[index];
			}
		}
		/*else
		{
			printf("VV = %llu, VVC = %f, VN = %f, %f, %f, VC = %d, %d, %d, VCS = %f\n",
				oVV[index],
				oVVC[index],
				oVN[index].x(), oVN[index].y(), oVN[index].z(),
				oVC[index].x(), oVC[index].y(), oVC[index].z(),
				oVCS[index]);
		}*/
	});
	cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	if (1) //hash 珥湲고 
	{
		resetHashTable(globalHash_info_host, st);

		auto executionPolicy = thrust::cuda::par_nosync.on(st);
		thrust::fill(executionPolicy, voxel_value_repos.begin(), voxel_value_repos.end(), FLT_MAX);// VOXEL_INVALID);
		thrust::fill(executionPolicy, voxel_valueCnt_repos.begin(), voxel_valueCnt_repos.end(), 0);
		thrust::fill(executionPolicy, voxel_nms_repos.begin(), voxel_nms_repos.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		thrust::fill(executionPolicy, voxel_clrs_repos.begin(), voxel_clrs_repos.end(), Eigen::Vector3b(0, 0, 0));
		thrust::fill(executionPolicy, voxel_clrScores_repos.begin(), voxel_clrScores_repos.end(), 0.0f);
		VoxelExtraAttrib emptyExtraAttrib = { 0 };
		thrust::fill(executionPolicy, voxel_extraAttribs_repos.begin(), voxel_extraAttribs_repos.end(), emptyExtraAttrib);
	}

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		thrust::make_counting_iterator<size_t>(0),
		thrust::make_counting_iterator<size_t>(h_count),
		[_globalHash, _globalHash_info,
		tTable,
		tVV, tVVC, tVN, tVC, tVCS, tVEA,
		oVV, oVVC, oVN, oVC, oVCS, oVEA
		]__device__(size_t index) {
		//printf("%llu\n", tTable[index]);
		auto hashSlot_idx = _globalHash_info->get_insert_idx_func64_v4(tTable[index]);
		oVV[hashSlot_idx] = tVV[index];
		oVVC[hashSlot_idx] = tVVC[index];
		oVN[hashSlot_idx] = tVN[index];
		oVC[hashSlot_idx] = tVC[index];
		oVCS[hashSlot_idx] = tVCS[index];
		oVEA[hashSlot_idx] = tVEA[index];
	}
	);


	h_usedSize = 0;

	{ /// hash 복원 결과 사이즈 확인
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;


		pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

		kernel_iterateChkUsed_voxelCnt_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (globalHash_info);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

		checkCudaSync(st);
	}
	if (DoAlive) {
		if (*KeyRepos != nullptr) {
			delete[] KeyRepos;
			*KeyRepos = new HashEntry[h_count];
		}
		else
			*KeyRepos = new HashEntry[h_count];
	}
	cudaMemcpyAsync(*KeyRepos, thrust::raw_pointer_cast(backashtable.data()), sizeof(HashEntry) * h_count, cudaMemcpyDeviceToHost, st);
	//thrust::device_free(backashtable);
	//thrust::device_free(back_voxelValues);
	//thrust::device_free(back_voxelValueCounts);
	//thrust::device_free(back_voxelNormals);
	//thrust::device_free(back_voxelColors);
	//thrust::device_free(back_voxelColorScores);

	checkCudaSync(st);

	qDebug(" \tsh) [2] refreshHashTable usedCheck Size = %d", h_usedSize);

	return h_count;
}

uint32_t    MarchingCubes::backupHashTable(VoxelHashBackupInfo & voxelBackupInfo, bool  DoAlive, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return 0;

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize = 0;
	uint32_t* ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff.data());

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	/*if (DoAlive) {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelAliveCnt_wUsed_hashtable64_v1 << <gridsize, threadblocksize, 0, st >> > (globalHash_info, ext_used_Key);
		checkCudaErrors(cudaGetLastError());
	}
	else */ {
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (
			globalHash_info
			, ext_used_Key
			);
		checkCudaErrors(cudaGetLastError());
	}
	cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	//checkCudaSync(st);
	cudaDeviceSynchronize();

	qDebug("\n [1] usedCheck Size = %d", h_usedSize);

	voxelBackupInfo.Create(h_usedSize);

	if (h_usedSize == 0)
		return 0;

	size_t& repos_hBack_usedSize = voxelBackupInfo.usedSize;
	HashEntry** repos_hBack_table = &voxelBackupInfo.hashtable;
	voxel_value_t** repos_hBack_VV = &voxelBackupInfo.voxelValues;
	unsigned short** repos_hBack_VVC = &voxelBackupInfo.voxelValueCounts;
	Eigen::Vector3f** repos_hBack_VN = &voxelBackupInfo.voxelNormals;
	Eigen::Vector3b** repos_hBack_VC = &voxelBackupInfo.voxelColors;
	float** repos_hBack_VCS = &voxelBackupInfo.voxelColorScores;
	char** repos_hBack_VSG = &voxelBackupInfo.voxelSegmentations;
	VoxelExtraAttrib** repos_hBack_VEA = &voxelBackupInfo.voxelExtraAttribs;

	thrust::device_vector<voxel_value_t>& voxel_value_repos = m_MC_voxelValues;
	thrust::device_vector<unsigned short>& voxel_valueCnt_repos = m_MC_voxelValueCounts;
	thrust::device_vector<Eigen::Vector3f>& voxel_nms_repos = m_MC_voxelNormals;
	thrust::device_vector<Eigen::Vector3b>& voxel_clrs_repos = m_MC_voxelColors;
	thrust::device_vector<float>& voxel_clrScores_repos = m_MC_voxelColorScores;
	thrust::device_vector<char>& voxel_segmentations_repos = m_MC_voxelSegmentations;
	thrust::device_vector<VoxelExtraAttrib>& voxel_extraAttribs_repos = m_MC_voxelExtraAttribs;

	HashEntry* tTable;
	voxel_value_t* tVV;
	unsigned short* tVVC;
	Eigen::Vector3f* tVN;
	Eigen::Vector3b* tVC;
	float* tVCS;
	char* tVSG;
	VoxelExtraAttrib* tVEA;
	//unsigned char* tVF;

	cudaMalloc(&tTable, sizeof(HashEntry) * h_usedSize);
	cudaMalloc(&tVV, sizeof(voxel_value_t) * h_usedSize);
	cudaMalloc(&tVVC, sizeof(unsigned short) * h_usedSize);
	cudaMalloc(&tVN, sizeof(Eigen::Vector3f) * h_usedSize);
	cudaMalloc(&tVC, sizeof(Eigen::Vector3b) * h_usedSize);
	cudaMalloc(&tVCS, sizeof(float) * h_usedSize);
	cudaMalloc(&tVSG, sizeof(char) * h_usedSize);

	cudaMalloc(&tVEA, sizeof(VoxelExtraAttrib) * h_usedSize);

	auto oTable = globalHash_info_host->hashtable;
	const auto oVV = thrust::raw_pointer_cast(voxel_value_repos.data());
	const auto oVVC = thrust::raw_pointer_cast(voxel_valueCnt_repos.data());
	const auto oVN = thrust::raw_pointer_cast(voxel_nms_repos.data());
	const auto oVC = thrust::raw_pointer_cast(voxel_clrs_repos.data());
	const auto oVCS = thrust::raw_pointer_cast(voxel_clrScores_repos.data());
	const auto oVSG = thrust::raw_pointer_cast(voxel_segmentations_repos.data());
	const auto oVEA = thrust::raw_pointer_cast(voxel_extraAttribs_repos.data());

	//auto tTable = thrust::raw_pointer_cast(backashtable.data());
	//auto tVV = thrust::raw_pointer_cast(back_voxelValues.data());
	//auto tVVC = thrust::raw_pointer_cast(back_voxelValueCounts.data());
	//auto tVN = thrust::raw_pointer_cast(back_voxelNormals.data());
	//auto tVC = thrust::raw_pointer_cast(back_voxelColors.data());
	//auto tVCS = thrust::raw_pointer_cast(back_voxelColorScores.data());

	uint32_t* d_count; // 디바이스에서 카운트를 저장할 포인터
	uint32_t h_count = 0; // 호스트 카운트 값
			//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	cudaMalloc(&d_count, sizeof(uint32_t));
	cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st);


	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_used_buff.begin(),
		m_MC_used_buff.begin() + h_usedSize,
		[d_count,
		oTable, oVV, oVVC, oVN, oVC, oVCS, oVSG, oVEA,
		tTable, tVV, tVVC, tVN, tVC, tVCS, tVSG, tVEA
		]__device__(unsigned int& index) {
		if (
			FLT_VALID(oVV[index]) &&
			USHORT_VALID(oVVC[index]) &&
			FLT_VALID(oVN[index].x()) &&
			FLT_VALID(oVN[index].y()) &&
			FLT_VALID(oVN[index].z())
			)
		{
			uint32_t _prev_table_ = atomicAdd(d_count, 1);
			/*printf("[%d] VV = %f, VVC = %d, VN = %f, %f, %f, VC = %d, %d, %d, VCS = %f\n",
				_prev_table_,
				oVV[index],
				oVVC[index],
				oVN[index].x(), oVN[index].y(), oVN[index].z(),
				oVC[index].x(), oVC[index].y(), oVC[index].z(),
				oVCS[index]
				);*/
			tTable[_prev_table_] = oTable[index];
			tVV[_prev_table_] = oVV[index];
			tVVC[_prev_table_] = oVVC[index];
			tVN[_prev_table_] = oVN[index];
			tVC[_prev_table_] = oVC[index];
			tVCS[_prev_table_] = oVCS[index];
			tVSG[_prev_table_] = oVSG[index];
			tVEA[_prev_table_] = oVEA[index];
		}
	});
	checkCudaSync(st);
	cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	cudaMemcpyAsync(*repos_hBack_table, tTable, h_count * sizeof(HashEntry), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VV, tVV, h_count * sizeof(voxel_value_t), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VVC, tVVC, h_count * sizeof(unsigned short), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VN, tVN, h_count * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VC, tVC, h_count * sizeof(Eigen::Vector3b), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VCS, tVCS, h_count * sizeof(float), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VSG, tVSG, h_count * sizeof(char), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(*repos_hBack_VEA, tVEA, h_count * sizeof(VoxelExtraAttrib), cudaMemcpyDeviceToHost, st);

	//repos_hBack_allocSize = h_usedSize;
	repos_hBack_usedSize = h_count;



	if (0) //hash 珥湲고 
	{
		resetHashTable(globalHash_info_host, st);

		auto executionPolicy = thrust::cuda::par_nosync.on(st);
		thrust::fill(executionPolicy, voxel_value_repos.begin(), voxel_value_repos.end(), FLT_MAX);// VOXEL_INVALID);
		thrust::fill(executionPolicy, voxel_valueCnt_repos.begin(), voxel_valueCnt_repos.end(), 0);
		thrust::fill(executionPolicy, voxel_nms_repos.begin(), voxel_nms_repos.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));
		thrust::fill(executionPolicy, voxel_clrs_repos.begin(), voxel_clrs_repos.end(), Eigen::Vector3b(0, 0, 0));
		thrust::fill(executionPolicy, voxel_clrScores_repos.begin(), voxel_clrScores_repos.end(), 0.0f);
		thrust::fill(executionPolicy, voxel_segmentations_repos.begin(), voxel_segmentations_repos.end(), 0);
		VoxelExtraAttrib emptyExtraAttrib = { 0 };
		thrust::fill(executionPolicy, voxel_extraAttribs_repos.begin(), voxel_extraAttribs_repos.end(), emptyExtraAttrib);
	}


	//thrust::device_free(backashtable);
	//thrust::device_free(back_voxelValues);
	//thrust::device_free(back_voxelValueCounts);
	//thrust::device_free(back_voxelNormals);
	//thrust::device_free(back_voxelColors);
	//thrust::device_free(back_voxelColorScores);

	//checkCudaSync(st);

	cudaDeviceSynchronize();

	cudaFree(tTable);
	cudaFree(tVV);
	cudaFree(tVVC);
	cudaFree(tVN);
	cudaFree(tVC);
	cudaFree(tVCS);
	cudaFree(tVSG);
	cudaFree(tVEA);
	cudaFree(d_count);

	qDebug("\n [2] resultCheck Size = %d", h_count);

	return h_count;
}

uint32_t    MarchingCubes::restoreHashTable(VoxelHashBackupInfo & voxelBackupInfo, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return 0;

	/*size_t& repos_hBack_usedSize = regModule->m_h_usedSize[iDataType];

	HashEntry* hTable = regModule->m_h_hashtable[iDataType];
	voxel_value_t* hVV = regModule->m_h_voxelValues[iDataType];
	unsigned short* hVVC = regModule->m_h_voxelValueCounts[iDataType];
	Eigen::Vector3f* hVN = regModule->m_h_voxelNormals[iDataType];
	Eigen::Vector3b* hVC = regModule->m_h_voxelColors[iDataType];
	float* hVCS = regModule->m_h_voxelColorScores[iDataType];*/

	size_t& repos_hBack_usedSize = voxelBackupInfo.usedSize;

	HashEntry** hTable = &voxelBackupInfo.hashtable;
	voxel_value_t** hVV = &voxelBackupInfo.voxelValues;
	unsigned short** hVVC = &voxelBackupInfo.voxelValueCounts;
	Eigen::Vector3f** hVN = &voxelBackupInfo.voxelNormals;
	Eigen::Vector3b** hVC = &voxelBackupInfo.voxelColors;
	float** hVCS = &voxelBackupInfo.voxelColorScores;
	char** hVSG = &voxelBackupInfo.voxelSegmentations;
	VoxelExtraAttrib** hVEA = &voxelBackupInfo.voxelExtraAttribs;

	//size_t													back_usedSize;
	//thrust::device_ptr<uint64_t>							backashtable = thrust::device_malloc<uint64_t>(repos_hBack_usedSize);
	//thrust::device_ptr<voxel_value_t>						back_voxelValues = thrust::device_malloc<voxel_value_t>(repos_hBack_usedSize);
	//thrust::device_ptr<unsigned short>					back_voxelValueCounts = thrust::device_malloc<unsigned short>(repos_hBack_usedSize);
	//thrust::device_ptr<Eigen::Vector3f>					back_voxelNormals = thrust::device_malloc<Eigen::Vector3f>(repos_hBack_usedSize);
	//thrust::device_ptr<Eigen::Vector3b>	back_voxelColors = thrust::device_malloc<Eigen::Vector3b>(repos_hBack_usedSize);
	//thrust::device_ptr<float>								back_voxelColorScores = thrust::device_malloc<float>(repos_hBack_usedSize);

	thrust::device_vector<HashEntry>			backashtable(repos_hBack_usedSize);
	thrust::device_vector<voxel_value_t>		back_voxelValues(repos_hBack_usedSize);
	thrust::device_vector<unsigned short>		back_voxelValueCounts(repos_hBack_usedSize);
	thrust::device_vector<Eigen::Vector3f>		back_voxelNormals(repos_hBack_usedSize);
	thrust::device_vector<Eigen::Vector3b>		back_voxelColors(repos_hBack_usedSize);
	thrust::device_vector<float>				back_voxelColorScores(repos_hBack_usedSize);
	thrust::device_vector<char>					back_voxelSegmentations(repos_hBack_usedSize);
	thrust::device_vector<VoxelExtraAttrib>		back_voxelExtraAttribs(repos_hBack_usedSize);

	auto tTable = thrust::raw_pointer_cast(backashtable.data());
	auto tVV = thrust::raw_pointer_cast(back_voxelValues.data());
	auto tVVC = thrust::raw_pointer_cast(back_voxelValueCounts.data());
	auto tVN = thrust::raw_pointer_cast(back_voxelNormals.data());
	auto tVC = thrust::raw_pointer_cast(back_voxelColors.data());
	auto tVCS = thrust::raw_pointer_cast(back_voxelColorScores.data());
	auto tVSG = thrust::raw_pointer_cast(back_voxelSegmentations.data());
	auto tVEA = thrust::raw_pointer_cast(back_voxelExtraAttribs.data());

	cudaMemcpyAsync(tTable, *hTable, repos_hBack_usedSize * sizeof(HashEntry), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVV, *hVV, repos_hBack_usedSize * sizeof(voxel_value_t), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVVC, *hVVC, repos_hBack_usedSize * sizeof(unsigned short), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVN, *hVN, repos_hBack_usedSize * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVC, *hVC, repos_hBack_usedSize * sizeof(Eigen::Vector3b), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVCS, *hVCS, repos_hBack_usedSize * sizeof(float), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVSG, *hVSG, repos_hBack_usedSize * sizeof(char), cudaMemcpyHostToDevice, st);
	cudaMemcpyAsync(tVEA, *hVEA, repos_hBack_usedSize * sizeof(VoxelExtraAttrib), cudaMemcpyHostToDevice, st);

	thrust::device_vector<voxel_value_t>& voxel_value_repos = m_MC_voxelValues;
	thrust::device_vector<unsigned short>& voxel_valueCnt_repos = m_MC_voxelValueCounts;
	thrust::device_vector<Eigen::Vector3f>& voxel_nms_repos = m_MC_voxelNormals;
	thrust::device_vector<Eigen::Vector3b>& voxel_clrs_repos = m_MC_voxelColors;
	thrust::device_vector<float>& voxel_clrScores_repos = m_MC_voxelColorScores;
	thrust::device_vector<char>& voxel_segmentations_repos = m_MC_voxelSegmentations;
	thrust::device_vector<VoxelExtraAttrib>& voxel_extraAttribs_repos = m_MC_voxelExtraAttribs;

	auto dVV = thrust::raw_pointer_cast(voxel_value_repos.data());
	auto dVVC = thrust::raw_pointer_cast(voxel_valueCnt_repos.data());
	auto dVN = thrust::raw_pointer_cast(voxel_nms_repos.data());
	auto dVC = thrust::raw_pointer_cast(voxel_clrs_repos.data());
	auto dVCS = thrust::raw_pointer_cast(voxel_clrScores_repos.data());
	auto dVSG = thrust::raw_pointer_cast(voxel_segmentations_repos.data());
	auto dVEA = thrust::raw_pointer_cast(voxel_extraAttribs_repos.data());

	auto _globalHash_info = globalHash_info;
	auto _globalHash = globalHash_info_host->hashtable;
	//auto _globalHash_value = globalHash_value;

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		thrust::make_counting_iterator<size_t>(0),
		thrust::make_counting_iterator<size_t>(repos_hBack_usedSize),
		[_globalHash, _globalHash_info,
		tTable,
		tVV, tVVC, tVN, tVC, tVCS, tVSG, tVEA,
		dVV, dVVC, dVN, dVC, dVCS, dVSG, dVEA
		]__device__(size_t index) {
		//printf("%llu\n", tTable[index]);
		auto hashSlot_idx = _globalHash_info->get_insert_idx_func64_v4(tTable[index]);
		dVV[hashSlot_idx] = tVV[index];
		dVVC[hashSlot_idx] = tVVC[index];
		dVN[hashSlot_idx] = tVN[index];
		dVC[hashSlot_idx] = tVC[index];
		dVCS[hashSlot_idx] = tVCS[index];
		dVSG[hashSlot_idx] = tVSG[index];
		dVEA[hashSlot_idx] = tVEA[index];
	}
	);


	uint32_t h_usedSize = 0;

	{ /// hash 복원 결과 사이즈 확인
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_hashtable64_v2, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;


		pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

		kernel_iterateChkUsed_voxelCnt_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (globalHash_info);
		checkCudaErrors(cudaGetLastError());
		cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

		if (0) {
			uint32_t* ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff.data());

			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2, 0, 0));
			int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

			pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

			kernel_iterateChkUsed_voxelCnt_wUsed_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (globalHash_info, ext_used_Key);
			checkCudaErrors(cudaGetLastError());
			cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

			thrust::for_each(
				thrust::cuda::par_nosync(*alloc_).on(st),
				m_MC_used_buff.begin(),
				m_MC_used_buff.begin() + 50,
				[
					dVV, dVVC, dVN, dVC, dVCS, dVSG
				]__device__(unsigned int& index) {
				printf(" kk  VV = %f, VVC = %d, VN = %f, %f, %f, VC = %d, %d, %d, VCS = %f VSG = %d \n",
					dVV[index],
					dVVC[index],
					dVN[index].x(), dVN[index].y(), dVN[index].z(),
					dVC[index].x(), dVC[index].y(), dVC[index].z(),
					dVCS[index],
					dVSG[index]
				);
			});
		}
		checkCudaSync(st);
	}

	/*	thrust::device_free(backashtable);
		thrust::device_free(back_voxelValues);
		thrust::device_free(back_voxelValueCounts);
		thrust::device_free(back_voxelNormals);
		thrust::device_free(back_voxelColors);
		thrust::device_free(back_voxelColorScores);*/

	qDebug(" \n[1] restoreHashTable usedCheck Size = %d", h_usedSize);
	//SaveVoxelValues(GetSaveDataFolderPath() + "\\VoxelValues.ply", alloc_, st);
	return h_usedSize;
}

uint32_t	MarchingCubes::resetHashTable_voxel(CUstream_st * st) {
	if (!globalVoxelValuesInitialized) return 0;

	resetHashTable(globalHash_info_host, st);

	auto executionPolicy = thrust::cuda::par_nosync.on(st);
	thrust::fill(executionPolicy, m_MC_voxelValues.begin(), m_MC_voxelValues.end(), FLT_MAX);// VOXEL_INVALID);
	//mscho	@20250228
	thrust::fill(executionPolicy, m_MC_voxelValueCounts.begin(), m_MC_voxelValueCounts.end(), USHRT_MAX);

	thrust::fill(executionPolicy, m_MC_voxelNormals.begin(), m_MC_voxelNormals.end(), Eigen::Vector3f(0.0f, 0.0f, 0.0f));
	thrust::fill(executionPolicy, m_MC_voxelColors.begin(), m_MC_voxelColors.end(), Eigen::Vector3b(0, 0, 0));
	thrust::fill(executionPolicy, m_MC_voxelColorScores.begin(), m_MC_voxelColorScores.end(), 0.0f);
	VoxelExtraAttrib emptyExtraAttrib = { 0 };
	thrust::fill(executionPolicy, m_MC_voxelExtraAttribs.begin(), m_MC_voxelExtraAttribs.end(), emptyExtraAttrib);
	thrust::fill(executionPolicy, m_MC_voxelSegmentations.begin(), m_MC_voxelSegmentations.end(), 0);

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize;

	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateChkUsed_voxelCnt_hashtable64_v2, 0, 0));
	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	kernel_iterateChkUsed_voxelCnt_hashtable64_v2 << <gridsize, threadblocksize, 0, st >> > (globalHash_info);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpyAsync(&h_usedSize, &globalHash_info->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	return h_usedSize;
}

#ifdef USE_MESH_BASE
uint32_t	MarchingCubes::resetHashTable_triangle(CUstream_st * st) {
	if (!globalVoxelValuesInitialized) return 0;
	thrust::device_vector< Eigen::Vector<uint32_t, 3>>& voxel_tri_repos = pRegistration->m_MC_triangles;
	resetHashTable(hInfo_global_tri_host, hTable_global_tri, hTable_global_tri_value, st);

	thrust::fill(thrust::cuda::par_nosync.on(st), voxel_tri_repos.begin(), voxel_tri_repos.end(), Eigen::Vector<uint32_t, 3>(UINT_MAX, UINT_MAX, UINT_MAX));


	pHashManager->reset_hashKeyCountValue64(hInfo_global_tri, st);

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize;

	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateUsedCntOnly_triangle_hashtable64_v1, 0, 0));
	int gridsize = ((uint32_t)hInfo_global_tri_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	kernel_iterateUsedCntOnly_triangle_hashtable64_v1 << <gridsize, threadblocksize, 0, st >> > (
		hInfo_global_tri
		, hTable_global_tri
		, hTable_global_tri_value
		, thrust::raw_pointer_cast(voxel_tri_repos.data())
		);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpyAsync(&h_usedSize, &hInfo_global_tri->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	return h_usedSize;
}

uint32_t	MarchingCubes::resetHashTable_vertex(CUstream_st * st) {
	if (!globalVoxelValuesInitialized) return 0;
	thrust::device_vector<Eigen::Vector3f>& vtx_pos_repos = pRegistration->m_MC_VTX_pos;
	thrust::device_vector<Eigen::Vector3f>& vtx_nm_repos = pRegistration->m_MC_VTX_nm;
	thrust::device_vector<Eigen::Vector3b>& vtx_color_repos = pRegistration->m_MC_VTX_color;
	thrust::device_vector<uint32_t>& vtx_dupCnt_repos = pRegistration->m_MC_VTX_dupCnt;

	resetHashTable(hInfo_global_vtx_host, hTable_global_vtx, hTable_global_vtx_value, st);

	thrust::fill(thrust::cuda::par_nosync.on(st), vtx_pos_repos.begin(), vtx_pos_repos.end(), Eigen::Vector3f(0.f, 0.f, 0.f));
	thrust::fill(thrust::cuda::par_nosync.on(st), vtx_nm_repos.begin(), vtx_nm_repos.end(), Eigen::Vector3f(0.f, 0.f, 0.f));
	thrust::fill(thrust::cuda::par_nosync.on(st), vtx_color_repos.begin(), vtx_color_repos.end(), Eigen::Vector3b(0, 0, 0));
	thrust::fill(thrust::cuda::par_nosync.on(st), vtx_dupCnt_repos.begin(), vtx_dupCnt_repos.end(), (uint32_t)0);

	pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);

	int mingridsize;
	int threadblocksize;
	uint32_t h_usedSize;

	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, kernel_iterateUsedCntOnly_vtx_hashtable64_v1, 0, 0));
	int gridsize = ((uint32_t)hInfo_global_vtx_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	kernel_iterateUsedCntOnly_vtx_hashtable64_v1 << <gridsize, threadblocksize, 0, st >> > (
		hInfo_global_vtx
		, hTable_global_vtx
		, hTable_global_vtx_value
		);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpyAsync(&h_usedSize, &hInfo_global_vtx->Count_HashTableUsed, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	return h_usedSize;
}
#endif

__global__ void Kernel_GatherOccupiedVoxelIndices(
	MarchingCubes::ExecutionInfo info,
	unsigned int* output
	, uint32_t block_size
	, uint32_t block_capacity
	, uint32_t block_data_last
	, uint32_t * Count_AllUsedSize
	, unsigned long long* AllUsedCacheArray
)
{
	if (nullptr == info.gridSlotIndexCache) return;

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > block_capacity - 1) return;

	//printf("threadid : %d\n", threadid);

	uint32_t    block_base = threadid * block_size;
	uint32_t    block_data_count;

	if (threadid == block_capacity - 1 && block_data_last != 0)
		block_data_count = block_data_last;
	else
		block_data_count = block_size;


	for (uint32_t i = 0; i < block_data_count; i++)
	{
		uint32_t    slot_idx = block_base + i;
		auto hs_key = info.globalHash[slot_idx];
		if (hs_key.Exists() && (info.globalHash[slot_idx].value != kEmpty8))
		{
			//	mscho	@20240523
			auto voxel_value = info.voxelValues[slot_idx];
			auto voxel_count = info.voxelValueCounts[slot_idx];

			if (FLT_VALID(voxel_value) && USHORT_VALID(voxel_count) && voxel_count != 0)
			{
				voxel_value /= (float)voxel_count;
				if (voxel_value < -0.25 || voxel_value > +0.25)	continue;

				auto x_idx = hs_key.x;
				auto y_idx = hs_key.y;
				auto z_idx = hs_key.z;

				if (x_idx < info.cache.localMinGlobalIndexX ||
					y_idx < info.cache.localMinGlobalIndexY ||
					z_idx < info.cache.localMinGlobalIndexZ) continue;

				//shshin @240417 margin 1->0 : Extract 들어오기전에 이미  margin크기만큼 localMinMax를 수정함.
				if (false == info.local.ContainsWithMargin(x_idx, y_idx, z_idx, 1))
				{
					continue;
				}
				else if (true == info.local.ContainsWithMargin(x_idx, y_idx, z_idx))
				{
					uint32_t _prev_table_ = atomicAdd(&(info.globalHashInfo->Count_HashTableUsed), 1);
					output[_prev_table_] = slot_idx;
				}

				auto xCacheIndex = x_idx - info.cache.localMinGlobalIndexX;
				auto yCacheIndex = y_idx - info.cache.localMinGlobalIndexY;
				auto zCacheIndex = z_idx - info.cache.localMinGlobalIndexZ;

				auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
					yCacheIndex * info.cache.voxelCountX + xCacheIndex;

				if (cacheIndex < (size_t)info.cache.voxelCount)
				{
					info.gridSlotIndexCache[cacheIndex] = slot_idx;
					uint32_t Idx = atomicAdd(Count_AllUsedSize, 1);
					AllUsedCacheArray[Idx] = cacheIndex;
				}
			}
		}
	}
}
#endif

//	mscho	@20240523
//	데이타를 모으는 구간을 입력받아서, 처리하도록 한다.
__global__ void Kernel_GatherOccupiedVoxelIndices_v2(
	MarchingCubes::ExecutionInfo info
	, unsigned int* output
	, float		min_tsdf
	, float		max_tsdf
	, uint32_t block_size
	, uint32_t block_capacity
	, uint32_t block_data_last
	, uint32_t * Count_AllUsedSize
	, unsigned long long* AllUsedCacheArray
)
{
	if (nullptr == info.gridSlotIndexCache) return;

#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > block_capacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < block_capacity; threadid++) {
#endif

		uint32_t    block_base = threadid * block_size;
		uint32_t    block_data_count;

		if (threadid == block_capacity - 1 && block_data_last != 0)
			block_data_count = block_data_last;
		else
			block_data_count = block_size;


		for (uint32_t i = 0; i < block_data_count; i++)
		{
			uint32_t    slot_idx = block_base + i;
			auto& hs_key = info.globalHash[slot_idx];

			if ((hs_key.Exists()) && (info.globalHash[slot_idx].value != kEmpty8))
			{
				//	mscho	@20240523
				auto voxel_value = info.voxelValues[slot_idx];
				auto voxel_count = info.voxelValueCounts[slot_idx];

				//	mscho	@20240524
				if (FLT_VALID(voxel_value) && fabs(voxel_value) < 2500.f && USHORT_VALID(voxel_count) && voxel_count != 0)
				{
					voxel_value /= (float)voxel_count;
					if (voxel_value < min_tsdf || voxel_value > max_tsdf) 	continue;

					auto x_idx = hs_key.x;
					auto y_idx = hs_key.y;
					auto z_idx = hs_key.z;

					if (x_idx < info.cache.localMinGlobalIndexX ||
						y_idx < info.cache.localMinGlobalIndexY ||
						z_idx < info.cache.localMinGlobalIndexZ) continue;

					//shshin @240417 margin 1->0 : Extract 들어오기전에 이미  margin크기만큼 localMinMax를 수정함.
					if (false == info.local.ContainsWithMargin(x_idx, y_idx, z_idx, 1))
					{
						continue;
					}
					else if (true == info.local.ContainsWithMargin(x_idx, y_idx, z_idx))
					{
						uint32_t _prev_table_ = atomicAdd(&(info.globalHashInfo->Count_HashTableUsed), (uint32_t)1);
						output[_prev_table_] = slot_idx;
					}

					auto xCacheIndex = x_idx - info.cache.localMinGlobalIndexX;
					auto yCacheIndex = y_idx - info.cache.localMinGlobalIndexY;
					auto zCacheIndex = z_idx - info.cache.localMinGlobalIndexZ;

					auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
						yCacheIndex * info.cache.voxelCountX + xCacheIndex;
					if (cacheIndex < (size_t)info.cache.voxelCount)
					{
						info.gridSlotIndexCache[cacheIndex] = slot_idx;
						uint32_t Idx = atomicAdd(Count_AllUsedSize, (uint32_t)1);
						AllUsedCacheArray[Idx] = cacheIndex;
					}
				}

			}
		}
	}
	}

//	mscho	@20240717
//  내부 for문을 제거하고, 전체를 kernel로 돌리는 함수로 변경한다.
//	데이타를 모으는 구간을 입력받아서, 처리하도록 한다.
//	내부 로직이 복잡해지면, Loop를 돌지 않는 것이 속도상에서 유리한 것으로 보인다.
__global__ void Kernel_GatherOccupiedVoxelIndices_v3(
	MarchingCubes::ExecutionInfo info
	, unsigned int* output
	, float		min_tsdf
	, float		max_tsdf
	, uint32_t block_size
	, uint32_t block_capacity
	, uint32_t block_data_last
	, uint32_t * Count_AllUsedSize
	, unsigned long long* AllUsedCacheArray
	, int margin
)
{
	if (nullptr == info.gridSlotIndexCache) return;
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > block_capacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 16384)
	for (int threadid = 0; threadid < block_capacity; threadid++) {
#endif

		uint32_t    slot_idx = threadid;
		auto& hs_key = info.globalHash[slot_idx];

		if (hs_key.Exists() && (info.globalHash[slot_idx].value != kEmpty8))
		{
			//	mscho	@20240523
			auto voxel_value = info.voxelValues[slot_idx];
			auto voxel_count = VOXELCNT_VALUE(info.voxelValueCounts[slot_idx]);

			//	mscho	@20240524
			if (FLT_VALID(voxel_value) && fabs(voxel_value) < 2500.f && USHORT_VALID(voxel_count) && voxel_count != 0)
			{
				voxel_value /= (float)voxel_count;
				if (voxel_value < min_tsdf || voxel_value > max_tsdf)
					kernel_return;

				auto x_idx = hs_key.x;
				auto y_idx = hs_key.y;
				auto z_idx = hs_key.z;

				if (x_idx < info.cache.localMinGlobalIndexX ||
					y_idx < info.cache.localMinGlobalIndexY ||
					z_idx < info.cache.localMinGlobalIndexZ)
					kernel_return;

				//shshin @240417 margin 1->0 : Extract 들어오기전에 이미  margin크기만큼 localMinMax를 수정함.
				//	mscho	@20240805
				if (false == info.local.ContainsWithMargin_v2(x_idx, y_idx, z_idx, 0))
				{
					kernel_return;
				}
				//	mscho	@20240805
				//	3차원 Voxel에서 voxel 중심으로부터 가장 멀어질 수 있는 값이 value의 최대 상하한 값이므로
				//  +0.0866 / -0.0866 : 이내의 값을 가지고 있는 voxel들만이, point가 추출될 수 있는 voxel이다..
				//	이 범위를 벗어나는 voxel은 zero crossing이 발생하지 않거나, 1개 이상의 범위에서 발생한다
				//	pjkang: 하드코딩된 0.0866 제거 - 모든 누적 복셀 추출을 위해 TSDF 범위만 사용
				else if (true == info.local.ContainsWithMargin_v2(x_idx, y_idx, z_idx, margin * -1))
				{
					// TSDF 범위 체크는 이미 위에서 수행되었으므로 추가 체크 없이 포함
					{
						uint32_t _prev_table_ = atomicAdd(&(info.globalHashInfo->Count_HashTableUsed), (uint32_t)1);
						output[_prev_table_] = slot_idx;
					}
				}

				auto xCacheIndex = x_idx - info.cache.localMinGlobalIndexX;
				auto yCacheIndex = y_idx - info.cache.localMinGlobalIndexY;
				auto zCacheIndex = z_idx - info.cache.localMinGlobalIndexZ;

				auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
					yCacheIndex * info.cache.voxelCountX + xCacheIndex;
				if (cacheIndex < (size_t)info.cache.voxelCount)
				{
					info.gridSlotIndexCache[cacheIndex] = slot_idx;
					uint32_t Idx = atomicAdd(Count_AllUsedSize, (uint32_t)1);
					AllUsedCacheArray[Idx] = cacheIndex;
				}
			}
		}
	}
	}


//	mscho	@20250714
//	kernel thread갯수가 너무 많아서, block으로 나눠서, 실행하는 버젼을 만들었다.
//	다른 thread가 동시에 실행되기 어려워서, 나누어 놓았다.
//	빠르게 실행하고, 빠질 수 있는 thread가 동시에 실행 될 수 있도록
__global__ void Kernel_GatherOccupiedVoxelIndices_v4(
	MarchingCubes::ExecutionInfo info
	, unsigned int* output
	, float		min_tsdf
	, float		max_tsdf
	, uint32_t block_size
	, uint32_t block_capacity
	, uint32_t block_n
	, uint32_t * Count_AllUsedSize
	, unsigned long long* AllUsedCacheArray
	, int margin
)
{
	if (nullptr == info.gridSlotIndexCache) return;
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > block_capacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 16384)
	for (int threadid = 0; threadid < block_capacity; threadid++) {
#endif

		uint32_t    slot_idx = threadid + block_size * block_n;
		auto& hs_key = info.globalHash[slot_idx];

		if (hs_key.Exists() && (info.globalHash[slot_idx].value != kEmpty8))
		{
			//	mscho	@20240523
			auto voxel_value = info.voxelValues[slot_idx];
			auto voxel_count = VOXELCNT_VALUE(info.voxelValueCounts[slot_idx]);

			//	mscho	@20240524
			if (FLT_VALID(voxel_value) && fabs(voxel_value) < 2500.f && USHORT_VALID(voxel_count) && voxel_count != 0)
			{
				voxel_value /= (float)voxel_count;
				if (voxel_value < min_tsdf || voxel_value > max_tsdf)
					kernel_return;

				auto x_idx = hs_key.x;
				auto y_idx = hs_key.y;
				auto z_idx = hs_key.z;

				if (x_idx < info.cache.localMinGlobalIndexX ||
					y_idx < info.cache.localMinGlobalIndexY ||
					z_idx < info.cache.localMinGlobalIndexZ)
					kernel_return;

				//shshin @240417 margin 1->0 : Extract 들어오기전에 이미  margin크기만큼 localMinMax를 수정함.
				//	mscho	@20240805
				if (false == info.local.ContainsWithMargin_v2(x_idx, y_idx, z_idx, 0))
				{
					kernel_return;
				}
				//	mscho	@20240805
				//	3차원 Voxel에서 voxel 중심으로부터 가장 멀어질 수 있는 값이 value의 최대 상하한 값이므로
				//  +0.0866 / -0.0866 : 이내의 값을 가지고 있는 voxel들만이, point가 추출될 수 있는 voxel이다..
				//	이 범위를 벗어나는 voxel은 zero crossing이 발생하지 않거나, 1개 이상의 범위에서 발생한다
				//	pjkang: 하드코딩된 0.0866 제거 - 모든 누적 복셀 추출을 위해 TSDF 범위만 사용
				else if (true == info.local.ContainsWithMargin_v2(x_idx, y_idx, z_idx, margin * -1))
				{
					// TSDF 범위 체크는 이미 위에서 수행되었으므로 추가 체크 없이 포함
					{
						uint32_t _prev_table_ = atomicAdd(&(info.globalHashInfo->Count_HashTableUsed), (uint32_t)1);
						output[_prev_table_] = slot_idx;
					}
				}

				auto xCacheIndex = x_idx - info.cache.localMinGlobalIndexX;
				auto yCacheIndex = y_idx - info.cache.localMinGlobalIndexY;
				auto zCacheIndex = z_idx - info.cache.localMinGlobalIndexZ;

				auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
					yCacheIndex * info.cache.voxelCountX + xCacheIndex;
				if (cacheIndex < (size_t)info.cache.voxelCount)
				{
					info.gridSlotIndexCache[cacheIndex] = slot_idx;
					uint32_t Idx = atomicAdd(Count_AllUsedSize, (uint32_t)1);
					AllUsedCacheArray[Idx] = cacheIndex;
				}
			}
		}
	}
	}

#ifndef BUILD_FOR_CPU
__device__ int64_t make_int64(float dist, int index) {
	int scaled_dist = static_cast<int>(dist * 1e6);  // 스케일링 팩터를 1e6으로 설정하여 정밀도를 높임
	return (static_cast<int64_t>(scaled_dist) << 32) | static_cast<int64_t>(index);
}
__device__ unsigned long long int make_ullint(float dist, int index) {
	int scaled_dist = static_cast<int>(dist * 1e6);  // 스케일링 팩터를 1e6으로 설정하여 정밀도를 높임
	return (static_cast<unsigned long long int>(scaled_dist) << 32) | static_cast<unsigned long long int>(index);
}

__device__ float extract_distance(int64_t value) {
	int scaled_dist = static_cast<int>(value >> 32);
	return static_cast<float>(scaled_dist) / 1e6;
}
__device__ float extract_distance(unsigned long long int value) {
	int scaled_dist = static_cast<int>(value >> 32);
	return static_cast<float>(scaled_dist) / 1e6;
}

__device__ int extract_index(int64_t value) {
	return static_cast<int>(value & 0xFFFFFFFF);
}

__device__ int extract_index(unsigned long long int value) {
	return static_cast<int>(value & 0xFFFFFFFF);
}
__global__ void restoreSearchResult_v1
(
	int size,
	int* indices,
	float* distance,
	int64_t * result
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > size - 1) return;

	indices[threadid] = extract_index(result[threadid]);
	if (indices[threadid] == -1)
	{
		distance[threadid] = std::numeric_limits<float>::infinity();
	}
	else
	{
		float dist = extract_distance(result[threadid]);
		distance[threadid] = powf(dist, 2.0f);
	}
	//indices[threadid] = (uint32_t)(result[threadid] & 0xFFFFFFFF);
	//
	//int64_t extractedDistanceInt = (result[threadid] >> 32);
	//distance[threadid] = powf((float)extractedDistanceInt / 10000000.0f, 2.0f);
}


__global__ void searchCacheVoxelKernel_v1(
	Eigen::Vector3f * inputPoints,
	Eigen::Vector3f * targetPoints,
	unsigned int* gridSlotIndexCache,
	float voxelSize,
	float3 minBound,
	int3 voxelCount,
	float radius,
	int size,
	int range,
	int64_t * result,
	int3 localMinGlobalIndex)
{
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalSearchPoints = (2 * range + 1) * (2 * range + 1) * (2 * range + 1);

	if (globalThreadId >= size * totalSearchPoints) return;

	int pointIdx = globalThreadId / totalSearchPoints;
	int searchPointIdx = globalThreadId % totalSearchPoints;

	int dz = searchPointIdx / ((2 * range + 1) * (2 * range + 1)) - range;
	int dy = (searchPointIdx / (2 * range + 1)) % (2 * range + 1) - range;
	int dx = searchPointIdx % (2 * range + 1) - range;


	Eigen::Vector3f pointPos = inputPoints[pointIdx];

	auto xGlobalIndex = (size_t)floorf((pointPos.x() - minBound.x) / voxelSize);
	auto yGlobalIndex = (size_t)floorf((pointPos.y() - minBound.y) / voxelSize);
	auto zGlobalIndex = (size_t)floorf((pointPos.z() - minBound.z) / voxelSize);

	auto xCacheIndex = (int)xGlobalIndex - localMinGlobalIndex.x;
	auto yCacheIndex = (int)yGlobalIndex - localMinGlobalIndex.y;
	auto zCacheIndex = (int)zGlobalIndex - localMinGlobalIndex.z;

	int zIdx = zCacheIndex + dz;
	int yIdx = yCacheIndex + dy;
	int xIdx = xCacheIndex + dx;

	if (zIdx < 0 || zIdx >= voxelCount.z || yIdx < 0 || yIdx >= voxelCount.y || xIdx < 0 || xIdx >= voxelCount.x) return;

	auto cacheIndex = zIdx * voxelCount.x * voxelCount.y + yIdx * voxelCount.x + xIdx;
	unsigned int targetIdx = gridSlotIndexCache[cacheIndex];

	// gridSlotIndexCache 경계 검사
	if (cacheIndex < 0 || cacheIndex >= voxelCount.x * voxelCount.y * voxelCount.z) {
		printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		return;
	}

	if (targetIdx != UINT32_MAX) {
		//	mscho	@20250527
		//Eigen::Vector3f target = targetPoints[targetIdx];			
		//float dist = norm3df((pointPos.x() - target.x()), (pointPos.y() - target.y()), (pointPos.z() - target.z()));

		Eigen::Vector3d point_d = pointPos.cast<double>();
		Eigen::Vector3d target_d = targetPoints[targetIdx].cast<double>();
		//double  dist_d = norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		float  dist = (float)norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		if (dist < radius)
		{
			//dist *= 10000000.0f;
			//int64_t tmp = ((int64_t)dist << 32) + (int64_t)targetIdx;
			int64_t tmp = make_int64(dist, targetIdx);
			atomicMin(&result[pointIdx], tmp);
		}
	}
}

//	mscho	@20250619
__global__ void searchCacheVoxelKernel_v3(
	Eigen::Vector3f * inputPoints,
	Eigen::Vector3f * inputNormals,
	Eigen::Vector3f * targetPoints,
	Eigen::Vector3f * targetNormals,
	unsigned int* gridSlotIndexCache,
	float normal_dot_th,
	float voxelSize,
	float3 minBound,
	int3 voxelCount,
	float radius,
	int size,
	int range,
	int64_t * result,
	int3 localMinGlobalIndex
)
{
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalSearchPoints = (2 * range + 1) * (2 * range + 1) * (2 * range + 1);

	if (globalThreadId >= size * totalSearchPoints) return;
	int pointIdx = globalThreadId / totalSearchPoints;
	int searchPointIdx = globalThreadId % totalSearchPoints;

	int dz = searchPointIdx / ((2 * range + 1) * (2 * range + 1)) - range;
	int dy = (searchPointIdx / (2 * range + 1)) % (2 * range + 1) - range;
	int dx = searchPointIdx % (2 * range + 1) - range;


	Eigen::Vector3f pointPos = inputPoints[pointIdx];
	Eigen::Vector3f pointNor = inputNormals[pointIdx];

	auto xGlobalIndex = (size_t)floorf((pointPos.x() - minBound.x) / voxelSize);
	auto yGlobalIndex = (size_t)floorf((pointPos.y() - minBound.y) / voxelSize);
	auto zGlobalIndex = (size_t)floorf((pointPos.z() - minBound.z) / voxelSize);

	auto xCacheIndex = (int)xGlobalIndex - localMinGlobalIndex.x;
	auto yCacheIndex = (int)yGlobalIndex - localMinGlobalIndex.y;
	auto zCacheIndex = (int)zGlobalIndex - localMinGlobalIndex.z;

	int zIdx = zCacheIndex + dz;
	int yIdx = yCacheIndex + dy;
	int xIdx = xCacheIndex + dx;

	if (zIdx < 0 || zIdx >= voxelCount.z || yIdx < 0 || yIdx >= voxelCount.y || xIdx < 0 || xIdx >= voxelCount.x) return;

	auto cacheIndex = zIdx * voxelCount.x * voxelCount.y + yIdx * voxelCount.x + xIdx;
	unsigned int targetIdx = gridSlotIndexCache[cacheIndex];

	// gridSlotIndexCache 경계 검사
	if (cacheIndex < 0 || cacheIndex >= voxelCount.x * voxelCount.y * voxelCount.z) {
		printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		return;
	}

	if (targetIdx != UINT32_MAX) {
		//	mscho	@20250527
		//Eigen::Vector3f target = targetPoints[targetIdx];			
		//float dist = norm3df((pointPos.x() - target.x()), (pointPos.y() - target.y()), (pointPos.z() - target.z()));

		Eigen::Vector3d point_d = pointPos.cast<double>();
		Eigen::Vector3d target_d = targetPoints[targetIdx].cast<double>();

		//double  dist_d = norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		float  dist = (float)norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		if (dist < radius)
		{
			float axis_dot_pair = pointNor.dot(targetNormals[targetIdx]);
			//	source / target 의 두개의 포인트의 normal의 방향이 일정 각도 이내일때에만 pair 로 사용하도록 한다.
			//	이렇게 하면 얇은 면의 ICP 에서 반대면을 사용하거나.. noise 가 많은 source 의 경우에 이상하게 ICP되지 않을 수 있다

			if (axis_dot_pair > normal_dot_th)
			{
				int64_t tmp = make_int64(dist, targetIdx);
				atomicMin(&result[pointIdx], tmp);
			}
			//else
			//{
			//	//std::cout << pointNor << "...." << targetNormals[targetIdx] << "...." << axis_dot_pair << std::endl;
			//	printf("pointNor: (%f, %f, %f)....targetNormals[%d]: (%f, %f, %f)....axis_dot_pair: (%f  , threshold : %f)\n",
			//		pointNor.x(), pointNor.y(), pointNor.z(),
			//		targetIdx,
			//		targetNormals[targetIdx].x(), targetNormals[targetIdx].y(), targetNormals[targetIdx].z(),
			//		axis_dot_pair, normal_dot_th);
			//}
		}
	}
}


//	mscho	@20250704
__global__ void searchCacheVoxelKernel_v4(
	Eigen::Vector3f * inputPoints,
	Eigen::Vector3f * inputNormals,
	Eigen::Vector3f * targetPoints,
	Eigen::Vector3f * targetNormals,
	unsigned int* gridSlotIndexCache,
	float normal_dot_th,
	float voxelSize,
	float3 minBound,
	int3 voxelCount,
	float radius,
	int size,
	int range,
	const int	scale,
	int64_t * result,
	int3 localMinGlobalIndex
)
{
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalSearchPoints = (2 * range + 1) * (2 * range + 1) * (2 * range + 1);

	//	mscho	@20250708
	//if (radius > 0.3 && globalThreadId % scale != 0)	return;

	globalThreadId *= scale;
	if (globalThreadId >= size * totalSearchPoints) return;

	int pointIdx = globalThreadId / totalSearchPoints;
	int searchPointIdx = globalThreadId % totalSearchPoints;

	int dz = searchPointIdx / ((2 * range + 1) * (2 * range + 1)) - range;
	int dy = (searchPointIdx / (2 * range + 1)) % (2 * range + 1) - range;
	int dx = searchPointIdx % (2 * range + 1) - range;


	Eigen::Vector3f pointPos = inputPoints[pointIdx];
	Eigen::Vector3f pointNor = inputNormals[pointIdx];

	auto xGlobalIndex = (size_t)floorf((pointPos.x() - minBound.x) / voxelSize);
	auto yGlobalIndex = (size_t)floorf((pointPos.y() - minBound.y) / voxelSize);
	auto zGlobalIndex = (size_t)floorf((pointPos.z() - minBound.z) / voxelSize);

	auto xCacheIndex = (int)xGlobalIndex - localMinGlobalIndex.x;
	auto yCacheIndex = (int)yGlobalIndex - localMinGlobalIndex.y;
	auto zCacheIndex = (int)zGlobalIndex - localMinGlobalIndex.z;

	int zIdx = zCacheIndex + dz;
	int yIdx = yCacheIndex + dy;
	int xIdx = xCacheIndex + dx;

	if (zIdx < 0 || zIdx >= voxelCount.z || yIdx < 0 || yIdx >= voxelCount.y || xIdx < 0 || xIdx >= voxelCount.x) return;

	auto cacheIndex = zIdx * voxelCount.x * voxelCount.y + yIdx * voxelCount.x + xIdx;
	unsigned int targetIdx = gridSlotIndexCache[cacheIndex];

	// gridSlotIndexCache 경계 검사
	if (cacheIndex < 0 || cacheIndex >= voxelCount.x * voxelCount.y * voxelCount.z) {
		printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		return;
	}

	if (targetIdx != UINT32_MAX) {
		//	mscho	@20250527
		//Eigen::Vector3f target = targetPoints[targetIdx];			
		//float dist = norm3df((pointPos.x() - target.x()), (pointPos.y() - target.y()), (pointPos.z() - target.z()));

		Eigen::Vector3d point_d = pointPos.cast<double>();
		Eigen::Vector3d target_d = targetPoints[targetIdx].cast<double>();

		//double  dist_d = norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		float  dist = (float)norm3d((pointPos.x() - target_d.x()), (pointPos.y() - target_d.y()), (pointPos.z() - target_d.z()));
		if (dist < radius)
		{
			float axis_dot_pair = pointNor.dot(targetNormals[targetIdx]);
			//	source / target 의 두개의 포인트의 normal의 방향이 일정 각도 이내일때에만 pair 로 사용하도록 한다.
			//	이렇게 하면 얇은 면의 ICP 에서 반대면을 사용하거나.. noise 가 많은 source 의 경우에 이상하게 ICP되지 않을 수 있다

			if (axis_dot_pair > normal_dot_th)
			{
				int64_t tmp = make_int64(dist, targetIdx);
				atomicMin(&result[pointIdx], tmp);
			}
			//else
			//{
			//	//std::cout << pointNor << "...." << targetNormals[targetIdx] << "...." << axis_dot_pair << std::endl;
			//	printf("pointNor: (%f, %f, %f)....targetNormals[%d]: (%f, %f, %f)....axis_dot_pair: (%f  , threshold : %f)\n",
			//		pointNor.x(), pointNor.y(), pointNor.z(),
			//		targetIdx,
			//		targetNormals[targetIdx].x(), targetNormals[targetIdx].y(), targetNormals[targetIdx].z(),
			//		axis_dot_pair, normal_dot_th);
			//}
		}
	}
}

void MarchingCubes::searchCacheVoxel_v1(
	cached_allocator * alloc_, CUstream_st * st,
	thrust::device_vector<Eigen::Vector3f>&inputPoints,
	thrust::device_vector<Eigen::Vector3f>&targetPoints,
	size_t size,
	float radius,
	int max_nn,
	thrust::device_vector<int>&indices,
	thrust::device_vector<float>&distance,
	thrust::device_vector<int64_t>&result
)
{
	NvtxRangeCuda nvtxPrint("searchCache");
	thrust::fill(thrust::cuda::par_nosync.on(st), result.begin(), result.end(), std::numeric_limits<int64_t>::max());
	__printLastCudaError("0_searchCacheVoxel_v1", __FILE__, __LINE__);
	auto info = exeInfo;
	auto _globalScanAreaAABB = m_globalScanAreaAABB;
	auto voxelSize = m_voxelSize;

	float3 minBound = make_float3(_globalScanAreaAABB.min().x(), _globalScanAreaAABB.min().y(), _globalScanAreaAABB.min().z());
	int3 voxelCount = make_int3(info.cache.voxelCountX, info.cache.voxelCountY, info.cache.voxelCountZ);
	int3 localMinGlobalIndex = make_int3(info.cache.localMinGlobalIndexX, info.cache.localMinGlobalIndexY, info.cache.localMinGlobalIndexZ);

	int range = ceil(radius / voxelSize);

	int numBlocks = size * ((range * 2 + 1) * (range * 2 + 1) * (range * 2 + 1));

	__printLastCudaError("1_searchCacheVoxel_v1", __FILE__, __LINE__);
	searchCacheVoxelKernel_v1 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
		thrust::raw_pointer_cast(inputPoints.data()),
		thrust::raw_pointer_cast(targetPoints.data()),
		info.gridSlotIndexCache,
		voxelSize,
		minBound,
		voxelCount,
		radius,
		size,
		range,
		thrust::raw_pointer_cast(result.data()),
		localMinGlobalIndex
		);
	checkCudaSync(st);
	//qDebug("--------------- stepEnd ---------------");

	__printLastCudaError("2_searchCacheVoxel_v1", __FILE__, __LINE__);
	restoreSearchResult_v1 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (size
		, thrust::raw_pointer_cast(indices.data())
		, thrust::raw_pointer_cast(distance.data())
		, thrust::raw_pointer_cast(result.data())
		);
	__printLastCudaError("3_searchCacheVoxel_v1", __FILE__, __LINE__);
	return;
}

//	mscho	@20250619
void MarchingCubes::searchCacheVoxel_v3(
	cached_allocator * alloc_, CUstream_st * st,
	thrust::device_vector<Eigen::Vector3f>&inputPoints,
	thrust::device_vector<Eigen::Vector3f>&inputNormals,
	thrust::device_vector<Eigen::Vector3f>&targetPoints,
	thrust::device_vector<Eigen::Vector3f>&targetNormals,
	float	normal_dot_th,
	size_t size,
	float radius,
	int max_nn,
	thrust::device_vector<int>&indices,
	thrust::device_vector<float>&distance,
	thrust::device_vector<int64_t>&result
)
{
	NvtxRangeCuda nvtxPrint("searchCacheVoxelKernel");
	thrust::fill(thrust::cuda::par_nosync.on(st), result.begin(), result.end(), std::numeric_limits<int64_t>::max());
	if (FOR_TEST_PRINT)__printLastCudaError("0_searchCacheVoxel_v3", __FILE__, __LINE__);
	auto info = exeInfo;
	auto _globalScanAreaAABB = m_globalScanAreaAABB;
	auto voxelSize = m_voxelSize;

	float3 minBound = make_float3(_globalScanAreaAABB.min().x(), _globalScanAreaAABB.min().y(), _globalScanAreaAABB.min().z());
	int3 voxelCount = make_int3(info.cache.voxelCountX, info.cache.voxelCountY, info.cache.voxelCountZ);
	int3 localMinGlobalIndex = make_int3(info.cache.localMinGlobalIndexX, info.cache.localMinGlobalIndexY, info.cache.localMinGlobalIndexZ);

	int range = ceil(radius / voxelSize);

	int numBlocks = size * ((range * 2 + 1) * (range * 2 + 1) * (range * 2 + 1));

	if (FOR_TEST_PRINT)__printLastCudaError("1_searchCacheVoxel_v3", __FILE__, __LINE__);

	//	mscho	@20250704
	if (radius < 0.3)
	{
		searchCacheVoxelKernel_v3 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
			thrust::raw_pointer_cast(inputPoints.data()),
			thrust::raw_pointer_cast(inputNormals.data()),
			thrust::raw_pointer_cast(targetPoints.data()),
			thrust::raw_pointer_cast(targetNormals.data()),
			info.gridSlotIndexCache,
			normal_dot_th,
			voxelSize,
			minBound,
			voxelCount,
			radius,
			size,
			range,
			thrust::raw_pointer_cast(result.data()),
			localMinGlobalIndex
			);
	}
	else
	{
		//	mscho	@20250708
		int	scale = 8;
		numBlocks /= scale;
		searchCacheVoxelKernel_v4 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
			thrust::raw_pointer_cast(inputPoints.data()),
			thrust::raw_pointer_cast(inputNormals.data()),
			thrust::raw_pointer_cast(targetPoints.data()),
			thrust::raw_pointer_cast(targetNormals.data()),
			info.gridSlotIndexCache,
			normal_dot_th,
			voxelSize,
			minBound,
			voxelCount,
			radius,
			size,
			range,
			scale,
			thrust::raw_pointer_cast(result.data()),
			localMinGlobalIndex
			);
	}
	//	mscho	@20250708
	//checkCudaSync(st);
	//qDebug("--------------- stepEnd ---------------");

	if (FOR_TEST_PRINT)__printLastCudaError("2_searchCacheVoxel_v3", __FILE__, __LINE__);
	restoreSearchResult_v1 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (size
		, thrust::raw_pointer_cast(indices.data())
		, thrust::raw_pointer_cast(distance.data())
		, thrust::raw_pointer_cast(result.data())
		);
	if (FOR_TEST_PRINT)__printLastCudaError("3_searchCacheVoxel_v3", __FILE__, __LINE__);
	return;
}

__device__ void insert_result_old(unsigned long long int* result, float dist, int index, int max_nn) {
	unsigned long long int new_value = make_ullint(dist, index);

	while (true) {
		int max_index = 0;
		float max_dist = extract_distance(result[0]);

		// Find the largest distance in the current results
		for (int i = 1; i < max_nn; ++i) {
			float current_dist = extract_distance(result[i]);
			if (current_dist > max_dist) {
				max_dist = current_dist;
				max_index = i;
			}
		}

		// Check if we should replace the largest distance
		if (dist < max_dist) {
			unsigned long long int old_value = result[max_index];
			// Use atomicCAS to replace the value atomically
			if (atomicCAS(&result[max_index], old_value, new_value) == old_value) {
				break; // Successfully replaced the value
			}
		}
		else {
			break; // No need to replace any values
		}
	}
}
__device__ void insert_result(unsigned long long int* result, float dist, int index, int max_nn) {
	unsigned long long int new_value = make_ullint(dist, index);

	for (int i = 0; i < max_nn; ++i) {
		if (atomicCAS(&result[i], std::numeric_limits<int64_t>::max(), new_value) == std::numeric_limits<int64_t>::max()) {
			break;
		}
	}
}
__global__ void restoreSearchResult_v2
(
	int size,
	int max_nn,
	int* indices,
	float* distance,
	unsigned long long int* result
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= size) return;

	for (int i = 0; i < max_nn; ++i) {
		indices[threadid * max_nn + i] = extract_index(result[threadid * max_nn + i]);
		if (indices[threadid * max_nn + i] == -1) {
			distance[threadid * max_nn + i] = std::numeric_limits<float>::infinity();
		}
		else {
			float dist = extract_distance(result[threadid * max_nn + i]);
			distance[threadid * max_nn + i] = powf(dist, 2.0f);
		}
	}
}

__global__ void searchCacheVoxelKernel_v2(
	Eigen::Vector3f * inputPoints,
	Eigen::Vector3f * targetPoints,
	unsigned int* gridSlotIndexCache,
	float voxelSize,
	float3 minBound,
	int3 voxelCount,
	float radius,
	int size,
	int range,
	unsigned long long int* result,
	int3 localMinGlobalIndex,
	int max_nn)
{
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	int totalSearchPoints = (2 * range + 1) * (2 * range + 1) * (2 * range + 1);

	if (globalThreadId >= size * totalSearchPoints) return;

	int pointIdx = globalThreadId / totalSearchPoints;
	int searchPointIdx = globalThreadId % totalSearchPoints;

	int dz = searchPointIdx / ((2 * range + 1) * (2 * range + 1)) - range;
	int dy = (searchPointIdx / (2 * range + 1)) % (2 * range + 1) - range;
	int dx = searchPointIdx % (2 * range + 1) - range;

	Eigen::Vector3f pointPos = inputPoints[pointIdx];

	auto xGlobalIndex = (size_t)floorf((pointPos.x() - minBound.x) / voxelSize);
	auto yGlobalIndex = (size_t)floorf((pointPos.y() - minBound.y) / voxelSize);
	auto zGlobalIndex = (size_t)floorf((pointPos.z() - minBound.z) / voxelSize);

	auto xCacheIndex = (int)xGlobalIndex - localMinGlobalIndex.x;
	auto yCacheIndex = (int)yGlobalIndex - localMinGlobalIndex.y;
	auto zCacheIndex = (int)zGlobalIndex - localMinGlobalIndex.z;

	int zIdx = zCacheIndex + dz;
	int yIdx = yCacheIndex + dy;
	int xIdx = xCacheIndex + dx;

	if (zIdx < 0 || zIdx >= voxelCount.z || yIdx < 0 || yIdx >= voxelCount.y || xIdx < 0 || xIdx >= voxelCount.x) return;

	auto cacheIndex = zIdx * voxelCount.x * voxelCount.y + yIdx * voxelCount.x + xIdx;
	unsigned int targetIdx = gridSlotIndexCache[cacheIndex];

	if (targetIdx != UINT32_MAX) {
		Eigen::Vector3f target = targetPoints[targetIdx];
		float dist = norm3df((pointPos.x() - target.x()), (pointPos.y() - target.y()), (pointPos.z() - target.z()));

		if (dist <= radius) {
			insert_result(&result[pointIdx * max_nn], dist, targetIdx, max_nn);
		}
	}
}

void MarchingCubes::searchCacheVoxel_v2(
	cached_allocator * alloc_, CUstream_st * st,
	thrust::device_vector<Eigen::Vector3f>&inputPoints,
	thrust::device_vector<Eigen::Vector3f>&targetPoints,
	size_t size,
	float radius,
	int max_nn,
	thrust::device_vector<int>&indices,
	thrust::device_vector<float>&distance,
	thrust::device_vector<unsigned long long>&result
)
{
	NvtxRangeCuda nvtxPrint("searchCacheVoxel_v2");
	if (result.size() < max_nn * size)
		result.resize(max_nn * size * 1.5);

	thrust::fill(thrust::cuda::par_nosync.on(st), result.begin(), result.end(), std::numeric_limits<int64_t>::max());
	auto info = exeInfo;
	auto _globalScanAreaAABB = m_globalScanAreaAABB;
	auto voxelSize = m_voxelSize;

	float3 minBound = make_float3(_globalScanAreaAABB.min().x(), _globalScanAreaAABB.min().y(), _globalScanAreaAABB.min().z());
	int3 voxelCount = make_int3(info.cache.voxelCountX, info.cache.voxelCountY, info.cache.voxelCountZ);
	int3 localMinGlobalIndex = make_int3(info.cache.localMinGlobalIndexX, info.cache.localMinGlobalIndexY, info.cache.localMinGlobalIndexZ);

	int range = ceil(radius / voxelSize);

	int numBlocks = size * ((range * 2 + 1) * (range * 2 + 1) * (range * 2 + 1));

	searchCacheVoxelKernel_v2 << <BLOCKS_PER_GRID_THREAD_N(numBlocks, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
		thrust::raw_pointer_cast(inputPoints.data()),
		thrust::raw_pointer_cast(targetPoints.data()),
		info.gridSlotIndexCache,
		voxelSize,
		minBound,
		voxelCount,
		radius,
		size,
		range,
		thrust::raw_pointer_cast(result.data()),
		localMinGlobalIndex,
		max_nn
		);

	restoreSearchResult_v2 << <BLOCKS_PER_GRID_THREAD_N(size, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (
		size,
		max_nn,
		thrust::raw_pointer_cast(indices.data()),
		thrust::raw_pointer_cast(distance.data()),
		thrust::raw_pointer_cast(result.data())
		);
	return;
}

void MarchingCubes::setCache_ICPTarget(
	std::shared_ptr<HPointNormalCloud> input, size_t size,
	thrust::device_vector<unsigned long long>&points_cacheIdx_main,
	int cnt_success,
	cached_allocator * alloc_, CUstream_st * st,
	bool chkData
)
{
	NvtxRangeCuda nvtxPrint("build cache");
	thrust::device_vector<Eigen::Vector3f>& inputPoints = input->points_;
	auto info = exeInfo;
	auto _globalScanAreaAABB = m_globalScanAreaAABB;
	auto voxelSize = m_voxelSize;

	auto _repos_cache_voxel = thrust::raw_pointer_cast(points_cacheIdx_main.data());
	uint32_t* Count_AllUsedSize = used_cnt_HashVoxel;
	cudaMemsetAsync(Count_AllUsedSize, 0, sizeof(uint32_t), st);

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		make_tuple_iterator(inputPoints.begin(), thrust::make_counting_iterator<int>(0)),
		make_tuple_iterator(inputPoints.begin() + size, thrust::make_counting_iterator<int>(size)),
		[info, _globalScanAreaAABB, voxelSize, _repos_cache_voxel, Count_AllUsedSize]__device__(auto tu)
	{
		Eigen::Vector3f pointPos = thrust::get<0>(tu);
		int cntIdx = thrust::get<1>(tu);

		//	mscho	@20240805

		//if (pointPos.x() < _globalScanAreaAABB.min().x())	printf("\n\n\n\n\t\t\t\t (x)==> %f < %f\n\n\n", pointPos.x(), _globalScanAreaAABB.min().x());
		//if (pointPos.y() < _globalScanAreaAABB.min().y())	printf("\n\n\n\n\t\t\t\t (y)==> %f < %f\n\n\n", pointPos.y(), _globalScanAreaAABB.min().y());
		//if (pointPos.z() < _globalScanAreaAABB.min().z())	printf("\n\n\n\n\t\t\t\t (z)==> %f < %f\n\n\n", pointPos.z(), _globalScanAreaAABB.min().z());

		auto xGlobalIndex = (size_t)floorf(pointPos.x() / voxelSize - _globalScanAreaAABB.min().x() / voxelSize + 2500.0f) - 2500;
		auto yGlobalIndex = (size_t)floorf(pointPos.y() / voxelSize - _globalScanAreaAABB.min().y() / voxelSize + 2500.0f) - 2500;
		auto zGlobalIndex = (size_t)floorf(pointPos.z() / voxelSize - _globalScanAreaAABB.min().z() / voxelSize + 2500.0f) - 2500;

		//	mscho	@20240813
		if (xGlobalIndex < info.cache.localMinGlobalIndexX || yGlobalIndex < info.cache.localMinGlobalIndexY || zGlobalIndex < info.cache.localMinGlobalIndexZ)				return;

		auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

		if (xCacheIndex > info.cache.voxelCountX || yCacheIndex > info.cache.voxelCountY || zCacheIndex > info.cache.voxelCountZ)				return;

		auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
			yCacheIndex * info.cache.voxelCountX + xCacheIndex;
		if (cacheIndex >= info.cache.voxelCount)				return;

		uint32_t prev_idx = atomicCAS(&(info.gridSlotIndexCache[cacheIndex]), UINT32_MAX, cntIdx);
		if (prev_idx == UINT32_MAX)
		{
			uint32_t Idx = atomicAdd(Count_AllUsedSize, 1);
			_repos_cache_voxel[Idx] = cacheIndex;
		}
	}
	);
	cudaMemcpyAsync(used_cnt_HashVoxel_h, Count_AllUsedSize, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	if (chkData)
		checkCacheData(input, points_cacheIdx_main, cnt_success, alloc_, st);
	//uint32_t h_Count_AllUsedSize = 0;
	//cudaMemcpy(&h_Count_AllUsedSize, Count_AllUsedSize, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	//qDebug("cache Size %d -> %d", size, h_Count_AllUsedSize);
}

void MarchingCubes::checkCacheData(
	std::shared_ptr<HPointNormalCloud> input,
	thrust::device_vector<unsigned long long>&points_cacheIdx_main,
	int cnt_success,
	cached_allocator * alloc_, CUstream_st * st)
{
	auto inPoints = thrust::raw_pointer_cast(input->points_.data());
	auto inNormal = thrust::raw_pointer_cast(input->normals_.data());

	uint32_t* Count_AllUsedSize = used_cnt_HashVoxel;
	uint32_t h_Count_AllUsedSize = 0;
	cudaMemcpy(&h_Count_AllUsedSize, Count_AllUsedSize, sizeof(uint32_t), cudaMemcpyDeviceToHost);

	thrust::device_vector<Eigen::Vector3f> outPoints_(h_Count_AllUsedSize);
	thrust::device_vector<Eigen::Vector3f> outNormals_(h_Count_AllUsedSize);
	auto outPoints = thrust::raw_pointer_cast(outPoints_.data());
	auto outNormals = thrust::raw_pointer_cast(outNormals_.data());

	auto info = exeInfo;
	auto _repos_cache_voxel = thrust::raw_pointer_cast(points_cacheIdx_main.data());

	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		thrust::make_counting_iterator<int>(0),
		thrust::make_counting_iterator<int>(h_Count_AllUsedSize),
		[info, inPoints, inNormal, _repos_cache_voxel, outPoints, outNormals]__device__(const int idx)
	{
		auto cacheIndex = _repos_cache_voxel[idx];
		int cntIdx = info.gridSlotIndexCache[cacheIndex];

		outPoints[idx] = inPoints[cntIdx];
		outNormals[idx] = inNormal[cntIdx];
	}
	);
	checkCudaSync(st);

	char szTemp[128];
	sprintf(szTemp, "%04d", cnt_success);
	//pRegistration->
	//	plyFileWrite_Ex_deviceData(
	//	GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_com_PC_in.ply",
	//	input->m_nowSize,
	//	inPoints,
	//	inNormal
	//);
	plyFileWrite_Ex_deviceData(
		pSettings->GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_com_PC_out.ply",
		st,
		h_Count_AllUsedSize,
		outPoints,
		outNormals
	);
}
#endif

#pragma endregion

__device__ void Device_ForEachNeighbor(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info, size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex, uint32_t * gridSlotIndexCache,
	voxel_value_t * voxelValues, Eigen::Vector3b * voxelColors,
	voxel_value_t vvc, Eigen::Vector3f * pc, Eigen::Vector3f * voxelMin, Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac, unsigned int* apcount)
{
	size_t x = xCacheIndex + (xOffset);
	size_t y = yCacheIndex + (yOffset);
	size_t z = zCacheIndex + (zOffset);
	if (x < info->cache.voxelCountX &&
		y < info->cache.voxelCountY &&
		z < info->cache.voxelCountZ) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto vvs = voxelValues[voxelValuesIndex];
			if (VOXEL_INVALID != vvs)
			{
				if ((vvc > 0 && vvs < 0) || (vvc < 0 && vvs > 0))
				{
					auto ps = info->cache.GetGlobalPosition(x, y, z);
					auto ip = Interpolation(0.0f, *pc, ps, VV2D(vvc), VV2D(vvs));
					if ((voxelMin->x() <= ip.x() && ip.x() < voxelMax->x()) &&
						(voxelMin->y() <= ip.y() && ip.y() < voxelMax->y()) &&
						(voxelMin->z() <= ip.z() && ip.z() < voxelMax->z()))
					{
						if (vvs > 0)
						{
							Eigen::Vector3f normal = ip - (*pc);
							normal = normal / norm3df(normal.x(), normal.y(), normal.z());
							(*an) += normal;
						}
						else
						{
							Eigen::Vector3f normal = (*pc) - ip;
							normal = normal / norm3df(normal.x(), normal.y(), normal.z());
							(*an) += normal;
						}
						if (voxelColors)
						{
							auto cs = voxelColors[voxelValuesIndex];
							(*ac) += Eigen::Vector3f(
								(float)cs.x(),
								(float)cs.y(),
								(float)cs.z());
						}
						(*ap) += ip;
						(*apcount)++;
					}
				}
			}
		}
	}
}

#ifndef BUILD_FOR_CPU
// shshin @240403 point Insert 구조 진행
__global__ void Kernel_ExtractVoxelPoints_v2(
	MarchingCubes::ExecutionInfo info,
	unsigned int* slotIndices, uint32_t * gridSlotIndexCache,

	HashKey64 * hashinfo_vtx, uint8_t * hashTable_vtx_value,

	voxel_value_t * voxelValues, Eigen::Vector3b * voxelColors,

	Eigen::Vector3f * pointPositions,
	Eigen::Vector3f * pointNormals,
	Eigen::Vector3b * pointColors,
	uint32_t * vtx_dupCnt_repos,

	bool use26Direction = true)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > info.globalHashInfo->Count_HashTableUsed - 1) return;

	auto slotIndex = slotIndices[threadid];
	auto key = info.globalHash[slotIndex];

	auto xGlobalIndex = key.x;
	auto yGlobalIndex = key.y;
	auto zGlobalIndex = key.z;

	auto pc = info.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

	Eigen::Vector3f voxelMin = pc - Eigen::Vector3f(
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f);
	Eigen::Vector3f voxelMax = pc + Eigen::Vector3f(
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f);

	auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

	auto vvc = voxelValues[slotIndex];
	if (VOXEL_INVALID == vvc) return;

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	unsigned int apcount = 0;

#define ForEachNeighbor(xOffset, yOffset, zOffset)\
Device_ForEachNeighbor((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
gridSlotIndexCache, voxelValues, voxelColors, vvc, &pc, &voxelMin, &voxelMax, &ap, &an, &ac, &apcount);\

	if (false == use26Direction)
	{
		//ForEachNeighbor(0, 0, 0);

		ForEachNeighbor(-1, 0, 0);
		ForEachNeighbor(1, 0, 0);
		ForEachNeighbor(0, -1, 0);
		ForEachNeighbor(0, 1, 0);
		ForEachNeighbor(0, 0, -1);
		ForEachNeighbor(0, 0, 1);
	}
	else
	{
		ForEachNeighbor(-1, -1, -1);
		ForEachNeighbor(-1, -1, 0);
		ForEachNeighbor(-1, -1, 1);

		ForEachNeighbor(-1, 0, -1);
		ForEachNeighbor(-1, 0, 0);
		ForEachNeighbor(-1, 0, 1);

		ForEachNeighbor(-1, 1, -1);
		ForEachNeighbor(-1, 1, 0);
		ForEachNeighbor(-1, 1, 1);

		ForEachNeighbor(0, -1, -1);
		ForEachNeighbor(0, -1, 0);
		ForEachNeighbor(0, -1, 1);

		ForEachNeighbor(0, 0, -1);
		//ForEachNeighbor(0, 0, 0);
		ForEachNeighbor(0, 0, 1);

		ForEachNeighbor(0, 1, -1);
		ForEachNeighbor(0, 1, 0);
		ForEachNeighbor(0, 1, 1);

		ForEachNeighbor(1, -1, -1);
		ForEachNeighbor(1, -1, 0);
		ForEachNeighbor(1, -1, 1);

		ForEachNeighbor(1, 0, -1);
		ForEachNeighbor(1, 0, 0);
		ForEachNeighbor(1, 0, 1);

		ForEachNeighbor(1, 1, -1);
		ForEachNeighbor(1, 1, 0);
		ForEachNeighbor(1, 1, 1);
	}

	if (apcount > 1)
	{
		Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
		Eigen::Vector3f pointNormal = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
		Eigen::Vector3b pointColor = Eigen::Vector3b(0, 0, 0);
		if (pointNormals)
		{
			pointNormal = Eigen::Vector3f(an / (float)apcount);
		}
		if (pointColors)
		{
			auto cc = Eigen::Vector3f(ac / (float)apcount);
			pointColor = Eigen::Vector3b(
				(unsigned char)(cc.x() * 255.0f),
				(unsigned char)(cc.y() * 255.0f),
				(unsigned char)(cc.z() * 255.0f));
		}
		hashinfo_vtx->insert_vertex_func64_v5(
			vtx_dupCnt_repos,
			pointPositions,
			pointNormals,
			pointColors
			, pointPosition
			, pointNormal
			, pointColor
			, true
			, key
		);
	}
}

__device__ void Device_ForEachNeighbor_bugfix(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info, size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex, uint32_t * gridSlotIndexCache,
	voxel_value_t * voxelValues, Eigen::Vector3b * voxelColors, Eigen::Vector3f voxelNormal,
	voxel_value_t vvc, Eigen::Vector3f * pc, Eigen::Vector3f * voxelMin, Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac, unsigned int* apcount)
{
	size_t x = xCacheIndex + (xOffset);
	size_t y = yCacheIndex + (yOffset);
	size_t z = zCacheIndex + (zOffset);
	if (x < info->cache.voxelCountX &&
		y < info->cache.voxelCountY &&
		z < info->cache.voxelCountZ) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto vvs = voxelValues[voxelValuesIndex];
			if (VOXEL_INVALID != vvs)
			{
				if ((vvc > 0 && vvs < 0) || (vvc < 0 && vvs > 0))
				{
					auto ps = info->cache.GetGlobalPosition(x, y, z);
					auto ip = Interpolation(0.0f, *pc, ps, VV2D(vvc), VV2D(vvs));
					if ((voxelMin->x() <= ip.x() && ip.x() < voxelMax->x()) &&
						(voxelMin->y() <= ip.y() && ip.y() < voxelMax->y()) &&
						(voxelMin->z() <= ip.z() && ip.z() < voxelMax->z()))
					{
						if (ip == (*pc))
						{
							Eigen::Vector3f normal = voxelNormal;
							(*an) += normal;
						}
						else if (vvs > 0)
						{
							Eigen::Vector3f normal = ip - (*pc);
							normal = normal / norm3df(normal.x(), normal.y(), normal.z());


							(*an) += normal;
						}
						else
						{
							Eigen::Vector3f normal = (*pc) - ip;
							normal = normal / norm3df(normal.x(), normal.y(), normal.z());
							(*an) += normal;
						}
						if (voxelColors)
						{
							auto cs = voxelColors[voxelValuesIndex];
							(*ac) += Eigen::Vector3f(
								(float)cs.x(),
								(float)cs.y(),
								(float)cs.z());
						}
						(*ap) += ip;
						(*apcount)++;
					}
				}
			}
		}
	}
}

//	mscho	@20240422	==> @20240521
__device__ void Device_ForEachNeighbor_v3(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info, size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex, uint32_t * gridSlotIndexCache,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3b * voxelColors, Eigen::Vector3f * voxelNormals,
	//float normal_coeff,
	voxel_value_t vvc, Eigen::Vector3f * pc, Eigen::Vector3f _voxelNormal, Eigen::Vector3f * voxelMin, Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac, unsigned int* apcount)
{
	size_t x = xCacheIndex + (xOffset);
	size_t y = yCacheIndex + (yOffset);
	size_t z = zCacheIndex + (zOffset);
	if (x < info->cache.voxelCountX &&
		y < info->cache.voxelCountY &&
		z < info->cache.voxelCountZ) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto vvs = voxelValues[voxelValuesIndex];
			//	mscho	@20240521 (New)

			//auto vvn = voxelNormals[voxelValuesIndex].normalized();

			//auto center_voxelValuesIndex = gridSlotIndexCache[
			//	zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
			//		+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];

			auto c_normal = _voxelNormal;// voxelNormals[center_voxelValuesIndex].normalized();

			if (VOXEL_INVALID != vvs)
			{
				auto cnt = voxelValueCounts[voxelValuesIndex];
				if (0 == cnt) return;
				vvs /= (float)voxelValueCounts[voxelValuesIndex];
				//vvs /= voxelValueCounts[voxelValuesIndex];
				if (-2.0f < vvs && vvs > 2.0f) return;
				if ((vvc > 0 && vvs < 0) || (vvc < 0 && vvs > 0))
				{
					if (fabsf(vvc) > 0.1f && fabsf(vvs) > 0.1f) return;
					auto ps = info->cache.GetGlobalPosition(x, y, z);

					//Eigen::Vector3f tsdf_tgt;
					//if (vvs > 0.f)
					//	tsdf_tgt = ps + vvn * normal_coeff;
					//else
					//	tsdf_tgt = ps - vvn * normal_coeff;

					auto ip = Interpolation(0.0f, *pc, ps, VV2D(vvc), VV2D(vvs));
					//auto ip_tsdf = Interpolation(0.0f, *tsdf_point, tsdf_tgt, VV2D(vvc), VV2D(vvs));

					if ((voxelMin->x() <= ip.x() && ip.x() < voxelMax->x()) &&
						(voxelMin->y() <= ip.y() && ip.y() < voxelMax->y()) &&
						(voxelMin->z() <= ip.z() && ip.z() < voxelMax->z()))
					{
						//	mscho	@20240521 (New)
						//if (true)
						//{
						//	if (vvs > 0)
						//	{
						//		Eigen::Vector3f normal = ip - (*pc);
						//		float distance = norm3df(normal.x(), normal.y(), normal.z());
						//		if (distance > 0)
						//		{
						//			Eigen::Vector3f calc_normal = normal / distance;
						//			float normal_dot = calc_normal.dot(c_normal);
						//			if (normal_dot < 0.f)
						//				return;
						//			Eigen::Vector3f sum_normal = calc_normal * 10.f;
						//			sum_normal += c_normal;
						//			normal = sum_normal.normalized();
						//			

						//			normal_dot = calc_normal.dot(c_normal);
						//			normal_dot = fabsf(normal_dot);
						//			float	inverse_dot = 1.0f - normal_dot;
						//			//inverse_dot = powf(inverse_dot, 0.8f);
						//			inverse_dot = powf(inverse_dot, 0.5f);
						//			auto	surface_normal = inverse_dot * calc_normal + (1.0 - inverse_dot) * c_normal;
						//			sum_normal = surface_normal;
						//			normal = sum_normal.normalized();
						//		}
						//		else
						//			normal = c_normal;
						//		(*an) += normal;
						//	}
						//	else
						//	{
						//		Eigen::Vector3f normal = (*pc) - ip;
						//		float distance = norm3df(normal.x(), normal.y(), normal.z());
						//		if (distance > 0)
						//		{
						//			Eigen::Vector3f calc_normal = normal / distance;
						//			float normal_dot = calc_normal.dot(c_normal);
						//			if (normal_dot < 0.f)
						//				return;
						//			Eigen::Vector3f sum_normal = calc_normal * 10.f;
						//			sum_normal += c_normal;
						//			normal = sum_normal.normalized();

						//			normal_dot = calc_normal.dot(c_normal);
						//			normal_dot = fabsf(normal_dot);
						//			float	inverse_dot = 1.0f - normal_dot;
						//			//inverse_dot = powf(inverse_dot, 0.8f);
						//			inverse_dot = powf(inverse_dot, 0.5f);
						//			auto	surface_normal = inverse_dot * calc_normal + (1.0 - inverse_dot) * c_normal;
						//			sum_normal = surface_normal;
						//			normal = sum_normal.normalized();
						//		}
						//		else
						//			normal = c_normal;
						//		(*an) += normal;
						//	}
						//}

						//	mscho	@20240521 (New)
						if (true)
						{
							if (vvs > 0)
							{
								Eigen::Vector3f normal = ip - (*pc);
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								(*an) += normal;
							}
							else
							{
								Eigen::Vector3f normal = (*pc) - ip;
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								(*an) += normal;
							}
						}
						else
							(*an) += c_normal;
						//(*an) += voxelNormals[voxelValuesIndex];

						if (voxelColors)
						{
							auto cs = voxelColors[voxelValuesIndex];
							(*ac) += Eigen::Vector3f(
								(float)cs.x(),
								(float)cs.y(),
								(float)cs.z());
						}
						(*ap) += ip;
						(*apcount)++;
					}
				}
			}
		}
	}
}
#endif

//	mscho	@20240422	==> @20240521 ==> @20240524
__device__ void Device_ForEachNeighbor_v4(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info, size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex, uint32_t * gridSlotIndexCache,
	VoxelExtraAttrib * voxelExtraAttribs,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3b * voxelColors, Eigen::Vector3f * voxelNormals,
	char* voxelSegmentations,
	uint32_t * voxel_valid,
	voxel_value_t vvc, Eigen::Vector3f * pc, Eigen::Vector3f _voxelNormal, Eigen::Vector3f * voxelMin, Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac, unsigned int* apcount, int* toothCnt, int* patchID)
{
	auto x = (int)xCacheIndex + (int)(xOffset);
	auto y = (int)yCacheIndex + (int)(yOffset);
	auto z = (int)zCacheIndex + (int)(zOffset);
	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto vvs = voxelValues[voxelValuesIndex];
			//	mscho	@20240521 (New)

			//auto vvn = voxelNormals[voxelValuesIndex].normalized();

			//auto center_voxelValuesIndex = gridSlotIndexCache[
			//	zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
			//		+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];

			auto c_normal = _voxelNormal;// voxelNormals[center_voxelValuesIndex].normalized();

			if (VOXEL_INVALID != vvs)
			{
				if (fabs(vvs) > 2900.f)
				{
					(*voxel_valid) = 0;
					return;
				}

				auto cnt = VOXELCNT_VALUE(voxelValueCounts[voxelValuesIndex]);
				if (0 == cnt) return;

				vvs /= (float)cnt;
				//vvs /= voxelValueCounts[voxelValuesIndex];
				if (vvs < -1.0f || vvs > 1.0f) return;
				if ((vvc > 0 && vvs < 0) || (vvc < 0 && vvs > 0))
				{
					if (fabsf(vvc) > 0.2f && fabsf(vvs) > 0.2f) return;
					auto ps = info->cache.GetGlobalPosition(x, y, z);

					//Eigen::Vector3f tsdf_tgt;
					//if (vvs > 0.f)
					//	tsdf_tgt = ps + vvn * normal_coeff;
					//else
					//	tsdf_tgt = ps - vvn * normal_coeff;

					auto ip = Interpolation(0.0f, *pc, ps, VV2D(vvc), VV2D(vvs));
					//auto ip_tsdf = Interpolation(0.0f, *tsdf_point, tsdf_tgt, VV2D(vvc), VV2D(vvs));

					if ((voxelMin->x() <= ip.x() && ip.x() < voxelMax->x()) &&
						(voxelMin->y() <= ip.y() && ip.y() < voxelMax->y()) &&
						(voxelMin->z() <= ip.z() && ip.z() < voxelMax->z()))
					{
						//	mscho	@20240521 (New)
						//if (true)
						//{
						//	if (vvs > 0)
						//	{
						//		Eigen::Vector3f normal = ip - (*pc);
						//		float distance = norm3df(normal.x(), normal.y(), normal.z());
						//		if (distance > 0)
						//		{
						//			Eigen::Vector3f calc_normal = normal / distance;
						//			float normal_dot = calc_normal.dot(c_normal);
						//			if (normal_dot < 0.f)
						//				return;
						//			Eigen::Vector3f sum_normal = calc_normal * 10.f;
						//			sum_normal += c_normal;
						//			normal = sum_normal.normalized();
						//			

						//			normal_dot = calc_normal.dot(c_normal);
						//			normal_dot = fabsf(normal_dot);
						//			float	inverse_dot = 1.0f - normal_dot;
						//			//inverse_dot = powf(inverse_dot, 0.8f);
						//			inverse_dot = powf(inverse_dot, 0.5f);
						//			auto	surface_normal = inverse_dot * calc_normal + (1.0 - inverse_dot) * c_normal;
						//			sum_normal = surface_normal;
						//			normal = sum_normal.normalized();
						//		}
						//		else
						//			normal = c_normal;
						//		(*an) += normal;
						//	}
						//	else
						//	{
						//		Eigen::Vector3f normal = (*pc) - ip;
						//		float distance = norm3df(normal.x(), normal.y(), normal.z());
						//		if (distance > 0)
						//		{
						//			Eigen::Vector3f calc_normal = normal / distance;
						//			float normal_dot = calc_normal.dot(c_normal);
						//			if (normal_dot < 0.f)
						//				return;
						//			Eigen::Vector3f sum_normal = calc_normal * 10.f;
						//			sum_normal += c_normal;
						//			normal = sum_normal.normalized();

						//			normal_dot = calc_normal.dot(c_normal);
						//			normal_dot = fabsf(normal_dot);
						//			float	inverse_dot = 1.0f - normal_dot;
						//			//inverse_dot = powf(inverse_dot, 0.8f);
						//			inverse_dot = powf(inverse_dot, 0.5f);
						//			auto	surface_normal = inverse_dot * calc_normal + (1.0 - inverse_dot) * c_normal;
						//			sum_normal = surface_normal;
						//			normal = sum_normal.normalized();
						//		}
						//		else
						//			normal = c_normal;
						//		(*an) += normal;
						//	}
						//}

						//	mscho	@20240521 (New)
						if (true)
						{
							if (vvs > 0)
							{
								Eigen::Vector3f normal = ip - (*pc);
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								(*an) += normal;
							}
							else
							{
								Eigen::Vector3f normal = (*pc) - ip;
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								(*an) += normal;
							}
						}
						else
							(*an) += c_normal;
						//(*an) += voxelNormals[voxelValuesIndex];

						if (voxelColors)
						{
							auto cs = voxelColors[voxelValuesIndex];
							(*ac) += Eigen::Vector3f(
								(float)cs.x(),
								(float)cs.y(),
								(float)cs.z());
						}
						(*ap) += ip;
						(*apcount)++;

						(*patchID) += voxelExtraAttribs[voxelValuesIndex].startPatchID;
						if (toothCnt != nullptr)
							(*toothCnt) += (int)voxelSegmentations[voxelValuesIndex];
					}
				}
			}
		}
		else {
			auto center_voxelValuesIndex = gridSlotIndexCache[
				zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
					+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];
			Eigen::Vector3f pp(xOffset, yOffset, zOffset);
			pp.normalize();
			if (pp.dot(_voxelNormal) > 0.95f && voxelValueCounts[center_voxelValuesIndex] > 20) {
				auto c_normal = _voxelNormal;
				(*an) += _voxelNormal;

				auto cs = voxelColors[center_voxelValuesIndex];
				(*ac) += Eigen::Vector3f(
					(float)cs.x(),
					(float)cs.y(),
					(float)cs.z());
				(*ap) += (*pc - c_normal * vvc);
				(*apcount)++;
				(*patchID) += voxelExtraAttribs[center_voxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[center_voxelValuesIndex];
			}

		}
	}
}

#ifndef BUILD_FOR_CPU

//	mscho	@20240422	==> @20240523
__device__ void Device_ForEachNeighbor_v4_shared(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info, size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex, uint32_t * gridSlotIndexCache,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3b * voxelColors, Eigen::Vector3f * voxelNormals,
	//float normal_coeff,
	voxel_value_t vvc, Eigen::Vector3f * pc, Eigen::Vector3f _voxelNormal, Eigen::Vector3f * voxelMin, Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac, unsigned int* apcount, uint32_t voxel_idx)
{
	size_t x = xCacheIndex + (xOffset);
	size_t y = yCacheIndex + (yOffset);
	size_t z = zCacheIndex + (zOffset);

	*ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	*an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	*ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	*apcount = 0;

	if (voxel_idx == 13) return;
	if (x < info->cache.voxelCountX &&
		y < info->cache.voxelCountY &&
		z < info->cache.voxelCountZ) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto vvs = voxelValues[voxelValuesIndex];
			//	mscho	@20240521 (New)

			//auto vvn = voxelNormals[voxelValuesIndex].normalized();

			//auto center_voxelValuesIndex = gridSlotIndexCache[
			//	zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
			//		+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];

			auto c_normal = _voxelNormal;// voxelNormals[center_voxelValuesIndex].normalized();

			if (VOXEL_INVALID != vvs)
			{
				vvs /= voxelValueCounts[voxelValuesIndex];
				if ((vvc > 0 && vvs < 0) || (vvc < 0 && vvs > 0))
				{
					if (fabsf(vvc) > 0.1f && fabsf(vvs) > 0.1f) return;
					auto ps = info->cache.GetGlobalPosition(x, y, z);

					//Eigen::Vector3f tsdf_tgt;
					//if (vvs > 0.f)
					//	tsdf_tgt = ps + vvn * normal_coeff;
					//else
					//	tsdf_tgt = ps - vvn * normal_coeff;

					auto ip = Interpolation(0.0f, *pc, ps, VV2D(vvc), VV2D(vvs));
					//auto ip_tsdf = Interpolation(0.0f, *tsdf_point, tsdf_tgt, VV2D(vvc), VV2D(vvs));

					if ((voxelMin->x() <= ip.x() && ip.x() < voxelMax->x()) &&
						(voxelMin->y() <= ip.y() && ip.y() < voxelMax->y()) &&
						(voxelMin->z() <= ip.z() && ip.z() < voxelMax->z()))
					{
						//	mscho	@20240521 (New)
						Eigen::Vector3f normal;
						if (true)
						{
							if (vvs > 0)
							{
								normal = ip - (*pc);
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								//an[voxel_idx] += normal;
							}
							else
							{
								normal = (*pc) - ip;
								float distance = norm3df(normal.x(), normal.y(), normal.z());
								if (distance > 0)
								{
									Eigen::Vector3f calc_normal = normal / distance;
									float normal_dot = calc_normal.dot(c_normal);
									if (normal_dot < 0.f)
										return;
									normal = calc_normal.normalized();
								}
								else
									normal = c_normal;
								//(*an) += normal;
							}
						}
						else
							normal = c_normal;

						//atomicAdd(&an[voxel_idx].x(), normal.x());
						//atomicAdd(&an[voxel_idx].y(), normal.y());
						//atomicAdd(&an[voxel_idx].z(), normal.z());

						(*an) += normal;

						//(*an) += voxelNormals[voxelValuesIndex];

						if (voxelColors)
						{
							auto cs = voxelColors[voxelValuesIndex];
							(*ac) += Eigen::Vector3f(
								(float)cs.x(),
								(float)cs.y(),
								(float)cs.z());
							//atomicAdd(&ac[voxel_idx].x(), (float)cs.x() / 255.0f);
							//atomicAdd(&ac[voxel_idx].y(), (float)cs.y() / 255.0f);
							//atomicAdd(&ac[voxel_idx].z(), (float)cs.z() / 255.0f);

						}
						//atomicAdd(&ap[voxel_idx].x(), ip.x());
						//atomicAdd(&ap[voxel_idx].y(), ip.y());
						//atomicAdd(&ap[voxel_idx].z(), ip.z());

						//atomicAdd(&apcount[voxel_idx], 1);

						(*ap) += ip;
						(*apcount) = 1;
					}
				}
			}
		}
	}

}


// shshin @240409 cache 직접연결 구조 진행
__global__ void Kernel_ExtractVoxelPoints_v3(
	MarchingCubes::ExecutionInfo info,
	unsigned int* slotIndices, uint32_t * gridSlotIndexCache,

	HashKey64 * hashinfo_vtx,

	voxel_value_t * voxelValues,
	Eigen::Vector3f * voxelNormals,
	unsigned short* voxelValueCounts,
	Eigen::Vector3b * voxelColors,

	Eigen::Vector3f * pointPositions,
	Eigen::Vector3f * pointNormals,
	Eigen::Vector3b * pointColors,

	uint64_t * output,

	bool use26Direction = true
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > info.globalHashInfo->Count_HashTableUsed - 1) return;

	auto slotIndex = slotIndices[threadid];
	auto key = info.globalHash[slotIndex];

	auto xGlobalIndex = key.x;
	auto yGlobalIndex = key.y;
	auto zGlobalIndex = key.z;

	auto pc = info.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

	Eigen::Vector3f voxelMin = pc - Eigen::Vector3f(
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f);
	Eigen::Vector3f voxelMax = pc + Eigen::Vector3f(
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f,
		info.global.voxelSize * 0.5f);

	auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

	auto vvc = voxelValues[slotIndex];
	if (VOXEL_INVALID == vvc) return;

	auto voxelNormal = voxelNormals[slotIndex];
	auto voxelValueCount = voxelValueCounts[slotIndex];

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	unsigned int apcount = 0;

#define ForEachNeighbor_bugfix(xOffset, yOffset, zOffset)\
Device_ForEachNeighbor_bugfix((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
gridSlotIndexCache, voxelValues, voxelColors, voxelNormal, vvc, &pc, &voxelMin, &voxelMax, &ap, &an, &ac, &apcount);\

	if (false == use26Direction)
	{
		//ForEachNeighbor(0, 0, 0);

		ForEachNeighbor_bugfix(-1, 0, 0);
		ForEachNeighbor_bugfix(1, 0, 0);
		ForEachNeighbor_bugfix(0, -1, 0);
		ForEachNeighbor_bugfix(0, 1, 0);
		ForEachNeighbor_bugfix(0, 0, -1);
		ForEachNeighbor_bugfix(0, 0, 1);
	}
	else
	{
		ForEachNeighbor_bugfix(-1, -1, -1);
		ForEachNeighbor_bugfix(-1, -1, 0);
		ForEachNeighbor_bugfix(-1, -1, 1);

		ForEachNeighbor_bugfix(-1, 0, -1);
		ForEachNeighbor_bugfix(-1, 0, 0);
		ForEachNeighbor_bugfix(-1, 0, 1);

		ForEachNeighbor_bugfix(-1, 1, -1);
		ForEachNeighbor_bugfix(-1, 1, 0);
		ForEachNeighbor_bugfix(-1, 1, 1);

		ForEachNeighbor_bugfix(0, -1, -1);
		ForEachNeighbor_bugfix(0, -1, 0);
		ForEachNeighbor_bugfix(0, -1, 1);

		ForEachNeighbor_bugfix(0, 0, -1);
		//ForEachNeighbor_bugfix(0, 0, 0);
		ForEachNeighbor_bugfix(0, 0, 1);

		ForEachNeighbor_bugfix(0, 1, -1);
		ForEachNeighbor_bugfix(0, 1, 0);
		ForEachNeighbor_bugfix(0, 1, 1);

		ForEachNeighbor_bugfix(1, -1, -1);
		ForEachNeighbor_bugfix(1, -1, 0);
		ForEachNeighbor_bugfix(1, -1, 1);

		ForEachNeighbor_bugfix(1, 0, -1);
		ForEachNeighbor_bugfix(1, 0, 0);
		ForEachNeighbor_bugfix(1, 0, 1);

		ForEachNeighbor_bugfix(1, 1, -1);
		ForEachNeighbor_bugfix(1, 1, 0);
		ForEachNeighbor_bugfix(1, 1, 1);
	}

	if (apcount > 1)
	{
		uint32_t _prev_table_ = atomicAdd(&(hashinfo_vtx->Count_HashTableUsed), 1);

		auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
			yCacheIndex * info.cache.voxelCountX + xCacheIndex;

		info.gridSlotIndexCache_pts[cacheIndex] = _prev_table_;
		/*if(_prev_table_ < 10)
			printf("%llu, %d, %lld\n", cacheIndex, _prev_table_, key);*/

		if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, -4))
		{
			output[_prev_table_] = key.KeyValue();
		}

		pointPositions[_prev_table_] = Eigen::Vector3f(ap / (float)apcount);

		if (!VECTOR3F_VALID_(pointPositions[_prev_table_]))
			printf("Extract pt Invalid!\n");


		/*if (FLT_VALID(voxelNormal.z()) && voxelValueCount > 0)
			pointNormals[_prev_table_] = (voxelNormal / (float)voxelValueCount).normalized();
		else*/
		pointNormals[_prev_table_] = Eigen::Vector3f(an / (float)apcount).normalized();;
		//qDebug("an = %f, %f, %f  | apcount = %d | res = %f, %f, %f \n", an.x(), an.y(), an.z(), apcount, pointNormals[_prev_table_].x(), pointNormals[_prev_table_].y(), pointNormals[_prev_table_].z());

		auto cc = Eigen::Vector3f(ac / (float)apcount);
		pointColors[_prev_table_] = Eigen::Vector3b(
			(unsigned char)(cc.x() * 255.0f),
			(unsigned char)(cc.y() * 255.0f),
			(unsigned char)(cc.z() * 255.0f));
	}
}
#endif

__device__ void Device_ForEachNeighbor_v5(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info,
	size_t xCacheIndex,
	size_t yCacheIndex,
	size_t zCacheIndex,
	const uint32_t * gridSlotIndexCache,
	const VoxelExtraAttrib * voxelExtraAttribs,
	const voxel_value_t * voxelValues,
	const unsigned short* voxelValueCounts,
	const Eigen::Vector3b * voxelColors,
	const Eigen::Vector3f * voxelNormals,
	const char* voxelSegmentations,
	voxel_value_t centerVoxelValue,
	Eigen::Vector3f centerPos,
	Eigen::Vector3f centerNormal,
	Eigen::Vector3f * voxelMin,
	Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * accumulatedPos,
	Eigen::Vector3f * accumulatedNormal,
	Eigen::Vector3f * accumulatedColor,
	unsigned int* countOfAccumulation,
	int* toothCnt,
	int* patchID)
{
	auto x = (int)xCacheIndex + (int)(xOffset);
	auto y = (int)yCacheIndex + (int)(yOffset);
	auto z = (int)zCacheIndex + (int)(zOffset);
	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto neighborVoxelValue = VV2D(voxelValues[voxelValuesIndex]);
			auto neighborNormal = voxelNormals[voxelValuesIndex];

			if (VOXEL_INVALID == neighborVoxelValue) return;
			auto neighborVoxelValueCnt = VOXELCNT_VALUE(voxelValueCounts[voxelValuesIndex]);
			if (0 == neighborVoxelValueCnt) return;
			neighborVoxelValue /= (float)neighborVoxelValueCnt;

			if (neighborVoxelValue < -1.0f || neighborVoxelValue > 1.0f) return;
			if (centerVoxelValue * neighborVoxelValue >= 0) return;
			if (fabsf(centerVoxelValue) > ONE_VOXEL_SIZE * 2.0f && fabsf(neighborVoxelValue) > ONE_VOXEL_SIZE * 2.0f) return;

			auto interpolatedPos = Interpolation(0.0f, centerPos, info->cache.GetGlobalPosition(x, y, z), VV2D(centerVoxelValue), VV2D(neighborVoxelValue));

			if ((voxelMin->x() <= interpolatedPos.x() && interpolatedPos.x() < voxelMax->x()) &&
				(voxelMin->y() <= interpolatedPos.y() && interpolatedPos.y() < voxelMax->y()) &&
				(voxelMin->z() <= interpolatedPos.z() && interpolatedPos.z() < voxelMax->z()))
			{
				if (neighborVoxelValue > 0)
				{
					Eigen::Vector3f normal = interpolatedPos - centerPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						normal = calcNormal.normalized();
					}
					else
						normal = centerNormal;
					(*accumulatedNormal) += normal;
				}
				else
				{
					Eigen::Vector3f normal = centerPos - interpolatedPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						normal = calcNormal.normalized();
					}
					else
						normal = centerNormal;
					(*accumulatedNormal) += normal;
				}

				if (voxelColors != nullptr) {
					auto color = voxelColors[voxelValuesIndex];
					(*accumulatedColor) += Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f;
				}
				(*accumulatedPos) += interpolatedPos;
				(*countOfAccumulation)++;

				(*patchID) += voxelExtraAttribs[voxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[voxelValuesIndex];
			}
		}
		else if (0) {
			auto centerVoxelValuesIndex = gridSlotIndexCache[
				zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
					+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];
			Eigen::Vector3f pp(xOffset, yOffset, zOffset);
			pp.normalize();
			if (pp.dot(centerNormal) > 0.95f && voxelValueCounts[centerVoxelValuesIndex] > 20) {
				(*accumulatedNormal) += centerNormal;

				auto cs = voxelColors[centerVoxelValuesIndex];
				(*accumulatedColor) += Eigen::Vector3f(
					(float)cs.x() / 255.0f,
					(float)cs.y() / 255.0f,
					(float)cs.z() / 255.0f);
				(*accumulatedPos) += (centerPos - centerNormal * centerVoxelValue);
				(*countOfAccumulation)++;
				(*patchID) += voxelExtraAttribs[centerVoxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[centerVoxelValuesIndex];
			}

		}
	}
}

//	mscho	@20240422
//	Point extract함수를 수정한다..
__global__ void Kernel_ExtractVoxelPoints_v4(
	MarchingCubes::ExecutionInfo info
	, unsigned int* slotIndices
	, uint32_t * gridSlotIndexCache

	, voxel_value_t * voxelValues
	, Eigen::Vector3f * voxelNormals
	, unsigned short* voxelValueCounts
	, Eigen::Vector3b * voxelColors
	, char* voxelSegmentations
	, VoxelExtraAttrib * voxelExtraAttribs

	, Eigen::Vector3f * pointPositions
	, const Eigen::Vector3f pointHalfPositions
	, Eigen::Vector3f * pointNormals
	, Eigen::Vector3b * pointColors
	, unsigned char* pointMaterialIDs
	, unsigned short* pointStartPatchIDs

	, uint64_t * output
	, uint32_t * cnt_output_extract
	, uint32_t * cnt_output_contains

	, bool use26Direction = true
	, bool filtering = true
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > info.globalHashInfo->Count_HashTableUsed - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < info.globalHashInfo->Count_HashTableUsed; threadid++) {
#endif

		auto slotIndex = slotIndices[threadid];
		auto& key = info.globalHash[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto pc = info.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

		Eigen::Vector3f voxelMin = pc - pointHalfPositions;
		Eigen::Vector3f voxelMax = pc + pointHalfPositions;

		auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

		auto vvc = voxelValues[slotIndex];
		//	mscho	@20240524
		if (VOXEL_INVALID == vvc)
			kernel_return;
		if (fabsf(vvc) > 999.0)
			kernel_return;
		//	mscho	@20240521 (New)
		auto _voxelNormal = voxelNormals[slotIndex];
		if (!FLT_VALID(_voxelNormal.x()))
			kernel_return;
		_voxelNormal.normalize();
		auto voxelValueCount = VOXELCNT_VALUE(voxelValueCounts[slotIndex]);
		if (voxelValueCount < 1)
			kernel_return;
		if (filtering && voxelValueCount < 2)
			kernel_return;
		//	mscho	@20240523
		//	Voxelcount==0 일때, 데이타를 살리는 것은 안맞는거 같음.. 삭제
		//if (0 == voxelValueCount)
		//{
		//	uint32_t _prev_table_ = atomicAdd(cnt_output_extract, 1);
		//	auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
		//		yCacheIndex * info.cache.voxelCountX + xCacheIndex;
		//	info.gridSlotIndexCache_pts[cacheIndex] = _prev_table_;
		//	if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, -4))
		//		//if (info.local.Contains(xGlobalIndex, yGlobalIndex, zGlobalIndex))
		//	{
		//		uint32_t _prev_idx = atomicAdd(cnt_output_contains, 1);
		//		output[_prev_idx] = key;
		//	}
		//	pointPositions[_prev_table_] = pc;
		//	pointNormals[_prev_table_] = voxelNormals[slotIndex];
		//	auto cc = voxelColors[slotIndex];
		//	pointColors[_prev_table_] = Eigen::Vector3b(
		//		(unsigned char)(cc.x() * 255.0f),
		//		(unsigned char)(cc.y() * 255.0f),
		//		(unsigned char)(cc.z() * 255.0f));

		//	//pointNormals[_prev_table_] = tsdf_normal;

		//	return;
		//}

		vvc = voxelValues[slotIndex] / (voxel_value_t)voxelValueCount;
		//voxelNormal.normalize();
		//Eigen::Vector3f tsdf_point;
		//	mscho	@20240521 (New)
		//float	normal_coeff = 0.f;
		//if (vvc > 0.f)
		//	tsdf_point = pc + tsdf_normal * normal_coeff;
		//else
		//	tsdf_point = pc - tsdf_normal * normal_coeff;

		//tsdf_point = pc;

		//	mscho	@20240523
		//	현재 Voxel의 Voxelvalue가 0이라고 하더라도.. Normal떄문에 검색을 해야 한다..
		//	주변점과의 관계를 이용해서, Normal을 뽑아내야지... 다음에서 Normal을 평가할 수도 있고
		//  Detail이 살아있는 Normal을 사용할 수 있다.

		//if (0.0f == vvc)
		//{
		//	uint32_t _prev_table_ = atomicAdd(cnt_output_extract, 1);
		//	auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
		//		yCacheIndex * info.cache.voxelCountX + xCacheIndex;
		//	info.gridSlotIndexCache_pts[cacheIndex] = _prev_table_;
		//	if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, -4))
		//		//if (info.local.Contains(xGlobalIndex, yGlobalIndex, zGlobalIndex))
		//	{
		//		uint32_t _prev_idx = atomicAdd(cnt_output_contains, 1);
		//		output[_prev_idx] = key;
		//	}
		//	pointPositions[_prev_table_] = pc;
		//	pointNormals[_prev_table_] = voxelNormals[slotIndex];
		//	auto cc = voxelColors[slotIndex];
		//	pointColors[_prev_table_] = Eigen::Vector3b(
		//		(unsigned char)(cc.x() * 255.0f),
		//		(unsigned char)(cc.y() * 255.0f),
		//		(unsigned char)(cc.z() * 255.0f));

		//	//pointNormals[_prev_table_] = tsdf_normal;

		//	return;
		//}

		//	mscho	@20240523
		auto _voxelColor = voxelColors[slotIndex];
		bool	bIsoSurface = false;
		int toothCnt = 0;
		int patchID = 0;
		if (fabsf(vvc) <= 0.001)
		{
			bIsoSurface = true;
		}


		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		//	mscho	 @20240524
		uint32_t	voxel_valid = 1;
		//	mscho	@20240521 (New)
#define ForEachNeighbor_v4(xOffset, yOffset, zOffset)\
Device_ForEachNeighbor_v4((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
gridSlotIndexCache, voxelExtraAttribs, voxelValues, voxelValueCounts, voxelColors, voxelNormals, voxelSegmentations, \
&voxel_valid, vvc, &pc, _voxelNormal, &voxelMin, &voxelMax, &ap, &an, &ac, &apcount, &toothCnt, & patchID);\


		//if (vvc == 0)
		//	printf("vvc is zero , (%f %f %f)\n", vvc, tsdf_point.x(), tsdf_point.y(), tsdf_point.z());
		if (false == use26Direction)
		{
			//ForEachNeighbor_v3(0, 0, 0);

			ForEachNeighbor_v4(-1, 0, 0);
			ForEachNeighbor_v4(1, 0, 0);
			ForEachNeighbor_v4(0, -1, 0);
			ForEachNeighbor_v4(0, 1, 0);
			ForEachNeighbor_v4(0, 0, -1);
			ForEachNeighbor_v4(0, 0, 1);
		}
		else
		{
			//	mscho	 @20240524
			ForEachNeighbor_v4(-1, -1, -1);
			ForEachNeighbor_v4(-1, -1, 0);
			ForEachNeighbor_v4(-1, -1, 1);

			ForEachNeighbor_v4(-1, 0, -1);
			ForEachNeighbor_v4(-1, 0, 0);
			ForEachNeighbor_v4(-1, 0, 1);

			ForEachNeighbor_v4(-1, 1, -1);
			ForEachNeighbor_v4(-1, 1, 0);
			ForEachNeighbor_v4(-1, 1, 1);

			ForEachNeighbor_v4(0, -1, -1);
			ForEachNeighbor_v4(0, -1, 0);
			ForEachNeighbor_v4(0, -1, 1);

			ForEachNeighbor_v4(0, 0, -1);
			//ForEachNeighbor_v4(0, 0, 0);
			ForEachNeighbor_v4(0, 0, 1);

			ForEachNeighbor_v4(0, 1, -1);
			ForEachNeighbor_v4(0, 1, 0);
			ForEachNeighbor_v4(0, 1, 1);

			ForEachNeighbor_v4(1, -1, -1);
			ForEachNeighbor_v4(1, -1, 0);
			ForEachNeighbor_v4(1, -1, 1);

			ForEachNeighbor_v4(1, 0, -1);
			ForEachNeighbor_v4(1, 0, 0);
			ForEachNeighbor_v4(1, 0, 1);

			ForEachNeighbor_v4(1, 1, -1);
			ForEachNeighbor_v4(1, 1, 0);
			ForEachNeighbor_v4(1, 1, 1);
		}

		//	mscho	@20240523 ==> @20240527	==> @20240530

#define		CACHE_MARGIN_MIN	-6
#define		CACHE_MARGIN_MAX	-2

		//	mscho	@20240620
		uint32_t	min_count = 1;	//	mscho	@202406025
		if (apcount > min_count && voxel_valid)
		{
			uint32_t _prev_table_ = atomicAdd(cnt_output_extract, (uint32_t)1);
			auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
				yCacheIndex * info.cache.voxelCountX + xCacheIndex;
			if (cacheIndex >= (size_t)info.cache.voxelCount)
				kernel_return;
			info.gridSlotIndexCache_pts[cacheIndex] = _prev_table_;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, CACHE_MARGIN_MIN, CACHE_MARGIN_MAX))
				//if (info.local.Contains(xGlobalIndex, yGlobalIndex, zGlobalIndex))
			{
				uint32_t _prev_idx = atomicAdd(cnt_output_contains, (uint32_t)1);
				output[_prev_idx] = key.KeyValue();
			}
			if (bIsoSurface)
			{
				pointPositions[_prev_table_] = pc;
			}
			else
			{
				pointPositions[_prev_table_] = Eigen::Vector3f(ap / (float)apcount);
			}
			pointNormals[_prev_table_] = an.normalized();
			auto cc = Eigen::Vector3f(ac / (float)apcount);
			pointColors[_prev_table_] = Eigen::Vector3b(
				(unsigned char)(cc.x() * 255.0f),
				(unsigned char)(cc.y() * 255.0f),
				(unsigned char)(cc.z() * 255.0f));

			if (toothCnt / (int)apcount > -5) {
				pointMaterialIDs[_prev_table_] = 255;
#ifdef SEGMENTATION_COLOR_DEBUG
				pointColors[_prev_table_] = Eigen::Vector3b(255, 0, 0);
#endif//SEGMENTATION_COLOR_DEBUG
			}
			else {
				pointMaterialIDs[_prev_table_] = 0;
#ifdef SEGMENTATION_COLOR_DEBUG
				pointColors[_prev_table_] = Eigen::Vector3b(0, 255, 0);
#endif//SEGMENTATION_COLOR_DEBUG
			}

			pointStartPatchIDs[_prev_table_] = patchID / apcount;

			//pointNormals[_prev_table_] = tsdf_normal;
		}
		else if (bIsoSurface && voxel_valid)
		{
			uint32_t _prev_table_ = atomicAdd(cnt_output_extract, (uint32_t)1);
			auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
				yCacheIndex * info.cache.voxelCountX + xCacheIndex;
			info.gridSlotIndexCache_pts[cacheIndex] = _prev_table_;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, CACHE_MARGIN_MIN, CACHE_MARGIN_MAX))
				//if (info.local.Contains(xGlobalIndex, yGlobalIndex, zGlobalIndex))
			{
				uint32_t _prev_idx = atomicAdd(cnt_output_contains, (uint32_t)1);
				output[_prev_idx] = key.KeyValue();
			}
			pointPositions[_prev_table_] = pc;
			pointNormals[_prev_table_] = _voxelNormal;
			pointColors[_prev_table_] = _voxelColor;
			if (voxelSegmentations[slotIndex] >= -5) {
				pointMaterialIDs[_prev_table_] = 255;
#ifdef SEGMENTATION_COLOR_DEBUG
				pointColors[_prev_table_] = Eigen::Vector3b(255, 0, 0);
#endif//SEGMENTATION_COLOR_DEBUG
			}
			else {
				pointMaterialIDs[_prev_table_] = 0;
#ifdef SEGMENTATION_COLOR_DEBUG
				pointColors[_prev_table_] = Eigen::Vector3b(0, 255, 0);
#endif//SEGMENTATION_COLOR_DEBUG
			}
			pointStartPatchIDs[_prev_table_] = voxelExtraAttribs[slotIndex].startPatchID;
			//pointColors[_prev_table_] = Eigen::Vector3b(
			//	(unsigned char)(0 * 255.0f),
			//	(unsigned char)(0 * 255.0f),
			//	(unsigned char)(1.0 * 255.0f));
		}
	}
	}

__device__ void Device_ForEachNeighbor_v5_enhanced(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info,
	size_t xCacheIndex,
	size_t yCacheIndex,
	size_t zCacheIndex,
	const uint32_t * gridSlotIndexCache,
	const VoxelExtraAttrib * voxelExtraAttribs,
	const voxel_value_t * voxelValues,
	const unsigned short* voxelValueCounts,
	const Eigen::Vector3b * voxelColors,
	const Eigen::Vector3f * voxelNormals,
	const char* voxelSegmentations,
	voxel_value_t centerVoxelValue,
	Eigen::Vector3f centerPos,
	Eigen::Vector3f centerNormal,
	Eigen::Vector3f * voxelMin,
	Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * accumulatedPos,
	Eigen::Vector3f * accumulatedNormal,
	Eigen::Vector3f * accumulatedColor,
	unsigned int* countOfAccumulation,
	int* toothCnt,
	int* patchID)
{
	auto x = (int)xCacheIndex + (int)(xOffset);
	auto y = (int)yCacheIndex + (int)(yOffset);
	auto z = (int)zCacheIndex + (int)(zOffset);
	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto neighborVoxelValue = VV2D(voxelValues[voxelValuesIndex]);


			auto neighborNormal = voxelNormals[voxelValuesIndex];

			if (VOXEL_INVALID == neighborVoxelValue) return;
			auto neighborVoxelValueCnt = VOXELCNT_VALUE(voxelValueCounts[voxelValuesIndex]);
			if (0 == neighborVoxelValueCnt) return;
			neighborVoxelValue /= (float)neighborVoxelValueCnt;

			//	mscho	@20250207
			//if (neighborVoxelValue < -1.0f || neighborVoxelValue > 1.0f) return;				
			//if (fabsf(neighborVoxelValue) < 0.00001f)
			//{
			//	return;
			//}

			Eigen::Vector3f neighborPos = info->cache.GetGlobalPosition(x, y, z);
			bool bSameSign = false;
			// voxel value의 부호가 같다는 의미는, isosurface 의 같은 방향에 있는 두개의 voxel이라는 의미임
			// center / neighbor voxel의 voxelvalue의 부호가 같은 경우에... 일정한 조건안에 있으면..
			// 이를 사용하도록 한다.
			if ((centerVoxelValue > 0.f && neighborVoxelValue > 0.f) || (centerVoxelValue < 0.f && neighborVoxelValue < 0.f))
			{
				// voxel 중심거리를 구한다.
				//auto center_distance = norm3df(centerPos.x() - neighborPos.x(), centerPos.y() - neighborPos.y(), centerPos.z() - neighborPos.z());
				//// 주변점의 voxelvalue가 중심거리의 절대값보다 작으면... return한다.
				//if (fabsf(neighborVoxelValue) < center_distance)
				//{
				//	//printf("\nneighborVoxelValue = %+f, center_distance = %+f\n", neighborVoxelValue, center_distance);
				//	return;
				//}
				//else
				//{
				// bSameSign = true;
				// //printf("\nneighborVoxelValue = %+f, center_distance = %+f\n", neighborVoxelValue, center_distance);
				//}

				//if (fabs(centerVoxelValue) >= ONE_VOXEL_SIZE * 0.5 || fabs(neighborVoxelValue) >= ONE_VOXEL_SIZE * 0.5)
				//	return;
				//else
				//	bSameSign = true;

				//	mscho	@20250207
				//	센터복셀과 주변복셀의 부호가 같고, 모두 0.05이하의  voxelvalue를 가지고 있다면,
				//	두개모두에서 iso surface가 생성될 수 있는 경우이다...
				if (fabs(centerVoxelValue) < ONE_VOXEL_SIZE * 0.5 * 1.414 && fabs(neighborVoxelValue) < ONE_VOXEL_SIZE * 0.5 * 1.414)
					bSameSign = true;
				else /*if(fabs(centerVoxelValue) > ONE_VOXEL_SIZE)*/
					return;
			}
			else
			{
				//	mscho	@20250207
				//	복셀의 부호가 다른 경우가 기본적으로 extraction하는 경우이다
				//float	distance = norm3df(
				//	centerPos.x() - neighborPos.x()
				//	,centerPos.y() - neighborPos.y()
				//	,centerPos.z() - neighborPos.z()
				//);
				//if (distance > 1.5)
				//{
				//	return;
				//}
				//if (centerVoxelValue >= 0.f && neighborVoxelValue < -1.5f) return;
				//else if (centerVoxelValue < 0.f && neighborVoxelValue > +1.5f) return;
			}
			//	mscho	@20250207
			//if (fabsf(centerVoxelValue) > 2.0f * ONE_VOXEL_SIZE && fabsf(neighborVoxelValue) > 2.0f * ONE_VOXEL_SIZE) return;

			//	mscho	@20250207
			//auto interpolatedPos = Interpolation(0.0f, centerPos, neighborPos, VV2D(centerVoxelValue), VV2D(neighborVoxelValue));
			Eigen::Vector3f		interpolatedPos;

			if (!bSameSign)
				interpolatedPos = Interpolation(0.0f, centerPos, neighborPos, VV2D(centerVoxelValue), VV2D(neighborVoxelValue));
			else
			{
				//float	absCenterValue = fabs(neighborVoxelValue);
				//float	absNeiValue = fabs(centerVoxelValue);
				float	ave_value = (centerVoxelValue * 0.7 + neighborVoxelValue * 0.3);
				auto	edge_pos = centerPos * 0.7 + neighborPos * 0.3;
				interpolatedPos = edge_pos - centerNormal * ave_value;
			}

			if ((voxelMin->x() <= interpolatedPos.x() && interpolatedPos.x() < voxelMax->x()) &&
				(voxelMin->y() <= interpolatedPos.y() && interpolatedPos.y() < voxelMax->y()) &&
				(voxelMin->z() <= interpolatedPos.z() && interpolatedPos.z() < voxelMax->z()))
			{
				//	mscho	@20250207
				//if (bSameSign)
				//	printf("\nneighborVoxelValue = %+f, centerVoxelValue = %+f\n", neighborVoxelValue, centerVoxelValue);
				if (neighborVoxelValue > 0)
				{
					//	mscho	@20250207
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						normal = (interpolatedPos - centerPos).normalized() * 7 + ave_normal * 3;
					}
					else
						normal = interpolatedPos - centerPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						//	mscho	@20250110
						//  Normal을 보다 더 날카롭게 한다.
						//else //if (dotNormal < 0.6f)
						//	calcNormal = dotNormal * centerNormal + (1.0 - dotNormal) * calcNormal;
						normal = calcNormal.normalized();
					}
					else
						normal = centerNormal;
					(*accumulatedNormal) += normal;
				}
				else
				{
					//	mscho	@20250207
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						normal = (centerPos - interpolatedPos).normalized() * 7 + ave_normal * 3;
					}
					else
						normal = centerPos - interpolatedPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						normal = calcNormal.normalized();
					}
					else
						normal = centerNormal;
					(*accumulatedNormal) += normal;
				}

				if (voxelColors != nullptr) {
					auto color = voxelColors[voxelValuesIndex];
					(*accumulatedColor) += Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f;
				}
				(*accumulatedPos) += interpolatedPos;
				(*countOfAccumulation)++;

				(*patchID) += voxelExtraAttribs[voxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[voxelValuesIndex];
			}
		}
		else {
			auto centerVoxelValuesIndex = gridSlotIndexCache[
				zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
					+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];
			Eigen::Vector3f pp(xOffset, yOffset, zOffset);
			pp.normalize();
			//해당 위치의 복셀데이터가 없는 경우, 중심점의 노말을 고려하여
			//해당 지점의 값을 대표할 수 있다고 판단되면(노말 각도와 중심점의 정보 누적 횟수) 
			//해당 지점의 데이터에 중심점의 정보를 변형하여 업데이트 하도록 한다.
			//이를 통해 신뢰도가 높은 복셀의 정보를 이용하여 extract에서 데이터 충실도를 높이는 효과를 노린다.
			if (pp.dot(centerNormal) > 0.95f && voxelValueCounts[centerVoxelValuesIndex] > 20) {
				(*accumulatedNormal) += centerNormal;

				auto cs = voxelColors[centerVoxelValuesIndex];
				(*accumulatedColor) += Eigen::Vector3f(
					(float)cs.x() / 255.0f,
					(float)cs.y() / 255.0f,
					(float)cs.z() / 255.0f);
				(*accumulatedPos) += (centerPos - centerNormal * centerVoxelValue);
				(*countOfAccumulation)++;
				(*patchID) += voxelExtraAttribs[centerVoxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[centerVoxelValuesIndex];
			}

		}
	}
}

//	mscho	@20250228
//	normal 계산하는 부분과 동일한 부호를 가지는 voxel의 extraction 알고리즘이 변경되었다.
__device__ void Device_ForEachNeighbor_v6_enhanced(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info,
	size_t xCacheIndex,
	size_t yCacheIndex,
	size_t zCacheIndex,
	const uint32_t * gridSlotIndexCache,
	const VoxelExtraAttrib * voxelExtraAttribs,
	const voxel_value_t * voxelValues,
	const unsigned short* voxelValueCounts,
	const Eigen::Vector3b * voxelColors,
	const Eigen::Vector3f * voxelNormals,
	const char* voxelSegmentations,
	voxel_value_t centerVoxelValue,
	Eigen::Vector3f centerPos,
	Eigen::Vector3f centerNormal,
	Eigen::Vector3f * voxelMin,
	Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * accumulatedPos,
	Eigen::Vector3f * accumulatedNormal,
	Eigen::Vector3f * accumulatedColor,
	unsigned int* countOfAccumulation,
	float	normal_ext_weight,
	int* toothCnt,
	int* patchID)
{
	auto x = (int)xCacheIndex + (int)(xOffset);
	auto y = (int)yCacheIndex + (int)(yOffset);
	auto z = (int)zCacheIndex + (int)(zOffset);
	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto neighborVoxelValue = VV2D(voxelValues[voxelValuesIndex]);


			auto neighborNormal = voxelNormals[voxelValuesIndex];

			if (VOXEL_INVALID == neighborVoxelValue) return;
			auto neighborVoxelValueCnt = VOXELCNT_VALUE(voxelValueCounts[voxelValuesIndex]);
			if (0 == neighborVoxelValueCnt) return;
			neighborVoxelValue /= (float)neighborVoxelValueCnt;

			Eigen::Vector3f neighborPos = info->cache.GetGlobalPosition(x, y, z);
			bool bSameSign = false;
			// voxel value의 부호가 같다는 의미는, isosurface 의 같은 방향에 있는 두개의 voxel이라는 의미임
			// center / neighbor voxel의 voxelvalue의 부호가 같은 경우에... 일정한 조건안에 있으면..
			// 이를 사용하도록 한다.
			//	mscho	@20250228
			if ((centerVoxelValue >= 0.f && neighborVoxelValue >= 0.f) || (centerVoxelValue < 0.f && neighborVoxelValue < 0.f))
			{
				//	mscho	@20250207	==> @20250228
				//	센터복셀과 주변복셀의 부호가 같고, 모두 0.05이하의  voxelvalue를 가지고 있다면,
				//	두개모두에서 iso surface가 생성될 수 있는 경우이다...
				if (fabs(centerVoxelValue) < ONE_VOXEL_SIZE * 0.707 && fabs(neighborVoxelValue) < ONE_VOXEL_SIZE * 0.707)
					bSameSign = true;
				//	mscho @20250312 같은 부호의 복셀을 Extract했을 때 두터운 포인트 클라우드가 나오는 문제가 있어서 아래 주석처리
				//else /*if(fabs(centerVoxelValue) > ONE_VOXEL_SIZE)*/
				return;
			}
			else
			{
				//	mscho	@20250207
				//	복셀의 부호가 다른 경우가 기본적으로 extraction하는 경우이다
				//float	distance = norm3df(
				//	centerPos.x() - neighborPos.x()
				//	,centerPos.y() - neighborPos.y()
				//	,centerPos.z() - neighborPos.z()
				//);
				//if (distance > 1.5)
				//{
				//	return;
				//}
				//if (centerVoxelValue >= 0.f && neighborVoxelValue < -1.5f) return;
				//else if (centerVoxelValue < 0.f && neighborVoxelValue > +1.5f) return;
			}

			Eigen::Vector3f		interpolatedPos;

			if (!bSameSign)
				interpolatedPos = Interpolation(0.0f, centerPos, neighborPos, VV2D(centerVoxelValue), VV2D(neighborVoxelValue));
			else
			{
				//float	absCenterValue = fabs(neighborVoxelValue);
				//float	absNeiValue = fabs(centerVoxelValue);
				float	ave_value = (centerVoxelValue * 0.7 + neighborVoxelValue * 0.3);
				auto	edge_pos = centerPos * 0.7 + neighborPos * 0.3;
				interpolatedPos = edge_pos - centerNormal * ave_value;
			}

			if ((voxelMin->x() <= interpolatedPos.x() && interpolatedPos.x() < voxelMax->x()) &&
				(voxelMin->y() <= interpolatedPos.y() && interpolatedPos.y() < voxelMax->y()) &&
				(voxelMin->z() <= interpolatedPos.z() && interpolatedPos.z() < voxelMax->z()))
			{

				if (neighborVoxelValue > 0)
				{
					//	mscho	@20250207	==> @20250228
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						Eigen::Vector3f calcNormal = (interpolatedPos - centerPos).normalized();
						//	mscho	@20250228
						normal = calcNormal * normal_ext_weight + ave_normal * (1.0 - normal_ext_weight);
						float dotNormal = normal.dot(centerNormal);
						if (dotNormal < 0.3f)
						{
							normal = centerNormal;
						}
					}
					else
						normal = interpolatedPos - centerPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						//	mscho	@20250529
						if (dotNormal < -0.1f)
							return;
						//	mscho	@20250110
						//  Normal을 보다 더 날카롭게 한다.
						//else //if (dotNormal < 0.6f)
						//	calcNormal = dotNormal * centerNormal + (1.0 - dotNormal) * calcNormal;
						else
						{
							//	mscho	@20250228
							normal = calcNormal.normalized();
							//	mscho	@20250529
							if (dotNormal < 0.3f && false)
							{
								normal = centerNormal;
							}
						}
					}
					else
						normal = centerNormal;
					//normal = centerNormal;
					(*accumulatedNormal) += normal;
				}
				else
				{
					//	mscho	@20250207	==> @20250228
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						//	mscho	@20250228
						Eigen::Vector3f calcNormal = (centerPos - interpolatedPos).normalized();
						//	mscho	@20250228
						normal = calcNormal * normal_ext_weight + ave_normal * (1.0 - normal_ext_weight);
						float dotNormal = normal.dot(centerNormal);
						if (dotNormal < 0.3f)
						{
							normal = centerNormal;
						}
					}
					else
						normal = centerPos - interpolatedPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						//	mscho	@20250529
						if (dotNormal < -0.1f)
							return;
						else
						{
							//	mscho	@20250228
							normal = calcNormal.normalized();
							//	mscho	@20250529
							if (dotNormal < 0.3f && false)
							{
								normal = centerNormal;
							}
						}
					}
					else
						normal = centerNormal;
					//normal = centerNormal;
					(*accumulatedNormal) += normal;
				}

				if (voxelColors != nullptr) {
					auto color = voxelColors[voxelValuesIndex];
					(*accumulatedColor) += Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f;
				}
				(*accumulatedPos) += interpolatedPos;
				(*countOfAccumulation)++;

				(*patchID) += voxelExtraAttribs[voxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[voxelValuesIndex];
			}
		}
		else {
			auto centerVoxelValuesIndex = gridSlotIndexCache[
				zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
					+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];
			Eigen::Vector3f pp(xOffset, yOffset, zOffset);
			pp.normalize();
			//해당 위치의 복셀데이터가 없는 경우, 중심점의 노말을 고려하여
			//해당 지점의 값을 대표할 수 있다고 판단되면(노말 각도와 중심점의 정보 누적 횟수) 
			//해당 지점의 데이터에 중심점의 정보를 변형하여 업데이트 하도록 한다.
			//이를 통해 신뢰도가 높은 복셀의 정보를 이용하여 extract에서 데이터 충실도를 높이는 효과를 노린다.
			if (pp.dot(centerNormal) > 0.95f && voxelValueCounts[centerVoxelValuesIndex] > 20) {
				(*accumulatedNormal) += centerNormal;

				auto cs = voxelColors[centerVoxelValuesIndex];
				(*accumulatedColor) += Eigen::Vector3f(
					(float)cs.x() / 255.0f,
					(float)cs.y() / 255.0f,
					(float)cs.z() / 255.0f);
				(*accumulatedPos) += (centerPos - centerNormal * centerVoxelValue);
				(*countOfAccumulation)++;
				(*patchID) += voxelExtraAttribs[centerVoxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[centerVoxelValuesIndex];
			}

		}
	}
}

__device__ void Device_ForEachNeighbor_forClustering(short xOffset, short yOffset, short zOffset,
	MarchingCubes::ExecutionInfo * info,
	size_t xCacheIndex,
	size_t yCacheIndex,
	size_t zCacheIndex,
	const uint32_t * gridSlotIndexCache,
	const VoxelExtraAttrib * voxelExtraAttribs,
	const voxel_value_t * voxelValues,
	const unsigned short* voxelValueCounts,
	const Eigen::Vector3b * voxelColors,
	const Eigen::Vector3f * voxelNormals,
	const char* voxelSegmentations,
	voxel_value_t centerVoxelValue,
	Eigen::Vector3f centerPos,
	Eigen::Vector3f centerNormal,
	unsigned int* centerLabel,
	Eigen::Vector3f * voxelMin,
	Eigen::Vector3f * voxelMax,
	Eigen::Vector3f * accumulatedPos,
	Eigen::Vector3f * accumulatedNormal,
	Eigen::Vector3f * accumulatedColor,
	unsigned int* countOfAccumulation,
	float	normal_ext_weight,
	int* toothCnt,
	int* patchID)
{
	auto x = (int)xCacheIndex + (int)(xOffset);
	auto y = (int)yCacheIndex + (int)(yOffset);
	auto z = (int)zCacheIndex + (int)(zOffset);
	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ))
	{
		auto voxelValuesIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != voxelValuesIndex)
		{
			auto neighborVoxelValue = VV2D(voxelValues[voxelValuesIndex]);
			if (VOXEL_INVALID == neighborVoxelValue) return;

			auto neighborNormal = voxelNormals[voxelValuesIndex];
			auto neighborLabel = voxelExtraAttribs[voxelValuesIndex].label;

			auto neighborVoxelValueCnt = VOXELCNT_VALUE(voxelValueCounts[voxelValuesIndex]);
			if (0 == neighborVoxelValueCnt) return;

			neighborVoxelValue /= (float)neighborVoxelValueCnt;

			Eigen::Vector3f neighborPos = info->cache.GetGlobalPosition(x, y, z);

			bool bSameSign = false;

			// voxel value의 부호가 같다는 의미는, isosurface 의 같은 방향에 있는 두개의 voxel이라는 의미임
			// center / neighbor voxel의 voxelvalue의 부호가 같은 경우에... 일정한 조건안에 있으면..
			// 이를 사용하도록 한다.
			//	mscho	@20250228
			if ((centerVoxelValue >= 0.f && neighborVoxelValue >= 0.f) || (centerVoxelValue < 0.f && neighborVoxelValue < 0.f))
			{
				//	mscho	@20250207	==> @20250228
				//	센터복셀과 주변복셀의 부호가 같고, 모두 0.05이하의  voxelvalue를 가지고 있다면,
				//	두개모두에서 iso surface가 생성될 수 있는 경우이다...
				if (fabs(centerVoxelValue) < ONE_VOXEL_SIZE * 0.707 && fabs(neighborVoxelValue) < ONE_VOXEL_SIZE * 0.707)
					bSameSign = true;
				//	mscho @20250312 같은 부호의 복셀을 Extract했을 때 두터운 포인트 클라우드가 나오는 문제가 있어서 아래 주석처리
				//else /*if(fabs(centerVoxelValue) > ONE_VOXEL_SIZE)*/
				return;
			}
			else
			{
				//	mscho	@20250207
				//	복셀의 부호가 다른 경우가 기본적으로 extraction하는 경우이다
				//float	distance = norm3df(
				//	centerPos.x() - neighborPos.x()
				//	,centerPos.y() - neighborPos.y()
				//	,centerPos.z() - neighborPos.z()
				//);
				//if (distance > 1.5)
				//{
				//	return;
				//}
				//if (centerVoxelValue >= 0.f && neighborVoxelValue < -1.5f) return;
				//else if (centerVoxelValue < 0.f && neighborVoxelValue > +1.5f) return;
			}

			Eigen::Vector3f		interpolatedPos;

			if (!bSameSign)
				interpolatedPos = Interpolation(0.0f, centerPos, neighborPos, VV2D(centerVoxelValue), VV2D(neighborVoxelValue));
			else
			{
				//float	absCenterValue = fabs(neighborVoxelValue);
				//float	absNeiValue = fabs(centerVoxelValue);
				float	ave_value = (centerVoxelValue * 0.7 + neighborVoxelValue * 0.3);
				auto	edge_pos = centerPos * 0.7 + neighborPos * 0.3;
				interpolatedPos = edge_pos - centerNormal * ave_value;
			}

			if ((voxelMin->x() <= interpolatedPos.x() && interpolatedPos.x() < voxelMax->x()) &&
				(voxelMin->y() <= interpolatedPos.y() && interpolatedPos.y() < voxelMax->y()) &&
				(voxelMin->z() <= interpolatedPos.z() && interpolatedPos.z() < voxelMax->z()))
			{
				if (*centerLabel > 0 && neighborLabel > 0 && *centerLabel > neighborLabel)
					(*centerLabel) = neighborLabel;

				if (neighborVoxelValue > 0)
				{
					//	mscho	@20250207	==> @20250228
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						Eigen::Vector3f calcNormal = (interpolatedPos - centerPos).normalized();
						//	mscho	@20250228
						normal = calcNormal * normal_ext_weight + ave_normal * (1.0 - normal_ext_weight);
						float dotNormal = normal.dot(centerNormal);
						if (dotNormal < 0.3f)
						{
							normal = centerNormal;
						}
					}
					else
						normal = interpolatedPos - centerPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						//	mscho	@20250110
						//  Normal을 보다 더 날카롭게 한다.
						//else //if (dotNormal < 0.6f)
						//	calcNormal = dotNormal * centerNormal + (1.0 - dotNormal) * calcNormal;
						else
						{
							//	mscho	@20250228
							normal = calcNormal.normalized();
							if (dotNormal < 0.3f)
							{
								normal = centerNormal;
							}
						}
					}
					else
						normal = centerNormal;
					//normal = centerNormal;
					(*accumulatedNormal) += normal;
				}
				else
				{
					//	mscho	@20250207	==> @20250228
					Eigen::Vector3f normal;
					if (bSameSign)
					{
						auto ave_normal = centerNormal * 0.7 + neighborNormal * 0.3;
						//	mscho	@20250228
						Eigen::Vector3f calcNormal = (centerPos - interpolatedPos).normalized();
						//	mscho	@20250228
						normal = calcNormal * normal_ext_weight + ave_normal * (1.0 - normal_ext_weight);
						float dotNormal = normal.dot(centerNormal);
						if (dotNormal < 0.3f)
						{
							normal = centerNormal;
						}
					}
					else
						normal = centerPos - interpolatedPos;
					float distance = norm3df(normal.x(), normal.y(), normal.z());
					if (distance > 0)
					{
						Eigen::Vector3f calcNormal = normal / distance;
						float dotNormal = calcNormal.dot(centerNormal);
						if (dotNormal < 0.f)
							return;
						else
						{
							//	mscho	@20250228
							normal = calcNormal.normalized();
							if (dotNormal < 0.3f)
							{
								normal = centerNormal;
							}
						}
					}
					else
						normal = centerNormal;
					//normal = centerNormal;
					(*accumulatedNormal) += normal;
				}

				if (voxelColors != nullptr) {
					auto color = voxelColors[voxelValuesIndex];
					(*accumulatedColor) += Eigen::Vector3f(color.x(), color.y(), color.z()) / 255.0f;
				}
				(*accumulatedPos) += interpolatedPos;
				(*countOfAccumulation)++;

				(*patchID) += voxelExtraAttribs[voxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[voxelValuesIndex];
			}
		}
		else {
			auto centerVoxelValuesIndex = gridSlotIndexCache[
				zCacheIndex * (info->cache.voxelCountX * info->cache.voxelCountY)
					+ yCacheIndex * (info->cache.voxelCountX) + xCacheIndex];
			Eigen::Vector3f pp(xOffset, yOffset, zOffset);
			pp.normalize();
			//해당 위치의 복셀데이터가 없는 경우, 중심점의 노말을 고려하여
			//해당 지점의 값을 대표할 수 있다고 판단되면(노말 각도와 중심점의 정보 누적 횟수) 
			//해당 지점의 데이터에 중심점의 정보를 변형하여 업데이트 하도록 한다.
			//이를 통해 신뢰도가 높은 복셀의 정보를 이용하여 extract에서 데이터 충실도를 높이는 효과를 노린다.
			if (pp.dot(centerNormal) > 0.95f && voxelValueCounts[centerVoxelValuesIndex] > 20) {
				(*accumulatedNormal) += centerNormal;

				auto cs = voxelColors[centerVoxelValuesIndex];
				(*accumulatedColor) += Eigen::Vector3f(
					(float)cs.x() / 255.0f,
					(float)cs.y() / 255.0f,
					(float)cs.z() / 255.0f);
				(*accumulatedPos) += (centerPos - centerNormal * centerVoxelValue);
				(*countOfAccumulation)++;
				(*patchID) += voxelExtraAttribs[centerVoxelValuesIndex].startPatchID;
				if (toothCnt != nullptr)
					(*toothCnt) += (int)voxelSegmentations[centerVoxelValuesIndex];
			}

		}
	}
}

__global__ void Kernel_ExtractVoxelPoints_v5(
	MarchingCubes::ExecutionInfo info
	, unsigned int* slotIndices
	, uint32_t * gridSlotIndexCache

	, voxel_value_t * voxelValues
	, Eigen::Vector3f * voxelNormals
	, unsigned short* voxelValueCounts
	, Eigen::Vector3b * voxelColors
	, char* voxelSegmentations
	, VoxelExtraAttrib * voxelExtraAttribs

	// 출력
	, Eigen::Vector3f * pointPositions
	, const Eigen::Vector3f pointHalfPositions
	, Eigen::Vector3f * pointNormals
	, Eigen::Vector3b * pointColors
	, VoxelExtraAttrib * pointExtraAttribs

	, HashKey * output
	, uint32_t * cntOutputExtract
	, uint32_t * cntOutputContains
	, unsigned long long* UsedCacheArray_extract

	, bool use26Direction = true
	, bool filtering = true
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > info.globalHashInfo->Count_HashTableUsed - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadId = 0; threadId < info.globalHashInfo->Count_HashTableUsed; threadId++) {
#endif

		auto slotIndex = slotIndices[threadId];
		const auto key = info.globalHash[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto pc = info.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

		Eigen::Vector3f voxelMin = pc - pointHalfPositions;
		Eigen::Vector3f voxelMax = pc + pointHalfPositions;

		auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
			yCacheIndex * info.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= (size_t)info.cache.voxelCount)
			kernel_return;

		//	mscho	@20250228
		auto centerVoxelValue = voxelValues[slotIndex];
		if (!FLT_VALID(centerVoxelValue))
			//if (VOXEL_INVALID == centerVoxelValue)
			kernel_return;
		auto centerVoxelNormal = voxelNormals[slotIndex];
		if (!FLT_VALID(centerVoxelNormal.x()))
			kernel_return;
		centerVoxelNormal.normalize();
		auto centerVoxelValueCount = VOXELCNT_VALUE(voxelValueCounts[slotIndex]);
		if (centerVoxelValueCount < 1 || (filtering && centerVoxelValueCount < 2))
			kernel_return;

		auto centerVoxelLabel = voxelExtraAttribs[slotIndex].label;

		centerVoxelValue = voxelValues[slotIndex] / (voxel_value_t)centerVoxelValueCount;

		//=========================================================================================
		//	mscho	@20250207	==> @20250228
		//	추출하고자 하는 Point의 voxelvalue를 적용한 point의 위치가 해당 voxel을 넘어간다면..
		//	해당 voxel에서는 isor surface가 생성되지 않을 위치이므로....
		//	return하도록 한다.
		auto iso_pos = pc - centerVoxelNormal * centerVoxelValue;

		if ((voxelMin.x() > iso_pos.x() || iso_pos.x() > voxelMax.x()) ||
			(voxelMin.y() > iso_pos.y() || iso_pos.y() > voxelMax.y()) ||
			(voxelMin.z() > iso_pos.z() || iso_pos.z() > voxelMax.z()))
			kernel_return;
		//========================================================================================

		const auto centerVoxelColor = voxelColors[slotIndex];
		const VoxelExtraAttrib centerVoxelExtraAttrib = voxelExtraAttribs[slotIndex];
		const bool bIsoSurface = fabsf(centerVoxelValue) <= 0.001;
		int toothCnt = 0;
		int patchID = 0;
		Eigen::Vector3f accumulatedPoint(0, 0, 0);
		Eigen::Vector3f accumulatedNormal(0, 0, 0);
		Eigen::Vector3f accumulatedColor(0, 0, 0);
		uint32_t repesentLabel = 0;
		unsigned int countOfAccumulation = 0;
#ifdef USE_CLUSTERING
#define ForEachNeighbor_v5(xOffset, yOffset, zOffset)\
	Device_ForEachNeighbor_forClustering((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
	gridSlotIndexCache, voxelExtraAttribs, voxelValues, voxelValueCounts, voxelColors, voxelNormals, voxelSegmentations, \
	centerVoxelValue, pc, centerVoxelNormal, &centerVoxelLabel, &voxelMin, &voxelMax, &accumulatedPoint, &accumulatedNormal, &accumulatedColor, &countOfAccumulation, 0.5, &toothCnt, &patchID);
#else
		//	mscho	@20250228
#define ForEachNeighbor_v5(xOffset, yOffset, zOffset)\
	Device_ForEachNeighbor_v6_enhanced((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
	gridSlotIndexCache, voxelExtraAttribs, voxelValues, voxelValueCounts, voxelColors, voxelNormals, voxelSegmentations, \
	centerVoxelValue, pc, centerVoxelNormal, &voxelMin, &voxelMax, &accumulatedPoint, &accumulatedNormal, &accumulatedColor, &countOfAccumulation, 0.5, &toothCnt, &patchID);
#endif


		if (false == use26Direction)
		{
			ForEachNeighbor_v5(-1, 0, 0);
			ForEachNeighbor_v5(1, 0, 0);
			ForEachNeighbor_v5(0, -1, 0);
			ForEachNeighbor_v5(0, 1, 0);
			ForEachNeighbor_v5(0, 0, -1);
			ForEachNeighbor_v5(0, 0, 1);
		}
		else
		{
			ForEachNeighbor_v5(-1, -1, -1);
			ForEachNeighbor_v5(-1, -1, 0);
			ForEachNeighbor_v5(-1, -1, 1);

			ForEachNeighbor_v5(-1, 0, -1);
			ForEachNeighbor_v5(-1, 0, 0);
			ForEachNeighbor_v5(-1, 0, 1);

			ForEachNeighbor_v5(-1, 1, -1);
			ForEachNeighbor_v5(-1, 1, 0);
			ForEachNeighbor_v5(-1, 1, 1);

			ForEachNeighbor_v5(0, -1, -1);
			ForEachNeighbor_v5(0, -1, 0);
			ForEachNeighbor_v5(0, -1, 1);

			ForEachNeighbor_v5(0, 0, -1);
			//ForEachNeighbor_v5(0, 0, 0);
			ForEachNeighbor_v5(0, 0, 1);

			ForEachNeighbor_v5(0, 1, -1);
			ForEachNeighbor_v5(0, 1, 0);
			ForEachNeighbor_v5(0, 1, 1);

			ForEachNeighbor_v5(1, -1, -1);
			ForEachNeighbor_v5(1, -1, 0);
			ForEachNeighbor_v5(1, -1, 1);

			ForEachNeighbor_v5(1, 0, -1);
			ForEachNeighbor_v5(1, 0, 0);
			ForEachNeighbor_v5(1, 0, 1);

			ForEachNeighbor_v5(1, 1, -1);
			ForEachNeighbor_v5(1, 1, 0);
			ForEachNeighbor_v5(1, 1, 1);
		}

		const int DEF_CACHE_MARGIN_MIN = -6;
		const int DEF_CACHE_MARGIN_MAX = -2;

		//	mscho	@20250207	==> @20250228
		//  Extract 되는 point가 너무 지저분해서, 0 -> 1로 올렸다.
		//const uint32_t minCount = 3;
		const uint32_t minCount = 2;
		//const uint32_t minCount = 0;
		//const uint32_t minCount = 1;
		if (countOfAccumulation > minCount)
		{
			uint32_t prevTableIndex = atomicAdd(cntOutputExtract, (uint32_t)1);

			UsedCacheArray_extract[prevTableIndex] = cacheIndex;

			info.gridSlotIndexCache_pts[cacheIndex] = prevTableIndex;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, DEF_CACHE_MARGIN_MIN, DEF_CACHE_MARGIN_MAX))
			{
				uint32_t prevIndex = atomicAdd(cntOutputContains, (uint32_t)1);
				output[prevIndex] = key.Key();
			}
			if (bIsoSurface)
			{
				pointPositions[prevTableIndex] = pc;
			}
			else
			{
				pointPositions[prevTableIndex] = accumulatedPoint / countOfAccumulation;
			}
			pointNormals[prevTableIndex] = accumulatedNormal.normalized();
			auto avgColor = accumulatedColor / countOfAccumulation;

			pointColors[prevTableIndex] = Eigen::Vector3b(
				(unsigned char)(avgColor.x() * 255.0f),
				(unsigned char)(avgColor.y() * 255.0f),
				(unsigned char)(avgColor.z() * 255.0f));

			pointExtraAttribs[prevTableIndex] = centerVoxelExtraAttrib;

			if (toothCnt / (int)countOfAccumulation > -5) {
				pointExtraAttribs[prevTableIndex].materialID = 255;
			}
			else {
				pointExtraAttribs[prevTableIndex].materialID = 0;
			}
			pointExtraAttribs[prevTableIndex].startPatchID = patchID / countOfAccumulation;
#ifdef USE_CLUSTERING
			pointExtraAttribs[prevTableIndex].label = centerVoxelLabel;
#endif
		}
		//	mscho	@20250527
		else if (bIsoSurface && false)
		{
			uint32_t prevTableIndex = atomicAdd(cntOutputExtract, (uint32_t)1);
			auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
				yCacheIndex * info.cache.voxelCountX + xCacheIndex;

			UsedCacheArray_extract[prevTableIndex] = cacheIndex;

			info.gridSlotIndexCache_pts[cacheIndex] = prevTableIndex;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, DEF_CACHE_MARGIN_MIN, DEF_CACHE_MARGIN_MAX))
			{
				uint32_t prevIndex = atomicAdd(cntOutputContains, (uint32_t)1);
				output[prevIndex] = key.Key();
			}
			pointPositions[prevTableIndex] = pc;
			pointNormals[prevTableIndex] = centerVoxelNormal;
			pointColors[prevTableIndex] = centerVoxelColor;
			if (voxelSegmentations[slotIndex] >= -5) {
				pointExtraAttribs[prevTableIndex].materialID = 255;
			}
			else {
				pointExtraAttribs[prevTableIndex].materialID = 0;
			}
			pointExtraAttribs[prevTableIndex] = centerVoxelExtraAttrib;
		}
	}
	}


//	mscho	@20250714
//	Block  단위로 나눠서 extract하는 기능을 추가
__global__ void Kernel_ExtractVoxelPoints_v6(
	MarchingCubes::ExecutionInfo info
	, unsigned int* slotIndices
	, uint32_t * gridSlotIndexCache

	, voxel_value_t * voxelValues
	, Eigen::Vector3f * voxelNormals
	, unsigned short* voxelValueCounts
	, Eigen::Vector3b * voxelColors
	, char* voxelSegmentations
	, VoxelExtraAttrib * voxelExtraAttribs

	// 출력
	, Eigen::Vector3f * pointPositions
	, const Eigen::Vector3f pointHalfPositions
	, Eigen::Vector3f * pointNormals
	, Eigen::Vector3b * pointColors
	, VoxelExtraAttrib * pointExtraAttribs

	, HashKey * output
	, uint32_t * cntOutputExtract
	, uint32_t * cntOutputContains
	, unsigned long long* UsedCacheArray_extract

	, unsigned int	block_size
	, unsigned int block_capacity
	, unsigned int block_n

	, bool use26Direction = true
	, bool filtering = true
)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	//if (threadId > info.globalHashInfo->Count_HashTableUsed - 1) return;
	if (threadId > block_capacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadId = 0; threadId < info.globalHashInfo->Count_HashTableUsed; threadId++) {
#endif
		threadId = threadId + block_n * block_size;

		auto slotIndex = slotIndices[threadId];
		const auto key = info.globalHash[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto pc = info.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

		Eigen::Vector3f voxelMin = pc - pointHalfPositions;
		Eigen::Vector3f voxelMax = pc + pointHalfPositions;

		auto xCacheIndex = xGlobalIndex - info.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - info.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - info.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
			yCacheIndex * info.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= (size_t)info.cache.voxelCount)
			kernel_return;

		//	mscho	@20250228
		auto centerVoxelValue = voxelValues[slotIndex];
		if (!FLT_VALID(centerVoxelValue))
			//if (VOXEL_INVALID == centerVoxelValue)
			kernel_return;
		auto centerVoxelNormal = voxelNormals[slotIndex];
		if (!FLT_VALID(centerVoxelNormal.x()))
			kernel_return;
		centerVoxelNormal.normalize();
		auto centerVoxelValueCount = VOXELCNT_VALUE(voxelValueCounts[slotIndex]);
		if (centerVoxelValueCount < 1 || (filtering && centerVoxelValueCount < 2))
			kernel_return;

		auto centerVoxelLabel = voxelExtraAttribs[slotIndex].label;

		centerVoxelValue = voxelValues[slotIndex] / (voxel_value_t)centerVoxelValueCount;

		//=========================================================================================
		//	mscho	@20250207	==> @20250228
		//	추출하고자 하는 Point의 voxelvalue를 적용한 point의 위치가 해당 voxel을 넘어간다면..
		//	해당 voxel에서는 isor surface가 생성되지 않을 위치이므로....
		//	return하도록 한다.
		auto iso_pos = pc - centerVoxelNormal * centerVoxelValue;

		if ((voxelMin.x() > iso_pos.x() || iso_pos.x() > voxelMax.x()) ||
			(voxelMin.y() > iso_pos.y() || iso_pos.y() > voxelMax.y()) ||
			(voxelMin.z() > iso_pos.z() || iso_pos.z() > voxelMax.z()))
			kernel_return;
		//========================================================================================

		const auto centerVoxelColor = voxelColors[slotIndex];
		const VoxelExtraAttrib centerVoxelExtraAttrib = voxelExtraAttribs[slotIndex];
		const bool bIsoSurface = fabsf(centerVoxelValue) <= 0.001;
		int toothCnt = 0;
		int patchID = 0;
		Eigen::Vector3f accumulatedPoint(0, 0, 0);
		Eigen::Vector3f accumulatedNormal(0, 0, 0);
		Eigen::Vector3f accumulatedColor(0, 0, 0);
		uint32_t repesentLabel = 0;
		unsigned int countOfAccumulation = 0;
#ifdef USE_CLUSTERING
#define ForEachNeighbor_v5(xOffset, yOffset, zOffset)\
	Device_ForEachNeighbor_forClustering((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
	gridSlotIndexCache, voxelExtraAttribs, voxelValues, voxelValueCounts, voxelColors, voxelNormals, voxelSegmentations, \
	centerVoxelValue, pc, centerVoxelNormal, &centerVoxelLabel, &voxelMin, &voxelMax, &accumulatedPoint, &accumulatedNormal, &accumulatedColor, &countOfAccumulation, 0.5, &toothCnt, &patchID);
#else
		//	mscho	@20250228
#define ForEachNeighbor_v5(xOffset, yOffset, zOffset)\
	Device_ForEachNeighbor_v6_enhanced((xOffset), (yOffset), (zOffset), &info, xCacheIndex, yCacheIndex, zCacheIndex,\
	gridSlotIndexCache, voxelExtraAttribs, voxelValues, voxelValueCounts, voxelColors, voxelNormals, voxelSegmentations, \
	centerVoxelValue, pc, centerVoxelNormal, &voxelMin, &voxelMax, &accumulatedPoint, &accumulatedNormal, &accumulatedColor, &countOfAccumulation, 0.5, &toothCnt, &patchID);
#endif


		if (false == use26Direction)
		{
			ForEachNeighbor_v5(-1, 0, 0);
			ForEachNeighbor_v5(1, 0, 0);
			ForEachNeighbor_v5(0, -1, 0);
			ForEachNeighbor_v5(0, 1, 0);
			ForEachNeighbor_v5(0, 0, -1);
			ForEachNeighbor_v5(0, 0, 1);
		}
		else
		{
			ForEachNeighbor_v5(-1, -1, -1);
			ForEachNeighbor_v5(-1, -1, 0);
			ForEachNeighbor_v5(-1, -1, 1);

			ForEachNeighbor_v5(-1, 0, -1);
			ForEachNeighbor_v5(-1, 0, 0);
			ForEachNeighbor_v5(-1, 0, 1);

			ForEachNeighbor_v5(-1, 1, -1);
			ForEachNeighbor_v5(-1, 1, 0);
			ForEachNeighbor_v5(-1, 1, 1);

			ForEachNeighbor_v5(0, -1, -1);
			ForEachNeighbor_v5(0, -1, 0);
			ForEachNeighbor_v5(0, -1, 1);

			ForEachNeighbor_v5(0, 0, -1);
			//ForEachNeighbor_v5(0, 0, 0);
			ForEachNeighbor_v5(0, 0, 1);

			ForEachNeighbor_v5(0, 1, -1);
			ForEachNeighbor_v5(0, 1, 0);
			ForEachNeighbor_v5(0, 1, 1);

			ForEachNeighbor_v5(1, -1, -1);
			ForEachNeighbor_v5(1, -1, 0);
			ForEachNeighbor_v5(1, -1, 1);

			ForEachNeighbor_v5(1, 0, -1);
			ForEachNeighbor_v5(1, 0, 0);
			ForEachNeighbor_v5(1, 0, 1);

			ForEachNeighbor_v5(1, 1, -1);
			ForEachNeighbor_v5(1, 1, 0);
			ForEachNeighbor_v5(1, 1, 1);
		}

		const int DEF_CACHE_MARGIN_MIN = -6;
		const int DEF_CACHE_MARGIN_MAX = -2;

		//	mscho	@20250207	==> @20250228
		//  Extract 되는 point가 너무 지저분해서, 0 -> 1로 올렸다.
		//const uint32_t minCount = 3;
		const uint32_t minCount = 2;
		//const uint32_t minCount = 0;
		//const uint32_t minCount = 1;
		if (countOfAccumulation > minCount)
		{
			uint32_t prevTableIndex = atomicAdd(cntOutputExtract, (uint32_t)1);

			UsedCacheArray_extract[prevTableIndex] = cacheIndex;

			info.gridSlotIndexCache_pts[cacheIndex] = prevTableIndex;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, DEF_CACHE_MARGIN_MIN, DEF_CACHE_MARGIN_MAX))
			{
				uint32_t prevIndex = atomicAdd(cntOutputContains, (uint32_t)1);
				output[prevIndex] = key.Key();
			}
			if (bIsoSurface)
			{
				pointPositions[prevTableIndex] = pc;
			}
			else
			{
				pointPositions[prevTableIndex] = accumulatedPoint / countOfAccumulation;
			}
			pointNormals[prevTableIndex] = accumulatedNormal.normalized();
			auto avgColor = accumulatedColor / countOfAccumulation;

			pointColors[prevTableIndex] = Eigen::Vector3b(
				(unsigned char)(avgColor.x() * 255.0f),
				(unsigned char)(avgColor.y() * 255.0f),
				(unsigned char)(avgColor.z() * 255.0f));

			pointExtraAttribs[prevTableIndex] = centerVoxelExtraAttrib;

			if (toothCnt / (int)countOfAccumulation > -5) {
				pointExtraAttribs[prevTableIndex].materialID = 255;
			}
			else {
				pointExtraAttribs[prevTableIndex].materialID = 0;
			}
			pointExtraAttribs[prevTableIndex].startPatchID = patchID / countOfAccumulation;
#ifdef USE_CLUSTERING
			pointExtraAttribs[prevTableIndex].label = centerVoxelLabel;
#endif
		}
		//	mscho	@20250527
		else if (bIsoSurface && false)
		{
			uint32_t prevTableIndex = atomicAdd(cntOutputExtract, (uint32_t)1);
			auto cacheIndex = zCacheIndex * info.cache.voxelCountX * info.cache.voxelCountY +
				yCacheIndex * info.cache.voxelCountX + xCacheIndex;

			UsedCacheArray_extract[prevTableIndex] = cacheIndex;

			info.gridSlotIndexCache_pts[cacheIndex] = prevTableIndex;
			if (info.local.ContainsWithMargin(xGlobalIndex, yGlobalIndex, zGlobalIndex, DEF_CACHE_MARGIN_MIN, DEF_CACHE_MARGIN_MAX))
			{
				uint32_t prevIndex = atomicAdd(cntOutputContains, (uint32_t)1);
				output[prevIndex] = key.Key();
			}
			pointPositions[prevTableIndex] = pc;
			pointNormals[prevTableIndex] = centerVoxelNormal;
			pointColors[prevTableIndex] = centerVoxelColor;
			if (voxelSegmentations[slotIndex] >= -5) {
				pointExtraAttribs[prevTableIndex].materialID = 255;
			}
			else {
				pointExtraAttribs[prevTableIndex].materialID = 0;
			}
			pointExtraAttribs[prevTableIndex] = centerVoxelExtraAttrib;
		}
	}
	}


#ifndef BUILD_FOR_CPU
void MarchingCubes::CaptureFrame(cached_allocator * alloc_, CUstream_st * st)
{
	SaveVoxelValues(pSettings->GetResourcesFolderPath() + "\\FrameCapture\\VoxelValues.ply", -1.0f, 1.0f, alloc_, st);
}
#endif

__device__ void Device_ForEachNeighborPt(
	short xOffset, short yOffset, short zOffset,
	const MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,
	Eigen::Vector3b * hash_vtx_clr,

	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3ui * ac,

	unsigned int* apcount,
	unsigned int* ancount)
{
	size_t x = xCacheIndex + (xOffset);
	size_t y = yCacheIndex + (yOffset);
	size_t z = zCacheIndex + (zOffset);
	if (x < info->cache.voxelCountX &&
		y < info->cache.voxelCountY &&
		z < info->cache.voxelCountZ) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex)
		{
			(*ap) += hash_vtx_pos[pointSlotIndex];
			(*an) += hash_vtx_nm[pointSlotIndex];
			(*ac) += hash_vtx_clr[pointSlotIndex].cast<unsigned int>();

			(*apcount)++;
			(*ancount)++;
		}
	}
}

//	mscho	@20240422
__device__ void Device_ForEachNeighborPt_v2(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f basePos, Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,
	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,
	Eigen::Vector3b * hash_vtx_clr,

	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3ui * ac,

	unsigned int* apcount,
	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	float	max_distance = norm3df((float)ave_step, (float)ave_step, (float)ave_step);
	max_distance *= 1.2f;

	float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{

			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;
			//	mscho	@20240521 (New)
			float	weight = (1.f - (distance / max_distance));
			float	wieght_ = powf(weight, 2.);
			int		iweight = (int)(wieght_ * 20.f);

			if (baseNormal.dot(normal) >= filteringValue) {
				Eigen::Vector3f vecP = hash_vtx_pos[pointSlotIndex] - basePos;
				Eigen::Vector3f disp = basePos + baseNormal.dot(vecP) * baseNormal;
				(*ap) += (disp * iweight);
				(*an) += (normal * iweight);
				auto cs = hash_vtx_clr[pointSlotIndex];
				(*ac) += cs.cast<unsigned int>() * iweight;

				(*apcount) += iweight;
			}
			//(*ancount)++;
		}
	}
}



#ifndef BUILD_FOR_CPU
//	mscho	@20240507 ==> @20240524
// neighbor normal 버퍼로 pos를 넣어주며,
// weight를 구해 point, nm, clor를 처리함.
__device__ void Device_ForEachNeighborPt_v3(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f basePos, Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,
	Eigen::Vector3b * hash_vtx_clr,

	Eigen::Vector3f * pos_nei,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3f * ac,

	unsigned int* apcount,
	unsigned int* neighborcount,
	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	float	max_distance = (float)ave_step;
	max_distance += 0.5f;
	int	step_width = ave_step * 2 + 1;
	auto nei_idx = (zOffset + ave_step) * step_width * step_width + (yOffset + ave_step) * step_width + (xOffset + ave_step);

	float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{

			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];

			float	weight = (1.f - (distance / max_distance));
			float	wieght_ = powf(weight, 2.);
			int		iweight = (int)(wieght_ * 10.f);

			if (baseNormal.dot(normal) > filteringValue) {
				Eigen::Vector3f vecP = hash_vtx_pos[pointSlotIndex] - basePos;
				Eigen::Vector3f disp = basePos + baseNormal.dot(vecP) * baseNormal;

				if (xOffset != 0 || yOffset != 0 || zOffset != 0)
					pos_nei[nei_idx] = hash_vtx_pos[pointSlotIndex];

				(*ap) += (disp * iweight);
				(*an) += (normal * iweight);
				auto cs = hash_vtx_clr[pointSlotIndex];
				(*ac) += (Eigen::Vector3f(
					(float)cs.x(),
					(float)cs.y(),
					(float)cs.z()) * iweight);

				(*apcount) += iweight;
				(*neighborcount)++;
			}
			//(*ancount)++;
		}
	}
}
#endif


//	mscho	@20250627
//	point와 nomal은 가중치를 줘도 괜찮을 듯 한데..
//  color noise 저감을 위한 수정
//  point / normal의 average를 위한 sum을 개별적으로 제어할 수 있도록 한다.
__device__ void Device_ForEachNeighborPt_v7(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f basePos, Eigen::Vector3f baseNormal, float filteringValue,
	const	bool bEnPos,
	const	bool bEnNor,
	const	bool bEnColor,

	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,
	Eigen::Vector3b * hash_vtx_clr,

	//Eigen::Vector3f* pos_nei,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3ui * ac,

	unsigned int* apcount,
	unsigned int* account,
	unsigned int* neighborcount,
	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	//	mscho	@20250228
	//	Max ditance를 너무 좁게 해 놓으면, 중심과 주변점의 가중치 차이가 너무 많이 나게 되는데..
	//	넓은 영역의 경우에는 의미가 있지만, 좁은 영역을 Averaging할 경우에는 거친 상황이 발생하게 된다..
	//	가우시안 값을 손대지 않고, Max distance를 늘이는 방법으로, smoothing의 정도를 조금 세게 한다.
	float	max_distance = (float)ave_step * 2;
	//	mscho	@20250529
	//max_distance += 0.5f;
	max_distance += 2.0f;

	//	mscho	@20250701
	//printf("ave_step = %d, %f - %f\n", ave_step, (float)ave_step * 0.2, norm3df((float)ave_step * 0.2, (float)ave_step * 0.2, (float)ave_step * 0.2));
	float	voxel_max_distance = norm3df((float)ave_step * 0.2, (float)ave_step * 0.2, (float)ave_step * 0.2);
	int	step_width = ave_step * 2 + 1;
	auto nei_idx = (zOffset + ave_step) * step_width * step_width + (yOffset + ave_step) * step_width + (xOffset + ave_step);

	float	distance = norm3df((float)xOffset, (float)yOffset, (float)zOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;

			//	mscho	@20250313
			auto PosDiff = basePos - hash_vtx_pos[pointSlotIndex];
			distance = norm3df(PosDiff.x(), PosDiff.y(), PosDiff.z());

			//	mscho	@20250701
			//float	weight = (1.f - (distance / max_distance));
			float	weight = (1.f - (distance / voxel_max_distance));
			if (weight < 0.f)	weight = 0.f;

			float	wieght_ = powf(weight, 4.);
			//int		iweight = (int)(wieght_ * 20.f);//	mscho	@20240507
			int		iweight = (int)(wieght_ * 100.f);//	mscho	@20250610
			//	평균에 있어서, 거리의 가중치를 높인다.,

			//float	wieght_ = powf(weight, 2.);
			//int		iweight = (int)(wieght_ * 10.f);//	mscho	@20240507

			//	mscho	@20240605
			int		noweight = iweight;
			//	엣지부분에서 color noise가 발생하는 원인이
			//	color ave에서 weighted를 하는 경우에 발생하였다..

			//	mscho	@20250110
			//	normal weight를 1==> iweight 로 바꿔본다.
			//int		colorweight = 1;

			int		colorweight = 1;

			//	mscho	@20240527
			if (baseNormal.dot(normal) > filteringValue) {
				if (bEnPos)
				{
					Eigen::Vector3f disp = hash_vtx_pos[pointSlotIndex];
					(*ap) += (disp * noweight);
					(*apcount) += noweight;
				}
				if (bEnColor)
				{
					auto cs = hash_vtx_clr[pointSlotIndex].cast<unsigned int>();
					(*ac) += cs * colorweight;
					(*account) += colorweight;
				}
				if (bEnNor)
				{
					float	wieght_normal = powf(weight, 0.5);
					(*an) += (normal * wieght_normal);	//	mscho	@20240620 ==> @20250110
				}
				(*neighborcount)++;
			}
		}
	}
}



//	mscho	@20240611
//	point와 nomal은 가중치를 줘도 괜찮을 듯 한데..
//  color noise 저감을 위한 수정
__device__ void Device_ForEachNeighborPt_v6(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f basePos, Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,
	Eigen::Vector3b * hash_vtx_clr,

	//Eigen::Vector3f* pos_nei,
	Eigen::Vector3f * ap, Eigen::Vector3f * an, Eigen::Vector3ui * ac,

	unsigned int* apcount,
	unsigned int* account,
	unsigned int* neighborcount,
	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	//	mscho	@20250228
	//	Max ditance를 너무 좁게 해 놓으면, 중심과 주변점의 가중치 차이가 너무 많이 나게 되는데..
	//	넓은 영역의 경우에는 의미가 있지만, 좁은 영역을 Averaging할 경우에는 거친 상황이 발생하게 된다..
	//	가우시안 값을 손대지 않고, Max distance를 늘이는 방법으로, smoothing의 정도를 조금 세게 한다.
	float	max_distance = (float)ave_step * 2;
	//	mscho	@20250529
	//max_distance += 0.5f;
	max_distance += 2.0f;
	int	step_width = ave_step * 2 + 1;
	auto nei_idx = (zOffset + ave_step) * step_width * step_width + (yOffset + ave_step) * step_width + (xOffset + ave_step);

	float	distance = norm3df((float)xOffset, (float)yOffset, (float)zOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;

			//	mscho	@20250313
			auto PosDiff = basePos - hash_vtx_pos[pointSlotIndex];
			distance = norm3df(PosDiff.x(), PosDiff.y(), PosDiff.z());

			float	weight = (1.f - (distance / max_distance));
			float	wieght_ = powf(weight, 4.);
			//int		iweight = (int)(wieght_ * 20.f);//	mscho	@20240507
			int		iweight = (int)(wieght_ * 100.f);//	mscho	@20250610
			//	평균에 있어서, 거리의 가중치를 높인다.,

			//float	wieght_ = powf(weight, 2.);
			//int		iweight = (int)(wieght_ * 10.f);//	mscho	@20240507

			//	mscho	@20240605
			int		noweight = iweight;
			//	엣지부분에서 color noise가 발생하는 원인이
			//	color ave에서 weighted를 하는 경우에 발생하였다..

			//	mscho	@20250110
			//	normal weight를 1==> iweight 로 바꿔본다.
			//int		colorweight = 1;
			float	wieght_normal = powf(weight, 0.5);
			int		colorweight = 1;

			//	mscho	@20240527
			if (baseNormal.dot(normal) > filteringValue) {
				//Eigen::Vector3f vecP = hash_vtx_pos[pointSlotIndex] - basePos;
				//Eigen::Vector3f disp = basePos + baseNormal.dot(vecP) * baseNormal;
				Eigen::Vector3f disp = hash_vtx_pos[pointSlotIndex];
				//if (xOffset != 0 || yOffset != 0 || zOffset != 0)
				//	pos_nei[nei_idx] = hash_vtx_pos[pointSlotIndex];

				(*ap) += (disp * noweight);
				(*an) += (normal * wieght_normal);	//	mscho	@20240620 ==> @20250110

				//	mscho	@20240530
				auto cs = hash_vtx_clr[pointSlotIndex].cast<unsigned int>();
				(*ac) += cs * colorweight;
				(*account) += colorweight;
				(*apcount) += noweight;
				(*neighborcount)++;
			}
			//(*ancount)++;
		}
	}
}

/// maxdistance Center, weight
__device__ void Device_ForEachNeighborOnlyNm_v4(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	const Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * an)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	//	mscho	@20250228
	//	msx distance 영역을 넓히는 방식으로 normal averaging에서 smoothing효과를 높인다.
	float	max_distance = (float)ave_step * 1.0;
	max_distance += 0.5f;

	//	mscho	@20240605
	//	bug 수정
	//float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);
	float	distance = norm3df((float)xOffset, (float)yOffset, (float)zOffset);


	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;

			//	mscho	@20240611	==> @20250228
			float	weight = (1.f - (distance / max_distance));
			//float	wieght_ = powf(weight, 2.);
			float	wieght_ = powf(weight, 4.);
			int		iweight = (int)(wieght_ * 4.f);

			if (baseNormal.dot(normal) > filteringValue) {

				//	mscho	@20240605	==> @20240611
				//(*an) += (normal * iweight);
				(*an) += (normal * wieght_);
				//(*an) += (normal);
			}
		}
	}
}

//	mscho	@20250626
/// maxdistance Center, weight
//	normal을 찾아서, sum을 하게되면 true return
__device__ bool Device_ForEachNeighborOnlyNm_v5(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	const Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * an)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	//	mscho	@20250228
	//	msx distance 영역을 넓히는 방식으로 normal averaging에서 smoothing효과를 높인다.
	float	max_distance = (float)ave_step * 1.0;
	max_distance += 0.5f;

	//	mscho	@20240605
	//	bug 수정
	//float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);
	float	distance = norm3df((float)xOffset, (float)yOffset, (float)zOffset);


	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return false;

			//	mscho	@20240611	==> @20250228
			float	weight = (1.f - (distance / max_distance));
			//float	wieght_ = powf(weight, 2.);
			float	wieght_ = powf(weight, 4.);
			int		iweight = (int)(wieght_ * 4.f);

			if (baseNormal.dot(normal) > filteringValue) {

				//	mscho	@20240605	==> @20240611
				//(*an) += (normal * iweight);
				(*an) += (normal * wieght_);
				//(*an) += (normal);
				return true;
			}
			else
				return false;
		}
		else
			return false;
	}
	else
		return false;
}



/// maxdistance Center, weight, neighbor Array
__device__ void Device_ForEachNeighborOnlyNm_v3(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	Eigen::Vector3f * hash_vtx_pos,
	Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * pos_nei,
	Eigen::Vector3f * an,
	uint32_t * apcount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	float	max_distance = (float)ave_step;
	max_distance += 0.5f;
	int	step_width = ave_step * 2 + 1;
	auto nei_idx = (zOffset + ave_step) * step_width * step_width + (yOffset + ave_step) * step_width + (xOffset + ave_step);

	//	mscho	@20240605
	//	bug 수정
	//float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);
	float	distance = norm3df((float)xOffset, (float)yOffset, (float)zOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = info->gridSlotIndexCache_pts[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			//	mscho	@20240521 (New)
			//float	weight = (1.f - (distance / max_distance));
			//float	wieght_ = powf(weight, 2.);
			//int		iweight = (int)(wieght_ * 10.f);

			if (baseNormal.dot(normal) > filteringValue) {

				if (xOffset != 0 || yOffset != 0 || zOffset != 0)
				{
					(*apcount)++;
					pos_nei[nei_idx] = hash_vtx_pos[pointSlotIndex];
				}
				//	mscho	@20240521 (New)
				//(*an) += (normal * iweight);
			}
		}
	}
}

//	mscho	@20240422
/// maxdistance *1.2 , weight
__device__ void Device_ForEachNeighborOnlyNm_v2(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * an,

	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	float	max_distance = norm3df((float)ave_step, (float)ave_step, (float)ave_step);
	max_distance *= 1.2f;

	float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;
			//	mscho	@20240521 (New)
			if (baseNormal.dot(normal) >= filteringValue) {
				float	weight = (1.f - (distance / max_distance));
				float	wieght_ = powf(weight, 2.);
				int		iweight = (int)(wieght_ * 20.f);
				(*an) += (normal * iweight);

				(*ancount) += iweight;
			}
		}
	}
}

//	mscho	@20240524
/// maxdistance *1.2 , weight
__device__ void Device_ForEachNeighborOnlyNm_CntPnt_v3(
	short xOffset, short yOffset, short zOffset,
	Eigen::Vector3f baseNormal, float filteringValue,
	int		ave_step,
	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * an,
	unsigned int* neighborcount,
	unsigned int* ancount)
{
	auto x = (int)xCacheIndex + (xOffset);
	auto y = (int)yCacheIndex + (yOffset);
	auto z = (int)zCacheIndex + (zOffset);

	float	max_distance = (float)ave_step;
	max_distance += 0.5f;
	float	distance = norm3df((float)xOffset, (float)xOffset, (float)xOffset);

	if ((0 <= x && x < info->cache.voxelCountX) &&
		(0 <= y && y < info->cache.voxelCountY) &&
		(0 <= z && z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex && distance < max_distance)
		{
			Eigen::Vector3f normal = hash_vtx_nm[pointSlotIndex];
			if (false == VECTOR3F_VALID_(normal))
				return;
			//	mscho	@20240521 (New)
			if (baseNormal.dot(normal) >= filteringValue) {
				//float	weight = (1.f - (distance / max_distance));
				//float	wieght_ = powf(weight, 2.);
				//int		iweight = (int)(wieght_ * 20.f);
				//(*an) += (normal * iweight);
				(*neighborcount)++;
				//(*ancount) += iweight;
			}
		}
	}
}

#ifndef BUILD_FOR_CPU
__device__ void Device_ForEachNeighborOnlyNm(
	short xOffset, short yOffset, short zOffset,

	MarchingCubes::ExecutionInfo * info,

	size_t xCacheIndex, size_t yCacheIndex, size_t zCacheIndex,

	uint32_t * gridSlotIndexCache,

	Eigen::Vector3f * hash_vtx_nm,

	Eigen::Vector3f * an,

	unsigned int* ancount)
{
	auto x = xCacheIndex + (xOffset);
	auto y = yCacheIndex + (yOffset);
	auto z = zCacheIndex + (zOffset);
	if ((x < info->cache.voxelCountX) &&
		(y < info->cache.voxelCountY) &&
		(z < info->cache.voxelCountZ)) {
		auto pointSlotIndex = gridSlotIndexCache[
			z * (info->cache.voxelCountX * info->cache.voxelCountY)
				+ y * (info->cache.voxelCountX) + x];
		if (kEmpty32 != pointSlotIndex)
		{
			(*an) += hash_vtx_nm[pointSlotIndex];

			(*ancount)++;
		}
	}
}
#endif

#ifndef BUILD_FOR_CPU
__global__ void Kernel_AvgPoints_v1(
	MarchingCubes::ExecutionInfo voxelInfo,
	unsigned int* slotIndices, uint32_t * gridSlotIndexCache,

	HashKey64 * hashinfo_vtx, uint64_t * hashTable_vtx, uint8_t * hashTable_vtx_value

	, Eigen::Vector3f * hash_vtx_pos
	, Eigen::Vector3f * hash_vtx_nm
	, Eigen::Vector3b * hash_vtx_clr

	, uint32_t * vtx_dupCnt_repos
	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, int refMargin_normal
	, int refMargin_point
)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > hashinfo_vtx->Count_HashTableUsed - 1) return;

	auto slotIndex = slotIndices[threadid];
	auto key = hashTable_vtx[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto pc = voxelInfo.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);

	Eigen::Vector3f voxelMin = pc - Eigen::Vector3f(
		voxelInfo.global.voxelSize * 0.5f,
		voxelInfo.global.voxelSize * 0.5f,
		voxelInfo.global.voxelSize * 0.5f);
	Eigen::Vector3f voxelMax = pc + Eigen::Vector3f(
		voxelInfo.global.voxelSize * 0.5f,
		voxelInfo.global.voxelSize * 0.5f,
		voxelInfo.global.voxelSize * 0.5f);

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

#define ForEachNeighborPt_v1(xOffset, yOffset, zOffset)\
Device_ForEachNeighborPt((xOffset), (yOffset), (zOffset), &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
gridSlotIndexCache, hash_vtx_pos, hash_vtx_nm, hash_vtx_clr, &ap, &an, &ac, &apcount, &ancount);\

#define ForEachNeighborOnlyNm_v1(xOffset, yOffset, zOffset)\
Device_ForEachNeighborOnlyNm((xOffset), (yOffset), (zOffset), &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
gridSlotIndexCache, hash_vtx_nm, &an, &ancount);\

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
				if (
					(x > (refMargin_point * -1) && x < refMargin_point) &&
					(y > (refMargin_point * -1) && y < refMargin_point) &&
					(z > (refMargin_point * -1) && z < refMargin_point)
					)
				{
					ForEachNeighborPt_v1(x, y, z);
				}
				else
				{
					ForEachNeighborOnlyNm_v1(x, y, z);
				}

			}
		}
	}

	//printf("apcount %d, ancount %d\n", apcount, ancount);

	if (apcount > 1)
	{
		Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
		//Eigen::Vector3f pointNormal = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
		//Eigen::Vector3b pointColor = Eigen::Vector3b(0, 0, 0);

		Eigen::Vector3f pointNormal = Eigen::Vector3f(an / (float)ancount);

		auto cc = (ac / apcount).cast<unsigned char>();
		Eigen::Vector3b pointColor = cc;

		uint32_t currntIdx = atomicAdd(count, 1);

		view_pos_repos[currntIdx] = pointPosition;
		view_nm_repos[currntIdx] = pointNormal;
		view_color_repos[currntIdx] = cc;
	}
	else
	{
		uint32_t currntIdx = atomicAdd(count, 1);

		view_pos_repos[currntIdx] = hash_vtx_pos[slotIndex];
		view_nm_repos[currntIdx] = hash_vtx_nm[slotIndex];
		view_color_repos[currntIdx] = hash_vtx_clr[slotIndex];
	}
}

__global__ void Kernel_AvgPoints_v2(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	HashKey64 * hashinfo_vtx

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, int refMargin_normal
	, int refMargin_point
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > hashinfo_vtx->Count_HashTableUsed - 1) return;

	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	/*if (slotIndex < 10)
		printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

#define ForEachNeighborPt_v2(xOffset, yOffset, zOffset)\
Device_ForEachNeighborPt((xOffset), (yOffset), (zOffset), &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &ancount);\

#define ForEachNeighborOnlyNm_v2(xOffset, yOffset, zOffset)\
Device_ForEachNeighborOnlyNm((xOffset), (yOffset), (zOffset), &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &ancount);\

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
				if (
					(x > (refMargin_point * -1) && x < refMargin_point + 1) &&
					(y > (refMargin_point * -1) && y < refMargin_point + 1) &&
					(z > (refMargin_point * -1) && z < refMargin_point + 1)
					)
				{
					ForEachNeighborPt_v2(x, y, z);
				}
				else
				{
					ForEachNeighborOnlyNm_v2(x, y, z);
				}

			}
		}
	}

	//printf("apcount %d, ancount %d\n", apcount, ancount);

	//if (apcount > 4)
	//{
	//	Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
	//	//Eigen::Vector3f pointNormal = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	//	//Eigen::Vector3b pointColor = Eigen::Vector3b(0, 0, 0);

	//	Eigen::Vector3f pointNormal = Eigen::Vector3f(an / (float)ancount);

	//	auto cc = Eigen::Vector3f(ac / (float)apcount);
	//	Eigen::Vector3b pointColor = Eigen::Vector3b(
	//		(unsigned char)(cc.x() * 255.0f),
	//		(unsigned char)(cc.y() * 255.0f),
	//		(unsigned char)(cc.z() * 255.0f));

	//	uint32_t currntIdx = atomicAdd(count, 1);

	//	view_pos_repos[currntIdx] = pointPosition;
	//	view_nm_repos[currntIdx] = pointNormal;
	//	view_color_repos[currntIdx] = cc;
	//}
	//else
	//{
	//	uint32_t currntIdx = atomicAdd(count, 1);

	//	view_pos_repos[currntIdx] = point_pos_buff[slotIndex];
	//	view_nm_repos[currntIdx] = point_nm_buff[slotIndex];
	//	view_color_repos[currntIdx] = Eigen::Vector3f(
	//		(float)(point_clr_buff[slotIndex].x() / 255.0f),
	//		(float)(point_clr_buff[slotIndex].y() / 255.0f),
	//		(float)(point_clr_buff[slotIndex].z() / 255.0f)); ;
	//}

	if (apcount < 5)
	{
		ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		ac = Eigen::Vector3ui(0, 0, 0);
		apcount = 0;
		ancount = 0;
		ForEachNeighborPt_v2(0, 0, 0);
	}

	if (apcount > 4)
	{
		Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
		Eigen::Vector3f pointNormal = Eigen::Vector3f(an / (float)ancount).normalized();
		auto cc = (ac / apcount).cast<unsigned char>();
		Eigen::Vector3b pointColor = cc;

		uint32_t currntIdx = atomicAdd(count, 1);

		if (!VECTOR3F_VALID_(pointPosition))
			printf("average PT Invalid!, count = %d\n", apcount);
		else if (apcount == 0)
			printf("average count = %d\n", apcount);

		view_pos_repos[currntIdx] = pointPosition;
		view_nm_repos[currntIdx] = pointNormal;
		view_color_repos[currntIdx] = cc;
		//printf("an = %f, %f, %f  | apcount = %d | res = %f, %f, %f \n", an.x(), an.y(), an.z(), apcount, pointNormal.x(), pointNormal.y(), pointNormal.z());
	}
}
#endif

//	mscho	@20240507
//	Normal을 재계산하여 사용하는 코드로 수정한다.
//	__global__ void Kernel_AvgPoints_v5(
//		MarchingCubes::ExecutionInfo voxelInfo,
//		uint64_t* slotIndices,
//
//		HashKey64* hashinfo_vtx
//
//		, Eigen::Vector3f* point_pos_buff
//		, Eigen::Vector3f* point_nm_buff
//		, Eigen::Vector3b* point_clr_buff
//
//		, uint32_t* count
//		, Eigen::Vector3f* view_pos_repos, Eigen::Vector3f* view_nm_repos, Eigen::Vector3f* view_color_repos
//		, int refMargin_normal
//		, int refMargin_point
//		, float filteringForPt
//		, float filteringForNm
//	)
//	{
//		unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
//		if (slotIndex > hashinfo_vtx->Count_HashTableUsed - 1) return;
//
//		auto key = slotIndices[slotIndex];
//
//		auto xGlobalIndex = (key >> 32) & 0xffff;
//		auto yGlobalIndex = (key >> 16) & 0xffff;
//		auto zGlobalIndex = (key) & 0xffff;
//
//		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
//		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
//		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;
//
//		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
//			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;
//
//		if (cacheIndex >= voxelInfo.cache.voxelCount) return;
//		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
//		if (pointSlotIndex == kEmpty32) return;
//		
//		Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
//		Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
//
//		/*if (slotIndex < 10)
//			printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/
//
//		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
//		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
//		Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
//		unsigned int apcount = 0;
//		unsigned int ancount = 0;
//		
//		const int		ave_width = refMargin_normal * 2 + 1;
//		int		ave_size = ave_width * ave_width * ave_width;
//		Eigen::Vector3f neighbor_pnts[15 * 15 * 15]; 
//		Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
//		for (int i = 0; i < (ave_width) * (ave_width) * (ave_width); i++)
//			neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
//
//
//#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
//Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
//voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);\
//
//#define ForEachNeighborNor_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
//Device_ForEachNeighborNormalPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
//voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);\
//
//		for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
//		{
//			for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
//			{
//				for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
//				{
//					if (
//						(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
//						(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
//						(z >= (refMargin_point * -1) && z < refMargin_point + 1)
//						)
//					{
//						ForEachNeighborPt_v4(x, y, z, basePos, baseNormal, filteringForPt);
//					}
//					else
//					{
//						ForEachNeighborNor_v4(x, y, z, basePos, baseNormal, filteringForPt);
//					}
//
//				}
//			}
//		}
//		int normal_mode = 1;
//		if (normal_mode == 0)
//		{
//			if (apcount > 0)
//			{
//				Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
//				Eigen::Vector3f pointNormal = an.normalized();
//				auto cc = Eigen::Vector3f(ac / (float)apcount);
//
//				Eigen::Vector3f extract_point;
//				Eigen::Vector3f point_off;
//				int		mode = 1;
//				if (mode == 0)
//				{
//					extract_point = pointPosition * 10.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
//				}
//				else if (mode == 1)
//				{
//					extract_point = pointPosition * 20.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
//				}
//				else if (mode == 2)
//				{
//					extract_point = pointPosition * 40.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
//				}
//
//				//Eigen::Vector3f ave_point = point_off;
//				Eigen::Vector3f ave_point = pointPosition;
//				Eigen::Vector3f ave_normal = pointNormal;
//				Eigen::Vector3f ave_color = cc;
//				uint32_t currntIdx = atomicAdd(count, 1);
//
//				view_pos_repos[currntIdx] = ave_point;
//				view_nm_repos[currntIdx] = ave_normal;
//				view_color_repos[currntIdx] = ave_color;
//			}
//		}
//		else if (normal_mode == 1)
//		{
//			if (apcount > 0)
//			{
//				Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);
//				Eigen::Vector3f pointNormal = an.normalized();
//				auto cc = Eigen::Vector3f(ac / (float)apcount);
//
//
//				Eigen::Vector3f extract_point;
//				Eigen::Vector3f point_off;
//				int		mode = 3;
//				if (mode == 0)
//				{
//					extract_point = pointPosition * 10.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
//				}
//				else if (mode == 1)
//				{
//					// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
//					extract_point = pointPosition * 20.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
//				}
//				else if (mode == 2)
//				{
//					// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
//					extract_point = pointPosition * 40.f;
//					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
//					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
//					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
//					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
//					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
//					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
//					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
//					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
//					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
//				}
//				else
//					point_off = pointPosition;
//
//				Eigen::Vector3f ave_point = point_off;
//				Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
//				Eigen::Vector3f ave_color = cc;
//				Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
//				int	valid_k = 0;
//				for (int i = 0; i < ave_size; i++)
//				{
//					if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
//					{
//						neighbor_pnts[valid_k] = neighbor_pnts[i];
//						valid_k++;
//					}
//				}
//				//printf("normal calc count = %d \n", valid_k);
//				if (valid_k >= 2)
//				{
//					int normal_cnt = 0;
//					for (int i = 0; i < valid_k; i++)
//					{
//						for (int k = i + 1; k < valid_k; k++)
//						{
//							Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
//							Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
//							Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];
//
//							Eigen::Vector3f	dir_ab = vec_ab.normalized();
//							Eigen::Vector3f	dir_ac = vec_ac.normalized();
//							Eigen::Vector3f	dir_bc = vec_bc.normalized();
//
//							float	dot_ab = baseNormal.dot(dir_ab);
//							float	dot_ac = baseNormal.dot(dir_ac);
//							float	dot_bc = baseNormal.dot(dir_bc);
//
//							if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
//								continue;
//
//							Eigen::Vector3f cross_product = Eigen::Vector3f(
//								vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
//								, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
//								, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
//							);
//							/*float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
//							new_normal = cross_product / distance;*/
//							new_normal = cross_product.normalized();
//							float	dot_normal = new_normal.dot(baseNormal);
//							if (dot_normal < 0)
//							{
//								new_normal *= -1;
//							}
//
//							dot_normal = new_normal.dot(baseNormal);
//
//							if (fabsf(dot_normal) > 0.2)
//							{
//								neighbor_normal += new_normal;
//								normal_cnt++;
//							}
//						}
//
//					}
//					if (normal_cnt > 0)
//					{
//						/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
//						ave_normal = neighbor_normal / distance;*/
//						ave_normal = neighbor_normal.normalized();
//					}
//				}
//				else
//					ave_normal = an.normalized();
//
//
//				uint32_t currntIdx = atomicAdd(count, 1);
//				view_pos_repos[currntIdx] = ave_point;
//				view_nm_repos[currntIdx] = ave_normal;
//				view_color_repos[currntIdx] = ave_color;
//			}
//		}
//	}

	//	mscho	@20240507
//	Normal을 재계산하여 사용하는 코드로 수정한다.
// if (valid_k >= 2)의 else 의경우 avg Normal을 사용하도록 함.

#ifndef BUILD_FOR_CPU
//#define AVG_PT_ON
__global__ void Kernel_AvgEstimateNormal_v1(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	if (cacheIndex >= voxelInfo.cache.voxelCount) return;
	auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
	if (pointSlotIndex == kEmpty32) return;

	Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
	Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();

	/*if (slotIndex < 10)
		printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

	const int		ave_width = refMargin_normal * 2 + 1;
	int		ave_size = ave_width * ave_width * ave_width;
	Eigen::Vector3f neighbor_pnts[15 * 15 * 15];
	Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < (ave_width) * (ave_width) * (ave_width); i++)
		neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#ifdef AVG_PT_ON
#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);
#endif

#define ForEachNeighborEstimateNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_pos_buff, point_nm_buff, neighbor_pnts, &an, &apcount);

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
#ifdef AVG_PT_ON
				if (
					(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
					(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
					(z >= (refMargin_point * -1) && z < refMargin_point + 1)
					)
				{
					ForEachNeighborPt_v4(x, y, z, basePos, baseNormal, filteringForPt);
				}
				else
#endif
				{
					ForEachNeighborEstimateNor_v1(x, y, z, baseNormal, filteringForPt);
				}

			}
		}
	}
	int normal_mode = 1;

	view_nm_tmp_repos[pointSlotIndex] = baseNormal;

	if (normal_mode == 0)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointNormal = an.normalized();

			view_nm_tmp_repos[pointSlotIndex] = pointNormal;

		}
	}
	else if (normal_mode == 1)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointPosition;
			Eigen::Vector3f pointNormal = an.normalized();
#ifdef AVG_PT_ON
			pointPosition = Eigen::Vector3f(ap / (float)apcount);
			auto cc = Eigen::Vector3f(ac / (float)apcount);


			Eigen::Vector3f extract_point;
			Eigen::Vector3f point_off;
			int		mode = 3;
			if (mode == 0)
			{
				extract_point = pointPosition * 10.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
			}
			else if (mode == 1)
			{
				// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
				extract_point = pointPosition * 20.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
			}
			else if (mode == 2)
			{
				// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
				extract_point = pointPosition * 40.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
			}
			else
				point_off = pointPosition;
#else
			pointPosition = basePos;
#endif

			Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
			Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			int	valid_k = 0;
			for (int i = 0; i < ave_size; i++)
			{
				if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
				{
					neighbor_pnts[valid_k] = neighbor_pnts[i];
					valid_k++;
				}
			}
			//printf("normal calc count = %d \n", valid_k);
			if (valid_k >= 2)
			{
				int normal_cnt = 0;
				for (int i = 0; i < valid_k; i++)
				{
					for (int k = i + 1; k < valid_k; k++)
					{
						Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
						Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
						Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];

						Eigen::Vector3f	dir_ab = vec_ab.normalized();
						Eigen::Vector3f	dir_ac = vec_ac.normalized();
						Eigen::Vector3f	dir_bc = vec_bc.normalized();

						float	dot_ab = baseNormal.dot(dir_ab);
						float	dot_ac = baseNormal.dot(dir_ac);
						float	dot_bc = baseNormal.dot(dir_bc);

						if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
							continue;

						Eigen::Vector3f cross_product = Eigen::Vector3f(
							vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
							, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
							, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
						);
						/*float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
						new_normal = cross_product / distance;*/
						new_normal = cross_product.normalized();
						float	dot_normal = new_normal.dot(baseNormal);
						if (dot_normal < 0)
						{
							new_normal *= -1;
						}

						dot_normal = new_normal.dot(baseNormal);

						if (fabsf(dot_normal) > 0.2)
						{
							neighbor_normal += new_normal;
							normal_cnt++;
						}
					}

				}
				if (normal_cnt > 0)
				{
					/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
					ave_normal = neighbor_normal / distance;*/
					ave_normal = neighbor_normal.normalized();
				}
				else
					ave_normal = pointNormal;
			}
			else
				ave_normal = pointNormal;

			view_nm_tmp_repos[pointSlotIndex] = ave_normal;
		}
	}
}

//	mscho	@20240520
__global__ void Kernel_AvgEstimateNormal_v2(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	uint32_t cnt_contains
	, Eigen::Vector3f * voxel_nm_buff
	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	if (cacheIndex >= voxelInfo.cache.voxelCount) return;
	auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
	if (pointSlotIndex == kEmpty32) return;

	Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
	Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
	/*if (slotIndex < 10)
		printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

	const int		ave_width = refMargin_normal * 2 + 1;
	int		ave_size = ave_width * ave_width * ave_width;
	Eigen::Vector3f neighbor_pnts[15 * 15 * 15];
	Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	for (int i = 0; i < (ave_width) * (ave_width) * (ave_width); i++)
		neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#ifdef AVG_PT_ON
#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);
#endif

#define ForEachNeighborEstimateNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_pos_buff, point_nm_buff, neighbor_pnts, &an, &apcount);

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
#ifdef AVG_PT_ON
				if (
					(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
					(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
					(z >= (refMargin_point * -1) && z < refMargin_point + 1)
					)
				{
					ForEachNeighborPt_v4(x, y, z, basePos, baseNormal, filteringForPt);
				}
				else
#endif
				{
					ForEachNeighborEstimateNor_v1(x, y, z, baseNormal, filteringForPt);
				}

			}
		}
	}
	int normal_mode = 1;
	view_nm_tmp_repos[pointSlotIndex] = baseNormal;

	if (normal_mode == 0)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointNormal = an.normalized();

			view_nm_tmp_repos[pointSlotIndex] = pointNormal;

		}
	}
	else if (normal_mode == 1)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointPosition;
			Eigen::Vector3f pointNormal = an.normalized();
#ifdef AVG_PT_ON
			pointPosition = Eigen::Vector3f(ap / (float)apcount);
			auto cc = Eigen::Vector3f(ac / (float)apcount);


			Eigen::Vector3f extract_point;
			Eigen::Vector3f point_off;
			int		mode = 3;
			if (mode == 0)
			{
				extract_point = pointPosition * 10.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
			}
			else if (mode == 1)
			{
				// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
				extract_point = pointPosition * 20.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
			}
			else if (mode == 2)
			{
				// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
				extract_point = pointPosition * 40.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
			}
			else
				point_off = pointPosition;
#else
			pointPosition = basePos;
#endif

			Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
			Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			int	valid_k = 0;
			for (int i = 0; i < ave_size; i++)
			{
				if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
				{
					neighbor_pnts[valid_k] = neighbor_pnts[i];
					valid_k++;
				}
			}
			//printf("normal calc count = %d \n", valid_k);
			//	mscho	@20240519
			if (valid_k >= 2)
			{
				int normal_cnt = 0;
				for (int i = 0; i < valid_k; i++)
				{
					for (int k = i + 1; k < valid_k; k++)
					{
						Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
						Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
						Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];

						Eigen::Vector3f	dir_ab = vec_ab.normalized();
						Eigen::Vector3f	dir_ac = vec_ac.normalized();
						Eigen::Vector3f	dir_bc = vec_bc.normalized();

						float	dot_ab = baseNormal.dot(dir_ab);
						float	dot_ac = baseNormal.dot(dir_ac);
						float	dot_bc = baseNormal.dot(dir_bc);

						if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
							continue;

						Eigen::Vector3f cross_product = Eigen::Vector3f(
							vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
							, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
							, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
						);
						/*float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
						new_normal = cross_product / distance;*/
						new_normal = cross_product.normalized();
						float	dot_normal = new_normal.dot(baseNormal);
						if (dot_normal < 0)
						{
							new_normal *= -1;
						}

						dot_normal = new_normal.dot(baseNormal);

						if (fabsf(dot_normal) > 0.2)
						{
							neighbor_normal += new_normal;
							normal_cnt++;
						}
					}

				}
				if (normal_cnt > 0)
				{
					/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
					ave_normal = neighbor_normal / distance;*/
					ave_normal = neighbor_normal.normalized();
				}
				else
					ave_normal = pointNormal;
			}
			else
				ave_normal = pointNormal;

			view_nm_tmp_repos[pointSlotIndex] = ave_normal;
		}
		else
		{
			//if("normal calc count ==================================\n");

		}
	}
}

//	mscho	@20240521 (New)
__global__ void Kernel_AvgEstimateNormal_v3(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	uint32_t cnt_contains
	, Eigen::Vector3f * voxel_nm_buff
	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	if (cacheIndex >= voxelInfo.cache.voxelCount) return;
	auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
	if (pointSlotIndex == kEmpty32) return;

	//	mscho	@20240521 (New)
	auto normalSlotIndex = voxelInfo.gridSlotIndexCache[cacheIndex];

	Eigen::Vector3f voxelNormal = (voxel_nm_buff[normalSlotIndex]).normalized();

	Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
	Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
	/*if (slotIndex < 10)
		printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

	const int		ave_width = refMargin_normal * 2 + 1;
	int		ave_size = ave_width * ave_width * ave_width;
	Eigen::Vector3f neighbor_pnts[5 * 5 * 5];
	Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

	//	mscho	@20240521 (New)
	for (int i = 0; i < ave_size; i++)
		neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#ifdef AVG_PT_ON
#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);
#endif

#define ForEachNeighborEstimateNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_pos_buff, point_nm_buff, neighbor_pnts, &an, &apcount);

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
#ifdef AVG_PT_ON
				if (
					(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
					(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
					(z >= (refMargin_point * -1) && z < refMargin_point + 1)
					)
				{
					ForEachNeighborPt_v4(x, y, z, basePos, baseNormal, filteringForPt);
				}
				else
#endif
				{
					//	mscho	@20240521 (New)
					ForEachNeighborEstimateNor_v1(x, y, z, voxelNormal, filteringForPt);
				}

			}
		}
	}
	int normal_mode = 1;
	view_nm_tmp_repos[pointSlotIndex] = voxelNormal;// baseNormal;
	//apcount = 0;
	if (normal_mode == 0)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointNormal = an.normalized();

			view_nm_tmp_repos[pointSlotIndex] = pointNormal;

		}
	}
	else if (normal_mode == 1)
	{
		if (apcount > 0)
		{
			Eigen::Vector3f pointPosition;
			Eigen::Vector3f pointNormal = an.normalized();
#ifdef AVG_PT_ON
			pointPosition = Eigen::Vector3f(ap / (float)apcount);
			auto cc = Eigen::Vector3f(ac / (float)apcount);


			Eigen::Vector3f extract_point;
			Eigen::Vector3f point_off;
			int		mode = 3;
			if (mode == 0)
			{
				extract_point = pointPosition * 10.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
			}
			else if (mode == 1)
			{
				// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
				extract_point = pointPosition * 20.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
			}
			else if (mode == 2)
			{
				// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
				extract_point = pointPosition * 40.f;
				//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
				point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
				//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
				//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
				//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
				//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
				point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
				point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
				point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
			}
			else
				point_off = pointPosition;
#else
			pointPosition = basePos;
#endif

			Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
			Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			int	valid_k = 0;
			for (int i = 0; i < ave_size; i++)
			{
				if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
				{
					neighbor_pnts[valid_k] = neighbor_pnts[i];
					valid_k++;
				}
			}
			//printf("normal calc count = %d \n", valid_k);
			//	mscho	@20240519
			//	mscho	@20240521 (New)
			if (valid_k >= 2)
			{
				int normal_cnt = 0;
				for (int i = 0; i < valid_k; i++)
				{
					for (int k = i + 1; k < valid_k; k++)
					{
						Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
						Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
						Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];

						Eigen::Vector3f	dir_ab = vec_ab.normalized();
						Eigen::Vector3f	dir_ac = vec_ac.normalized();
						Eigen::Vector3f	dir_bc = vec_bc.normalized();

						float	dot_ab = voxelNormal.dot(dir_ab);
						float	dot_ac = voxelNormal.dot(dir_ac);
						float	dot_bc = voxelNormal.dot(dir_bc);

						if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
							continue;

						Eigen::Vector3f cross_product = Eigen::Vector3f(
							vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
							, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
							, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
						);
						/*float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
						new_normal = cross_product / distance;*/
						new_normal = cross_product.normalized();
						float	dot_normal = new_normal.dot(voxelNormal);
						if (dot_normal < 0)
						{
							new_normal *= -1;
						}

						dot_normal = new_normal.dot(voxelNormal);

						//if (fabsf(dot_normal) > 0.2)
						{
							neighbor_normal += new_normal;
							normal_cnt++;
						}
					}

				}
				//if (normal_cnt > 0)
				//{
				//	/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
				//	ave_normal = neighbor_normal / distance;*/
				//	neighbor_normal.normalize();
				//	//neighbor_normal += (baseNormal * 2.0);
				//	//neighbor_normal = baseNormal;

				//	//	voxelNormal : source point cloud에서 계산된 normal -  제일 부드럽다
				//	//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다
				//	//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.

				//	float	normal_dot = neighbor_normal.dot(voxelNormal);
				//	normal_dot = fabsf(normal_dot);
				//	float	inverse_dot = 1.0f - normal_dot;
				//	inverse_dot = powf(inverse_dot, 0.3f);
				//	auto	surface_normal = inverse_dot * neighbor_normal + (1.0 - inverse_dot) * voxelNormal;
				//	neighbor_normal = surface_normal;

				//	ave_normal = neighbor_normal.normalized();
				//}

				if (normal_cnt > 0)
				{
					/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
					ave_normal = neighbor_normal / distance;*/
					neighbor_normal.normalize();
					//neighbor_normal += (baseNormal * 2.0);
					//neighbor_normal = baseNormal;

					//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
					//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
					//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
					const float	NeighborWeight = 0.5;
					float	normal_dot = neighbor_normal.dot(voxelNormal);
					normal_dot = fabsf(normal_dot);
					float	inverse_dot = 1.0f - normal_dot;
					inverse_dot = powf(inverse_dot, 0.4f);
					auto	surface_normal = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;

					ave_normal = surface_normal.normalized();
				}
				else
					ave_normal = pointNormal;
			}
			else
				ave_normal = pointNormal;

			view_nm_tmp_repos[pointSlotIndex] = ave_normal;
		}
		else
		{
			//if("normal calc count ==================================\n");

		}
	}
}
#endif

//	mscho	@20240523
__global__ void Kernel_AvgEstimateNormal_v4(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,
	uint32_t cnt_contains
	, Eigen::Vector3f * voxel_nm_buff
	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif

		auto key = slotIndices[slotIndex];


		auto xGlobalIndex = (key >> 32) & 0xffff;
		auto yGlobalIndex = (key >> 16) & 0xffff;
		auto zGlobalIndex = (key) & 0xffff;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		//	mscho	@20240521 (New)
		auto normalSlotIndex = voxelInfo.gridSlotIndexCache[cacheIndex];

		Eigen::Vector3f voxelNormal = (voxel_nm_buff[normalSlotIndex]).normalized();

		Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
		Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
		/*if (slotIndex < 10)
			printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

			//	mscho	@20240523
			//Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		//Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		//unsigned int ancount = 0;

		const int		ave_width = refMargin_normal * 2 + 1;
		int		ave_size = ave_width * ave_width * ave_width;
		Eigen::Vector3f neighbor_pnts[5 * 5 * 5];
		Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

		//	mscho	@20240521 (New)
		for (int i = 0; i < ave_size; i++)
			neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#ifdef AVG_PT_ON
#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);
#endif

#define ForEachNeighborEstimateNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_pos_buff, point_nm_buff, neighbor_pnts, &an, &apcount);

		//	mscho	@20240523
		int		idx_cnt = refMargin_normal <= 1 ? (3 * 3 * 3) : (5 * 5 * 5);
		for (int idx = 0; idx < idx_cnt; idx++)
		{
			ForEachNeighborEstimateNor_v1(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], voxelNormal, filteringForPt);
		}

		int normal_mode = 1;
		view_nm_tmp_repos[pointSlotIndex] = voxelNormal;// baseNormal;
		//apcount = 0;
		if (normal_mode == 0)
		{
			if (apcount > 0)
			{
				Eigen::Vector3f pointNormal = an.normalized();

				view_nm_tmp_repos[pointSlotIndex] = pointNormal;

			}
		}
		else if (normal_mode == 1)
		{
			if (apcount > 0)
			{
				Eigen::Vector3f pointPosition;
				Eigen::Vector3f pointNormal = an.normalized();
#ifdef AVG_PT_ON
				pointPosition = Eigen::Vector3f(ap / (float)apcount);
				auto cc = Eigen::Vector3f(ac / (float)apcount);


				Eigen::Vector3f extract_point;
				Eigen::Vector3f point_off;
				int		mode = 3;
				if (mode == 0)
				{
					extract_point = pointPosition * 10.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
				}
				else if (mode == 1)
				{
					// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
					extract_point = pointPosition * 20.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
				}
				else if (mode == 2)
				{
					// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
					extract_point = pointPosition * 40.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
				}
				else
					point_off = pointPosition;
#else
				pointPosition = basePos;
#endif

				Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
				Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				int	valid_k = 0;
				for (int i = 0; i < ave_size; i++)
				{
					if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
					{
						neighbor_pnts[valid_k] = neighbor_pnts[i];
						valid_k++;
					}
				}
				//printf("normal calc count = %d \n", valid_k);
				//	mscho	@20240519
				//	mscho	@20240521 (New)
				if (valid_k >= 2)
				{
					int normal_cnt = 0;
					for (int i = 0; i < valid_k; i++)
					{
						for (int k = i + 1; k < valid_k; k++)
						{
							Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
							Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
							Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];

							Eigen::Vector3f	dir_ab = vec_ab.normalized();
							Eigen::Vector3f	dir_ac = vec_ac.normalized();
							Eigen::Vector3f	dir_bc = vec_bc.normalized();

							float	dot_ab = voxelNormal.dot(dir_ab);
							float	dot_ac = voxelNormal.dot(dir_ac);
							float	dot_bc = voxelNormal.dot(dir_bc);

							if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
								continue;

							Eigen::Vector3f cross_product = Eigen::Vector3f(
								vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
								, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
								, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
							);
							/*float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
							new_normal = cross_product / distance;*/
							new_normal = cross_product.normalized();
							float	dot_normal = new_normal.dot(voxelNormal);
							if (dot_normal < 0)
							{
								new_normal *= -1;
							}

							dot_normal = new_normal.dot(voxelNormal);

							//if (fabsf(dot_normal) > 0.2)
							{
								neighbor_normal += new_normal;
								normal_cnt++;
							}
						}

					}
					//	mscho	@20240523
					//if (normal_cnt > 0)
					//{
					//	/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
					//	ave_normal = neighbor_normal / distance;*/
					//	neighbor_normal.normalize();
					//	//neighbor_normal += (baseNormal * 2.0);
					//	//neighbor_normal = baseNormal;

					//	//	voxelNormal : source point cloud에서 계산된 normal -  제일 부드럽다
					//	//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다
					//	//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.

					//	float	normal_dot = neighbor_normal.dot(voxelNormal);
					//	normal_dot = fabsf(normal_dot);
					//	float	inverse_dot = 1.0f - normal_dot;
					//	inverse_dot = powf(inverse_dot, 0.3f);
					//	auto	surface_normal = inverse_dot * neighbor_normal + (1.0 - inverse_dot) * voxelNormal;
					//	neighbor_normal = surface_normal;

					//	ave_normal = neighbor_normal.normalized();
					//}
					//	mscho	@20240523
					if (true)
					{
						//if (normal_cnt > 0)
						//{
						//	/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
						//	ave_normal = neighbor_normal / distance;*/
						//	neighbor_normal.normalize();
						//	//neighbor_normal += (baseNormal * 2.0);
						//	//neighbor_normal = baseNormal;

						//	//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
						//	//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
						//	//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
						//	const float	NeighborWeight = 0.5;
						//	float	normal_dot = neighbor_normal.dot(voxelNormal);
						//	normal_dot = fabsf(normal_dot);
						//	float	inverse_dot = min(1.0f - normal_dot, 1.0f);
						//	inverse_dot = powf(inverse_dot, 0.4f);
						//	auto	surface_normal = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;

						//	ave_normal = surface_normal.normalized();
						//}
						//else
						//	ave_normal = pointNormal;

						//	mscho	@20240527
						if (normal_cnt > 0)
						{
							/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
							ave_normal = neighbor_normal / distance;*/
							neighbor_normal.normalize();
							//neighbor_normal += (baseNormal * 2.0);
							//neighbor_normal = baseNormal;

							//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
							//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
							//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
								//	mscho	@20240610
							const float	NeighborWeight = 0.3;
							float	normal_dot = neighbor_normal.dot(voxelNormal);
							normal_dot = fabsf(normal_dot);
							float	inverse_dot = min(1.0f - normal_dot, 1.0f);
							inverse_dot = powf(inverse_dot, 0.3f);	// 승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.
							auto	surface_normal = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;

							ave_normal = surface_normal.normalized();
						}
						else
							ave_normal = pointNormal;
					}
					else
					{
						if (normal_cnt > 0)
						{
							neighbor_normal.normalize();
							//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
							//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
							//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
							const float	NeighborWeight = 0.5;
							float	normal_dot = neighbor_normal.dot(voxelNormal);
							normal_dot = fabsf(normal_dot);
							float	inverse_dot1 = 1.0f - normal_dot;
							inverse_dot1 = powf(inverse_dot1, 0.4f);
							auto	surface_normal_1st = inverse_dot1 * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot1) * voxelNormal;
							auto    _normal_1st = surface_normal_1st.normalized();

							//const float	NeighborWeight2 = 0.5;
							//float	normal_dot2 = neighbor_normal.dot(voxelNormal);	// normal_dot2 값이 1에 가깝다는 의미는 두개의 노말의 방향이 일치한다는 의미
							//normal_dot2 = fabsf(normal_dot2);
							//float	inverse_dot2 = 1.0f - normal_dot2;
							//inverse_dot2 = powf(inverse_dot2, 1.5f);
							//auto	surface_normal_2nd = inverse_dot2 * (_normal_1st * NeighborWeight2 + baseNormal * (1.0 - NeighborWeight2)) + (1.0 - inverse_dot2) * voxelNormal;
							//auto  ave_normal = surface_normal_2nd.normalized();

							auto  ave_normal = surface_normal_1st.normalized();

						}
						else
							ave_normal = pointNormal;
					}

				}
				else
					ave_normal = pointNormal;

				view_nm_tmp_repos[pointSlotIndex] = ave_normal;
			}
			else
			{
				//if("normal calc count ==================================\n");

			}
		}
	}
	}

//	mscho	@20240717
//	Normal estimation,을 생략해본다.
__global__ void Kernel_AvgEstimateNormal_v5(
	MarchingCubes::ExecutionInfo voxelInfo,
	const HashKey * slotIndices,
	uint32_t cnt_contains
	, const Eigen::Vector3f * voxel_nm_buff
	, const Eigen::Vector3f * point_pos_buff
	, const Eigen::Vector3f * point_nm_buff
	, const Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos // 출력
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, float normal_weight // 값을 올릴수록 디테일이 높아진다.(낮출수록 부드러워진다.)
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif

		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		//	mscho	@20240521 (New)
		auto normalSlotIndex = voxelInfo.gridSlotIndexCache[cacheIndex];

		Eigen::Vector3f voxelNormal = (voxel_nm_buff[normalSlotIndex]).normalized();

		Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
		Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();

		//Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		//unsigned int ancount = 0;

		const int		ave_width = refMargin_normal * 2 + 1;
		Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

		int normal_mode = 1;
		view_nm_tmp_repos[pointSlotIndex] = voxelNormal;// baseNormal;
		//apcount = 0;
		if (true)
		{
			Eigen::Vector3f pointPosition;
			pointPosition = basePos;

			Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
			Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

			//printf("normal calc count = %d \n", valid_k);
			//	mscho	@20240519
			//	mscho	@20240521 (New)
			if (true)
			{
				neighbor_normal = baseNormal;
				/*float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
				ave_normal = neighbor_normal / distance;*/
				neighbor_normal.normalize();
				//neighbor_normal += (baseNormal * 2.0);
				//neighbor_normal = baseNormal;

				//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
				//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
				//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
				//	mscho	@20240610
				const float	NeighborWeight = 0;// 0.3;
				float	normal_dot = neighbor_normal.dot(voxelNormal);
				normal_dot = fabsf(normal_dot);
				//float	inverse_dot = min(1.0f - normal_dot, 1.0f);

				// @20241121
				float	inverse_dot = min(normal_dot, 1.0f);

				//inverse_dot = powf(inverse_dot, 0.3f);	// 승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.
				// @20241121
				//inverse_dot = powf(inverse_dot, 0.3f);
				inverse_dot = powf(inverse_dot, 0.5f);

				auto	surface_nor_1st = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;
				auto    surface_nor_2nd = surface_nor_1st.normalized();
				// @20241121
				auto	surface_normal = surface_nor_2nd * normal_weight + (1.0f - normal_weight) * voxelNormal;
				ave_normal = surface_normal.normalized();
				view_nm_tmp_repos[pointSlotIndex] = ave_normal;
			}
		}
	}
	}

/*
// 더이상 사용되지 않음
__global__ void Kernel_AvgEstimateNormal_v4_HD(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t* slotIndices,
	uint32_t cnt_contains
	, Eigen::Vector3f* voxel_nm_buff
	, Eigen::Vector3f* point_pos_buff
	, Eigen::Vector3f* point_nm_buff
	, Eigen::Vector3b* point_clr_buff

	, Eigen::Vector3f* view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
#ifndef BUILD_FOR_CPU
		unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (slotIndex > cnt_contains - 1) return;
		{
#else
#pragma omp parallel for schedule(dynamic, 256)
		for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];


		auto xGlobalIndex = (key >> 32) & 0xffff;
		auto yGlobalIndex = (key >> 16) & 0xffff;
		auto zGlobalIndex = (key) & 0xffff;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			continue;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			continue;

		//	mscho	@20240521 (New)
		auto normalSlotIndex = voxelInfo.gridSlotIndexCache[cacheIndex];

		Eigen::Vector3f voxelNormal = (voxel_nm_buff[normalSlotIndex]).normalized();

		Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
		Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
		/#if (slotIndex < 10)
			printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);#/

			//	mscho	@20240523
			//Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		//Eigen::Vector3f ac = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		//unsigned int ancount = 0;

		const int		ave_width = refMargin_normal * 2 + 1;
		int		ave_size = ave_width * ave_width * ave_width;
		Eigen::Vector3f neighbor_pnts[7 * 7 * 7]; //neighbor_pnts[5 * 5 * 5];
		Eigen::Vector3f neighbor_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

		//	mscho	@20240521 (New)
		for (int i = 0; i < ave_size; i++)
			neighbor_pnts[i] = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);

#ifdef AVG_PT_ON
#define ForEachNeighborPt_v4(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v3((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, neighbor_pnts, &ap, &an, &ac, &apcount, &ancount);
#endif

#define ForEachNeighborEstimateNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_pos_buff, point_nm_buff, neighbor_pnts, &an, &apcount);

		//	mscho	@20240523
		int		idx_cnt = (refMargin_normal * 2 + 1) * (refMargin_normal * 2 + 1) * (refMargin_normal * 2 + 1);
		for (int idx = 0; idx < idx_cnt; idx++)
		{
			ForEachNeighborEstimateNor_v1(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], voxelNormal, filteringForPt);
		}

		int normal_mode = 1;
		view_nm_tmp_repos[pointSlotIndex] = voxelNormal;// baseNormal;
		//apcount = 0;
		if (normal_mode == 0)
		{
			if (apcount > 0)
			{
				Eigen::Vector3f pointNormal = an.normalized();

				view_nm_tmp_repos[pointSlotIndex] = pointNormal;

			}
		}
		else if (normal_mode == 1)
		{
			if (apcount > 0)
			{
				Eigen::Vector3f pointPosition;
				Eigen::Vector3f pointNormal = an.normalized();
#ifdef AVG_PT_ON
				pointPosition = Eigen::Vector3f(ap / (float)apcount);
				auto cc = Eigen::Vector3f(ac / (float)apcount);


				Eigen::Vector3f extract_point;
				Eigen::Vector3f point_off;
				int		mode = 3;
				if (mode == 0)
				{
					extract_point = pointPosition * 10.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.1f + 0.05f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.1f + 0.05f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.1f + 0.05f;
				}
				else if (mode == 1)
				{
					// Display voxel 을 연산 voxel / 2 크기, 체적으로 하면 1/8 로 한다
					extract_point = pointPosition * 20.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.05f + 0.025f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.05f + 0.025f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.05f + 0.025f;
				}
				else if (mode == 2)
				{
					// Display voxel 을 연산 voxel / 4 크기, 체적으로 하면 1/64 로 한다
					extract_point = pointPosition * 40.f;
					//	Voxel 의 중심으로 보내기 위해서는 일단 + 높으값을 더한다음에
					point_off = Eigen::Vector3f(extract_point.x() + 1000.f, extract_point.y() + 1000.f, extract_point.z() + 1000.f);
					//	floorf함수를 거쳐서, 소수점 아래 자리를 버린다음
					//	다시 1000을 빼면, 소수점을 제한한 값으로 바꾼게 된다..
					//	여기에 voxel size를 곱해주면....voxel 이 시작좌표로 가게 된다..
					//	여기에 0.5 voxel 크기를 더해주면, voxel center 좌표를 가리키게 된다.
					point_off.x() = ((float)floorf(point_off.x()) - 1000.f) * 0.025f + 0.025f;
					point_off.y() = ((float)floorf(point_off.y()) - 1000.f) * 0.025f + 0.025f;
					point_off.z() = ((float)floorf(point_off.z()) - 1000.f) * 0.025f + 0.025f;
				}
				else
					point_off = pointPosition;
#else
				pointPosition = basePos;
#endif

				Eigen::Vector3f ave_normal = baseNormal;// pointNormal;
				Eigen::Vector3f new_normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				int	valid_k = 0;
				for (int i = 0; i < ave_size; i++)
				{
					if (FLT_VALID(neighbor_pnts[i].x()) && FLT_VALID(neighbor_pnts[i].y()) && FLT_VALID(neighbor_pnts[i].z()))
					{
						neighbor_pnts[valid_k] = neighbor_pnts[i];
						valid_k++;
					}
				}
				//printf("normal calc count = %d \n", valid_k);
				//	mscho	@20240519
				//	mscho	@20240521 (New)
				if (valid_k >= 2)
				{
					int normal_cnt = 0;
					for (int i = 0; i < valid_k; i++)
					{
						for (int k = i + 1; k < valid_k; k++)
						{
							Eigen::Vector3f	vec_ab = neighbor_pnts[i] - pointPosition;
							Eigen::Vector3f	vec_ac = neighbor_pnts[k] - pointPosition;
							Eigen::Vector3f	vec_bc = neighbor_pnts[k] - neighbor_pnts[i];

							Eigen::Vector3f	dir_ab = vec_ab.normalized();
							Eigen::Vector3f	dir_ac = vec_ac.normalized();
							Eigen::Vector3f	dir_bc = vec_bc.normalized();

							float	dot_ab = voxelNormal.dot(dir_ab);
							float	dot_ac = voxelNormal.dot(dir_ac);
							float	dot_bc = voxelNormal.dot(dir_bc);

							if (fabsf(dot_ab) > 0.7 || fabsf(dot_ac) > 0.7 || fabsf(dot_bc) > 0.7)
								continue;

							Eigen::Vector3f cross_product = Eigen::Vector3f(
								vec_ab.y() * vec_ac.z() - vec_ab.z() * vec_ac.y()
								, vec_ab.z() * vec_ac.x() - vec_ab.x() * vec_ac.z()
								, vec_ab.x() * vec_ac.y() - vec_ab.y() * vec_ac.x()
							);
							/#float	distance = norm3df(cross_product.x(), cross_product.y(), cross_product.z());
							new_normal = cross_product / distance;#/
							new_normal = cross_product.normalized();
							float	dot_normal = new_normal.dot(voxelNormal);
							if (dot_normal < 0)
							{
								new_normal *= -1;
							}

							dot_normal = new_normal.dot(voxelNormal);

							//if (fabsf(dot_normal) > 0.2)
							{
								neighbor_normal += new_normal;
								normal_cnt++;
							}
						}

					}
					//	mscho	@20240523
					//if (normal_cnt > 0)
					//{
					//	/#float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
					//	ave_normal = neighbor_normal / distance;#/
					//	neighbor_normal.normalize();
					//	//neighbor_normal += (baseNormal * 2.0);
					//	//neighbor_normal = baseNormal;

					//	//	voxelNormal : source point cloud에서 계산된 normal -  제일 부드럽다
					//	//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다
					//	//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.

					//	float	normal_dot = neighbor_normal.dot(voxelNormal);
					//	normal_dot = fabsf(normal_dot);
					//	float	inverse_dot = 1.0f - normal_dot;
					//	inverse_dot = powf(inverse_dot, 0.3f);
					//	auto	surface_normal = inverse_dot * neighbor_normal + (1.0 - inverse_dot) * voxelNormal;
					//	neighbor_normal = surface_normal;

					//	ave_normal = neighbor_normal.normalized();
					//}
					//	mscho	@20240523
					if (true)
					{
						//if (normal_cnt > 0)
						//{
						//	/#float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
						//	ave_normal = neighbor_normal / distance;#/
						//	neighbor_normal.normalize();
						//	//neighbor_normal += (baseNormal * 2.0);
						//	//neighbor_normal = baseNormal;

						//	//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
						//	//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
						//	//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
						//	const float	NeighborWeight = 0.5;
						//	float	normal_dot = neighbor_normal.dot(voxelNormal);
						//	normal_dot = fabsf(normal_dot);
						//	float	inverse_dot = min(1.0f - normal_dot, 1.0f);
						//	inverse_dot = powf(inverse_dot, 0.4f);
						//	auto	surface_normal = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;

						//	ave_normal = surface_normal.normalized();
						//}
						//else
						//	ave_normal = pointNormal;

						//	mscho	@20240527
						if (normal_cnt > 0)
						{
							/#float	distance = norm3df(neighbor_normal.x(), neighbor_normal.y(), neighbor_normal.z());
							ave_normal = neighbor_normal / distance;#/
							neighbor_normal.normalize();
							//neighbor_normal += (baseNormal * 2.0);
							//neighbor_normal = baseNormal;

							//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
							//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
							//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
							const float	NeighborWeight = 0.2;
							float	normal_dot = neighbor_normal.dot(voxelNormal);
							normal_dot = fabsf(normal_dot);
							float	inverse_dot = min(1.0f - normal_dot, 1.0f);
							inverse_dot = powf(inverse_dot, 0.3f);	// 승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.
							auto	surface_normal = inverse_dot * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot) * voxelNormal;

							ave_normal = surface_normal.normalized();
						}
						else
							ave_normal = pointNormal;
					}
					else
					{
						if (normal_cnt > 0)
						{
							neighbor_normal.normalize();
							//	voxelNormal : source point cloud reconstruction 에서 계산된 normal -  제일 부드럽다. 많은 점들에서 Normal을 추출하였다.
							//  baseNormal : Extract에서 만들어지 Normal - 제일 거칠다... detail을 표현할 수 있다. 반지름 150um이내에서 모아온 점이다.
							//  neighbor_normal : 인접점들로 평가된 nomal - 중간정도의 detail이 표현된다.
							const float	NeighborWeight = 0.5;
							float	normal_dot = neighbor_normal.dot(voxelNormal);
							normal_dot = fabsf(normal_dot);
							float	inverse_dot1 = 1.0f - normal_dot;
							inverse_dot1 = powf(inverse_dot1, 0.4f);
							auto	surface_normal_1st = inverse_dot1 * (neighbor_normal * NeighborWeight + baseNormal * (1.0 - NeighborWeight)) + (1.0 - inverse_dot1) * voxelNormal;
							auto    _normal_1st = surface_normal_1st.normalized();

							//const float	NeighborWeight2 = 0.5;
							//float	normal_dot2 = neighbor_normal.dot(voxelNormal);	// normal_dot2 값이 1에 가깝다는 의미는 두개의 노말의 방향이 일치한다는 의미
							//normal_dot2 = fabsf(normal_dot2);
							//float	inverse_dot2 = 1.0f - normal_dot2;
							//inverse_dot2 = powf(inverse_dot2, 1.5f);
							//auto	surface_normal_2nd = inverse_dot2 * (_normal_1st * NeighborWeight2 + baseNormal * (1.0 - NeighborWeight2)) + (1.0 - inverse_dot2) * voxelNormal;
							//auto  ave_normal = surface_normal_2nd.normalized();

							auto  ave_normal = surface_normal_1st.normalized();

						}
						else
							ave_normal = pointNormal;
					}

				}
				else
					ave_normal = pointNormal;

				view_nm_tmp_repos[pointSlotIndex] = ave_normal;
			}
			else
			{
				//if("normal calc count ==================================\n");

			}
		}
		}
	}
	*/

__global__ void Kernel_AvgNormal_v1(
	MarchingCubes::ExecutionInfo voxelInfo,
	const HashKey * slotIndices,

	uint32_t cnt_contains

	// 입력
	, const Eigen::Vector3f * point_pos_buff
	, const Eigen::Vector3f * point_nm_buff
	, const Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos // 출력
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];
		//printf("test? %lld\n", key);

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex];
		if (false == VECTOR3F_VALID_(baseNormal))
		{
			view_nm_tmp_repos[pointSlotIndex] = baseNormal;
			kernel_return;
		}
		baseNormal = baseNormal.normalized();

		/*if (slotIndex < 10)
			printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		unsigned int ancount = 0;

		const int		ave_width = refMargin_normal * 2 + 1;
		int		ave_size = ave_width * ave_width * ave_width;

		//	mscho	@20250626
#define ForEachNeighborNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v4((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_nm_buff, &an);

#define ForEachNeighborNor_v2(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v5((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
			refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_nm_buff, &an)
		//	mscho	@20240523
		//for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
		//{
		//	for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		//	{
		//		for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
		//		{
		//			ForEachNeighborNor_v1(x, y, z, baseNormal, filteringForPt);
		//		}
		//	}
		//}

		//	mscho	@20240523
		//int		idx_cnt = refMargin_normal <= 1 ? (3 * 3 * 3) : (5 * 5 * 5);

		//	mscho	@20250626
		//	normal이 지저분해 보일 수 있으므로, 최소 10개의  point 는 찾아서, 평균을 구하도록 한다.
		int		idx_cnt = (5 * 5 * 5);
		int		pnt_cnt = 0;
		for (int idx = 0; idx < idx_cnt; idx++)
		{
			//ForEachNeighborNor_v1(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			auto bFound = ForEachNeighborNor_v2(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			if (bFound)
			{
				pnt_cnt++;
				// mscho	@20250627
				if (pnt_cnt > 15)
					break;
			}
		}

		view_nm_tmp_repos[pointSlotIndex] = baseNormal;

		if (an != Eigen::Vector3f(0.0f, 0.0f, 0.0f))
		{
			Eigen::Vector3f pointNormal = an.normalized();

			view_nm_tmp_repos[pointSlotIndex] = pointNormal;
		}
	}
	}

__global__ void Kernel_AvgNormal_v1_HD(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, Eigen::Vector3f * view_nm_tmp_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];
		//printf("test? %lld\n", key);

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex];
		if (false == VECTOR3F_VALID_(baseNormal))
		{
			view_nm_tmp_repos[pointSlotIndex] = baseNormal;
			kernel_return;
		}
		baseNormal = baseNormal.normalized();

		/*if (slotIndex < 10)
			printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		unsigned int apcount = 0;
		unsigned int ancount = 0;

		const int		ave_width = refMargin_normal * 2 + 1;
		int		ave_size = ave_width * ave_width * ave_width;

#define ForEachNeighborNor_v1(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v4((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), \
	refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex, point_nm_buff, &an);
		//	mscho	@20240523
		//for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
		//{
		//	for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		//	{
		//		for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
		//		{
		//			ForEachNeighborNor_v1(x, y, z, baseNormal, filteringForPt);
		//		}
		//	}
		//}

		//	mscho	@20240523
		int		idx_cnt = (3 * 3 * 3);
		if (refMargin_normal == 2) idx_cnt = (5 * 5 * 5);
		if (refMargin_normal >= 3) idx_cnt = (7 * 7 * 7);

		for (int idx = 0; idx < idx_cnt; idx++)
		{
			ForEachNeighborNor_v1(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
		}

		view_nm_tmp_repos[pointSlotIndex] = baseNormal;

		if (an != Eigen::Vector3f(0.0f, 0.0f, 0.0f))
		{
			Eigen::Vector3f pointNormal = an.normalized();

			view_nm_tmp_repos[pointSlotIndex] = pointNormal;
		}
	}
	}

//	mscho	@20240422
__global__ void Kernel_AvgPoints_v4(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	//qDebug("test2 \t");
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif

		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		//printf("cache %d, %d\n", cacheIndex, voxelInfo.cache.voxelCount - 1);

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			kernel_return;
		}

		baseNormal = baseNormal.normalized();

		//if (slotIndex < 10)
			//printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
		unsigned int apcount = 0;
		unsigned int ancount = 0;

#define ForEachNeighborPt_v3(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v2((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &ancount);

		//#define ForEachNeighborOnlyNm_v2(xOffset, yOffset, zOffset)\
		//Device_ForEachNeighborOnlyNm((xOffset), (yOffset), (zOffset), &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
		//voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &ancount);\

#define ForEachNeighborOnlyNm_v3(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v2((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &ancount);
		//for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
		//{
		//	for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		//	{
		//		for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
		//		{
		//			if (
		//				(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
		//				(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
		//				(z >= (refMargin_point * -1) && z < refMargin_point + 1)
		//				)
		//			{
		//				//ForEachNeighborPt_v2(x, y, z);

		//				ForEachNeighborPt_v3(x, y, z, basePos, baseNormal, filteringForPt);
		//				//ForEachNeighborOnlyNm_v3(x, y, z);
		//				//ForEachNeighborOnlyNm_v2(0, 0, 0);
		//			}
		//			else
		//			{
		//				ForEachNeighborOnlyNm_v3(x, y, z, baseNormal, filteringForNm);
		//				//ForEachNeighborOnlyNm_v2(0, 0, 0);
		//			}

		//		}
		//	}
		//}


		//	mscho	@20240523
		int		max_id = refMargin_normal > refMargin_point ? refMargin_normal : refMargin_point;
		int		second_id = refMargin_normal < refMargin_point ? refMargin_normal : refMargin_point;
		int		idx_cnt = (max_id <= 1 ? (3 * 3 * 3) : (5 * 5 * 5));
		int		idx_cnt2 = (second_id <= 1 ? (3 * 3 * 3) : (5 * 5 * 5));
		if (idx_cnt != idx_cnt2)
		{
			for (int idx = 0; idx < idx_cnt2; idx++)
			{
				ForEachNeighborPt_v3(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
				ForEachNeighborOnlyNm_v3(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			}
			for (int idx = idx_cnt2; idx < idx_cnt; idx++)
			{
				ForEachNeighborOnlyNm_v3(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			}

		}
		else
		{
			for (int idx = 0; idx < idx_cnt; idx++)
			{
				ForEachNeighborPt_v3(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
				ForEachNeighborOnlyNm_v3(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			}
		}

		//	mscho	@20240521 (New)
		if (apcount > 0)
		{
			//	mscho	@20240523
			//Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);// *10.0f;
			//Eigen::Vector3f pointNormal = an.normalized();
			//auto cc = Eigen::Vector3f(ac / (float)apcount);
			//Eigen::Vector3b pointColor = Eigen::Vector3b(
			//	(unsigned char)(cc.x() * 255.0f),
			//	(unsigned char)(cc.y() * 255.0f),
			//	(unsigned char)(cc.z() * 255.0f));

			//	mscho	@20240523
			//Eigen::Vector3f ave_point = pointPosition;
			//Eigen::Vector3f ave_normal = pointNormal;
			//Eigen::Vector3f ave_color = cc;
			uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
			//	mscho	@20240523
			view_pos_repos[currntIdx] = Eigen::Vector3f(ap / (float)apcount);
			view_nm_repos[currntIdx] = an.normalized();
			view_color_repos[currntIdx] = (ac / apcount).cast<unsigned char>();
		}
	}
	}

/* no longer used -> Kernel_AvgPoints_v7
//	mscho	@20240524
__global__ void Kernel_AvgPoints_v6(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t* slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f* point_pos_buff
	, Eigen::Vector3f* point_nm_buff
	, Eigen::Vector3b* point_clr_buff
	, unsigned char* point_mid_buff
	, uint32_t* count
	, Eigen::Vector3f* view_pos_repos, Eigen::Vector3f* view_nm_repos, Eigen::Vector3f* view_color_repos, unsigned char* view_mId_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, uint32_t	filterCount
	, bool isHDMode
)
{

#ifndef BUILD_FOR_CPU
		unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
		if (slotIndex > cnt_contains - 1) return;
		//qDebug("test2 \t");
		{
#else
#pragma omp parallel for schedule(dynamic, 256)
		for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif

		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = (key >> 32) & 0xffff;
		auto yGlobalIndex = (key >> 16) & 0xffff;
		auto zGlobalIndex = (key) & 0xffff;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		//printf("cache %d, %d\n", cacheIndex, voxelInfo.cache.voxelCount - 1);

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			continue;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			continue;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		//	mscho	@20240527
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex].normalized();
		Eigen::Vector3f baseColor = Eigen::Vector3f(point_clr_buff[pointSlotIndex].x() / 255.0
			, point_clr_buff[pointSlotIndex].y() / 255.0
			, point_clr_buff[pointSlotIndex].z() / 255.0
		);
		unsigned char baseMid = point_mid_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			continue;
		}

		//baseNormal = baseNormal.normalized();

		//if (slotIndex < 10)
			//printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = ap;
		Eigen::Vector3f ac = ap;
		unsigned int apcount = 0;
		unsigned int ancount = 0;
		unsigned int point_nei_cnt = 0;

		//	mscho	@20240611
		unsigned int account = 0;

//		//	mscho	@20240524
//#define ForEachNeighborPt_v5(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
//Device_ForEachNeighborPt_v5((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
//voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &point_nei_cnt, &ancount);


				//	mscho	@20240524
#define ForEachNeighborPt_v6(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v6((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

#define ForEachNeighborOnlyNm_v4(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_CntPnt_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &point_nei_cnt, &ancount);

		int		idx_cnt_1st = (3 * 3 * 3);
		int		idx_cnt_2nd = (5 * 5 * 5);
		if (isHDMode)
		{
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);
		}

		//	mscho	@20240611	==> @20240620
		//  idx_cnt_1st = idx_cnt_2nd;
		for (int idx = 0; idx < idx_cnt_1st; idx++)
		{
			//	mscho	@20240527
			//	point는 갯수만 세고, normal은 누적한다
			//ForEachNeighborPt_v5(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);

			//	mscho	@20240611
			ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
		}
		for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
		{
			//	point를 누적하다가, 정해진 갯수 이상이라는 것이 판단되면 탈출한다.
			ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			if (point_nei_cnt > filterCount)
				break; // 내부 루프 탈출
		}

		//	mscho	@20240521 (New)
		if (point_nei_cnt > filterCount)
		{
			//	mscho	@20240523
			//Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount);// *10.0f;
			//Eigen::Vector3f pointNormal = an.normalized();
			//auto cc = Eigen::Vector3f(ac / (float)apcount);
			//Eigen::Vector3b pointColor = Eigen::Vector3b(
			//	(unsigned char)(cc.x() * 255.0f),
			//	(unsigned char)(cc.y() * 255.0f),
			//	(unsigned char)(cc.z() * 255.0f));

			//	mscho	@20240523
			//Eigen::Vector3f ave_point = pointPosition;
			//Eigen::Vector3f ave_normal = pointNormal;
			//Eigen::Vector3f ave_color = cc;


				//	mscho	@202406011 ==> @20240620
				//	voxel에 중복으로 들어가는 포인트들이 있어서, 평균값이 해당복셀을 벗어냐면 제외시킨다.
			Eigen::Vector3f voxel_pos = Eigen::Vector3f((((float)xGlobalIndex / 10.0) - 250.0f), (((float)yGlobalIndex / 10.0) - 250.0f), (((float)zGlobalIndex / 10.0) - 250.0f));

			//	산술평균좌표를 구한다음
			Eigen::Vector3f ave_pos = Eigen::Vector3f(ap / (float)apcount);
			//	원래좌표와의 차이를 구한다
			Eigen::Vector3f diff_pos = basePos - ave_pos;
			//	평균법선을 normalize하고
			Eigen::Vector3f ave_normal = an.normalized();
			//	좌표의 차이와 평균노말과의 dot product를 연산
			float	diff_dot = ave_normal.dot(diff_pos);
			//	기준좌표에서 normal*dot 만큼을 빼주면..
			//	평균좌표의 법선방향으로만 평균이동량을 적용하는 것임..
			//	간략한 surface fitting
			Eigen::Vector3f new_pos = basePos - ave_normal * diff_dot;
			//if ((voxel_pos.x() <= ave_pos.x() && ave_pos.x() < voxel_pos.x() + 0.1f) &&
			//	(voxel_pos.y() <= ave_pos.y() && ave_pos.y() < voxel_pos.y() + 0.1f) &&
			//	(voxel_pos.z() <= ave_pos.z() && ave_pos.z() < voxel_pos.z() + 0.1f))
			{
				uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
				//	mscho	@20240523 ==> @20240620
				view_pos_repos[currntIdx] = new_pos;// ave_pos;
				view_nm_repos[currntIdx] = ave_normal;
				view_mId_repos[currntIdx] = baseMid;

				//	mscho	@20240530	==>	@20240611
				//	color는 weighted ave를 사용하지 않는다..
				//	별도의 count를 사용하도록 한다.

				if (true)
					view_color_repos[currntIdx] = Eigen::Vector3f(ac / (float)account);
				else
				{

					float	reli_score = (float)(point_nei_cnt - filterCount) / 80.f;
					reli_score = min(reli_score, 1.f);
					reli_score = powf(reli_score, 1.f);
					float r, g, b;
					if (0 <= reli_score && reli_score < 0.25)
					{
						r = 1.0;
						g = min<float>(reli_score, 0.25) / 0.25;
						b = 0;
					}
					else if (0.25 <= reli_score && reli_score < 0.50)
					{
						r = 1.0 - min<float>((reli_score - 0.25), 0.25) / 0.25;
						g = 1.0;
						b = 0;
					}
					else if (0.50 <= reli_score && reli_score < 0.75)
					{
						r = 0;
						g = 1.0;
						b = min<float>(reli_score - 0.5, 0.25) / 0.25;
					}
					else
					{
						r = 0;
						g = 1.0 - min<float>((reli_score - 0.75), 0.25) / 0.25;
						b = 1.0;
					}
					view_color_repos[currntIdx] = Eigen::Vector3f(r, g, b);
				}
			}
		}
		}
	}
	*/

__global__ void Kernel_AvgPoints_v7(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff
	, VoxelExtraAttrib * point_extra_attrib_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extra_attrib_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, uint32_t	filterCount
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, bool isHDMode
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = (key >> 32) & 0xffff;
		auto yGlobalIndex = (key >> 16) & 0xffff;
		auto zGlobalIndex = (key) & 0xffff;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex].normalized();
		Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

		//unsigned short basePatchID = point_extra_attrib_buff[pointSlotIndex].startPatchID;
		VoxelExtraAttrib baseExtraAttrib = point_extra_attrib_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			kernel_return;
		}

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = ap;
		Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
		unsigned int apcount = 0;
		unsigned int ancount = 0;
		unsigned int point_nei_cnt = 0;
		unsigned int account = 0;

		//	mscho	@20240524
#define ForEachNeighborPt_v6(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v6((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

#define ForEachNeighborOnlyNm_v4(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_CntPnt_v3((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &point_nei_cnt, &ancount);

		int		idx_cnt_1st = (3 * 3 * 3);
		int		idx_cnt_2nd = (5 * 5 * 5);

		//	mscho	@20250313
		if (isHDMode || refMargin_normal >= 2)
		{
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);
		}

		for (int idx = 0; idx < idx_cnt_1st; idx++)
		{
			ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
		}
		for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
		{
			ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			if (point_nei_cnt > filterCount)
				break;
		}

		if (point_nei_cnt > filterCount)
		{
			Eigen::Vector3f voxel_pos = Eigen::Vector3f((((float)xGlobalIndex / 10.0) - 250.0f), (((float)yGlobalIndex / 10.0) - 250.0f), (((float)zGlobalIndex / 10.0) - 250.0f));

			Eigen::Vector3f ave_pos = Eigen::Vector3f(ap / (float)apcount);
			Eigen::Vector3f diff_pos = basePos - ave_pos;
			Eigen::Vector3f ave_normal = an.normalized();
			float	diff_dot = ave_normal.dot(diff_pos);
			//Eigen::Vector3f new_pos = basePos - ave_normal * diff_dot;
			Eigen::Vector3f new_pos = ave_pos;
			bool copy = true;

			if (new_pos.x() < minBound.x() || new_pos.x() > maxBound.x()) copy = false;
			else if (new_pos.y() < minBound.y() || new_pos.y() > maxBound.y()) copy = false;
			else if (new_pos.z() < minBound.z() || new_pos.z() > maxBound.z()) copy = false;
			if (copy) {
				uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
				view_pos_repos[currntIdx] = new_pos;// ave_pos;
				view_nm_repos[currntIdx] = ave_normal;
				view_color_repos[currntIdx] = (ac / account).cast<unsigned char>();
				view_extra_attrib_repos[currntIdx] = baseExtraAttrib;
				//view_lock_repos[currntIdx] = 1.0f;// (baseExtraAttrib.flags & VOXEL_FLAG_LOCKED_BIT) ? 1.0f : 0.0f;
			}
		}
	}
	}

// mscho @20250316
// Average point가 center voxel의 범위를 벗어나면, 해당 voxel의 Limit에 위치하도록 한다.
__global__ void Kernel_AvgPoints_v8(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff
	, VoxelExtraAttrib * point_extra_attrib_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extra_attrib_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, uint32_t filterCount
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, bool isHDMode
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex].normalized();
		Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

		//unsigned short basePatchID = point_extra_attrib_buff[pointSlotIndex].startPatchID;
		VoxelExtraAttrib baseExtraAttrib = point_extra_attrib_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			kernel_return;
		}

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = ap;
		Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
		unsigned int apcount = 0;
		unsigned int ancount = 0;
		unsigned int point_nei_cnt = 0;
		unsigned int account = 0;

		// mscho @20240524
#define ForEachNeighborPt_v6(xOffset, yOffset, zOffset, _basePos, _baseNormal, filteringValue)\
Device_ForEachNeighborPt_v6((xOffset), (yOffset), (zOffset), (_basePos), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

#define ForEachNeighborOnlyNm_v4(xOffset, yOffset, zOffset, _baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_CntPnt_v3((xOffset), (yOffset), (zOffset), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &point_nei_cnt, &ancount);

		int  idx_cnt_1st = (3 * 3 * 3);
		int  idx_cnt_2nd = (5 * 5 * 5);

		// mscho @20250306
		if (isHDMode || refMargin_normal >= 2)
		{
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);
		}

		for (int idx = 0; idx < idx_cnt_1st; idx++)
		{
			ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
		}
		for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
		{
			ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
			if (point_nei_cnt > filterCount)
				break;
		}

		if (point_nei_cnt > filterCount)
		{
			Eigen::Vector3f voxel_pos = Eigen::Vector3f((((float)xGlobalIndex / 10.0) - 250.0f), (((float)yGlobalIndex / 10.0) - 250.0f), (((float)zGlobalIndex / 10.0) - 250.0f));

			Eigen::Vector3f ave_pos = Eigen::Vector3f(ap / (float)apcount);
			Eigen::Vector3f diff_pos = basePos - ave_pos;
			Eigen::Vector3f ave_normal = an.normalized();
			float diff_dot = ave_normal.dot(diff_pos);

			//	mscho	@20250620
			//	point averaging 을 surface fitting 방식으로 평균 Normal 과 점 이동량의  dot product 만큼만 적용하도록 한다.
			Eigen::Vector3f new_pos = basePos - ave_normal * diff_dot;
			//Eigen::Vector3f new_pos = ave_pos;
			bool copy = true;




			if (new_pos.x() < minBound.x() || new_pos.x() > maxBound.x()) copy = false;
			else if (new_pos.y() < minBound.y() || new_pos.y() > maxBound.y()) copy = false;
			else if (new_pos.z() < minBound.z() || new_pos.z() > maxBound.z()) copy = false;
			if (copy) {
				//	mscho	@20250801					
				Eigen::Vector3f BaseCenter = basePos;
				float pointHalfPositions = 0.5 * ONE_VOXEL_SIZE;
				BaseCenter.x() = floorf(basePos.x() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.y() = floorf(basePos.y() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.z() = floorf(basePos.z() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;

				Eigen::Vector3f voxelMin;
				Eigen::Vector3f voxelMax;

				voxelMin.x() = BaseCenter.x() - pointHalfPositions;
				voxelMin.y() = BaseCenter.y() - pointHalfPositions;
				voxelMin.z() = BaseCenter.z() - pointHalfPositions;
				voxelMax.x() = BaseCenter.x() + pointHalfPositions;
				voxelMax.y() = BaseCenter.y() + pointHalfPositions;
				voxelMax.z() = BaseCenter.z() + pointHalfPositions;

				if (new_pos.x() < voxelMin.x())   new_pos.x() = voxelMin.x();
				else if (new_pos.x() >= voxelMax.x()) new_pos.x() = voxelMax.x() - 0.0001f;
				if (new_pos.y() < voxelMin.y())   new_pos.y() = voxelMin.y();
				else if (new_pos.y() >= voxelMax.y()) new_pos.y() = voxelMax.y() - 0.0001f;
				if (new_pos.z() < voxelMin.z())   new_pos.z() = voxelMin.z();
				else if (new_pos.z() >= voxelMax.z()) new_pos.z() = voxelMax.z() - 0.0001f;

				uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
				view_pos_repos[currntIdx] = new_pos;// ave_pos;
				view_nm_repos[currntIdx] = ave_normal;
				view_color_repos[currntIdx] = (ac / account).cast<unsigned char>();
				view_extra_attrib_repos[currntIdx] = baseExtraAttrib;
				//view_lock_repos[currntIdx] = 1.0f;// (baseExtraAttrib.flags & VOXEL_FLAG_LOCKED_BIT) ? 1.0f : 0.0f;
			}
		}
	}
	}

// mscho @20250620
// Average point가 center voxel의 범위를 벗어나면, 해당 voxel의 Limit에 위치하도록 한다.
__global__ void Kernel_AvgPoints_v9(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff
	, VoxelExtraAttrib * point_extra_attrib_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extra_attrib_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, uint32_t filterCount
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, bool isHDMode
	, bool isMetalMode
	, bool isNewPattern	//	mscho	@20250624
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex].normalized();
		Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

		//unsigned short basePatchID = point_extra_attrib_buff[pointSlotIndex].startPatchID;
		VoxelExtraAttrib baseExtraAttrib = point_extra_attrib_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			kernel_return;
		}

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = ap;
		Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
		unsigned int apcount = 0;
		unsigned int ancount = 0;
		unsigned int point_nei_cnt = 0;
		unsigned int account = 0;

		// mscho @20240524
#define ForEachNeighborPt_v7(xOffset, yOffset, zOffset, _basePos, _baseNormal, filteringValue, bEnPos, bEnNor, bEnColor)\
Device_ForEachNeighborPt_v7((xOffset), (yOffset), (zOffset), (_basePos), (_baseNormal), (filteringValue), (bEnPos), (bEnNor), (bEnColor), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

//#define ForEachNeighborPt_v6(xOffset, yOffset, zOffset, _basePos, _baseNormal, filteringValue)\
//Device_ForEachNeighborPt_v6((xOffset), (yOffset), (zOffset), (_basePos), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
//voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

#define ForEachNeighborOnlyNm_v4(xOffset, yOffset, zOffset, _baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_CntPnt_v3((xOffset), (yOffset), (zOffset), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &point_nei_cnt, &ancount);

		int  idx_cnt_1st = (3 * 3 * 3);
		int  idx_cnt_2nd = (5 * 5 * 5);

		int		idx_cnt_enough = (3 * 3 * 3);
		bool	isEnoughPoint = false;
		bool	isWide = false;
		// mscho @20250306
		if (isHDMode || refMargin_normal >= 2)
		{
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);
			isWide = true;
		}

		//	flags에 현재 voxel의 딥러닝 material ID가 저장되어 있다.
		bool isMaterialMetal = (isMetalMode && (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT));
		if (isMaterialMetal)
		{
			//	Metal mode이고, compound 의  point deep learning id 가 metal 일때
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);

			//	mscho	@20250624	==> @20250625
			if (!isNewPattern)
				filterCount--;
			else
				filterCount++;
			for (int idx = 0; idx < idx_cnt_1st; idx++)
			{
				ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
				if (idx >= idx_cnt_enough - 1)
				{
					if (point_nei_cnt > filterCount)
					{
						isEnoughPoint = true;
						break;
					}
				}
			}
			if (!isEnoughPoint)
			{
				for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
				{
					ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
					if (point_nei_cnt > filterCount)
						break;
				}
			}
		}
		else
		{
			//	Metal mode가 아니고, 이고, 12Pixel pattern이 아니고..compound 의  point deep learning id 가 metal 일때
			//	딥러닝 결과가 메탈인 경우에는 average point th  를 1 낮춰서
			//	좀 쉽게 생성되도록 한다.
			//	mscho	@20250624

			if (!isNewPattern)
			{
				if (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT)
					filterCount--;
			}
			else
			{
				if (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT)
					filterCount++;
			}

			if (isWide)
			{
				for (int idx = 0; idx < idx_cnt_1st; idx++)
				{
					ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
					if (idx >= idx_cnt_enough - 1)
					{
						if (point_nei_cnt > filterCount)
						{
							isEnoughPoint = true;
							break;
						}
					}
				}
				if (!isEnoughPoint)
				{
					for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
					{
						ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
						if (point_nei_cnt > filterCount)
							break;
					}
				}
			}
			else
			{
				bool	_bEnPos = true;
				bool	_bEnNor = true;
				bool	_bEnColor = true;
				for (int idx = 0; idx < idx_cnt_1st; idx++)
				{
					//ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
					//	mscho	@20250625	==> @20250627
					//  filterCount	보다 주변점의 갯수가 많으면, 더 이상 average하지 말고 탈출하도록 한다.					 
					//if (idx >= idx_cnt_enough - 1)
					//{
					//	if (point_nei_cnt > filterCount)
					//	{
					//		isEnoughPoint = true;
					//		break;
					//	}
					//}
					//	mscho	@20250627
					ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);
					if (point_nei_cnt > filterCount * 3.0)
					{
						_bEnPos = false;
						_bEnNor = true;
						isEnoughPoint = true;
						break;
					}
					else if (point_nei_cnt > filterCount * 2.0)
					{
						_bEnPos = false;
						_bEnNor = true;
						isEnoughPoint = true;
					}
					else if (point_nei_cnt > filterCount)
					{
						_bEnPos = true;
						_bEnNor = true;
						isEnoughPoint = true;
					}
					//else if (point_nei_cnt > filterCount / 2)
					//{
					//	_bEnColor = false;
					//}
				}
				if (point_nei_cnt <= filterCount && !isEnoughPoint)
				{
					for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
					{
						//ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
						//if (point_nei_cnt > filterCount * 2)
						//	break;
						//	mscho	@20250627
						ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);
						if (point_nei_cnt > filterCount * 2.0)
						{
							_bEnPos = false;
							_bEnNor = true;
							break;
						}
						else if (point_nei_cnt > filterCount)
						{
							_bEnPos = false;
							_bEnNor = true;
							isEnoughPoint = true;
						}
						//else if (point_nei_cnt > filterCount / 2)
						//{
						//	_bEnColor = false;
						//}
					}
				}

			}
		}


		if (point_nei_cnt > filterCount)
		{
			Eigen::Vector3f voxel_pos = Eigen::Vector3f((((float)xGlobalIndex / 10.0) - 250.0f), (((float)yGlobalIndex / 10.0) - 250.0f), (((float)zGlobalIndex / 10.0) - 250.0f));

			Eigen::Vector3f ave_pos = Eigen::Vector3f(ap / (float)apcount);
			Eigen::Vector3f diff_pos = basePos - ave_pos;
			Eigen::Vector3f ave_normal = an.normalized();
			float diff_dot = ave_normal.dot(diff_pos);

			//	mscho	@20250620
			//	point averaging 을 surface fitting 방식으로 평균 Normal 과 점 이동량의  dot product 만큼만 적용하도록 한다.
			Eigen::Vector3f new_pos = basePos - ave_normal * diff_dot;
			//Eigen::Vector3f new_pos = ave_pos;
			bool copy = true;




			if (new_pos.x() < minBound.x() || new_pos.x() > maxBound.x()) copy = false;
			else if (new_pos.y() < minBound.y() || new_pos.y() > maxBound.y()) copy = false;
			else if (new_pos.z() < minBound.z() || new_pos.z() > maxBound.z()) copy = false;
			if (copy) {
				//	mscho	@20250801
				Eigen::Vector3f BaseCenter = basePos;
				float pointHalfPositions = 0.5 * ONE_VOXEL_SIZE;
				BaseCenter.x() = floorf(basePos.x() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.y() = floorf(basePos.y() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.z() = floorf(basePos.z() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;

				Eigen::Vector3f voxelMin;
				Eigen::Vector3f voxelMax;

				voxelMin.x() = BaseCenter.x() - pointHalfPositions;
				voxelMin.y() = BaseCenter.y() - pointHalfPositions;
				voxelMin.z() = BaseCenter.z() - pointHalfPositions;
				voxelMax.x() = BaseCenter.x() + pointHalfPositions;
				voxelMax.y() = BaseCenter.y() + pointHalfPositions;
				voxelMax.z() = BaseCenter.z() + pointHalfPositions;

				if (new_pos.x() < voxelMin.x())   new_pos.x() = voxelMin.x();
				else if (new_pos.x() >= voxelMax.x()) new_pos.x() = voxelMax.x() - 0.0001f;
				if (new_pos.y() < voxelMin.y())   new_pos.y() = voxelMin.y();
				else if (new_pos.y() >= voxelMax.y()) new_pos.y() = voxelMax.y() - 0.0001f;
				if (new_pos.z() < voxelMin.z())   new_pos.z() = voxelMin.z();
				else if (new_pos.z() >= voxelMax.z()) new_pos.z() = voxelMax.z() - 0.0001f;

				uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
				view_pos_repos[currntIdx] = new_pos;// ave_pos;
				view_nm_repos[currntIdx] = ave_normal;
				view_color_repos[currntIdx] = (ac / account).cast<unsigned char>();
				view_extra_attrib_repos[currntIdx] = baseExtraAttrib;
				//view_lock_repos[currntIdx] = 1.0f;// (baseExtraAttrib.flags & VOXEL_FLAG_LOCKED_BIT) ? 1.0f : 0.0f;
			}
		}
	}
	}

// mscho @20250627
// Average point가 center voxel의 범위를 벗어나면, 해당 voxel의 Limit에 위치하도록 한다.
__global__ void Kernel_AvgPoints_v10(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,

	uint32_t cnt_contains

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff
	, VoxelExtraAttrib * point_extra_attrib_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extra_attrib_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
	, uint32_t filterCount
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, bool isHDMode
	, bool isMetalMode
	, bool isNewPattern	//	mscho	@20250624
	, VoxelExtractMode voxelExtractMode
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif
		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = point_pos_buff[pointSlotIndex];
		Eigen::Vector3f baseNormal = point_nm_buff[pointSlotIndex].normalized();
		Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

		//unsigned short basePatchID = point_extra_attrib_buff[pointSlotIndex].startPatchID;
		VoxelExtraAttrib baseExtraAttrib = point_extra_attrib_buff[pointSlotIndex];

		if (false == VECTOR3F_VALID_(baseNormal))
		{
			kernel_return;
		}

		Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		Eigen::Vector3f an = ap;
		Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
		unsigned int apcount = 0;
		unsigned int ancount = 0;
		unsigned int point_nei_cnt = 0;
		unsigned int account = 0;

		// mscho @20240524
#define ForEachNeighborPt_v7(xOffset, yOffset, zOffset, _basePos, _baseNormal, filteringValue, bEnPos, bEnNor, bEnColor)\
Device_ForEachNeighborPt_v7((xOffset), (yOffset), (zOffset), (_basePos), (_baseNormal), (filteringValue), (bEnPos), (bEnNor), (bEnColor), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

//#define ForEachNeighborPt_v6(xOffset, yOffset, zOffset, _basePos, _baseNormal, filteringValue)\
//Device_ForEachNeighborPt_v6((xOffset), (yOffset), (zOffset), (_basePos), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
//voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &account, &point_nei_cnt, &ancount);

#define ForEachNeighborOnlyNm_v4(xOffset, yOffset, zOffset, _baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_CntPnt_v3((xOffset), (yOffset), (zOffset), (_baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &point_nei_cnt, &ancount);

		int  idx_cnt_1st = (3 * 3 * 3);
		int  idx_cnt_2nd = (5 * 5 * 5);

		int		idx_cnt_enough = (3 * 3 * 3);
		bool	isEnoughPoint = false;
		bool	isWide = false;
		// mscho @20250306
		if (isHDMode || refMargin_normal >= 2)
		{
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);
			isWide = true;
		}

		//	flags에 현재 voxel의 딥러닝 material ID가 저장되어 있다.
		bool isMaterialMetal = (isMetalMode && (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT));
		if (isMaterialMetal)
		{
			//	Metal mode이고, compound 의  point deep learning id 가 metal 일때
			idx_cnt_1st = (5 * 5 * 5);
			idx_cnt_2nd = (7 * 7 * 7);

			//	mscho	@20250624	==> @20250625
			if (!isNewPattern)
				filterCount--;
			else
				filterCount++;
			bool	_bEnPos = true;
			bool	_bEnNor = true;
			bool	_bEnColor = true;
			for (int idx = 0; idx < idx_cnt_1st; idx++)
			{
				ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);
				if (idx >= idx_cnt_enough - 1)
				{
					if (point_nei_cnt > filterCount)
					{
						isEnoughPoint = true;
						break;
					}
				}
				if (point_nei_cnt > filterCount * 3.0)
				{
					_bEnPos = false;
					_bEnNor = true;
					_bEnColor = false;
					isEnoughPoint = true;
					break;
				}
				else if (point_nei_cnt > filterCount * 2.0)
				{
					_bEnPos = false;
					_bEnNor = true;
					_bEnColor = false;
					isEnoughPoint = true;
				}
				else if (point_nei_cnt > filterCount)
				{
					_bEnPos = true;
					_bEnNor = true;
					_bEnColor = true;
					isEnoughPoint = true;
				}
			}
			if (!isEnoughPoint)
			{
				for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
				{
					ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
					if (point_nei_cnt > filterCount)
						break;
				}
			}
		}
		else
		{
			//	Metal mode가 아니고, 이고, 12Pixel pattern이 아니고..compound 의  point deep learning id 가 metal 일때
			//	딥러닝 결과가 메탈인 경우에는 average point th  를 1 낮춰서
			//	좀 쉽게 생성되도록 한다.
			//	mscho	@20250624

			if (!isNewPattern)
			{
				if (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT)
					filterCount--;
			}
			else
			{
				if (baseExtraAttrib.deepLearningClass == DL_METAL || baseExtraAttrib.deepLearningClass == DL_ABUTMENT)
					filterCount++;
			}

			if (isWide)
			{
				bool	_bEnPos = true;
				bool	_bEnNor = true;
				bool	_bEnColor = true;
				for (int idx = 0; idx < idx_cnt_1st; idx++)
				{
					ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);
					if (idx >= idx_cnt_enough - 1)
					{
						if (point_nei_cnt > filterCount)
						{
							isEnoughPoint = true;
							break;
						}
					}
					if (point_nei_cnt > filterCount * 3.0)
					{
						_bEnPos = false;
						_bEnNor = true;
						_bEnColor = false;
						isEnoughPoint = true;
						break;
					}
					else if (point_nei_cnt > filterCount * 2.0)
					{
						//_bEnPos = false;
						_bEnNor = true;
						//_bEnColor = false;
						isEnoughPoint = true;
					}
					else if (point_nei_cnt > filterCount)
					{
						_bEnPos = true;
						_bEnNor = true;
						_bEnColor = true;
						isEnoughPoint = true;
					}
				}
				if (!isEnoughPoint)
				{
					for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
					{
						ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
						if (point_nei_cnt > filterCount)
							break;
					}
				}
			}
			else
			{
				bool	_bEnPos = true;
				bool	_bEnNor = true;
				bool	_bEnColor = true;
				for (int idx = 0; idx < idx_cnt_1st; idx++)
				{
					//ForEachNeighborPt_v6(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt);
					//	mscho	@20250625	==> @20250627
					//  filterCount	보다 주변점의 갯수가 많으면, 더 이상 average하지 말고 탈출하도록 한다.					 

					//	mscho	@20250701
					ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);

					if (point_nei_cnt > filterCount * 3.0)
					{
						_bEnPos = false;
						_bEnNor = true;
						_bEnColor = false;
						isEnoughPoint = true;
						//break;
					}
					else if (point_nei_cnt > filterCount * 2.0)
					{
						_bEnPos = false;
						_bEnNor = true;
						_bEnColor = false;
						isEnoughPoint = true;
						//break;
					}
					else if (point_nei_cnt > filterCount)
					{
						//_bEnPos = true;
						//_bEnNor = true;
						//_bEnColor = true;
						//isEnoughPoint = true;
						_bEnPos = false;
						_bEnNor = true;
						_bEnColor = false;
						isEnoughPoint = true;
					}
				}
				if (point_nei_cnt <= filterCount && !isEnoughPoint)
				{
					for (int idx = idx_cnt_1st; idx < idx_cnt_2nd; idx++)
					{
						//ForEachNeighborOnlyNm_v4(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], baseNormal, filteringForPt);
						//if (point_nei_cnt > filterCount * 2)
						//	break;
						//	mscho	@20250627
						ForEachNeighborPt_v7(sc_voxel_neighbor_id[idx * 3], sc_voxel_neighbor_id[idx * 3 + 1], sc_voxel_neighbor_id[idx * 3 + 2], basePos, baseNormal, filteringForPt, _bEnPos, _bEnNor, _bEnColor);
						if (point_nei_cnt > filterCount * 2.0)
						{
							_bEnPos = false;
							_bEnNor = true;
							break;
						}
						else if (point_nei_cnt > filterCount)
						{
							_bEnPos = false;
							_bEnNor = true;
							isEnoughPoint = true;
						}
					}
				}

			}
		}

		if (point_nei_cnt > filterCount)
		{
			Eigen::Vector3f voxel_pos = Eigen::Vector3f((((float)xGlobalIndex / 10.0) - 250.0f), (((float)yGlobalIndex / 10.0) - 250.0f), (((float)zGlobalIndex / 10.0) - 250.0f));

			Eigen::Vector3f ave_pos = Eigen::Vector3f(ap / (float)apcount);
			Eigen::Vector3f diff_pos = basePos - ave_pos;
			Eigen::Vector3f ave_normal = an.normalized();
			float diff_dot = ave_normal.dot(diff_pos);

			//	mscho	@20250620
			//	point averaging 을 surface fitting 방식으로 평균 Normal 과 점 이동량의  dot product 만큼만 적용하도록 한다.
			Eigen::Vector3f new_pos = basePos - ave_normal * diff_dot;
			//Eigen::Vector3f new_pos = ave_pos;
			bool copy = true;
			if (new_pos.x() < minBound.x() || new_pos.x() > maxBound.x()) copy = false;
			else if (new_pos.y() < minBound.y() || new_pos.y() > maxBound.y()) copy = false;
			else if (new_pos.z() < minBound.z() || new_pos.z() > maxBound.z()) copy = false;
			if (copy) {
				//	mscho	@20250801
				Eigen::Vector3f BaseCenter = basePos;
				float pointHalfPositions = 0.5 * ONE_VOXEL_SIZE;
				BaseCenter.x() = floorf(basePos.x() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.y() = floorf(basePos.y() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;
				BaseCenter.z() = floorf(basePos.z() / ONE_VOXEL_SIZE) * ONE_VOXEL_SIZE + pointHalfPositions;;

				Eigen::Vector3f voxelMin;
				Eigen::Vector3f voxelMax;

				voxelMin.x() = BaseCenter.x() - pointHalfPositions;
				voxelMin.y() = BaseCenter.y() - pointHalfPositions;
				voxelMin.z() = BaseCenter.z() - pointHalfPositions;
				voxelMax.x() = BaseCenter.x() + pointHalfPositions;
				voxelMax.y() = BaseCenter.y() + pointHalfPositions;
				voxelMax.z() = BaseCenter.z() + pointHalfPositions;

				if (new_pos.x() < voxelMin.x())   new_pos.x() = voxelMin.x();
				else if (new_pos.x() >= voxelMax.x()) new_pos.x() = voxelMax.x() - 0.0001f;
				if (new_pos.y() < voxelMin.y())   new_pos.y() = voxelMin.y();
				else if (new_pos.y() >= voxelMax.y()) new_pos.y() = voxelMax.y() - 0.0001f;
				if (new_pos.z() < voxelMin.z())   new_pos.z() = voxelMin.z();
				else if (new_pos.z() >= voxelMax.z()) new_pos.z() = voxelMax.z() - 0.0001f;

				uint32_t currntIdx = atomicAdd(count, (uint32_t)1);
				view_pos_repos[currntIdx] = new_pos;// ave_pos;
				view_nm_repos[currntIdx] = ave_normal;
				view_color_repos[currntIdx] = (ac / account).cast<unsigned char>();
				view_extra_attrib_repos[currntIdx] = baseExtraAttrib;
				//view_lock_repos[currntIdx] = 1.0f;// (baseExtraAttrib.flags & VOXEL_FLAG_LOCKED_BIT) ? 1.0f : 0.0f;
			}
		}
	}
	}

#ifndef BUILD_FOR_CPU

__global__ void Kernel_AvgPosColor_v1(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,

	HashKey64 * hashinfo_vtx

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3f * result_point_pos_buff
	, Eigen::Vector3b * point_clr_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, int refMargin_normal
	, int refMargin_point
	, float filteringForPt
	, float filteringForNm
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > hashinfo_vtx->Count_HashTableUsed - 1) return;

	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	if (cacheIndex >= voxelInfo.cache.voxelCount) return;
	auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
	if (pointSlotIndex == kEmpty32) return;

	Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
	Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();

	/*if (slotIndex < 10)
		printf("%llu, %d, %lld\n", cacheIndex, voxelInfo.gridSlotIndexCache_pts[cacheIndex], key);*/

	Eigen::Vector3f ap = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f an = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3ui ac = Eigen::Vector3ui(0, 0, 0);
	unsigned int apcount = 0;
	unsigned int ancount = 0;

#define ForEachNeighborPt_v3(xOffset, yOffset, zOffset, basePos, baseNormal, filteringValue)\
Device_ForEachNeighborPt_v2((xOffset), (yOffset), (zOffset), (basePos), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_pos_buff, point_nm_buff, point_clr_buff, &ap, &an, &ac, &apcount, &ancount);\

#define ForEachNeighborOnlyNm_v3(xOffset, yOffset, zOffset, baseNormal, filteringValue)\
Device_ForEachNeighborOnlyNm_v2((xOffset), (yOffset), (zOffset), (baseNormal), (filteringValue), refMargin_normal, &voxelInfo, xCacheIndex, yCacheIndex, zCacheIndex,\
voxelInfo.gridSlotIndexCache_pts, point_nm_buff, &an, &ancount);\

	for (short x = (refMargin_normal * -1); x < refMargin_normal + 1; x++)
	{
		for (short y = (refMargin_normal * -1); y < refMargin_normal + 1; y++)
		{
			for (short z = (refMargin_normal * -1); z < refMargin_normal + 1; z++)
			{
				if (
					(x >= (refMargin_point * -1) && x < refMargin_point + 1) &&
					(y >= (refMargin_point * -1) && y < refMargin_point + 1) &&
					(z >= (refMargin_point * -1) && z < refMargin_point + 1)
					)
				{
					//ForEachNeighborPt_v2(x, y, z);

					ForEachNeighborPt_v3(x, y, z, basePos, baseNormal, filteringForPt);
					//ForEachNeighborOnlyNm_v3(x, y, z);
					//ForEachNeighborOnlyNm_v2(0, 0, 0);
				}
				else
				{
					ForEachNeighborOnlyNm_v3(x, y, z, baseNormal, filteringForNm);
					//ForEachNeighborOnlyNm_v2(0, 0, 0);
				}

			}
		}
	}


	if (apcount > 0)
	{
		Eigen::Vector3f pointPosition = Eigen::Vector3f(ap / (float)apcount) * 10.0f;
		Eigen::Vector3f point_off = Eigen::Vector3f(pointPosition.x() + 1000.f, pointPosition.y() + 1000.f, pointPosition.z() + 1000.f);
		float   px = point_off.x();
		float   py = point_off.y();
		float   pz = point_off.z();
		point_off.x() = ((float)floorf(px) - 1000.f) * 0.1f + 0.05f;
		point_off.y() = ((float)floorf(py) - 1000.f) * 0.1f + 0.05f;
		point_off.z() = ((float)floorf(pz) - 1000.f) * 0.1f + 0.05f;
		Eigen::Vector3f pointNormal = an.normalized();
		auto cc = (ac / apcount).cast<unsigned char>();
		Eigen::Vector3b pointColor = cc;

		Eigen::Vector3f ave_point = point_off;
		Eigen::Vector3f ave_normal = pointNormal;
		Eigen::Vector3b ave_color = cc;
		uint32_t currntIdx = atomicAdd(count, 1);

		view_pos_repos[currntIdx] = ave_point;
		view_nm_repos[currntIdx] = ave_normal;
		view_color_repos[currntIdx] = ave_color;
	}
}
#endif

void MarchingCubes::clearCache_Main(thrust::device_vector<unsigned long long>&points_cacheIdx_main,
	cached_allocator * alloc_, CUstream_st * st, bool clearAll)
{
	//	mscho	@20240805
	//unsigned int host_Count_HashTableUsed_all = 0;
	//uint32_t used_cnt_Host;
	//cudaMemcpy(&used_cnt_Host, used_cnt_HashVoxel, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//cudaMemcpyAsync(used_cnt_HashVoxel_h, used_cnt_HashVoxel, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	//checkCudaSync(st);
	//__printLastCudaError("after cudaMemcpyAsync", __FILE__, __LINE__);
	//	mscho	@20240805

	if (clearAll) {
#ifndef BUILD_FOR_CPU
		if (exeInfo.maxSize_cache < exeInfo.cache.voxelCount) if (FOR_TEST_PRINT)qDebug() << "\033[31mlocalCache Size Error\033[30m";
		cudaMemsetAsync(exeInfo.gridSlotIndexCache, 0xFF, sizeof(uint32_t) * exeInfo.maxSize_cache, st);
		checkCudaSync(st);
#else
		memset(exeInfo.gridSlotIndexCache, 0xFF, sizeof(uint32_t) * exeInfo.cache.voxelCount);
#endif
		* used_cnt_HashVoxel_h = 0;

	}
	if (*used_cnt_HashVoxel_h > 0)
	{
		NvtxRangeCuda nvtxPrint("@cache Clear");

		uint32_t* gridSlotIndexCache_raw = exeInfo.gridSlotIndexCache;
#ifndef BUILD_FOR_CPU
		thrust::device_ptr<uint32_t> gridSlotIndexCache = thrust::device_pointer_cast(exeInfo.gridSlotIndexCache);
		auto _repos_cache_voxel = thrust::raw_pointer_cast(points_cacheIdx_main.data());
#else
		qDebug("gridSlotIndexCache_raw: %p", gridSlotIndexCache_raw);
		qDebug("used_cnt_HashVoxel_h: %p", used_cnt_HashVoxel_h);
		qDebug("*used_cnt_HashVoxel_h: %ud", *used_cnt_HashVoxel_h);
		qDebug("used_cnt_HashVoxel: %p", used_cnt_HashVoxel);
		qDebug("*used_cnt_HashVoxel: %ud", *used_cnt_HashVoxel);

#endif

#ifndef BUILD_FOR_CPU
		auto executionPolicy = thrust::cuda::par_nosync(*alloc_).on(st);
#else
		auto executionPolicy = thrust::omp::par;
#endif
		thrust::for_each(
			executionPolicy,
			points_cacheIdx_main.begin(),
			points_cacheIdx_main.begin() + (*used_cnt_HashVoxel_h),	//	mscho	@20240805
			[gridSlotIndexCache_raw]__device__(unsigned long long cachePosIndex)
		{
			//printf("cache value %llu\n", cachePosIndex);
			gridSlotIndexCache_raw[cachePosIndex] = UINT32_MAX;
		});

		//thrust::for_each(
		//	thrust::cuda::par_nosync(*alloc_).on(st),
		//	gridSlotIndexCache,
		//	gridSlotIndexCache + exeInfo.cache.voxelCount,
		//	[]__device__(uint32_t cacheValue)
		//{
		//	if (cacheValue != UINT32_MAX)
		//	{
		//		printf("main cache value %d\n", cacheValue);
		//	}
		//});

		//	mscho	@20240805
		checkCudaSync(st);
	}
}

//	mscho	@20240521 (New)
void MarchingCubes::clearCache_Extract(thrust::device_vector<unsigned long long>&points_cacheIdx_extract,
	cached_allocator * alloc_, CUstream_st * st, bool clearAll
)
{
	if (clearAll) {
#ifndef BUILD_FOR_CPU
		if (exeInfo.maxSize_cache < exeInfo.cache.voxelCount) if (FOR_TEST_PRINT)qDebug() << "\033[31mlocalCache Size Error\033[30m";
		cudaMemsetAsync(exeInfo.gridSlotIndexCache_pts, 0xFF, sizeof(uint32_t) * exeInfo.maxSize_cache, st);
		checkCudaSync(st);
#else
		memset(exeInfo.gridSlotIndexCache_pts, 0xFF, sizeof(uint32_t) * exeInfo.maxSize_cache);
#endif
		* used_cnt_Extract_h = 0;
	}

	if (*used_cnt_Extract_h > 0)
	{
		NvtxRangeCuda nvtxPrint("@cache Clear");

		uint32_t* gridSlotIndexCache_pts_raw = exeInfo.gridSlotIndexCache_pts;
#ifndef BUILD_FOR_CPU
		thrust::device_ptr<uint32_t> gridSlotIndexCache = thrust::device_pointer_cast(exeInfo.gridSlotIndexCache_pts);
#else
		qDebug("gridSlotIndexCache_pts_raw: %p", gridSlotIndexCache_pts_raw);
		qDebug("used_cnt_Extract_h: %p", used_cnt_Extract_h);
		qDebug("*used_cnt_Extract_h: %ud", *used_cnt_Extract_h);
		qDebug("used_cnt_HashVoxel: %p", used_cnt_HashVoxel);
		qDebug("*used_cnt_HashVoxel: %ud", *used_cnt_HashVoxel);
#endif

#ifndef BUILD_FOR_CPU
		auto executionPolicy = thrust::cuda::par_nosync(*alloc_).on(st);
#else
		auto executionPolicy = thrust::omp::par;
#endif

		thrust::for_each(
			executionPolicy,
			points_cacheIdx_extract.begin(),
			points_cacheIdx_extract.begin() + (*used_cnt_Extract_h),	//	mscho	@20240805
			[gridSlotIndexCache_pts_raw]__device__(unsigned long long cachePosIndex)
		{
			/*if(cachePosIndex < 100)
				printf("cache value %llu\n", cachePosIndex);*/
			gridSlotIndexCache_pts_raw[cachePosIndex] = UINT32_MAX;
		});

#ifndef BUILD_FOR_CPU
		//thrust::for_each(
		//	thrust::cuda::par_nosync(*alloc_).on(st),
		//	gridSlotIndexCache,
		//	gridSlotIndexCache + exeInfo.maxSize_cache,
		//	[]__device__(uint32_t cacheValue)
		//{
		//	if (cacheValue != UINT32_MAX)
		//	{
		//		if (cacheValue < 100)
		//			printf("Extract cache value %d\n", cacheValue);
		//	}
		//});
#endif
			//	mscho	@20240805
		checkCudaSync(st);
	}
}

//	mscho	@20240521 (New)
void MarchingCubes::Extract_cache_clear_sync(
	CUstream_st * sub_st
)
{
	//	mscho	@20240521
#ifndef BUILD_FOR_CPU
	checkCudaSync(sub_st);
#endif
}

#ifndef BUILD_FOR_CPU
/*// 사용되지 않음 -> v3로 변경됨
void MarchingCubes::ExtractVoxelPoints_v2(
	cached_allocator* alloc_, CUstream_st* st,
	bool filtering
)
{
	auto _repos_pos = thrust::raw_pointer_cast(pRegistration->m_points_pos.data());
	auto _repos_nm = thrust::raw_pointer_cast(pRegistration->m_points_nm.data());
	auto _repos_color = thrust::raw_pointer_cast(pRegistration->m_points_color.data());
	auto _repos_material_id = thrust::raw_pointer_cast(pRegistration->m_material_id.data());
	auto _repos_startPatch = thrust::raw_pointer_cast(pRegistration->m_start_patch.data());
	auto voxelValues = thrust::raw_pointer_cast(pRegistration->m_MC_voxelValues.data());
	auto voxelNormals = thrust::raw_pointer_cast(pRegistration->m_MC_voxelNormals.data());
	auto voxelValueCounts = thrust::raw_pointer_cast(pRegistration->m_MC_voxelValueCounts.data());
	auto voxelColors = thrust::raw_pointer_cast(pRegistration->m_MC_voxelColors.data());
	auto voxelSegmentations = thrust::raw_pointer_cast(pRegistration->m_MC_voxelSegmentations.data());
	auto voxelStartPatchIDs = thrust::raw_pointer_cast(pRegistration->m_MC_voxelStartPatchIDs.data());

	auto ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_cache_voxel = thrust::raw_pointer_cast(pRegistration->m_points_cacheIdx_main.data());

	unsigned int* occupiedVoxelIndices = thrust::raw_pointer_cast(m_MC_used_buff.data());
	//	mscho	@20240521 (New)
	uint32_t* gridSlotIndexCache = exeInfo.gridSlotIndexCache;
	uint32_t* gridSlotIndexCache_pts = exeInfo.gridSlotIndexCache_pts;
	//cudaMemsetAsync(gridSlotIndexCache, 0xFF, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);
	//cudaMemsetAsync(gridSlotIndexCache_pts, 0xFF, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);

	{
		pHashManager->reset_hashKeyCountValue64(exeInfo.globalHashInfo, st);
		nvtxRangePushA("@Kernel_GatherOccupiedVoxelIndices");

		cudaMemsetAsync(used_cnt_HashVoxel, 0, sizeof(uint32_t), st);

		Kernel_GatherOccupiedVoxelIndices << <BLOCKS_PER_GRID_THREAD_N(exeInfo.blockIndex, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (
			exeInfo,
			occupiedVoxelIndices
			, exeInfo.blockSize
			, exeInfo.blockIndex
			, exeInfo.blockRemainder
			, used_cnt_HashVoxel
			, _repos_cache_voxel
			);
		checkCudaErrors(cudaGetLastError());

		nvtxRangePop();
		checkCudaSync(st);
	}

	unsigned int host_Count_HashTableUsed = 0;
	cudaMemcpyAsync(&host_Count_HashTableUsed, &exeInfo.globalHashInfo->Count_HashTableUsed, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	{

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ExtractVoxelPoints_v4, 0, 0));
		int gridsize = ((uint32_t)host_Count_HashTableUsed + threadblocksize - 1) / threadblocksize;

		qDebug("================================= [Hash] Local+Margin UsedCount: %d", host_Count_HashTableUsed);

		nvtxRangePushA("@Kernel_ExtractVoxelPoints");

		//pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);
		cudaMemsetAsync(used_cnt_Extract, 0, sizeof(uint32_t), st);
		cudaMemsetAsync(used_cnt_localContains, 0, sizeof(uint32_t), st);

		//	mscho	@20240422
		Eigen::Vector3f  _halfpoint = Eigen::Vector3f(
			exeInfo.global.voxelSize * 0.5f,
			exeInfo.global.voxelSize * 0.5f,
			exeInfo.global.voxelSize * 0.5f);

		//	mscho	@20240422
		if (gridsize > 0)
			Kernel_ExtractVoxelPoints_v4 << <gridsize, threadblocksize, 0, st >> > (
				exeInfo,
				occupiedVoxelIndices,
				gridSlotIndexCache,

				voxelValues,
				voxelNormals,
				voxelValueCounts,
				voxelColors,
				voxelSegmentations,
				voxelStartPatchIDs,

				_repos_pos,
				_halfpoint,
				_repos_nm,
				_repos_color,
				_repos_material_id,
				_repos_startPatch,

				ext_used_Key,
				used_cnt_Extract,
				used_cnt_localContains,

				true,
				filtering);
		checkCudaErrors(cudaGetLastError());

		nvtxRangePop();
		checkCudaSync(st);
	}
	if (0) {
		unsigned int host_Count_avgArea = 0;
		unsigned int host_Count_cacheArea = 0;
		cudaMemcpyAsync(&host_Count_avgArea, used_cnt_localContains, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
		cudaMemcpyAsync(&host_Count_cacheArea, used_cnt_Extract, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
		checkCudaSync(st);
		//qDebug("================================= [Cache] Local+Margin Count: %d", host_Count_cacheArea);
		//qDebug("================================= [Array] Local Count: %d", host_Count_avgArea);
	}
/# no longer used
		if (captureFlag == 1)
		{
			//Eigen::Vector3f* host_pos = new Eigen::Vector3f[pRegistration->m_points_pos.size()];
			//cudaMemcpyAsync(host_pos, _repos_pos, sizeof(Eigen::Vector3f) * pRegistration->m_points_pos.size(), cudaMemcpyDeviceToHost, st);

			//Eigen::Vector3f* host_nm = new Eigen::Vector3f[pRegistration->m_points_nm.size()];
			//cudaMemcpyAsync(host_nm, _repos_pos, sizeof(Eigen::Vector3f) * pRegistration->m_points_nm.size(), cudaMemcpyDeviceToHost, st);

			//Eigen::Vector3b* host_color = new Eigen::Vector3b[pRegistration->m_points_color.size()];
			//cudaMemcpyAsync(host_color, _repos_color, sizeof(Eigen::Vector3b) * pRegistration->m_points_color.size(), cudaMemcpyDeviceToHost, st);

			//PLYFormat ply;
			//for (size_t i = 0; i < pRegistration->m_points_pos.size(); i++)
			//{
			//	auto& p = host_pos[i];
			//	if (VECTOR3F_VALID_(p))
			//	{
			//		ply.AddPointFloat3(p.data());
			//		auto& n = host_nm[i];
			//		ply.AddNormalFloat3(n.data());
			//		auto& c = host_color[i];
			//		ply.AddColor((float)c.x() / 255.0f, (float)c.y() / 255.0f, (float)c.z() / 255.0f);
			//	}
			//}
			//ply.Serialize(GetResourcesFolderPath() + "\\Debug\\HD\\Extracted_BeforeAveraging.ply");
		}
#/
	}
	*/

#endif
	//	mscho	@20240523
	//	Extract를 하는 TSDF 범위를 지정해서, data를 모으는 기능을 추가한다.
uint32_t MarchingCubes::ExtractVoxelPoints_v3(
	float	min_tsdf,
	float	max_tadf,
	int margin,

	// output
	ExtractionPointCloud & point_cloud,
	thrust::device_vector<unsigned long long>&points_cacheIdx_main,
	thrust::device_vector<unsigned long long>&points_cacheIdx_extract,

	cached_allocator * alloc_, CUstream_st * st,
	bool filtering
)
{
	NvtxRangeCuda nvtxPrint("@sh/ExtractVoxelPoints", true, 0xFF00FF00);
	auto _repos_pos = thrust::raw_pointer_cast(point_cloud.points_.data());
	auto _repos_nm = thrust::raw_pointer_cast(point_cloud.normals_.data());
	auto _repos_color = thrust::raw_pointer_cast(point_cloud.colors_.data());
	auto _repos_extra_attrib = thrust::raw_pointer_cast(point_cloud.extraAttribs_.data());
	//auto _repos_flags = thrust::raw_pointer_cast(pRegistration->m_point_flags.data());

	const auto voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	const auto voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	const auto voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	const auto voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	const auto voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	const auto voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	auto ext_used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_cache_voxel = thrust::raw_pointer_cast(points_cacheIdx_main.data());
	auto _repos_cache_extract = thrust::raw_pointer_cast(points_cacheIdx_extract.data());

	unsigned int* occupiedVoxelIndices = thrust::raw_pointer_cast(m_MC_used_buff.data());

	//	mscho	@20240521 (New)
	uint32_t* gridSlotIndexCache = exeInfo.gridSlotIndexCache;
	uint32_t* gridSlotIndexCache_pts = exeInfo.gridSlotIndexCache_pts;

	//	Cache initial은 Integrate하기 바로전에 하도록 한다..
	//  다른 Stream으로 처리를 해서, 가능하다면 병렬처리가 될 수 있는 구조로 변경한다.
	//cudaMemsetAsync(gridSlotIndexCache, 0xFF, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);
	//cudaMemsetAsync(gridSlotIndexCache_pts, 0xFF, sizeof(uint32_t) * exeInfo.cache.voxelCount, st);

	{
#ifndef BUILD_FOR_CPU
		pHashManager->reset_hashKeyCountValue64(exeInfo.globalHashInfo, st);

		NvtxRangeCuda nvtxPrint("@Kernel_GatherOccupiedVoxelIndices");
		cudaMemsetAsync(used_cnt_HashVoxel, 0, sizeof(uint32_t), st);
#else
		pHashManager->reset_hashKeyCountValue64(exeInfo.globalHashInfo);
		*used_cnt_HashVoxel = 0;
#endif

		//	mscho	@20240717
		//	kernel 내부가 좀 복잡해지면서, block화 해서 내부에서 Loop를 도는 경우
		//	전체적인 수행속도가 낮아지는 경향이 있어서
		//	kernel단 한번씩만 수행하는 방식으로 복귀한다.
		if (exeInfo.blockSize == 1)
		{
			//	mscho	@20250714
//#ifndef BUILD_FOR_CPU
//				Kernel_GatherOccupiedVoxelIndices_v3 << <BLOCKS_PER_GRID_THREAD_N(exeInfo.blockIndex, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
//#else
//				Kernel_GatherOccupiedVoxelIndices_v3(
//#endif
//					exeInfo,
//					occupiedVoxelIndices
//					, min_tsdf			// Extract 대상이 되는 Voxel의 Voxel value min값
//					, max_tadf			// Extract 대상이 되는 Voxel의 Voxel value max값
//					, exeInfo.blockSize
//					, exeInfo.blockIndex
//					, exeInfo.blockRemainder
//					, used_cnt_HashVoxel
//					, _repos_cache_voxel
//					, margin
//				);

			unsigned int		block_n = 8;
			unsigned int		block_size = exeInfo.blockIndex / block_n;
			unsigned int		block_last = exeInfo.blockIndex - (block_n - 1) * block_size;
			for (unsigned int i = 0; i < block_n; i++)
			{
				size_t	thread_n = (size_t)(i == block_n - 1 ? block_last : block_size);
#ifndef BUILD_FOR_CPU
				Kernel_GatherOccupiedVoxelIndices_v4 << <BLOCKS_PER_GRID_THREAD_N(thread_n, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
				Kernel_GatherOccupiedVoxelIndices_v3(
#endif
					exeInfo,
					occupiedVoxelIndices
					, min_tsdf			// Extract 대상이 되는 Voxel의 Voxel value min값
					, max_tadf			// Extract 대상이 되는 Voxel의 Voxel value max값
					, block_size
					, (i == block_n - 1 ? block_last : block_size)
					, i
					, used_cnt_HashVoxel
					, _repos_cache_voxel
					, margin
				);
			}
		}
		else
#ifndef BUILD_FOR_CPU
					Kernel_GatherOccupiedVoxelIndices_v2 << <BLOCKS_PER_GRID_THREAD_N(exeInfo.blockIndex, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (
#else
						Kernel_GatherOccupiedVoxelIndices_v2(
#endif
							exeInfo,
							occupiedVoxelIndices
							, min_tsdf			// Extract 대상이 되는 Voxel의 Voxel value min값
							, max_tadf			// Extract 대상이 되는 Voxel의 Voxel value max값
							, exeInfo.blockSize
							, exeInfo.blockIndex
							, exeInfo.blockRemainder
							, used_cnt_HashVoxel
							, _repos_cache_voxel
						);
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaGetLastError());
		//	mscho	@20250207
		//checkCudaSync(st);
#endif
	}

	unsigned int host_Count_HashTableUsed = 0;
#ifndef BUILD_FOR_CPU
	//	mscho	@20250228
	//cudaMemcpyAsync(&host_Count_HashTableUsed, &exeInfo.globalHashInfo->Count_HashTableUsed, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(host_used_cnt_HashVoxel, &exeInfo.globalHashInfo->Count_HashTableUsed, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(used_cnt_HashVoxel_h, used_cnt_HashVoxel, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);
#else
	host_Count_HashTableUsed = exeInfo.globalHashInfo->Count_HashTableUsed;
	*used_cnt_HashVoxel_h = *used_cnt_HashVoxel;
#endif

	//	mscho	@20250228
	if (*host_used_cnt_HashVoxel == 0)
		return 0;
	else {
#ifndef BUILD_FOR_CPU
		//	mscho	@20250131

		//int mingridsize;
		//int threadblocksize;
		//checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ExtractVoxelPoints_v5, 0, 0));
		//int gridsize = ((uint32_t)host_Count_HashTableUsed + threadblocksize - 1) / threadblocksize;
#endif
			//	mscho	@20250228
			//qDebug("ExtractVoxelPoints_v3 [0] Local+Margin UsedCount: %d", host_Count_HashTableUsed);


			//pHashManager->reset_hashKeyCountValue64(hInfo_global_vtx, st);
#ifndef BUILD_FOR_CPU
		cudaMemsetAsync(used_cnt_Extract, 0, sizeof(uint32_t), st);
		cudaMemsetAsync(used_cnt_localContains, 0, sizeof(uint32_t), st);
#else
		* used_cnt_Extract = 0;
		*used_cnt_localContains = 0;
#endif

		//	mscho	@20240422
		Eigen::Vector3f  _halfpoint = Eigen::Vector3f(
			exeInfo.global.voxelSize * 0.5f,
			exeInfo.global.voxelSize * 0.5f,
			exeInfo.global.voxelSize * 0.5f);

		/// Extract Chache Clear Check
		//thrust::for_each(
		//	thrust::cuda::par_nosync(*alloc_).on(st),
		//	gridSlotIndexCache_pts,
		//	gridSlotIndexCache_pts + exeInfo.maxSize_cache,
		//	[]__device__(uint32_t cacheValue)
		//	{
		//	if (cacheValue != UINT32_MAX)
		//	{
		//		printf("preCheck Extract cache value %d\n", cacheValue);
		//	}
		//	});

		//	mscho	@20250714
		if (*host_used_cnt_HashVoxel > 100)
		{
			NvtxRangeCuda nvtxPrint("@Kernel_ExtractVoxelPoints_v6");
			unsigned int		block_n = 4;
			unsigned int		block_size = *host_used_cnt_HashVoxel / block_n;
			unsigned int		block_last = *host_used_cnt_HashVoxel - (block_n - 1) * block_size;

			for (unsigned int i = 0; i < block_n; i++)
			{
				size_t thread_n = (size_t)(i == block_n - 1 ? block_last : block_size);
#ifndef BUILD_FOR_CPU
				//	mscho	@20250131	==> @20250228
				//Kernel_ExtractVoxelPoints_v5 << <gridsize, threadblocksize, 0, st >> > (
				Kernel_ExtractVoxelPoints_v6 << <BLOCKS_PER_GRID_THREAD_N(thread_n, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (
#else
				Kernel_ExtractVoxelPoints_v5(
#endif
					exeInfo,
					occupiedVoxelIndices,
					gridSlotIndexCache,

					voxelValues,
					voxelNormals,
					voxelValueCounts,
					voxelColors,
					voxelSegmentations,
					voxelExtraAttribs,

					_repos_pos,
					_halfpoint,
					_repos_nm,
					_repos_color,
					_repos_extra_attrib,

					ext_used_Key,
					used_cnt_Extract,
					used_cnt_localContains,
					_repos_cache_extract,

					block_size,
					(i == block_n - 1 ? block_last : block_size),
					i,

					true,
					filtering);
				checkCudaErrors(cudaGetLastError());
			}
		}
		else
		{
			NvtxRangeCuda nvtxPrint("@Kernel_ExtractVoxelPoints_v5");
#ifndef BUILD_FOR_CPU
			//	mscho	@20250131	==> @20250228
			//Kernel_ExtractVoxelPoints_v5 << <gridsize, threadblocksize, 0, st >> > (
			Kernel_ExtractVoxelPoints_v5 << <BLOCKS_PER_GRID_THREAD_N(*host_used_cnt_HashVoxel, THRED256_PER_BLOCK), THRED256_PER_BLOCK, 0, st >> > (
#else
			Kernel_ExtractVoxelPoints_v5(
#endif
				exeInfo,
				occupiedVoxelIndices,
				gridSlotIndexCache,

				voxelValues,
				voxelNormals,
				voxelValueCounts,
				voxelColors,
				voxelSegmentations,
				voxelExtraAttribs,

				_repos_pos,
				_halfpoint,
				_repos_nm,
				_repos_color,
				_repos_extra_attrib,

				ext_used_Key,
				used_cnt_Extract,
				used_cnt_localContains,
				_repos_cache_extract,

				true,
				filtering);
			checkCudaErrors(cudaGetLastError());

			//	mscho	@20250207
			//checkCudaSync(st);
		}

	}
	//	mscho	@20250228
	//unsigned int host_Count_avgArea = 0;
	//unsigned int host_Count_cacheArea = 0;
#ifndef BUILD_FOR_CPU
	cudaMemcpyAsync(host_Count_avgArea, used_cnt_localContains, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	//cudaMemcpyAsync(&host_Count_cacheArea, used_cnt_Extract, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(used_cnt_Extract_h, used_cnt_Extract, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	thrust::host_vector<unsigned long long> hostCheck(points_cacheIdx_extract.begin(), points_cacheIdx_extract.begin() + *used_cnt_Extract_h);
#else
		//	mscho	@20250228
	* host_Count_avgArea = *used_cnt_localContains;
	//host_Count_cacheArea = *used_cnt_Extract;
	*used_cnt_Extract_h = *used_cnt_Extract;
#endif

	//	mscho	@20250228
	return *host_Count_avgArea;
}

/*
// no longer used -> AverageVoxelPoints_v4
uint32_t MarchingCubes::AverageVoxelPoints_v3(
	cached_allocator* alloc_, CUstream_st* st
	, Eigen::Vector3f* view_pos_repos
	, Eigen::Vector3f* view_nm_repos
	, Eigen::Vector3f* view_color_repos
	, unsigned char* view_mID_repos
	, float filteringForPt
	, float filteringForNm
	, int ref_margin_normal
	, int ref_margin_point
	, bool isHDMode
)
{
	nvtxRangePushA("@AverageVoxelPoints_v3");
	auto used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_pos = thrust::raw_pointer_cast(pRegistration->m_points_pos.data());
	auto _repos_nm = thrust::raw_pointer_cast(pRegistration->m_points_nm.data());
	auto _repos_color = thrust::raw_pointer_cast(pRegistration->m_points_color.data());
	auto _repos_material_id = thrust::raw_pointer_cast(pRegistration->m_material_id.data());
	//	shshin	@20240508
	auto _nm_Tmp = thrust::raw_pointer_cast(pRegistration->m_points_nm_avgTmp.data());


	//uint32_t* d_count; // 디바이스에서 카운트를 저장할 포인터
	uint32_t h_count = 0; // 호스트 카운트 값

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	//cudaMallocAsync(&d_count, sizeof(uint32_t), st);
	//checkCudaErrors(cudaMalloc(&d_count, sizeof(uint32_t)));
	//checkCudaErrors(cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st));
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaMemsetAsync(cnt_averageRes, 0, sizeof(uint32_t), st));
#else
		*cnt_averageRes = 0;
#endif


#ifndef BUILD_FOR_CPU
		unsigned int host_Count_HashTableUsed = 0;
		checkCudaErrors(cudaMemcpyAsync(&host_Count_HashTableUsed, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st));
		checkCudaSync(st);
#else
		unsigned int host_Count_HashTableUsed = *used_cnt_localContains;
#endif

		{
			qDebug("=================================host_Count_vtxCache : %d", host_Count_HashTableUsed);

			//	mscho	@20240521 (New)
			auto _voxelNormals = thrust::raw_pointer_cast(pRegistration->m_MC_voxelNormals.data());

			//	mscho	@20240523 ==> @20240527
			if (isHDMode)
			{
				qDebug("HDMode =================================host_Count_vtxCache : %d", host_Count_HashTableUsed);

#ifndef BUILD_FOR_CPU
				Kernel_AvgEstimateNormal_v4_HD << <BLOCKS_PER_GRID_THREAD_N((host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
				Kernel_AvgEstimateNormal_v4_HD(
#endif
					exeInfo,
					used_Key,
					//_neighbor_id_dev,
					host_Count_HashTableUsed,

					_voxelNormals,
					_repos_pos,
					_repos_nm,
					_repos_color,

					_nm_Tmp,

					ref_margin_normal,
					ref_margin_point,
					filteringForPt,
					filteringForNm
					);
			}
			else
			{
#ifndef BUILD_FOR_CPU
				Kernel_AvgEstimateNormal_v4 << <BLOCKS_PER_GRID_THREAD_N((host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
				Kernel_AvgEstimateNormal_v4(
#endif
					exeInfo,
					used_Key,
					//_neighbor_id_dev,
					host_Count_HashTableUsed,

					_voxelNormals,
					_repos_pos,
					_repos_nm,
					_repos_color,

					_nm_Tmp,

					ref_margin_normal,
					ref_margin_point,
					filteringForPt,
					filteringForNm
					);
			}
			checkCudaErrors(cudaGetLastError());

			if (isHDMode)
			{
#ifndef BUILD_FOR_CPU
				Kernel_AvgNormal_v1_HD << <BLOCKS_PER_GRID_THREAD_N(host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
				Kernel_AvgNormal_v1_HD(
#endif
					exeInfo,
					used_Key,

					host_Count_HashTableUsed,

					_repos_pos,
					_nm_Tmp,
					_repos_color,

					_repos_nm,

					isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
					isHDMode ? ref_margin_point + 1 : ref_margin_point,
					filteringForPt,
					filteringForNm //	mscho	@20240523	==> @20240611
					);
			}
			else
			{
				//	mscho	@20240524	==> @20240527
#ifndef BUILD_FOR_CPU
				Kernel_AvgNormal_v1 << <BLOCKS_PER_GRID_THREAD_N(host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
				Kernel_AvgNormal_v1(
#endif
					exeInfo,
					used_Key,

					host_Count_HashTableUsed,

					_repos_pos,
					_nm_Tmp,
					_repos_color,

					_repos_nm,

					ref_margin_normal - 1,	//	mscho	@20240620
					ref_margin_point,
					filteringForPt,
					filteringForNm//	mscho	@20240523	==> @20240611
					);
			}
			checkCudaErrors(cudaGetLastError());

			//	mscho	@20240524	==> @20240527
#ifndef BUILD_FOR_CPU
			Kernel_AvgPoints_v6 << <BLOCKS_PER_GRID_THREAD_N(host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgPoints_v6(
#endif
				exeInfo,
				used_Key,

				host_Count_HashTableUsed,

				_repos_pos,
				_repos_nm,
				_repos_color,
				_repos_material_id,

				//d_count,
				cnt_averageRes,

				view_pos_repos,
				view_nm_repos,
				view_color_repos,
				view_mID_repos,
				isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
				isHDMode ? ref_margin_point + 1 : ref_margin_point,
				filteringForPt,
				filteringForNm,	//	mscho	@20240523	==> @20240611
				//4//6
				isHDMode ? 20 : 4,	//	mscho	@20240611 ==> @20240620 ==> @20240625( 6 ==> 4 )
				isHDMode);
			checkCudaErrors(cudaGetLastError());
			//	mscho	@20240507
			//Kernel_AvgPoints_v4 << <gridsize, threadblocksize, 0, st >> > (
			//	exeInfo,
			//	used_Key,

			//	host_Count_HashTableUsed,

			//	_repos_pos,
			//	_repos_nm,
			//	_repos_color,

			//	d_count,

			//	view_pos_repos,
			//	view_nm_repos,
			//	view_color_repos,
			//	ref_margin_normal,
			//	ref_margin_point,
			//	filteringForPt,
			//	filteringForNm - 1	//	mscho	@20240523
			//	);
#ifndef BUILD_FOR_CPU
			checkCudaSync(st);
			checkCudaErrors(cudaMemcpyAsync(&h_count, cnt_averageRes, sizeof(uint32_t), cudaMemcpyDeviceToHost, st));
			checkCudaSync(st);
#else
			h_count = *cnt_averageRes;
#endif


#ifndef BUILD_FOR_CPU
			if (false && h_count > 0)
			{
				// 결과 사용
				Eigen::Vector3f* host_pts = new Eigen::Vector3f[h_count];
				Eigen::Vector3f* host_nms = new Eigen::Vector3f[h_count];
				Eigen::Vector3f* host_clrs = new Eigen::Vector3f[h_count];

				checkCudaErrors(cudaMemcpyAsync(host_pts, view_pos_repos, sizeof(Eigen::Vector3f) * h_count, cudaMemcpyDeviceToHost, st));
				checkCudaErrors(cudaMemcpyAsync(host_nms, view_nm_repos, sizeof(Eigen::Vector3f) * h_count, cudaMemcpyDeviceToHost, st));
				checkCudaErrors(cudaMemcpyAsync(host_clrs, view_color_repos, sizeof(Eigen::Vector3f) * h_count, cudaMemcpyDeviceToHost, st));

				static	int cntFile = 0;
				char szTemp[128];
				sprintf(szTemp, "%04d", cntFile++);
				qDebug("[%d] 조건을 만족하는 요소의 개수: %d", cntFile, h_count);
				std::string filename = GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_Avg_test" + ".ply";
				pRegistration->plyFileWrite_Ex(
					filename,
					h_count,
					host_pts,
					host_nms,
					host_clrs
				);

				delete[] host_pts;
				delete[] host_nms;
				delete[] host_clrs;
			}
#endif

		}
		//cudaFreeAsync(d_count, st);

		nvtxRangePop();

		checkCudaSync(st);

		return h_count;
	}
	*/

uint32_t MarchingCubes::AverageVoxelPoints_v4(
	cached_allocator * alloc_, CUstream_st * st
	// input 
	, ExtractionPointCloud & point_cloud
	// output
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, float filteringForPt
	, float filteringForNm
	, int ref_margin_normal
	, int ref_margin_point
)
{
	NvtxRangeCuda nvtxPrint("@AverageVoxelPoints_v4");

	auto used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_pos = thrust::raw_pointer_cast(point_cloud.points_.data());
	auto _repos_nm = thrust::raw_pointer_cast(point_cloud.normals_.data());
	auto _repos_color = thrust::raw_pointer_cast(point_cloud.colors_.data());

	//	shshin	@20240508
	auto _nm_Tmp = thrust::raw_pointer_cast(point_cloud.nm_avgTmp.data());

	uint32_t* d_count; // 디바이스에서 카운트를 저장할 포인터
	uint32_t h_count = 0; // 호스트 카운트 값

#ifndef BUILD_FOR_CPU
		//	mscho	@20240530
		//	cudaMallocAsync => cudaMalloc
		//cudaMallocAsync(&d_count, sizeof(uint32_t), st);
	cudaMalloc(&d_count, sizeof(uint32_t));
	cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st);
#else
	d_count = new uint32_t;
	*d_count = 0;
#endif

	unsigned int host_Count_HashTableUsed = 0;
#ifndef BUILD_FOR_CPU
	cudaMemcpyAsync(&host_Count_HashTableUsed, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);
#else
	host_Count_HashTableUsed = *used_cnt_localContains;
#endif

	{
		int mingridsize;
		int threadblocksize;

		//qDebug("=================================host_Count_vtxCache : %d", host_Count_HashTableUsed);
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_AvgNormal_v1, 0, 0));
		auto gridsize = ((uint32_t)host_Count_HashTableUsed + threadblocksize - 1) / threadblocksize;

		Kernel_AvgNormal_v1 << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_AvgNormal_v1(
#endif
			exeInfo,
			used_Key,

			host_Count_HashTableUsed,

			// input
			_repos_pos,
			_repos_nm,
			_repos_color,

			// output
			_nm_Tmp,

			ref_margin_normal,
			ref_margin_point,
			filteringForPt,
			filteringForNm
		);
		checkCudaErrors(cudaGetLastError());

#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_AvgPoints_v4, 0, 0));
		gridsize = ((uint32_t)host_Count_HashTableUsed + threadblocksize - 1) / threadblocksize;

		//	mscho	@20240507
		Kernel_AvgPoints_v4 << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_AvgPoints_v4(
#endif
			exeInfo,
			used_Key,

			host_Count_HashTableUsed,

			_repos_pos,
			_nm_Tmp,
			_repos_color,

			d_count,

			view_pos_repos,
			view_nm_repos,
			view_color_repos,
			ref_margin_normal,
			ref_margin_point,
			filteringForPt,
			filteringForNm
		);
		checkCudaErrors(cudaGetLastError());

#ifndef BUILD_FOR_CPU
		checkCudaSync(st);
		cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost, st);
#else
		h_count = *d_count;
#endif

#ifndef BUILD_FOR_CPU
		if (false && h_count > 0)
		{
			// 결과 사용

			Eigen::Vector3f* host_pts = new Eigen::Vector3f[h_count];
			Eigen::Vector3f* host_nms = new Eigen::Vector3f[h_count];
			Eigen::Vector3b* host_clrs = new Eigen::Vector3b[h_count];

			checkCudaErrors(cudaMemcpyAsync(host_pts, view_pos_repos, sizeof(Eigen::Vector3f) * h_count, cudaMemcpyDeviceToHost, st));
			checkCudaErrors(cudaMemcpyAsync(host_nms, view_nm_repos, sizeof(Eigen::Vector3f) * h_count, cudaMemcpyDeviceToHost, st));
			checkCudaErrors(cudaMemcpyAsync(host_clrs, view_color_repos, sizeof(Eigen::Vector3b) * h_count, cudaMemcpyDeviceToHost, st));

			static	int cntFile = 0;
			char szTemp[128];
			sprintf(szTemp, "%04d", cntFile++);
			qDebug("[%d] 조건을 만족하는 요소의 개수: %d", cntFile, h_count);
			std::string filename = pSettings->GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_Avg_test" + ".ply";
			plyFileWrite_Ex(
				filename,
				h_count,
				host_pts,
				host_nms,
				host_clrs
			);

			delete[] host_pts;
			delete[] host_nms;
			delete[] host_clrs;

		}
#endif
	}
#ifndef BUILD_FOR_CPU
	cudaFreeAsync(d_count, st);


	checkCudaSync(st);
#else
	delete d_count;
#endif
	return h_count;
}

//	mscho	@20250207
//	pointMag 를 인자로 받아서, average하는데, 주변점의 Minimum  갯수를 조정하도록 한다.
uint32_t MarchingCubes::AverageVoxelPoints_v5(
	cached_allocator * alloc_, CUstream_st * st
	, ExtractionPointCloud & point_cloud
	, float pointMag
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extraAttrib_repos
	, float filteringForPt
	, float filteringForNm
	, int ref_margin_normal
	, int ref_margin_point
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, int marginSize, float voxelSize
	, VoxelExtractMode voxelExtractMode
)
{
	NvtxRangeCuda nvtxPrint("@AverageVoxelPoints_v5");

	const bool isHDMode = voxelExtractMode == VoxelExtractMode::CompleteHD;

	auto used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_pos = thrust::raw_pointer_cast(point_cloud.points_.data());
	auto _repos_nm = thrust::raw_pointer_cast(point_cloud.normals_.data());
	auto _repos_color = thrust::raw_pointer_cast(point_cloud.colors_.data());
	auto _repos_extra_attrib = thrust::raw_pointer_cast(point_cloud.extraAttribs_.data());

	//	shshin	@20240508
	auto _nm_Tmp = thrust::raw_pointer_cast(point_cloud.nm_avgTmp.data());

	//uint32_t* d_count; // 디바이스에서 카운트를 저장할 포인터
	//	mscho	@20250207
	//uint32_t h_count = 0; // 호스트 카운트 값	

	Eigen::Vector3f min = minBound;
	Eigen::Vector3f max = maxBound;

	min.x() -= (voxelSize * (float)marginSize);
	min.y() -= (voxelSize * (float)marginSize);
	min.z() -= (voxelSize * (float)marginSize);

	max.x() += (voxelSize * (float)(marginSize));
	max.y() += (voxelSize * (float)(marginSize));
	max.z() += (voxelSize * (float)(marginSize));

	//	mscho	@20240530	==>@20250207
	//	cudaMallocAsync => cudaMalloc
	//cudaMallocAsync(&d_count, sizeof(uint32_t), st);
	//checkCudaErrors(cudaMalloc(&d_count, sizeof(uint32_t)));
	//checkCudaErrors(cudaMemsetAsync(d_count, 0, sizeof(uint32_t), st));

	//checkCudaErrors(cudaMemsetAsync(cnt_averageRes, 0, sizeof(uint32_t), st));

	//	mscho	
	// cudaMallocHost로 메모리를 잡아서 사용하도록 변경한다.
	//unsigned int host_Count_HashTableUsed = 0;
	checkCudaErrors(cudaMemcpyAsync(host_Count_HashTableUsed, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st));
	checkCudaSync(st);
	//	mscho	@20250207
	if (*host_Count_HashTableUsed <= 0) return 0;

	{
		//qDebug("Normal =================================host_Count_vtxCache : %d", host_Count_HashTableUsed);

		//	mscho	@20240521 (New)
		auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());

		//	mscho	@20240523 ==> @20240527: 기존에는 HD모드에 별도의 Averaging 공식을 적용했었으나, SD 공식을 개선하였으므로 HD도 동일한 공식을 적용하기로 함
		//if (isHDMode)
		//{
		//	qDebug("=================================host_Count_vtxCache : %d", host_Count_HashTableUsed);
		//	Kernel_AvgEstimateNormal_v5_HD << <BLOCKS_PER_GRID_THREAD_N((host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
		//		exeInfo,
		//		used_Key,
		//		//_neighbor_id_dev,
		//		host_Count_HashTableUsed,
		//		_voxelNormals,
		//		_repos_pos,
		//		_repos_nm,
		//		_repos_color,
		//		//_repos_material_id,
		//		_nm_Tmp,
		//		ref_margin_normal,
		//		ref_margin_point,
		//		filteringForPt,
		//		filteringForNm
		//		);
		//}
		//else
		{
			//	mscho	@20250110
			//  Normal의 배분율을 조금 수정한다.
			// average하는 목적에 따라 적절한 normal weight 값을 정해 준다.
			float normal_weight = 0.8f; // surface_normal(표면의  과 voxel_normal(여러 포인트 노말의 가중합)을 합성할 때, surface_normal의 가중치 (승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.)
			//if (bool applyingDifferentNormalAveragingTest = false) {
			//	switch (voxelExtractMode) {
			//	case VoxelExtractMode::ICPTarget:	normal_weight = 0.6f; break;
			//	case VoxelExtractMode::Complete:	normal_weight = 0.95f; break;
			//	case VoxelExtractMode::CompleteHD:	normal_weight = 0.95f; break;
			//	}
			//}
			//if (bool applyingDifferentNormalAveragingTest = true) 
			//{
			//	switch (voxelExtractMode) 
			//	{
			//		case VoxelExtractMode::ICPTarget:	normal_weight = 0.35f; break;
			//		case VoxelExtractMode::Complete:	normal_weight = 0.90f; break;
			//		case VoxelExtractMode::CompleteHD:	normal_weight = 0.90f; break;
			//	}
			//}

			//	mscho	@20250207
			if (bool applyingDifferentNormalAveragingTest = true)
			{
				switch (voxelExtractMode)
				{
				case VoxelExtractMode::ICPTarget:	normal_weight = 0.40f; break;
				case VoxelExtractMode::Complete:	normal_weight = 0.90f; break;
				case VoxelExtractMode::CompleteHD:	normal_weight = 0.90f; break;
				}
			}
			//	mscho	@20240717	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgEstimateNormal_v5 << <BLOCKS_PER_GRID_THREAD_N((*host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgEstimateNormal_v5(
#endif
				exeInfo,
				used_Key,
				//_neighbor_id_dev,
				*host_Count_HashTableUsed,//	mscho	@20250207

				// 입력
				_voxelNormals,
				_repos_pos,
				_repos_nm,
				_repos_color,

				_nm_Tmp, // averaged normal 출력

				ref_margin_normal,
				ref_margin_point,
				filteringForPt,
				filteringForNm,
				normal_weight
			);
		}
		checkCudaErrors(cudaGetLastError());

		if (voxelExtractMode == VoxelExtractMode::CompleteHD)
		{

#ifndef BUILD_FOR_CPU
			//	mscho	@20250207
			Kernel_AvgNormal_v1_HD << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1_HD(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
				isHDMode ? ref_margin_point + 1 : ref_margin_point,
				filteringForPt,
				filteringForNm //	mscho	@20240523	==> @20240611
			);
		}
		else
		{
			//	mscho	@20240524	==> @20240527	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgNormal_v1 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				ref_margin_normal - 1,	//	mscho	@20240620
				ref_margin_point,
				filteringForPt,
				filteringForNm//	mscho	@20240523	==> @20240611
			);
		}
		checkCudaErrors(cudaGetLastError());
		//	mscho	@20250207
		checkCudaErrors(cudaMemsetAsync(cnt_averageRes, 0, sizeof(uint32_t), st));

		uint32_t	FilterCntTh = isHDMode ? 20 : 5;
		// complete
		if (bool applyingDifferentNormalAveragingTest = true)
		{
			switch (voxelExtractMode)
			{
			case VoxelExtractMode::ICPTarget:	break;
			case VoxelExtractMode::Complete:	FilterCntTh *= pointMag; break;
			case VoxelExtractMode::CompleteHD:	FilterCntTh *= pointMag; break;
			}
		}

		//	mscho	@20240524	==> @20240527	==> @20250207
		// shshin @20240701
#ifndef BUILD_FOR_CPU
		Kernel_AvgPoints_v8 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
		Kernel_AvgPoints_v8(
#endif
			exeInfo,
			used_Key,

			*host_Count_HashTableUsed,//	mscho	@20250207

			_repos_pos,
			_repos_nm,
			_repos_color,
			_repos_extra_attrib,

			//d_count,
			cnt_averageRes,

			view_pos_repos,
			view_nm_repos,
			view_color_repos,
			view_extraAttrib_repos,
			isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
			isHDMode ? ref_margin_point + 1 : ref_margin_point,
			filteringForPt,
			filteringForNm,	//	mscho	@20240523	==> @20240611
			//4//6
			FilterCntTh,	//	mscho	@20240611 ==> @20240620 ==> @20240625( 6 ==> 4 ) ==> @20250207
			min,
			max,
			isHDMode);
		checkCudaErrors(cudaGetLastError());
		//	mscho	@20240507
		//Kernel_AvgPoints_v4 << <gridsize, threadblocksize, 0, st >> > (
		//	exeInfo,
		//	used_Key,

		//	host_Count_HashTableUsed,

		//	_repos_pos,
		//	_repos_nm,
		//	_repos_color,

		//	d_count,

		//	view_pos_repos,
		//	view_nm_repos,
		//	view_color_repos,
		//	ref_margin_normal,
		//	ref_margin_point,
		//	filteringForPt,
		//	filteringForNm - 1	//	mscho	@20240523
		//	);
		// 

		//	mscho	@20250207
		bool	bAveDebug = false;
		//checkCudaSync(st);
		checkCudaErrors(cudaMemcpyAsync(host_Count_AvePoints, cnt_averageRes, sizeof(uint32_t), cudaMemcpyDeviceToHost, st));

#ifndef BUILD_FOR_CPU
		//	mscho	@20250207
		if (bAveDebug)
		{
			// 결과 사용
			checkCudaSync(st);
			if (*host_Count_AvePoints > 0)
			{
				Eigen::Vector3f* host_pts = new Eigen::Vector3f[*host_Count_AvePoints];
				Eigen::Vector3f* host_nms = new Eigen::Vector3f[*host_Count_AvePoints];
				Eigen::Vector3f* host_clrs = new Eigen::Vector3f[*host_Count_AvePoints];

				checkCudaErrors(cudaMemcpyAsync(host_pts, view_pos_repos, sizeof(Eigen::Vector3f) * (*host_Count_AvePoints), cudaMemcpyDeviceToHost, st));
				checkCudaErrors(cudaMemcpyAsync(host_nms, view_nm_repos, sizeof(Eigen::Vector3f) * (*host_Count_AvePoints), cudaMemcpyDeviceToHost, st));
				checkCudaErrors(cudaMemcpyAsync(host_clrs, view_color_repos, sizeof(Eigen::Vector3f) * (*host_Count_AvePoints), cudaMemcpyDeviceToHost, st));

				static	int cntFile = 0;
				char szTemp[128];
				sprintf(szTemp, "%04d", cntFile++);
				qDebug("[%d] 조건을 만족하는 요소의 개수: %d", cntFile, *host_Count_AvePoints);
				std::string filename = pSettings->GetSaveDataFolderPath() + "\\" + std::string(szTemp) + "_Avg_test" + ".ply";
				plyFileWrite_Ex(
					filename,
					*host_Count_AvePoints,
					host_pts,
					host_nms,
					host_clrs
				);

				delete[] host_pts;
				delete[] host_nms;
				delete[] host_clrs;
			}
		}
#endif
	}
	//cudaFreeAsync(d_count, st);

	//	mscho	@20250207
	//cudaFreeAsync(d_count, st);
	checkCudaSync(st);

	return *host_Count_AvePoints;
}

uint32_t MarchingCubes::AverageVoxelPoints_v6(
	cached_allocator * alloc_, CUstream_st * st
	, ExtractionPointCloud & point_cloud
	, float pointMag
	, std::shared_ptr<pointcloud_Hios> outPC
	, float filteringForPt
	, float filteringForNm
	, int ref_margin_normal
	, int ref_margin_point
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, int marginSize, float voxelSize
	, VoxelExtractMode voxelExtractMode
)
{
	//	mscho	#30250306
	NvtxRangeCuda nvtxPrint("@AverageVoxelPoints_v6");
	const bool isHDMode = voxelExtractMode == VoxelExtractMode::CompleteHD;

	auto used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_pos = thrust::raw_pointer_cast(point_cloud.points_.data());
	auto _repos_nm = thrust::raw_pointer_cast(point_cloud.normals_.data());
	auto _repos_color = thrust::raw_pointer_cast(point_cloud.colors_.data());
	VoxelExtraAttrib* _repos_extraAttrib = thrust::raw_pointer_cast(point_cloud.extraAttribs_.data());

	//	shshin	@20240508
	auto _nm_Tmp = thrust::raw_pointer_cast(point_cloud.nm_avgTmp.data());

	Eigen::Vector3f min = minBound;
	Eigen::Vector3f max = maxBound;

	min.x() -= (voxelSize * (float)marginSize);
	min.y() -= (voxelSize * (float)marginSize);
	min.z() -= (voxelSize * (float)marginSize);

	max.x() += (voxelSize * (float)(marginSize));
	max.y() += (voxelSize * (float)(marginSize));
	max.z() += (voxelSize * (float)(marginSize));

	//	mscho	
	// cudaMallocHost로 메모리를 잡아서 사용하도록 변경한다.
	//unsigned int host_Count_HashTableUsed = 0;
	checkCudaErrors(cudaMemcpyAsync(host_Count_HashTableUsed, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st));
	checkCudaSync(st);
	//	mscho	@20250207
	if (*host_Count_HashTableUsed <= 0) return 0;

	{
		//	mscho	@20240521 (New)
		auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
		{
			//	mscho	@20250110
			//  Normal의 배분율을 조금 수정한다.
			// average하는 목적에 따라 적절한 normal weight 값을 정해 준다.
			float normal_weight = 0.8f; // surface_normal(표면의  과 voxel_normal(여러 포인트 노말의 가중합)을 합성할 때, surface_normal의 가중치 (승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.)

			//	mscho	@20250207	==> @20250228	==>	@20250313
			//	실시간 scan에서는 스캔중에 보이는 화면의 품질을 위해서, center voxel 가중치를 더 높인다.
			if (bool applyingDifferentNormalAveragingTest = true)
			{
				switch (voxelExtractMode)
				{
				case VoxelExtractMode::ICPTarget:	normal_weight = 0.10f; break;
					//case VoxelExtractMode::ICPTarget:	normal_weight = 0.80f; break;
				case VoxelExtractMode::Final:		normal_weight = 0.25f; break;
				case VoxelExtractMode::Complete:	normal_weight = 0.50f; break;
				case VoxelExtractMode::CompleteHD:	normal_weight = 0.50f; break;
				}
			}
			//	mscho	@20240717	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgEstimateNormal_v5 << <BLOCKS_PER_GRID_THREAD_N((*host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgEstimateNormal_v5(
#endif
				exeInfo,
				used_Key,
				//_neighbor_id_dev,
				*host_Count_HashTableUsed,//	mscho	@20250207

				// 입력
				_voxelNormals,
				_repos_pos,
				_repos_nm,
				_repos_color,

				_nm_Tmp, // averaged normal 출력

				ref_margin_normal,
				ref_margin_point,
				filteringForPt,
				filteringForNm,
				normal_weight
			);
		}
		checkCudaErrors(cudaGetLastError());

		if (voxelExtractMode == VoxelExtractMode::CompleteHD)
		{

#ifndef BUILD_FOR_CPU
			//	mscho	@20250207
			Kernel_AvgNormal_v1_HD << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1_HD(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
				isHDMode ? ref_margin_point + 1 : ref_margin_point,
				filteringForPt,
				filteringForNm //	mscho	@20240523	==> @20240611
			);
		}
		else
		{
			//	mscho	@20250313;
			//int	ref_margin_nor = (voxelExtractMode == VoxelExtractMode::ICPTarget ? ref_margin_normal + 1 : ref_margin_normal);
			int	ref_margin_normal_cnt = ref_margin_normal;
			ref_margin_normal_cnt = (ref_margin_normal_cnt > 2 ? 2 : ref_margin_normal_cnt);

			//	mscho	@20240524	==> @20240527	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgNormal_v1 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				ref_margin_normal_cnt,	//	mscho	@20240620	==> @20250228 (Normal 평균을 조금 더 내도록 한다... 실제로는 final이 중요하다.)	==> @20250313
				ref_margin_point,
				filteringForPt,
				filteringForNm//	mscho	@20240523	==> @20240611
			);
		}
		checkCudaErrors(cudaGetLastError());

		//	mscho	@20250207
		checkCudaErrors(cudaMemsetAsync(cnt_averageRes, 0, sizeof(uint32_t), st));

		//	mscho	@20250228 => @20250620	==> @20250627
		//uint32_t	FilterCntTh = isHDMode ? 20 : 4-1;
		uint32_t	FilterCntTh = 4;// 4;

		// complete
		//	mscho	@20250620
		if (bool applyingDifferentNormalAveragingTest = true)
		{
			switch (voxelExtractMode)
			{
			case VoxelExtractMode::ICPTarget:	break;
			case VoxelExtractMode::Complete:	FilterCntTh = ceil(FilterCntTh * pointMag); break;
			case VoxelExtractMode::CompleteHD:	FilterCntTh = ceil(FilterCntTh * pointMag * pointMag); break;
			}
		}
		//	mscho	@20250620
		//	Metal mode 인지 아닌지 판단해서...
		//  추가적인 기능을 구현한다.
		const bool isMetalMode = (pSettings->GetMetalMode() ? true : false);

		//	mscho	@20240524	==> @20240527	==> @20250207	==> @20250620
		// shshin @20240701

		//	mscho	@20250624	==> @20250627
		bool	b_isPattern12 = pSettings->IsPattern12();
#ifndef BUILD_FOR_CPU
		Kernel_AvgPoints_v10 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
		Kernel_AvgPoints_v8(
#endif
			exeInfo,
			used_Key,

			*host_Count_HashTableUsed,//	mscho	@20250207

			_repos_pos,
			_repos_nm,
			_repos_color,
			_repos_extraAttrib,

			//d_count,
			cnt_averageRes,

			thrust::raw_pointer_cast(outPC->points_.data()),//view_pos_repos,
			thrust::raw_pointer_cast(outPC->normals_.data()),//view_nm_repos,
			thrust::raw_pointer_cast(outPC->colors_.data()),//view_color_repos,
			thrust::raw_pointer_cast(outPC->extraAttribs_.data()),
			isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
			isHDMode ? ref_margin_point + 1 : ref_margin_point,
			filteringForPt,
			filteringForNm,	//	mscho	@20240523	==> @20240611
			//4//6
			FilterCntTh,	//	mscho	@20240611 ==> @20240620 ==> @20240625( 6 ==> 4 ) ==> @20250207
			min,
			max,
			isHDMode,
			isMetalMode,	// mscho	@20250620
			b_isPattern12,	//	mscho	@20250624
			voxelExtractMode
		);
		checkCudaErrors(cudaGetLastError());

		//	mscho	@20250207
		bool	bAveDebug = false;
		//checkCudaSync(st);
		checkCudaErrors(cudaMemcpyAsync(host_Count_AvePoints, cnt_averageRes, sizeof(uint32_t), cudaMemcpyDeviceToHost, st));

	}

	checkCudaSync(st);

	outPC->m_nowSize = *host_Count_AvePoints;

	return *host_Count_AvePoints;
}

uint32_t MarchingCubes::AverageVoxelPoints_v6(
	cached_allocator * alloc_, CUstream_st * st
	, ExtractionPointCloud & point_cloud // input 

	, float pointMag
	, Eigen::Vector3f * view_pos_repos
	, Eigen::Vector3f * view_nm_repos
	, Eigen::Vector3b * view_color_repos
	, VoxelExtraAttrib * view_extraAttrib_repos
	, float filteringForPt
	, float filteringForNm
	, int ref_margin_normal
	, int ref_margin_point
	, Eigen::Vector3f minBound, Eigen::Vector3f maxBound
	, int marginSize, float voxelSize
	, VoxelExtractMode voxelExtractMode
)
{
	//	mscho	#30250306
	NvtxRangeCuda nvtxPrint("@AverageVoxelPoints_v6_ex");
	const bool isHDMode = voxelExtractMode == VoxelExtractMode::CompleteHD;

	auto used_Key = thrust::raw_pointer_cast(m_MC_used_buff_pts.data());

	auto _repos_pos = thrust::raw_pointer_cast(point_cloud.points_.data());
	auto _repos_nm = thrust::raw_pointer_cast(point_cloud.normals_.data());
	auto _repos_color = thrust::raw_pointer_cast(point_cloud.colors_.data());
	VoxelExtraAttrib* _repos_extraAttrib = thrust::raw_pointer_cast(point_cloud.extraAttribs_.data());

	//	shshin	@20240508
	auto _nm_Tmp = thrust::raw_pointer_cast(point_cloud.nm_avgTmp.data());

	Eigen::Vector3f min = minBound;
	Eigen::Vector3f max = maxBound;

	min.x() -= (voxelSize * (float)marginSize);
	min.y() -= (voxelSize * (float)marginSize);
	min.z() -= (voxelSize * (float)marginSize);

	max.x() += (voxelSize * (float)(marginSize));
	max.y() += (voxelSize * (float)(marginSize));
	max.z() += (voxelSize * (float)(marginSize));

	//	mscho	
	// cudaMallocHost로 메모리를 잡아서 사용하도록 변경한다.
	//unsigned int host_Count_HashTableUsed = 0;
	checkCudaErrors(cudaMemcpyAsync(host_Count_HashTableUsed, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st));
	checkCudaSync(st);
	//	mscho	@20250207
	if (*host_Count_HashTableUsed <= 0) return 0;

	{

		//	mscho	@20240521 (New)
		auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());


		{
			//	mscho	@20250110
			//  Normal의 배분율을 조금 수정한다.
			// average하는 목적에 따라 적절한 normal weight 값을 정해 준다.
			float normal_weight = 0.8f; // surface_normal(표면의  과 voxel_normal(여러 포인트 노말의 가중합)을 합성할 때, surface_normal의 가중치 (승수를 높일수록, 부드러워진다. 디테일이 낮아진다....승수가 낮아지면, 그반대로 움직인다.)

			//	mscho	@20250207	==> @20250228	==>	@20250313
			//	실시간 scan에서는 스캔중에 보이는 화면의 품질을 위해서, center voxel 가중치를 더 높인다.
			if (bool applyingDifferentNormalAveragingTest = true)
			{
				switch (voxelExtractMode)
				{
				case VoxelExtractMode::ICPTarget:	normal_weight = 0.10f; break;
					//case VoxelExtractMode::ICPTarget:	normal_weight = 0.40f; break;
				case VoxelExtractMode::Final:		normal_weight = 0.25f; break;
				case VoxelExtractMode::Complete:	normal_weight = 0.50f; break;
				case VoxelExtractMode::CompleteHD:	normal_weight = 0.50f; break;
				}
			}
			//	mscho	@20240717	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgEstimateNormal_v5 << <BLOCKS_PER_GRID_THREAD_N((*host_Count_HashTableUsed), THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgEstimateNormal_v5(
#endif
				exeInfo,
				used_Key,
				//_neighbor_id_dev,
				*host_Count_HashTableUsed,//	mscho	@20250207

				// 입력
				_voxelNormals,
				_repos_pos,
				_repos_nm,
				_repos_color,

				_nm_Tmp, // averaged normal 출력

				ref_margin_normal,
				ref_margin_point,
				filteringForPt,
				filteringForNm,
				normal_weight
			);
		}
		checkCudaErrors(cudaGetLastError());

		if (voxelExtractMode == VoxelExtractMode::CompleteHD)
		{

#ifndef BUILD_FOR_CPU
			//	mscho	@20250207
			Kernel_AvgNormal_v1_HD << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1_HD(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
				isHDMode ? ref_margin_point + 1 : ref_margin_point,
				filteringForPt,
				filteringForNm //	mscho	@20240523	==> @20240611
			);
		}
		else
		{
			//	mscho	@20250313;
			//int	ref_margin_nor = (voxelExtractMode == VoxelExtractMode::ICPTarget ? ref_margin_normal + 1 : ref_margin_normal);
			int	ref_margin_normal_cnt = ref_margin_normal;
			ref_margin_normal_cnt = (ref_margin_normal_cnt > 2 ? 2 : ref_margin_normal_cnt);

			//	mscho	@20240524	==> @20240527	==> @20250207
#ifndef BUILD_FOR_CPU
			Kernel_AvgNormal_v1 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
			Kernel_AvgNormal_v1(
#endif
				exeInfo,
				used_Key,

				*host_Count_HashTableUsed,//	mscho	@20250207

				_repos_pos,
				_nm_Tmp,
				_repos_color,

				_repos_nm,

				ref_margin_normal_cnt,	//	mscho	@20240620	==> @20250228 (Normal 평균을 조금 더 내도록 한다... 실제로는 final이 중요하다.)	==> @20250313
				ref_margin_point,
				filteringForPt,
				filteringForNm//	mscho	@20240523	==> @20240611
			);
		}
		checkCudaErrors(cudaGetLastError());
		//	mscho	@20250207
		checkCudaErrors(cudaMemsetAsync(cnt_averageRes, 0, sizeof(uint32_t), st));

		//	mscho	@20250228
		//uint32_t	FilterCntTh = isHDMode ? 20 : 4-1;
		uint32_t	FilterCntTh = 4;

		// complete
		if (bool applyingDifferentNormalAveragingTest = true)
		{
			switch (voxelExtractMode)
			{
			case VoxelExtractMode::ICPTarget:	break;
			case VoxelExtractMode::Complete:	FilterCntTh *= pointMag; break;
			case VoxelExtractMode::CompleteHD:	FilterCntTh *= pointMag; break;
			}
		}

		//	mscho	@20240524	==> @20240527	==> @20250207
		// shshin @20240701
#ifndef BUILD_FOR_CPU
		Kernel_AvgPoints_v8 << <BLOCKS_PER_GRID_THREAD_N(*host_Count_HashTableUsed, THRED512_PER_BLOCK), THRED512_PER_BLOCK, 0, st >> > (
#else
		Kernel_AvgPoints_v8(
#endif
			exeInfo,
			used_Key,

			*host_Count_HashTableUsed,//	mscho	@20250207

			_repos_pos,
			_repos_nm,
			_repos_color,
			_repos_extraAttrib,

			//d_count,
			cnt_averageRes,

			view_pos_repos,
			view_nm_repos,
			view_color_repos,
			view_extraAttrib_repos,
			isHDMode ? ref_margin_normal + 1 : ref_margin_normal,
			isHDMode ? ref_margin_point + 1 : ref_margin_point,
			filteringForPt,
			filteringForNm,	//	mscho	@20240523	==> @20240611
			//4//6
			FilterCntTh,	//	mscho	@20240611 ==> @20240620 ==> @20240625( 6 ==> 4 ) ==> @20250207
			min,
			max,
			isHDMode);
		checkCudaErrors(cudaGetLastError());

		//	mscho	@20250207
		bool	bAveDebug = false;
		//checkCudaSync(st);
		checkCudaErrors(cudaMemcpyAsync(host_Count_AvePoints, cnt_averageRes, sizeof(uint32_t), cudaMemcpyDeviceToHost, st));

	}

	checkCudaSync(st);

	return *host_Count_AvePoints;
}

#ifndef BUILD_FOR_CPU
__global__ void Kernel_GeneratePointNormalsUsingPCA_Debug(
	MarchingCubes::ExecutionInfo voxelInfo,
	uint64_t * slotIndices,
	uint32_t cnt_contains,
	HashKey64 * hashinfo_vtx

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, uint32_t numberOfNeighborToSearch
	, bool useOutlierRemoval
	, Eigen::Vector3f * debugPoints
	, Eigen::Vector3f * debugNormals
	, Eigen::Vector3b * debugColors
)
{
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	//qDebug("test2 \t");

	auto key = slotIndices[slotIndex];

	auto xGlobalIndex = (key >> 32) & 0xffff;
	auto yGlobalIndex = (key >> 16) & 0xffff;
	auto zGlobalIndex = (key) & 0xffff;

	auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
	auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
	auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

	auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
		yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

	//printf("cache %d, %d\n", cacheIndex, voxelInfo.cache.voxelCount - 1);

	if (cacheIndex >= voxelInfo.cache.voxelCount) return;
	auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
	if (pointSlotIndex == kEmpty32) return;

	Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
	Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();
	Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

	uint32_t currentIndex = atomicAdd(count, 1);

	if (nullptr != debugPoints)
	{
		debugPoints[currentIndex] = basePos;
	}

	if (nullptr != debugNormals)
	{
		debugNormals[currentIndex] = baseNormal;
	}

	if (nullptr != debugColors)
	{
		debugColors[currentIndex] = baseColor;
	}
}
#endif

__global__ void Kernel_GeneratePointNormalsUsingPCA_Comulant(
	MarchingCubes::ExecutionInfo voxelInfo,
	HashKey * slotIndices,
	uint32_t cnt_contains
#ifdef USE_MESH_BASE
	, HashKey64 * hashinfo_vtx
#endif

	, Eigen::Vector3f * point_pos_buff
	, Eigen::Vector3f * point_nm_buff
	, Eigen::Vector3b * point_clr_buff

	, uint32_t * count
	, Eigen::Vector3f * view_pos_repos, Eigen::Vector3f * view_nm_repos, Eigen::Vector3b * view_color_repos
	, uint32_t numberOfNeighborToSearch
	, bool useOutlierRemoval
	, Eigen::Vector3f * debugPoints
	, Eigen::Vector3f * debugNormals
	, Eigen::Vector3b * debugColors
)
{
#ifndef BUILD_FOR_CPU
	unsigned int slotIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (slotIndex > cnt_contains - 1) return;
	//qDebug("test2 \t");
	{
#else
#pragma omp parallel for
	for (int threadid = 0; threadid < cnt_contains; threadid++) {
		unsigned int slotIndex = threadid;
#endif

		auto key = slotIndices[slotIndex];

		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto xCacheIndex = xGlobalIndex - voxelInfo.cache.localMinGlobalIndexX;
		auto yCacheIndex = yGlobalIndex - voxelInfo.cache.localMinGlobalIndexY;
		auto zCacheIndex = zGlobalIndex - voxelInfo.cache.localMinGlobalIndexZ;

		auto cacheIndex = zCacheIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
			yCacheIndex * voxelInfo.cache.voxelCountX + xCacheIndex;

		//printf("cache %d, %d\n", cacheIndex, voxelInfo.cache.voxelCount - 1);

		if (cacheIndex >= voxelInfo.cache.voxelCount)
			kernel_return;
		auto pointSlotIndex = voxelInfo.gridSlotIndexCache_pts[cacheIndex];
		if (pointSlotIndex == kEmpty32)
			kernel_return;

		Eigen::Vector3f basePos = (point_pos_buff[pointSlotIndex]);
		Eigen::Vector3f baseNormal = (point_nm_buff[pointSlotIndex]).normalized();

		//baseNormal = voxelInfo.voxelNormals[pointSlotIndex].normalized();
		{
			HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
			uint32_t hashSlot_idx = voxelInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
			if (hashSlot_idx == kEmpty32)
			{
				baseNormal = voxelInfo.voxelNormals[hashSlot_idx].normalized();
			}
		}

		Eigen::Vector3b baseColor = point_clr_buff[pointSlotIndex];

		uint32_t currentIndex = atomicAdd(count, (uint32_t)1);

		Eigen::Matrix<float, 9, 1> cumulants;
		cumulants.setZero();
		auto failed = false;
		{ // Make Culumant
			uint32_t found = 0;
			int offset = 0;
			uint32_t fallbackCount = 100;
			uint32_t loopCount = 0;
			while (found < numberOfNeighborToSearch)
			{
				for (int zOffset = -offset; zOffset <= offset; zOffset++)
				{
					auto zIndex = (int)zCacheIndex + zOffset;
					if (0 > zIndex) continue;
					if (zIndex >= voxelInfo.cache.voxelCountZ) continue;

					for (int yOffset = -offset; yOffset <= offset; yOffset++)
					{
						auto yIndex = (int)yCacheIndex + yOffset;
						if (0 > yIndex) continue;
						if (yIndex >= voxelInfo.cache.voxelCountY) continue;

						for (int xOffset = -offset; xOffset <= offset; xOffset++)
						{
							auto xIndex = (int)xCacheIndex + xOffset;
							if (0 > xIndex) continue;
							if (xIndex >= voxelInfo.cache.voxelCountX) continue;

							if ((xOffset == -offset || xOffset == offset) ||
								(yOffset == -offset || yOffset == offset) ||
								(zOffset == -offset || zOffset == offset))
							{
								auto neighborCacheIndex = zIndex * voxelInfo.cache.voxelCountX * voxelInfo.cache.voxelCountY +
									yIndex * voxelInfo.cache.voxelCountX + xIndex;
								auto neighborSlotIndex = voxelInfo.gridSlotIndexCache_pts[neighborCacheIndex];
								if (neighborSlotIndex == kEmpty32) continue;
								Eigen::Vector3f neighborPos = (point_pos_buff[neighborSlotIndex]);

								cumulants(0) += neighborPos(0);
								cumulants(1) += neighborPos(1);
								cumulants(2) += neighborPos(2);
								cumulants(3) += neighborPos(0) * neighborPos(0);
								cumulants(4) += neighborPos(0) * neighborPos(1);
								cumulants(5) += neighborPos(0) * neighborPos(2);
								cumulants(6) += neighborPos(1) * neighborPos(1);
								cumulants(7) += neighborPos(1) * neighborPos(2);
								cumulants(8) += neighborPos(2) * neighborPos(2);
								found++;
							}

							loopCount++;
							if (loopCount == fallbackCount)
							{
								found = numberOfNeighborToSearch;
								failed = true;
							}

							if (found >= numberOfNeighborToSearch) break;
						}
						if (found >= numberOfNeighborToSearch) break;
					}
					if (found >= numberOfNeighborToSearch) break;
				}
				if (found >= numberOfNeighborToSearch) break;

				offset++;
			}
			cumulants /= (float)found;
		}

		Eigen::Matrix3f covarianceMatrix;
		covarianceMatrix(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
		covarianceMatrix(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
		covarianceMatrix(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
		covarianceMatrix(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
		covarianceMatrix(1, 0) = covarianceMatrix(0, 1);
		covarianceMatrix(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
		covarianceMatrix(2, 0) = covarianceMatrix(0, 2);
		covarianceMatrix(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
		covarianceMatrix(2, 1) = covarianceMatrix(1, 2);

		auto eig_val_vec = FastEigen3x3(covarianceMatrix);
		int min_id;
		thrust::get<0>(eig_val_vec).minCoeff(&min_id);
		auto normal = thrust::get<1>(eig_val_vec).col(min_id).normalized();
		normal = (normal.norm() == 0.0) ? baseNormal : normal;
		if (normal.dot(baseNormal) < 0)
		{
			normal = -normal;
		}

		normal = baseNormal;

		if (failed)
		{
			if (useOutlierRemoval)
			{
				normal = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
				point_clr_buff[pointSlotIndex] = Eigen::Vector3b(0, 0, 0);
			}
		}

		point_nm_buff[pointSlotIndex] = normal;

		if (nullptr != debugPoints)
		{
			debugPoints[currentIndex] = basePos;
		}

		if (nullptr != debugNormals)
		{
			debugNormals[currentIndex] = normal;
		}

		if (nullptr != debugColors)
		{
			if (false == failed)
			{
				debugColors[currentIndex] = baseColor;
			}
			else
			{
				debugColors[currentIndex] = Eigen::Vector3b(255, 0, 0);
			}
		}
	}
	}

void MarchingCubes::LoadTransformsFromFile(vector<Eigen::Matrix4f>&transforms, const string & filename)
{
	auto tfile = ifstream();
	tfile.open(filename.c_str(), ios::in);
	std::string line;
	while (std::getline(tfile, line)) {
		//std::cout << "Read line: " << line << std::endl;
		stringstream ss(line);

		string word;
		ss >> word;
		Eigen::Matrix4f m;
		for (size_t r = 0; r < 4; r++)
		{
			for (size_t c = 0; c < 4; c++)
			{
				ss >> (*(m.data() + 4 * r + c));
			}
		}

		transforms.push_back(m);
	}
}

#ifndef BUILD_FOR_CPU
void LoadFrameFiles(cached_allocator * alloc_, CUstream_st * st, const string & dataRoot)
{
	//		vector<Eigen::Matrix4f> transforms_0;
	//		vector<Eigen::Matrix4f> transforms_45;
	//		LoadTransformsFromFile(transforms_0, dataRoot + "\\transform_0.txt");
	//		LoadTransformsFromFile(transforms_45, dataRoot + "\\transform_45.txt");
	//
	//		qDebug("Calling MarchingCubes::Integrate_v4()");
	//
	//		for (size_t i = 0; i < transforms_0.size() - 1; i++)
	//			//for (size_t i = 1; i < 2; i++)
	//		{
	//			pRegistration->__load_plyFile_pointcloud(pRegistration->m_src_0, i, "_source_0", const_cast<CUstream_st*>(pRegistration->GetStream()));
	//			pRegistration->__load_plyFile_pointcloud(pRegistration->m_src_45, i, "_source_45", const_cast<CUstream_st*>(pRegistration->GetStream()));
	//			pRegistration->m_src_0->rm_PC_nanf(pRegistration->source_PC, &pRegistration->alloc, pRegistration->GetStream());
	//			pRegistration->__GetAxisAlignedBoundingBox(*pRegistration->source_PC, *pRegistration->tmp_PC1);
	//			MarchingCubes::Integrate_v4(
	//				nullptr,
	//				nullptr,
	//				pRegistration->m_src_0,
	//				pRegistration->m_src_45,
	//				transforms_0[i],
	//				transforms_45[i],
	//				src_max_bound_0,
	//				src_min_bound_0,
	//				src_max_bound_45,
	//				src_min_bound_45,
	//				pRegistration->m_MC_depthMap,
	//				pRegistration->m_MC_voxelValues,
	//				pRegistration->m_MC_voxelValueCounts,
	//				pRegistration->m_MC_voxelNormals,
	//				pRegistration->m_MC_voxelColors,
	//				pRegistration->m_dlp_position,
	//				&pRegistration->alloc, pRegistration->GetStream()
	//				, pRegistration->m_MC_triangles
	//				, pRegistration->m_MC_cntMap
	//				, pRegistration->m_MC_colorMap
	//				, pRegistration->m_MC_normalMap);
	//
	//			qDebug("%d intagrated", i);
	//		}
}

__global__ void Kernel_LoadEXYZFile(
	float _voxelSize, size_t _globalVoxelCountX, size_t _globalVoxelCountY, size_t _globalVoxelCountZ, size_t _globalVoxelCount, Eigen::AlignedBox3f _globalScanAreaAABB,
	HashKey64 * _globalHash_info, Eigen::Vector3f * points, voxel_value_t * voxelValues, unsigned short* voxelValueCounts,
	Eigen::Vector3f * _globalVoxelNormals, voxel_value_t * _globalVoxelValues, unsigned short* _globalVoxelValueCounts,
	size_t noPoints)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > noPoints - 1) return;

	size_t index = threadid;

	auto& v = points[index];

	auto xGlobalIndex = (size_t)(floorf(v.x() / _voxelSize + 2500.f));
	auto yGlobalIndex = (size_t)(floorf(v.y() / _voxelSize + 2500.f));
	auto zGlobalIndex = (size_t)(floorf(v.z() / _voxelSize + 2500.f));

	HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
	auto i = _globalHash_info->get_insert_idx_func64_v4(key);
	if (kEmpty32 != i)
	{
		_globalVoxelNormals[i] = v;
		_globalVoxelValues[i] = voxelValues[index];
		_globalVoxelValueCounts[i] = voxelValueCounts[index];
	}
}

#if 0 // 사용되지 않음
void MarchingCubes::LoadEXYZFile(cached_allocator * alloc_, CUstream_st * st, const std::string & fileName)
{
	InitGlobalVoxelValues();

	float voxelSize = 0.1f;

	Eigen::AlignedBox3f globalScanAreaAABB;
	globalScanAreaAABB.extend(Eigen::Vector3f(-250.0f, -250.0f, -250.0f));
	globalScanAreaAABB.extend(Eigen::Vector3f(250.0f, 250.0f, 250.0f));

	auto globalVoxelCountX = (size_t)((globalScanAreaAABB.max() - globalScanAreaAABB.min()).x() / voxelSize);
	auto globalVoxelCountY = (size_t)((globalScanAreaAABB.max() - globalScanAreaAABB.min()).y() / voxelSize);
	auto globalVoxelCountZ = (size_t)((globalScanAreaAABB.max() - globalScanAreaAABB.min()).z() / voxelSize);
	auto globalVoxelCount = (size_t)(globalVoxelCountX * globalVoxelCountY * globalVoxelCountZ);

	EXYZFormat exyz;
	exyz.Deserialize(fileName);

	Eigen::AlignedBox3f localScanAreaAABB;
	localScanAreaAABB = exyz.GetAABB();

	//localScanAreaAABB.extend(Eigen::Vector3f(-10.0f, -10.0f, -10.0f));
	//localScanAreaAABB.extend(Eigen::Vector3f(10.0f, 10.0f, 10.0f));
	//localScanAreaAABB.extend(Eigen::Vector3f(-250.0f, -250.0f, -250.0f));
	//localScanAreaAABB.extend(Eigen::Vector3f(250.0f, 250.0f, 250.0f));

	auto localVoxelCountX = (size_t)((localScanAreaAABB.max() - localScanAreaAABB.min()).x() / voxelSize);
	auto localVoxelCountY = (size_t)((localScanAreaAABB.max() - localScanAreaAABB.min()).y() / voxelSize);
	auto localVoxelCountZ = (size_t)((localScanAreaAABB.max() - localScanAreaAABB.min()).z() / voxelSize);
	auto localVoxelCount = (size_t)(localVoxelCountX * localVoxelCountY * localVoxelCountZ);

	qDebug("%s loaded", fileName.c_str());

	auto& host_points_raw = exyz.GetPoints();
	auto& host_voxelValues_raw = exyz.GetVoxelValues();
	auto& host_voxelValueCounts_raw = exyz.GetVoxelValueCounts();
	size_t noPoints = host_points_raw.size() / 3;

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc

	Eigen::Vector3f* points;
	//cudaMallocAsync(&points, sizeof(Eigen::Vector3f) * noPoints, st);
	cudaMalloc(&points, sizeof(Eigen::Vector3f) * noPoints);
	cudaMemcpyAsync(points, host_points_raw.data(), sizeof(Eigen::Vector3f) * noPoints, cudaMemcpyHostToDevice, st);

	voxel_value_t* voxelValues;
	//cudaMallocAsync(&voxelValues, sizeof(voxel_value_t) * noPoints, st);
	cudaMalloc(&voxelValues, sizeof(voxel_value_t) * noPoints);
	cudaMemcpyAsync(voxelValues, host_voxelValues_raw.data(), sizeof(voxel_value_t) * noPoints, cudaMemcpyHostToDevice, st);

	unsigned short* voxelValueCounts;
	//cudaMallocAsync(&voxelValueCounts, sizeof(unsigned short) * noPoints, st);
	cudaMalloc(&voxelValueCounts, sizeof(unsigned short) * noPoints);
	cudaMemcpyAsync(voxelValueCounts, host_voxelValueCounts_raw.data(), sizeof(unsigned short) * noPoints, cudaMemcpyHostToDevice, st);

	if (noPoints >= m_MC_voxelValues.size())
	{
		m_MC_voxelNormals.resize(noPoints);
		m_MC_voxelValues.resize(noPoints);
		m_MC_voxelValueCounts.resize(noPoints);
	}

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_voxelNormals.begin(),
		m_MC_voxelNormals.end(),
		Eigen::Vector3f(0.0f, 0.0f, 0.0f));

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_voxelValues.begin(),
		m_MC_voxelValues.end(),
		VOXEL_INVALID);

	thrust::fill(
		thrust::cuda::par_nosync(*alloc_).on(st),
		m_MC_voxelValueCounts.begin(),
		m_MC_voxelValueCounts.end(),
		0);

	checkCudaSync(st);

	auto _globalVoxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _globalVoxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _globalVoxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());

	auto _globalHash_info = globalHash_info;
	//auto _globalHash = globalHash_info->hashtable;

	auto _voxelSize = voxelSize;
	auto _globalVoxelCountX = globalVoxelCountX;
	auto _globalVoxelCountY = globalVoxelCountY;
	auto _globalVoxelCountZ = globalVoxelCountZ;
	auto _globalVoxelCount = globalVoxelCount;
	auto _globalScanAreaAABB = globalScanAreaAABB;

	/*
	thrust::for_each(
		thrust::cuda::par_nosync(*alloc_).on(st),
		thrust::make_counting_iterator<size_t>(0),
		thrust::make_counting_iterator<size_t>(noPoints),
		[_voxelSize, _globalVoxelCountX, _globalVoxelCountY, _globalVoxelCountZ, _globalVoxelCount, _globalScanAreaAABB,
		_globalHash_info, points, voxelValues, voxelValueCounts, _globalVoxelPositions, _globalVoxelValues, _globalVoxelValueCounts,
		noPoints]
	__device__(size_t index) {
		auto& v = points[index];

		auto xGlobalIndex = (size_t)((v.x() - _globalScanAreaAABB.min().x()) / _voxelSize);
		auto yGlobalIndex = (size_t)((v.y() - _globalScanAreaAABB.min().y()) / _voxelSize);
		auto zGlobalIndex = (size_t)((v.z() - _globalScanAreaAABB.min().z()) / _voxelSize);

		//printf("%llu %llu %llu\n", xGlobalIndex, yGlobalIndex, zGlobalIndex);

		HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		auto i = get_hashtable_insert_idx_func64(_globalHash_info, key);

		////printf("[%llu:%d]global index %llu, %llu, %llu, xyz: %f, %f, %f\n", key, i, xGlobalIndex, yGlobalIndex, zGlobalIndex, v.x(), v.y(), v.z());

		_globalVoxelPositions[i] = v;
		_globalVoxelValues[i] = voxelValues[i];
		_globalVoxelValueCounts[i] = voxelValueCounts[i];
	});
	*/

	{
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_LoadEXYZFile, 0, 0));
		int gridsize = ((uint32_t)noPoints + threadblocksize - 1) / threadblocksize;

		qDebug("grid = %d, block = %d, thread = %d , localVoxelCount = %d", gridsize, threadblocksize, threadblocksize, noPoints);

		NvtxRangeCuda nvtxPrint("@Arron / SaveSurfacePointsAveraging / Kernel_LoadEXYZFile");
		Kernel_LoadEXYZFile << <gridsize, threadblocksize, 0, st >> > (
			_voxelSize, _globalVoxelCountX, _globalVoxelCountY, _globalVoxelCountZ, _globalVoxelCount, _globalScanAreaAABB,
			_globalHash_info, points, voxelValues, voxelValueCounts, _globalVoxelNormals, _globalVoxelValues, _globalVoxelValueCounts,
			noPoints);

		//Kernel_PrintVoxelValues << <gridsize, threadblocksize, 0, st >> > (
		//	_voxelSize, _globalVoxelCountX, _globalVoxelCountY, _globalVoxelCountZ, _globalVoxelCount, _globalScanAreaAABB,
		//	_globalHash_info, points, voxelValues, voxelValueCounts, _globalVoxelPositions, _globalVoxelValues, _globalVoxelValueCounts,
		//	noPoints);
		checkCudaSync(st);

		//fflush(stdout);
	}

	cudaFreeAsync(points, st);
	cudaFreeAsync(voxelValues, st);
	cudaFreeAsync(voxelValueCounts, st);

	checkCudaSync(st);
}
#endif

#endif

__global__ void Kernel_SaveVoxelValues(
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * resultPositions, Eigen::Vector3f * resultNormals, Eigen::Vector4f * resultColors,
	unsigned int* numberOfVoxelPositions, float voxelValueRangeMin, float voxelValueRangeMax)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 16384)
	for (int threadid = 0; threadid < globalHash_info->HashTableCapacity; threadid++) {
#endif

		auto key = globalHash_info->hashtable[threadid];
		if (key.Exists() && (kEmpty8 != globalHash_info->hashtable[threadid].value))
		{
			auto vv = voxelValues[threadid];
			if (VOXEL_INVALID != vv && USHORT_VALID(voxelValueCounts[threadid]))
			{
				if (voxelValueCounts[threadid] < 2) {
					kernel_return;
				}

				auto v = VV2D(vv);
				v /= voxelValueCounts[threadid];

				if (voxelValueRangeMin <= v && v <= voxelValueRangeMax)
				{
					auto x_idx = key.x;
					auto y_idx = key.y;
					auto z_idx = key.z;

					auto x = (float)x_idx * 0.1f + 0.05f - 250.0f;
					auto y = (float)y_idx * 0.1f + 0.05f - 250.0f;
					auto z = (float)z_idx * 0.1f + 0.05f - 250.0f;


					auto oldIndex = atomicAdd(numberOfVoxelPositions, (unsigned int)1);
					resultPositions[oldIndex] = Eigen::Vector3f(x, y, z);

					resultNormals[oldIndex] = voxelNormals[threadid];

					auto r = (float)(voxelColors[threadid].x()) / 255.0f;
					auto g = (float)(voxelColors[threadid].y()) / 255.0f;
					auto b = (float)(voxelColors[threadid].z()) / 255.0f;
					auto a = 1.0f;
					/*if (a > 1.0f) a = 1.0f;
					if (a < -1.0f) a = -1.0f;
					a = a + 1.0f;
					a = (a / 2) * 255.0f;*/

					r = a;
					g = a;
					b = a;

					if (v > 1.0f) v = 1.0f;
					if (v < -1.0f) v = -1.0f;

					r = v > 0.01f ? v : 0.0f;
					b = fabsf(v) <= 0.01f ? 1.0f : 0.0f;
					g = v < -0.01f ? -v : 0.0f;
					a = 1.0f;

					resultColors[oldIndex] = Eigen::Vector4f(r, g, b, a);
				}
			}
		}
	}
	}

#ifndef BUILD_FOR_CPU
void MarchingCubes::SaveVoxelValues(const string & filename, float voxelValueRangeMin, float voxelValueRangeMax, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return;

	qDebug("SaveVoxelValues()");

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	Eigen::Vector3f* voxelPositions = nullptr;
	cudaMalloc(&voxelPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

	Eigen::Vector3f* voxelNormals = nullptr;
	cudaMalloc(&voxelNormals, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

	Eigen::Vector4f* voxelColors = nullptr;
	cudaMalloc(&voxelColors, sizeof(Eigen::Vector4f) * globalHash_info_host->HashTableCapacity);

	unsigned int* numberOfVoxelPositions = nullptr;
	cudaMalloc(&numberOfVoxelPositions, sizeof(unsigned int));
	cudaMemsetAsync(numberOfVoxelPositions, 0, sizeof(unsigned int), st);

	checkCudaSync(st);

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_SaveVoxelValues, 0, 0));
	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	Kernel_SaveVoxelValues << <gridsize, threadblocksize, 0, st >> > (globalHash_info,
		thrust::raw_pointer_cast(m_MC_voxelValues.data()),
		thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
		thrust::raw_pointer_cast(m_MC_voxelColors.data()),
		thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
		voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
		voxelValueRangeMin, voxelValueRangeMax);

	checkCudaSync(st);

	unsigned int host_numberOfVoxelValues = 0;
	cudaMemcpyAsync(&host_numberOfVoxelValues, numberOfVoxelPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_voxelPositions = new Eigen::Vector3f[host_numberOfVoxelValues];
	cudaMemcpyAsync(host_voxelPositions, voxelPositions,
		sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_voxelNormals = new Eigen::Vector3f[host_numberOfVoxelValues];
	cudaMemcpyAsync(host_voxelNormals, voxelNormals,
		sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

	Eigen::Vector4f* host_voxelColors = new Eigen::Vector4f[host_numberOfVoxelValues];
	cudaMemcpyAsync(host_voxelColors, voxelColors,
		sizeof(Eigen::Vector4f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	qDebug("host_numberOfVoxelValues : %d", host_numberOfVoxelValues);

	PLYFormat ply;

	for (size_t i = 0; i < host_numberOfVoxelValues; i++)
	{
		auto& v = host_voxelPositions[i];
		if (VECTOR3F_VALID_(v))
		{
			ply.AddPointFloat3(v.data());
			auto& normal = host_voxelNormals[i];
			normal.normalize();
			ply.AddNormalFloat3(normal.data());
			auto& color = host_voxelColors[i];
			ply.AddColorFloat4(color.data());
		}
	}
	ply.Serialize(filename);

	delete[] host_voxelPositions;
	delete[] host_voxelNormals;
	delete[] host_voxelColors;

	cudaFree(voxelPositions);
	cudaFree(voxelNormals);
	cudaFree(voxelColors);
	cudaFree(numberOfVoxelPositions);
}
#else
// CPU 코드
void MarchingCubes::SaveVoxelValues(const string & filename, float voxelValueRangeMin, float voxelValueRangeMax, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info == nullptr) return;

	qDebug("SaveVoxelValues()");

	std::vector<Eigen::Vector3f> voxelPositions(globalHash_info->HashTableCapacity);
	std::vector<Eigen::Vector3f> voxelNormals(globalHash_info->HashTableCapacity);
	std::vector<Eigen::Vector4f> voxelColors(globalHash_info->HashTableCapacity);

	unsigned int numberOfVoxelPositions = 0;

	Kernel_SaveVoxelValues(globalHash_info,
		thrust::raw_pointer_cast(pRegistration->m_MC_voxelValues.data()),
		thrust::raw_pointer_cast(pRegistration->m_MC_voxelNormals.data()),
		thrust::raw_pointer_cast(pRegistration->m_MC_voxelColors.data()),
		thrust::raw_pointer_cast(pRegistration->m_MC_voxelValueCounts.data()),
		voxelPositions.data(), voxelNormals.data(), voxelColors.data(), &numberOfVoxelPositions,
		voxelValueRangeMin, voxelValueRangeMax);

	qDebug("host_numberOfVoxelValues : %d", numberOfVoxelPositions);

	PLYFormat ply;

	for (size_t i = 0; i < numberOfVoxelPositions; i++)
	{
		auto& v = voxelPositions[i];
		if (VECTOR3F_VALID_(v))
		{
			ply.AddPointFloat3(v.data());
			auto& normal = voxelNormals[i];
			normal.normalize();
			ply.AddNormalFloat3(normal.data());
			// 임시 주석
			//auto& color = voxelColors[i];
			//ply.AddColorFloat4(color.data());
		}
	}
	ply.Serialize(filename);

}
#endif

// 디버그 목적으로 DepthNormalMap을 저장한다.
void MarchingCubes::SaveDepthNormalMap(
#ifndef BUILD_FOR_CPU
	const thrust::device_vector<Eigen::Vector3f>&dev_depthMap,
	const thrust::device_vector<Eigen::Vector3f>&dev_normalMap,
	const thrust::device_vector<unsigned int>&dev_colorMap,
	cached_allocator * alloc_, CUstream_st * st,
#else
	const thrust::host_vector<Eigen::Vector3f>&depthMap,
	const thrust::host_vector<Eigen::Vector3f>&normalMap,
	const thrust::host_vector<unsigned int>&colorMap,
#endif
	const std::string & filename)
{
#ifndef BUILD_FOR_CPU
	thrust::host_vector<Eigen::Vector3f> depthMap(dev_depthMap.size());
	thrust::host_vector<Eigen::Vector3f> normalMap(dev_normalMap.size());
	thrust::host_vector<unsigned int> colorMap(dev_colorMap.size());

	cudaStreamSynchronize(st);
	thrust::copy(dev_depthMap.begin(), dev_depthMap.end(), depthMap.begin());
	thrust::copy(dev_normalMap.begin(), dev_normalMap.end(), normalMap.begin());
	thrust::copy(dev_colorMap.begin(), dev_colorMap.end(), colorMap.begin());
	cudaStreamSynchronize(st);

#endif
	PLYFormat ply;

	for (size_t i = 0; i < depthMap.size(); i++) {
		const Eigen::Vector3f& point = depthMap[i];
		if (FLT_VALID(point.z())) {
			ply.AddPointFloat3(point.data());
			ply.AddNormalFloat3(normalMap[i].data());
			ply.AddColor(
				static_cast<unsigned char>(colorMap[i * 3 + 0]),
				static_cast<unsigned char>(colorMap[i * 3 + 1]),
				static_cast<unsigned char>(colorMap[i * 3 + 2]));
		}
	}
	ply.Serialize(filename);
}

#ifndef BUILD_FOR_CPU
__global__ void Kernel_SaveVoxelKeys(
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, HashKey * voxelKeys, unsigned int* numberOfVoxelKeys)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity - 1) return;

	auto key = globalHash_info->hashtable[threadid];
	if (key.Exists())
	{
		auto vv = voxelValues[threadid];
		if (VOXEL_INVALID != vv)
		{
			auto oldIndex = atomicAdd(numberOfVoxelKeys, 1);
			voxelKeys[oldIndex] = key.Key();
		}
	}
}

void MarchingCubes::SaveVoxelKeys(const std::string & filename, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return;

	qDebug("SaveVoxelKeys()");

	HashKey* voxelKeys = nullptr;
	cudaMalloc(&voxelKeys, sizeof(HashKey) * globalHash_info_host->HashTableCapacity);

	unsigned int* numberOfVoxelKeys = nullptr;
	cudaMalloc(&numberOfVoxelKeys, sizeof(unsigned int));
	cudaMemsetAsync(numberOfVoxelKeys, 0, sizeof(unsigned int), st);

	checkCudaSync(st);

	auto voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_SaveVoxelKeys, 0, 0));
	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	Kernel_SaveVoxelKeys << <gridsize, threadblocksize, 0, st >> > (globalHash_info,
		voxelValues, voxelKeys, numberOfVoxelKeys);

	checkCudaSync(st);

	unsigned int host_numberOfVoxelKeys = 0;
	cudaMemcpyAsync(&host_numberOfVoxelKeys, numberOfVoxelKeys, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

	uint64_t* host_voxelKeys = new uint64_t[host_numberOfVoxelKeys];
	cudaMemcpyAsync(host_voxelKeys, voxelKeys,
		sizeof(HashKey) * host_numberOfVoxelKeys, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	qDebug("host_numberOfVoxelKeys : %d", host_numberOfVoxelKeys);

	PLYFormat ply;
	for (size_t i = 0; i < host_numberOfVoxelKeys; i++)
	{
		auto& key = host_voxelKeys[i];
		auto x_idx = (key >> 32) & 0xffff;
		auto y_idx = (key >> 16) & 0xffff;
		auto z_idx = (key) & 0xffff;
		//qDebug("key : %llu, %d, %d, %d", key, x_idx, y_idx, z_idx);

		float x = x_idx * 0.1f - 250.0f;
		float y = y_idx * 0.1f - 250.0f;
		float z = z_idx * 0.1f - 250.0f;

		ply.AddPoint(x, y, z);
	}
	ply.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\Temp\\Temp.ply");

	ofstream ofs;
	ofs.open(filename, ios::out | ios::binary);
	if (ofs.is_open())
	{
		uint64_t numberOfKeys = host_numberOfVoxelKeys;
		ofs.write((char*)&numberOfKeys, sizeof(uint64_t));
		ofs.write((char*)host_voxelKeys, sizeof(HashKey) * numberOfKeys);
		ofs.close();
	}
}

__global__ void Kernel_LoadVoxelValues(
	float _voxelSize, size_t _globalVoxelCountX, size_t _globalVoxelCountY, size_t _globalVoxelCountZ, size_t _globalVoxelCount, Eigen::AlignedBox3f _globalScanAreaAABB,
	HashKey64 * _globalHash_info, uint8_t * _globalHashValue, Eigen::Vector3f * points, voxel_value_t * voxelValues, unsigned short* voxelValueCounts,
	Eigen::Vector3f * _globalVoxelNormals, voxel_value_t * _globalVoxelValues, unsigned short* _globalVoxelValueCounts,
	size_t noPoints)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > noPoints - 1) return;

	size_t index = threadid;

	auto& v = points[index];

	auto xGlobalIndex = (size_t)(floorf(v.x() / _voxelSize + 2500.f));
	auto yGlobalIndex = (size_t)(floorf(v.y() / _voxelSize + 2500.f));
	auto zGlobalIndex = (size_t)(floorf(v.z() / _voxelSize + 2500.f));

	HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
	auto i = _globalHash_info->get_insert_idx_func64_v4(key);
	if (kEmpty32 != i)
	{
		_globalVoxelNormals[i] = v;
		_globalVoxelValues[i] = voxelValues[index];
		_globalVoxelValueCounts[i] = voxelValueCounts[index];
	}
}

#if 0 // 사용되지 않음
void MarchingCubes::LoadVoxelValues(const string & filename, cached_allocator * alloc_, CUstream_st * st)
{
	qDebug("LoadVoxelValues()");

	PLYFormat ply;
	ply.Deserialize(filename);

	qDebug("points : %llu, normals : %llu, colors : %llu"
		, ply.GetPoints().size() / 3
		, ply.GetNormals().size() / 3
		, ply.UseAlpha() ? ply.GetColors().size() / 4 : ply.GetColors().size() / 3);
}
#endif

void MarchingCubes::SaveCurrentPatchSurfacePoints(const std::string & filename, HMesh & View_VTX, cached_allocator * alloc_, CUstream_st * st)
{
	auto _repos_pos = View_VTX.points;
	auto _repos_nm = View_VTX.normals;
	auto _repos_color = View_VTX.colors;

	//auto _repos_pos = thrust::raw_pointer_cast(pRegistration->m_points_pos.data());
	//auto _repos_nm = thrust::raw_pointer_cast(pRegistration->m_points_nm.data());
	//auto _repos_color = thrust::raw_pointer_cast(pRegistration->m_points_color.data());

	unsigned int host_used_cnt_localContains = 0;
	cudaMemcpyAsync(&host_used_cnt_localContains, used_cnt_localContains, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	if (0 == host_used_cnt_localContains)
	{
		cudaMemcpyAsync(&host_used_cnt_localContains, &exeInfo.globalHashInfo->Count_HashTableUsed, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
		checkCudaSync(st);
	}

	Eigen::Vector3f* host_pos = new Eigen::Vector3f[host_used_cnt_localContains];
	cudaMemcpyAsync(host_pos, _repos_pos, sizeof(Eigen::Vector3f) * host_used_cnt_localContains, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_nm = new Eigen::Vector3f[host_used_cnt_localContains];
	cudaMemcpyAsync(host_nm, _repos_nm, sizeof(Eigen::Vector3f) * host_used_cnt_localContains, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* host_color = new Eigen::Vector3b[host_used_cnt_localContains];
	cudaMemcpyAsync(host_color, _repos_color, sizeof(Eigen::Vector3b) * host_used_cnt_localContains, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	PLYFormat ply;
	for (size_t i = 0; i < host_used_cnt_localContains; i++)
	{
		auto& p = host_pos[i];
		if (VECTOR3F_VALID_(p))
		{
			ply.AddPointFloat3(p.data());
			auto& n = host_nm[i];
			ply.AddNormalFloat3(n.data());
			auto& c = host_color[i];
			ply.AddColor(c);

			//ply.AddColor((float)c.x() / 255.0f, (float)c.y() / 255.0f, (float)c.z() / 255.0f);
		}
	}
	ply.Serialize(filename);

	delete[] host_pos;
	delete[] host_nm;
	delete[] host_color;
}

void MarchingCubes::SavePointsForRendering(const std::string & filename, HMesh & View_VTX, cached_allocator * alloc_, CUstream_st * st)
{
	size_t host_View_vtxSize = View_VTX.vertexCount;

	Eigen::Vector3f* host_View_VTX_pos = new Eigen::Vector3f[host_View_vtxSize];
	cudaMemcpyAsync(host_View_VTX_pos, View_VTX.points, sizeof(Eigen::Vector3f) * host_View_vtxSize, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_View_VTX_nm = new Eigen::Vector3f[host_View_vtxSize];
	cudaMemcpyAsync(host_View_VTX_nm, View_VTX.normals, sizeof(Eigen::Vector3f) * host_View_vtxSize, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* host_View_VTX_color = new Eigen::Vector3b[host_View_vtxSize];
	cudaMemcpyAsync(host_View_VTX_color, View_VTX.colors, sizeof(Eigen::Vector3b) * host_View_vtxSize, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	PLYFormat ply;
	for (size_t i = 0; i < host_View_vtxSize; i++)
	{
		auto& p = host_View_VTX_pos[i];
		auto& n = host_View_VTX_nm[i];
		auto& c = host_View_VTX_color[i];

		ply.AddPointFloat3(p.data());
		ply.AddNormalFloat3(n.data());
		ply.AddColor(c);
	}

	ply.Serialize(filename);

	delete[] host_View_VTX_pos;
	delete[] host_View_VTX_nm;
	delete[] host_View_VTX_color;
}

void MarchingCubes::LoadPointsForRendering(const std::string & filename, HMesh & View_VTX, cached_allocator * alloc_, CUstream_st * st)
{
	PLYFormat ply;
	ply.Deserialize(filename);

	View_VTX.vertexCount = ply.GetPoints().size() / 3;

	cudaMemcpyAsync(
		View_VTX.points,
		ply.GetPoints().data(),
		sizeof(float) * ply.GetPoints().size(),
		cudaMemcpyHostToDevice, st);

	cudaMemcpyAsync(
		View_VTX.normals,
		ply.GetPoints().data(),
		sizeof(float) * ply.GetNormals().size(),
		cudaMemcpyHostToDevice, st);

	cudaMemcpyAsync(
		View_VTX.colors,
		ply.GetColors().data(),
		sizeof(unsigned char) * ply.GetColors().size(),
		cudaMemcpyHostToDevice, st);

	checkCudaSync(st);
}

__global__ void Kernel_GetPointIndicesInRegion(
	Eigen::Vector3f center,
	float radius,
	Eigen::Vector3f * points,
	size_t pointCount,
	unsigned int* indices,
	unsigned int* count)
{
	unsigned int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIndex > pointCount - 1) return;

	auto& p = points[pointIndex];
	if ((center - p).norm() <= radius)
	{
		auto arrayIndex = atomicAdd(count, 1);
		indices[arrayIndex] = pointIndex;
	}
}

void MarchingCubes::GetPointIndicesInSphere(
	HMesh & View_VTX,
	const Eigen::Vector3f & center,
	float radius,
	unsigned int* indices,
	unsigned int* count,
	cached_allocator * alloc_, CUstream_st * st)
{
	size_t host_View_vtxSize = View_VTX.vertexCount;

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GetPointIndicesInRegion, 0, 0));
	int gridsize = (host_View_vtxSize + threadblocksize - 1) / threadblocksize;
	Kernel_GetPointIndicesInRegion << <gridsize, threadblocksize, 0, st >> > (
		center,
		radius,
		View_VTX.points,
		host_View_vtxSize,
		indices,
		count);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);
}

__global__ void Kernel_GetPointsInSphere(
	Eigen::Vector3f * input_points,
	size_t pointCount,
	Eigen::Vector3f center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * count)
{
	unsigned int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIndex > pointCount - 1) return;

	auto& p = input_points[pointIndex];
	if ((center - p).norm() <= radius)
	{
		auto arrayIndex = atomicAdd(count, 1);
		output_points[arrayIndex] = p;
	}
}

__global__ void Kernel_GetPointsNotInSphere(
	Eigen::Vector3f * input_points,
	size_t pointCount,
	Eigen::Vector3f center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * count)
{
	unsigned int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIndex > pointCount - 1) return;

	auto& p = input_points[pointIndex];
	if ((center - p).norm() > radius)
	{
		auto arrayIndex = atomicAdd(count, 1);
		output_points[arrayIndex] = p;
	}
}

void MarchingCubes::GetPointsInSphere(
	HMesh & View_VTX,
	const Eigen::Vector3f & center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * output_count,
	cached_allocator * alloc_, CUstream_st * st)
{
	size_t host_View_vtxSize = View_VTX.vertexCount;

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GetPointsInSphere, 0, 0));
	int gridsize = (host_View_vtxSize + threadblocksize - 1) / threadblocksize;
	Kernel_GetPointsInSphere << <gridsize, threadblocksize, 0, st >> > (
		View_VTX.points,
		View_VTX.vertexCount,
		center,
		radius,
		output_points,
		output_count);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);
}

void MarchingCubes::GetPointsNotInSphere(
	HMesh & View_VTX,
	const Eigen::Vector3f & center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * output_count,
	cached_allocator * alloc_, CUstream_st * st)
{
	size_t host_View_vtxSize = View_VTX.vertexCount;

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GetPointsNotInSphere, 0, 0));
	int gridsize = (host_View_vtxSize + threadblocksize - 1) / threadblocksize;
	Kernel_GetPointsNotInSphere << <gridsize, threadblocksize, 0, st >> > (
		View_VTX.points,
		View_VTX.vertexCount,
		center,
		radius,
		output_points,
		output_count);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);
}

__global__ void Kernel_GetPointsNotInCylinder(
	Eigen::Vector3f * input_points,
	size_t pointCount,
	Eigen::Vector3f center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * count)
{
	unsigned int pointIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (pointIndex > pointCount - 1) return;

	auto& p = input_points[pointIndex];
	auto dx = center.x() - p.x();
	auto dy = center.y() - p.y();
	if (sqrtf(dx * dx + dy * dy) > radius)
	{
		auto arrayIndex = atomicAdd(count, 1);
		output_points[arrayIndex] = p;
	}
}

void MarchingCubes::GetPointsNotInCylinder(
	HMesh & View_VTX,
	const Eigen::Vector3f & center,
	float radius,
	Eigen::Vector3f * output_points,
	size_t * output_count,
	cached_allocator * alloc_, CUstream_st * st)
{
	size_t host_View_vtxSize = View_VTX.vertexCount;

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_GetPointsNotInCylinder, 0, 0));
	int gridsize = (host_View_vtxSize + threadblocksize - 1) / threadblocksize;
	Kernel_GetPointsNotInCylinder << <gridsize, threadblocksize, 0, st >> > (
		View_VTX.points,
		View_VTX.vertexCount,
		center,
		radius,
		output_points,
		output_count);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);
}

__global__ void Kernel_DeleteVoxelDatasUsingPointsIndices(
	MarchingCubes::ExecutionInfo exeInfo,
	Eigen::AlignedBox3f globalScanAreaAABB,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * voxelNormals,
	Eigen::Vector3b * voxelColors,
	float* voxelColorScores,
	char* voxelSegmentations,
	VoxelExtraAttrib * voxelExtraAttribs,
	Eigen::Vector3f * view_pos_repos,
	Eigen::Vector3f * view_nm_repos,
	Eigen::Vector3b * view_color_repos,
	unsigned int* indices,
	unsigned int indexCount,
	float radius)
{
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > indexCount - 1) return;

	auto index = indices[threadId];
	auto point = view_pos_repos[index];

	auto xGlobalIndex = (size_t)floorf(point.x() / exeInfo.global.voxelSize - globalScanAreaAABB.min().x() / exeInfo.global.voxelSize);
	auto yGlobalIndex = (size_t)floorf(point.y() / exeInfo.global.voxelSize - globalScanAreaAABB.min().y() / exeInfo.global.voxelSize);
	auto zGlobalIndex = (size_t)floorf(point.z() / exeInfo.global.voxelSize - globalScanAreaAABB.min().z() / exeInfo.global.voxelSize);

	HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
	auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);

	if (i != kEmpty32)
	{
		voxelValues[i] = VOXEL_INVALID;
		voxelValueCounts[i] = 0;
		voxelNormals[i] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
		voxelColors[i] = Eigen::Vector3b(0, 0, 0);
		voxelColorScores[i] = 0.0f;
		voxelSegmentations[i] = 0;
		voxelExtraAttribs[i] = { 0, };

		size_t range = radius / exeInfo.global.voxelSize;

		for (size_t zIndex = zGlobalIndex - range; zIndex <= zGlobalIndex + range; zIndex++)
		{
			if (zIndex > exeInfo.global.voxelCountZ) continue;

			for (size_t yIndex = yGlobalIndex - range; yIndex <= yGlobalIndex + range; yIndex++)
			{
				if (yIndex > exeInfo.global.voxelCountY) continue;

				for (size_t xIndex = xGlobalIndex - range; xIndex <= xGlobalIndex + range; xIndex++)
				{
					if (xIndex > exeInfo.global.voxelCountX) continue;

					auto dx = (float)xIndex - (float)xGlobalIndex;
					auto dy = (float)yIndex - (float)yGlobalIndex;
					auto dz = (float)zIndex - (float)zGlobalIndex;
					auto ddx = dx * dx;
					auto ddy = dy * dy;
					auto ddz = dz * dz;
					if (sqrtf(ddx + ddy + ddz) * exeInfo.global.voxelSize <= radius)
					{
						HashKey nkey(xIndex, yIndex, zIndex);
						auto ni = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(nkey);
						if (ni != kEmpty32)
						{
							voxelValues[ni] = VOXEL_INVALID;
							voxelValueCounts[ni] = 0;
							voxelNormals[ni] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
							voxelColors[ni] = Eigen::Vector3b(0, 0, 0);
							voxelColorScores[ni] = 0.0f;
							voxelSegmentations[ni] = 0;
							voxelExtraAttribs[ni] = { 0, };
						}
					}
				}
			}
		}
	}
}

void MarchingCubes::DeleteVoxelDatasUsingPointsIndices(
	Eigen::Vector3f * view_pos_repos,
	Eigen::Vector3f * view_nm_repos,
	Eigen::Vector3b * view_color_repos,
	unsigned int* indices,
	unsigned int* indexCount,
	float radius,
	cached_allocator * alloc_, CUstream_st * st)
{
	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);

	unsigned int host_indexCount = 0;
	cudaMemcpyAsync(&host_indexCount, indexCount, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	qDebug("host_indexCount : %d", host_indexCount);

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_DeleteVoxelDatasUsingPointsIndices, 0, 0));
	int gridsize = (host_indexCount + threadblocksize - 1) / threadblocksize;
	Kernel_DeleteVoxelDatasUsingPointsIndices << <gridsize, threadblocksize, 0, st >> > (
		exeInfo,
		globalScanAreaAABB,
		_voxelValues,
		_voxelValueCounts,
		_voxelNormals,
		_voxelColors,
		_voxelColorScores,
		_voxelSegmentations,
		_voxelExtraAttribs,
		view_pos_repos,
		view_nm_repos,
		view_color_repos,
		indices,
		host_indexCount,
		radius);

	checkCudaSync(st);

	//SaveVoxelValues(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\VoxelValues_hole.ply", alloc_, st);
}

void MarchingCubes::Test_DeleteVoxelDataUsingPointsIndices(HMesh & View_VTX, cached_allocator * alloc_, CUstream_st * st)
{
	SavePointsForRendering(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\RenderedPoints.ply", View_VTX, alloc_, st);

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	unsigned int* d_indices = nullptr;
	cudaMalloc(&d_indices, sizeof(unsigned int) * View_VTX.vertexCount);
	cudaMemsetAsync(&d_indices, 0, sizeof(unsigned int) * View_VTX.vertexCount, st);

	unsigned int* d_count = nullptr;
	cudaMalloc(&d_count, sizeof(unsigned int));
	cudaMemsetAsync(&d_count, 0, sizeof(unsigned int), st);

	checkCudaSync(st);

	MarchingCubes::GetPointIndicesInSphere(
		View_VTX,
		Eigen::Vector3f(0.0f, 0.0f, 0.0f),
		5.0f,
		d_indices,
		d_count,
		alloc_, st);

	MarchingCubes::DeleteVoxelDatasUsingPointsIndices(
		View_VTX.points,
		View_VTX.normals,
		View_VTX.colors,
		d_indices, d_count,
		1.0f,
		alloc_, st);

	//MarchingCubes::SavePointsForRenderingWithMask(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\RenderedPointsWithMask.ply",
	//	mask, &pRegistration->alloc, pRegistration->GetStream());

	MarchingCubes::SerializeVolume(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\VoxelValues_hole.vol", alloc_, st);

	cudaFreeAsync(d_indices, st);
	cudaFreeAsync(d_count, st);
}

#endif

__global__ void Kernel_KeepAliveVoxelDatasUsingPointArray(
	MarchingCubes::ExecutionInfo exeInfo,
	Eigen::AlignedBox3f globalScanAreaAABB,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * voxelNormals,
	Eigen::Vector3b * voxelColors,
	float* voxelColorScores,
	char* voxelSegmentations,
	VoxelExtraAttrib * voxelExtraAttribs,
	Eigen::Vector3f * input_points,
	size_t input_count,
	float radius)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > (unsigned int)input_count - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadId = 0; threadId < (int)input_count; threadId++) {
#endif

		auto point = input_points[threadId];

		auto xGlobalIndex = (size_t)floorf(point.x() / exeInfo.global.voxelSize - globalScanAreaAABB.min().x() / exeInfo.global.voxelSize);
		auto yGlobalIndex = (size_t)floorf(point.y() / exeInfo.global.voxelSize - globalScanAreaAABB.min().y() / exeInfo.global.voxelSize);
		auto zGlobalIndex = (size_t)floorf(point.z() / exeInfo.global.voxelSize - globalScanAreaAABB.min().z() / exeInfo.global.voxelSize);

		HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);

		if (i != kEmpty32)
		{
			exeInfo.globalHash[i].value = 7;

			size_t range = radius / exeInfo.global.voxelSize;
			float radius2 = radius * radius / exeInfo.global.voxelSize;

			for (size_t zIndex = zGlobalIndex - range; zIndex <= zGlobalIndex + range; zIndex++)
			{
				if (zIndex > exeInfo.global.voxelCountZ) continue;

				auto dz = (float)zIndex - (float)zGlobalIndex;
				auto ddz = dz * dz;
				for (size_t yIndex = yGlobalIndex - range; yIndex <= yGlobalIndex + range; yIndex++)
				{
					if (yIndex > exeInfo.global.voxelCountY) continue;

					auto dy = (float)yIndex - (float)yGlobalIndex;
					auto ddy = dy * dy;
					for (size_t xIndex = xGlobalIndex - range; xIndex <= xGlobalIndex + range; xIndex++)
					{
						if (xIndex > exeInfo.global.voxelCountX) continue;

						auto dx = (float)xIndex - (float)xGlobalIndex;
						auto ddx = dx * dx;
						if ((ddx + ddy + ddz) <= radius2)
						{
							HashKey nkey(xIndex, yIndex, zIndex);
							auto ni = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(nkey);
							if (ni != kEmpty32)
							{
								exeInfo.globalHash[ni].value = 7;
							}
						}
					}
				}
			}
		}
	}
	}

__global__ void Kernel_DeleteTheRest(
	MarchingCubes::ExecutionInfo exeInfo,
	Eigen::AlignedBox3f globalScanAreaAABB,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * voxelNormals,
	Eigen::Vector3b * voxelColors,
	float* voxelColorScores,
	char* voxelSegmentations,
	VoxelExtraAttrib * voxelExtraAttribs,
	Eigen::Vector3f * input_points,
	size_t input_count)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;
	{
#else
#pragma omp parallel for schedule(dynamic, 256)
	for (int threadid = 0; threadid < exeInfo.globalHashInfo->HashTableCapacity; threadid++) {
#endif

		auto& key = exeInfo.globalHash[threadid];
		if (key.Exists())
		{
			if (7 != exeInfo.globalHash[threadid].value)
			{
				voxelValues[threadid] = VOXEL_INVALID;
				voxelValueCounts[threadid] = SHRT_MAX;
				voxelNormals[threadid] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				voxelColors[threadid] = Eigen::Vector3b(0, 0, 0);
				voxelColorScores[threadid] = 0;
				voxelExtraAttribs[threadid] = { 0, };
			}
			else if (7 == exeInfo.globalHash[threadid].value)
			{
				exeInfo.globalHash[threadid].value = 1;
			}
		}
	}
	}

void MarchingCubes::KeepAliveVoxelDatasUsingPointArray(
	Eigen::Vector3f * points,
	size_t pointCount,
	float radius,
	cached_allocator * alloc_, CUstream_st * st,
	bool isHtoD, bool deleteTheRest)
{
	NvtxRangeCuda nvtxPrint("@Final/chackAlive", true, true, 0xFF0000FF);
	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

#ifndef BUILD_FOR_CPU
	//auto _view_pos_repos = pRegistration->m_View_VTX.points;
	//auto _view_nm_repos = pRegistration->m_View_VTX.normals;
	//auto _view_color_repos = pRegistration->m_View_VTX.colors;
#endif
	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);

	//	mscho	@20240530
//	cudaMallocAsync => cudaMalloc

	Eigen::Vector3f* input_points = nullptr;
#ifndef BUILD_FOR_CPU
	if (isHtoD)
	{
		cudaMalloc(&input_points, sizeof(Eigen::Vector3f) * pointCount);
		cudaMemcpyAsync(input_points, points, sizeof(Eigen::Vector3f) * pointCount, cudaMemcpyHostToDevice, st);
	}
	else
		input_points = points;
#else
	input_points = points;
#endif

	checkCudaSync(st);

	int mingridsize;
	int threadblocksize;

	if (pointCount) {
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_KeepAliveVoxelDatasUsingPointArray, 0, 0));
		int gridsize = (pointCount + threadblocksize - 1) / threadblocksize;
		Kernel_KeepAliveVoxelDatasUsingPointArray << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_KeepAliveVoxelDatasUsingPointArray(
#endif
			exeInfo,
			globalScanAreaAABB,
			_voxelValues,
			_voxelValueCounts,
			_voxelNormals,
			_voxelColors,
			_voxelColorScores,
			_voxelSegmentations,
			_voxelExtraAttribs,
			input_points,
			pointCount,
			radius);
		checkCudaErrors(cudaGetLastError());
	}

	checkCudaSync(st);

	if (deleteTheRest)
	{
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_DeleteTheRest, 0, 0));
		int gridsize = (exeInfo.globalHashInfo_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;
		Kernel_DeleteTheRest << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_DeleteTheRest(
#endif
			exeInfo,
			globalScanAreaAABB,
			_voxelValues,
			_voxelValueCounts,
			_voxelNormals,
			_voxelColors,
			_voxelColorScores,
			_voxelSegmentations,
			_voxelExtraAttribs,
			input_points,
			pointCount);
		checkCudaErrors(cudaGetLastError());
		checkCudaSync(st);
	}

#ifndef BUILD_FOR_CPU
	if (isHtoD)
	{
		cudaFree(input_points);
	}
#endif
	//SaveVoxelValues(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\VoxelValues_hole.ply", alloc_, st);
}

__global__ void Kernel_KeysToPoints(MarchingCubes::ExecutionInfo exeInfo, HashKey * keys, size_t numberOfKeys, Eigen::Vector3f * points, float scale)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > (unsigned int)numberOfKeys - 1) return;
	{
#else
	for (int threadId = 0; threadId < int(numberOfKeys); threadId++) {
#endif

		auto key = keys[threadId];

		auto globalIndexX = key.x;
		auto globalIndexY = key.y;
		auto globalIndexZ = key.z;

		float x = ((float)globalIndexX * exeInfo.global.voxelSize + exeInfo.global.globalMin.x()) * scale;
		float y = ((float)globalIndexY * exeInfo.global.voxelSize + exeInfo.global.globalMin.y()) * scale;
		float z = ((float)globalIndexZ * exeInfo.global.voxelSize + exeInfo.global.globalMin.z()) * scale;

		points[threadId] = Eigen::Vector3f(x, y, z);
	}
	}

void MarchingCubes::KeepAliveVoxelDatasUsingVoxelKeys(
	HashKey * keys,
	size_t numberOfKeys,
	float radius,
	float scale,
	cached_allocator * alloc_, CUstream_st * st,
	bool isHtoD, bool deleteTheRest)
{
	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

#ifndef BUILD_FOR_CPU
	//auto _view_pos_repos = pRegistration->m_View_VTX.points;
	//auto _view_nm_repos = pRegistration->m_View_VTX.normals;
	//auto _view_color_repos = pRegistration->m_View_VTX.colors;
#endif

	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);

	//	mscho	@20240530
//	cudaMallocAsync => cudaMalloc

	HashKey* d_keys = nullptr;
	Eigen::Vector3f* input_points = nullptr;
#ifndef BUILD_FOR_CPU
	cudaMallocAsync(&input_points, sizeof(Eigen::Vector3f) * numberOfKeys, st);
	checkCudaSync(st);
#else
	input_points = new Eigen::Vector3f[numberOfKeys];
#endif

	if (isHtoD)
	{
#ifndef BUILD_FOR_CPU
		cudaMallocAsync(&d_keys, sizeof(HashKey) * numberOfKeys, st);
		cudaMemcpyAsync(d_keys, keys, sizeof(HashKey) * numberOfKeys, cudaMemcpyHostToDevice, st);

		checkCudaSync(st);
#else
		d_keys = new uint64_t[numberOfKeys];
		memcpy(d_keys, keys, sizeof(HashKey) * numberOfKeys);
#endif


#ifndef BUILD_FOR_CPU
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_KeysToPoints, 0, 0));
		int gridsize = (numberOfKeys + threadblocksize - 1) / threadblocksize;

		qDebug("numberOfKeys : %llu", numberOfKeys);

		Kernel_KeysToPoints << <gridsize, threadblocksize, 0, st >> > (exeInfo, d_keys, numberOfKeys, input_points, scale);

		checkCudaSync(st);

#else
		Kernel_KeysToPoints(exeInfo, d_keys, numberOfKeys, input_points, scale);
#endif

#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaFreeAsync(d_keys, st));
#else
		delete[] d_keys;
#endif
	}
	else
	{
#ifndef BUILD_FOR_CPU
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_KeysToPoints, 0, 0));
		int gridsize = (numberOfKeys + threadblocksize - 1) / threadblocksize;
		Kernel_KeysToPoints << <gridsize, threadblocksize, 0, st >> > (exeInfo, keys, numberOfKeys, input_points, scale);
#else
		Kernel_KeysToPoints(exeInfo, keys, numberOfKeys, input_points, scale);
#endif
	}

	checkCudaSync(st);

	qDebug("Kernel_KeysToPoints Done");

#ifndef BUILD_FOR_CPU
	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_KeepAliveVoxelDatasUsingPointArray, 0, 0));
	int gridsize = (numberOfKeys + threadblocksize - 1) / threadblocksize;
	Kernel_KeepAliveVoxelDatasUsingPointArray << <gridsize, threadblocksize, 0, st >> > (
#else
	Kernel_KeepAliveVoxelDatasUsingPointArray(
#endif
		exeInfo,
		globalScanAreaAABB,
		_voxelValues,
		_voxelValueCounts,
		_voxelNormals,
		_voxelColors,
		_voxelColorScores,
		_voxelSegmentations,
		_voxelExtraAttribs,
		input_points,
		numberOfKeys,
		radius);
	checkCudaErrors(cudaGetLastError());

	checkCudaSync(st);

	if (deleteTheRest)
	{
#ifndef BUILD_FOR_CPU
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_DeleteTheRest, 0, 0));
		int gridsize = (exeInfo.globalHashInfo_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;
		Kernel_DeleteTheRest << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_DeleteTheRest(
#endif
			exeInfo,
			globalScanAreaAABB,
			_voxelValues,
			_voxelValueCounts,
			_voxelNormals,
			_voxelColors,
			_voxelColorScores,
			_voxelSegmentations,
			_voxelExtraAttribs,
			input_points,
			numberOfKeys);
		checkCudaErrors(cudaGetLastError());
		checkCudaSync(st);
	}
#ifndef BUILD_FOR_CPU		
	checkCudaErrors(cudaFreeAsync(input_points, st));
#else
	delete[] input_points;
#endif
	//SaveVoxelValues(pSettings->GetResourcesFolderPath() + "\\Debug\\Temp\\VoxelValues_hole.ply", alloc_, st);
}

#ifndef BUILD_FOR_CPU
__global__ void Kernel_SaveAliveVoxels(
	MarchingCubes::ExecutionInfo exeInfo,
	voxel_value_t * input_voxelValues,
	unsigned short* input_voxelValueCounts,
	Eigen::Vector3f * input_voxelNormals,
	Eigen::Vector3b * input_voxelColors,
	float* input_voxelColorScores,
	char* input_voxelSegmentations,

	Eigen::Vector3f * output_points,
	Eigen::Vector3f * output_normals,
	Eigen::Vector3b * output_colors,
	size_t * output_count)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;

	auto key = exeInfo.globalHash[threadid];
	//auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
	if (key.Exists() && (7 == exeInfo.globalHash[threadid].value))
	{
		auto xGlobalIndex = key.x;
		auto yGlobalIndex = key.y;
		auto zGlobalIndex = key.z;

		auto index = atomicAdd(output_count, 1);
		output_points[index] = exeInfo.global.GetGlobalPosition(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		output_normals[index] = input_voxelNormals[threadid];
		output_colors[index] = input_voxelColors[threadid];
	}
}

void MarchingCubes::SaveAliveVoxels(cached_allocator * alloc_, CUstream_st * st)
{
	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc

	Eigen::Vector3f* output_points = nullptr;
	cudaMalloc(&output_points, sizeof(Eigen::Vector3f) * exeInfo.globalHashInfo_host->HashTableCapacity);
	cudaMemsetAsync(&output_points, 0, sizeof(Eigen::Vector3f) * exeInfo.globalHashInfo_host->HashTableCapacity, st);

	Eigen::Vector3f* output_normals = nullptr;
	cudaMalloc(&output_normals, sizeof(Eigen::Vector3f) * exeInfo.globalHashInfo_host->HashTableCapacity);
	cudaMemsetAsync(&output_normals, 0, sizeof(Eigen::Vector3f) * exeInfo.globalHashInfo_host->HashTableCapacity, st);

	Eigen::Vector3b* output_colors = nullptr;
	cudaMalloc(&output_colors, sizeof(Eigen::Vector3b) * exeInfo.globalHashInfo_host->HashTableCapacity);
	cudaMemsetAsync(&output_colors, 0, sizeof(Eigen::Vector3b) * exeInfo.globalHashInfo_host->HashTableCapacity, st);

	Eigen::size_t* output_count = nullptr;
	cudaMalloc(&output_count, sizeof(size_t));
	cudaMemsetAsync(&output_count, 0, sizeof(size_t), st);

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_SaveAliveVoxels, 0, 0));
	int gridsize = (exeInfo.globalHashInfo_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;
	Kernel_SaveAliveVoxels << <gridsize, threadblocksize, 0, st >> > (
		exeInfo,
		_voxelValues,
		_voxelValueCounts,
		_voxelNormals,
		_voxelColors,
		_voxelColorScores,
		_voxelSegmentations,
		output_points,
		output_normals,
		output_colors,
		output_count);

	checkCudaSync(st);

	size_t host_output_count = 0;
	cudaMemcpyAsync(&host_output_count, output_count, sizeof(size_t), cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	Eigen::Vector3f* host_output_points = new Eigen::Vector3f[host_output_count];
	cudaMemcpyAsync(host_output_points, output_points, sizeof(Eigen::Vector3f) * host_output_count, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_output_normals = new Eigen::Vector3f[host_output_count];
	cudaMemcpyAsync(host_output_normals, output_normals, sizeof(Eigen::Vector3f) * host_output_count, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* host_output_colors = new Eigen::Vector3b[host_output_count];
	cudaMemcpyAsync(host_output_colors, output_colors, sizeof(Eigen::Vector3b) * host_output_count, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	cudaFreeAsync(output_points, st);
	cudaFreeAsync(output_normals, st);
	cudaFreeAsync(output_colors, st);
	cudaFreeAsync(output_count, st);

	PLYFormat ply;
	for (size_t i = 0; i < host_output_count; i++)
	{
		auto& p = host_output_points[i];
		auto& n = host_output_normals[i];
		auto& c = host_output_colors[i];

		ply.AddPointFloat3(p.data());
		ply.AddNormalFloat3(n.data());
		ply.AddColor(c.x(), c.y(), c.z());
	}
	ply.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\HD\\AliveVoxels.ply");

	delete[] host_output_points;
	delete[] host_output_normals;
	delete[] host_output_colors;
}

void MarchingCubes::Test_KeepAliveVoxelDatasUsingPointArray(HMesh & View_VTX, cached_allocator * alloc_, CUstream_st * st)
{
	//SavePointsForRendering(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\RenderedPoints.ply", alloc_, st);

	size_t host_View_vtxSize = View_VTX.vertexCount;
	//	mscho	@20240530
//	cudaMallocAsync => cudaMalloc
	Eigen::Vector3f* d_points;
	cudaMalloc(&d_points, sizeof(Eigen::Vector3f) * host_View_vtxSize);
	cudaMemsetAsync(d_points, 0, sizeof(Eigen::Vector3f) * host_View_vtxSize, st);

	size_t* d_count;
	cudaMalloc(&d_count, sizeof(size_t));
	cudaMemsetAsync(d_count, 0, sizeof(size_t), st);

	GetPointsNotInCylinder(
		View_VTX,
		Eigen::Vector3f(0.0f, 0.0f, 0.0f),
		5.0f,
		d_points,
		d_count,
		alloc_, st);

	checkCudaSync(st);

	size_t host_pointCount;
	cudaMemcpyAsync(&host_pointCount, d_count, sizeof(size_t), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	Eigen::Vector3f* host_points = new Eigen::Vector3f[host_pointCount];
	cudaMemcpyAsync(host_points, d_points, sizeof(Eigen::Vector3f) * host_pointCount, cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	for (size_t i = 0; i < host_pointCount; i++)
	{
		auto& p = host_points[i];
	}

	KeepAliveVoxelDatasUsingPointArray(
		host_points,
		host_pointCount,
		1.0f,
		alloc_, st);

	SaveAliveVoxels(alloc_, st);
	delete[] host_points;
}

__global__ void Kernel_ClearVoxelHashValue(MarchingCubes::ExecutionInfo exeInfo)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;
	exeInfo.globalHash[threadid].value = 0;
}
#endif

__global__ void Kernel_ApplyEditResult(
	MarchingCubes::ExecutionInfo exeInfo,
	//Eigen::AlignedBox3f globalScanAreaAABB,
	//voxel_value_t* voxelValues,
	//unsigned short* voxelValueCounts,
	//Eigen::Vector3f* voxelNormals,
	//Eigen::Vector3b* voxelColors,
	//float* voxelColorScores,
	//char* voxelSegmentations,
	//unsigned short* voxelStartPatchIDs,
	Eigen::Vector3f * inputPoints,
	int* inputPointsFlags,
	size_t numberOfInputPoints,
	float aliveRadius, float deletedRadius, float pointMag)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > (unsigned int)numberOfInputPoints - 1) return;
	{
#else
	for (int threadId = 0; threadId < int(numberOfInputPoints); threadId++) {
#endif

		auto point = inputPoints[threadId];
		point.x() *= pointMag;
		point.y() *= pointMag;
		point.z() *= pointMag;

		auto xGlobalIndex = (size_t)floorf(point.x() / exeInfo.global.voxelSize - exeInfo.global.globalMin.x() / exeInfo.global.voxelSize);
		auto yGlobalIndex = (size_t)floorf(point.y() / exeInfo.global.voxelSize - exeInfo.global.globalMin.y() / exeInfo.global.voxelSize);
		auto zGlobalIndex = (size_t)floorf(point.z() / exeInfo.global.voxelSize - exeInfo.global.globalMin.z() / exeInfo.global.voxelSize);

		HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
		auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);

		if (i != kEmpty32)
		{
			if (inputPointsFlags[threadId] != 0)
			{
				if (4 == exeInfo.globalHash[i].value)
				{
					exeInfo.globalHash[i].value = 6;
				}
				else if (6 == exeInfo.globalHash[i].value)
				{
					exeInfo.globalHash[i].value = 6;
				}
				else
				{
					exeInfo.globalHash[i].value = 7;
				}

				size_t range = aliveRadius < exeInfo.global.voxelSize ? 1 : (size_t)ceilf(aliveRadius / exeInfo.global.voxelSize);

				for (size_t zIndex = zGlobalIndex - range; zIndex <= zGlobalIndex + range; zIndex++)
				{
					if (zIndex == zGlobalIndex || zIndex > exeInfo.global.voxelCountZ) continue;

					for (size_t yIndex = yGlobalIndex - range; yIndex <= yGlobalIndex + range; yIndex++)
					{
						if (yIndex == yGlobalIndex || yIndex > exeInfo.global.voxelCountY) continue;

						for (size_t xIndex = xGlobalIndex - range; xIndex <= xGlobalIndex + range; xIndex++)
						{
							if (xIndex == xGlobalIndex || xIndex > exeInfo.global.voxelCountX) continue;

							auto dx = (float)xIndex - (float)xGlobalIndex;
							auto dy = (float)yIndex - (float)yGlobalIndex;
							auto dz = (float)zIndex - (float)zGlobalIndex;
							auto ddx = dx * dx;
							auto ddy = dy * dy;
							auto ddz = dz * dz;
							if (sqrtf(ddx + ddy + ddz) * exeInfo.global.voxelSize <= aliveRadius)
							{
								HashKey nkey(xIndex, yIndex, zIndex);
								auto ni = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(nkey);
								if (ni != kEmpty32)
								{
									//if (7 == exeInfo.globalHash[ni].value)
									//{
									//	exeInfo.globalHash[ni].value = 7;
									//}
									//else if (4 == exeInfo.globalHash[ni].value)
									//{
									//	exeInfo.globalHash[ni].value = 6;
									//}
									//else if (6 == exeInfo.globalHash[ni].value)
									//{
									//	exeInfo.globalHash[ni].value = 6;
									//}
									//else
									//{
									//	exeInfo.globalHash[ni].value = 4;
									//}

									if (4 == exeInfo.globalHash[ni].value)
									{
										exeInfo.globalHash[ni].value = 6;
									}
									else if (6 == exeInfo.globalHash[ni].value)
									{
										exeInfo.globalHash[ni].value = 6;
									}
									else
									{
										exeInfo.globalHash[ni].value = 7;
									}
								}
							}
						}
					}
				}
			}
			else
			{
				if (7 == exeInfo.globalHash[i].value)
				{
					exeInfo.globalHash[i].value = 6;
				}
				else
				{
					exeInfo.globalHash[i].value = 4;
				}

				size_t range = deletedRadius < exeInfo.global.voxelSize ? 1 : (size_t)ceilf(deletedRadius / exeInfo.global.voxelSize);

				for (size_t zIndex = zGlobalIndex - range; zIndex <= zGlobalIndex + range; zIndex++)
				{
					if (zIndex == zGlobalIndex || zIndex > exeInfo.global.voxelCountZ) continue;

					for (size_t yIndex = yGlobalIndex - range; yIndex <= yGlobalIndex + range; yIndex++)
					{
						if (yIndex == yGlobalIndex || yIndex > exeInfo.global.voxelCountY) continue;

						for (size_t xIndex = xGlobalIndex - range; xIndex <= xGlobalIndex + range; xIndex++)
						{
							if (xIndex == xGlobalIndex || xIndex > exeInfo.global.voxelCountX) continue;

							auto dx = (float)xIndex - (float)xGlobalIndex;
							auto dy = (float)yIndex - (float)yGlobalIndex;
							auto dz = (float)zIndex - (float)zGlobalIndex;
							auto ddx = dx * dx;
							auto ddy = dy * dy;
							auto ddz = dz * dz;
							if (sqrtf(ddx + ddy + ddz) * exeInfo.global.voxelSize <= deletedRadius)
							{
								HashKey nkey(xIndex, yIndex, zIndex);
								auto ni = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(nkey);
								if (ni != kEmpty32)
								{
									if (7 == exeInfo.globalHash[ni].value)
									{
										exeInfo.globalHash[ni].value = 6;
									}
									else if (6 == exeInfo.globalHash[ni].value)
									{
										exeInfo.globalHash[ni].value = 6;
									}
									else
									{
										exeInfo.globalHash[ni].value = 4;
									}
								}
							}
						}
					}
				}
			}
		}
	}
	}

__global__ void Kernel_SetVoxelExtraAttrib(
	MarchingCubes::ExecutionInfo exeInfo,
	const Eigen::Vector3f * inputPoints,
	const VoxelExtraAttrib * extraAttribs,
	size_t pointCount, float pointMag,
	VoxelExtraAttrib * voxelExtraAttribs)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId > (unsigned int)pointCount - 1)
		return;

	{
#else
	for (int threadId = 0; threadId < int(pointCount); threadId++) {
#endif
		Eigen::Vector3f point = inputPoints[threadId] * pointMag;
		uint32_t iVoxel = exeInfo.LookupHashIndex(point);
		if (iVoxel != kEmpty32) {
			voxelExtraAttribs[iVoxel] = extraAttribs[threadId];
		}
	}
	}

__global__ void Kernel_RemoveVoxelDataByEditResult(
	MarchingCubes::ExecutionInfo exeInfo,
	Eigen::AlignedBox3f globalScanAreaAABB,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	Eigen::Vector3f * voxelNormals,
	Eigen::Vector3b * voxelColors,
	float* voxelColorScores,
	char* voxelSegmentations,
	VoxelExtraAttrib * voxelExtraAttribs)
{
#ifndef BUILD_FOR_CPU
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;
	{
#else
	for (int threadid = 0; threadid < exeInfo.globalHashInfo->HashTableCapacity; threadid++) {
#endif

		auto& key = exeInfo.globalHash[threadid];
		if (key.Exists())
		{
			//if (7 != exeInfo.globalHash[threadid].value)
			//{
			//	voxelValues[threadid] = VOXEL_INVALID;
			//	voxelValueCounts[threadid] = SHRT_MAX;
			//	voxelNormals[threadid] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
			//	voxelColors[threadid] = Eigen::Vector3b(0, 0, 0);
			//	voxelColorScores[threadid] = 0;
			//	voxelStartPatchIDs[threadid] = 0;
			//}
			//else
			//{
			//	exeInfo.globalHash[threadid].value = 1;
			//}


			if (1 == exeInfo.globalHash[threadid].value)
			{
				voxelValues[threadid] = VOXEL_INVALID;
				voxelValueCounts[threadid] = SHRT_MAX;
				voxelNormals[threadid] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				voxelColors[threadid] = Eigen::Vector3b(0, 0, 0);
				voxelColorScores[threadid] = 0;
				voxelExtraAttribs[threadid] = { 0, };
			}
			else if (4 == exeInfo.globalHash[threadid].value)
			{
				voxelValues[threadid] = VOXEL_INVALID;
				voxelValueCounts[threadid] = SHRT_MAX;
				voxelNormals[threadid] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				voxelColors[threadid] = Eigen::Vector3b(0, 0, 0);
				voxelColorScores[threadid] = 0;
				voxelExtraAttribs[threadid] = { 0, };
			}
			else if (6 == exeInfo.globalHash[threadid].value)
			{
				voxelValues[threadid] = VOXEL_INVALID;
				voxelValueCounts[threadid] = SHRT_MAX;
				voxelNormals[threadid] = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
				voxelColors[threadid] = Eigen::Vector3b(0, 0, 0);
				voxelColorScores[threadid] = 0;
				voxelExtraAttribs[threadid] = { 0, };
			}
			else if (7 == exeInfo.globalHash[threadid].value)
			{
				exeInfo.globalHash[threadid].value = 1;
			}
			else
			{
				//printf("?????????? : %d\n"[threadid]);
				voxelValueCounts[threadid] = 0;
			}
		}
	}
	}

// Pause 상태의 사용자 편집(Trimming, Lock) 내용을 복셀데이터에 반영한다.
void MarchingCubes::ApplyEditResult(
	Eigen::Vector3f * inputPoints,
	int* inputPointsFlags,
	size_t numberOfInputPoints,
	float aliveRadius, float deletedRadius, float pointMag,
	cached_allocator * alloc_, CUstream_st * st, bool isHtoD)
{
	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);

	Eigen::Vector3f* input_points = nullptr;
	int* input_points_flags = nullptr;
#ifndef BUILD_FOR_CPU
	if (isHtoD)
	{
		cudaMalloc(&input_points, sizeof(Eigen::Vector3f) * numberOfInputPoints);
		cudaMemcpyAsync(input_points, inputPoints, sizeof(Eigen::Vector3f) * numberOfInputPoints, cudaMemcpyHostToDevice, st);

		cudaMalloc(&input_points_flags, sizeof(int) * numberOfInputPoints);
		cudaMemcpyAsync(input_points_flags, inputPointsFlags, sizeof(int) * numberOfInputPoints, cudaMemcpyHostToDevice, st);
	}
	else
#endif
	{
		input_points = inputPoints;
		input_points_flags = inputPointsFlags;
	}

	checkCudaSync(st);

	int mingridsize;
	int threadblocksize;

	if (numberOfInputPoints) {
		//{
		//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ClearVoxelHashValue, 0, 0));
		//	int gridsize = (numberOfInputPoints + threadblocksize - 1) / threadblocksize;
		//	Kernel_ClearVoxelHashValue << <gridsize, threadblocksize, 0, st >> > (exeInfo);
		//	checkCudaErrors(cudaGetLastError());
		//	checkCudaSync(st);
		//}

		{
#ifndef BUILD_FOR_CPU
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ApplyEditResult, 0, 0));
			int gridsize = (numberOfInputPoints + threadblocksize - 1) / threadblocksize;
			Kernel_ApplyEditResult << <gridsize, threadblocksize, 0, st >> > (
#else
			Kernel_ApplyEditResult(
#endif
				exeInfo,
				input_points,
				input_points_flags,
				numberOfInputPoints,
				aliveRadius, deletedRadius, pointMag);
			checkCudaErrors(cudaGetLastError());
			checkCudaSync(st);
		}

		{
#ifndef BUILD_FOR_CPU
			checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_RemoveVoxelDataByEditResult, 0, 0));
			int gridsize = (exeInfo.globalHashInfo_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;
			Kernel_RemoveVoxelDataByEditResult << <gridsize, threadblocksize, 0, st >> > (
#else
			Kernel_RemoveVoxelDataByEditResult(
#endif
				exeInfo,
				globalScanAreaAABB,
				_voxelValues,
				_voxelValueCounts,
				_voxelNormals,
				_voxelColors,
				_voxelColorScores,
				_voxelSegmentations,
				_voxelExtraAttribs);
			checkCudaErrors(cudaGetLastError());
			checkCudaSync(st);
		}
	}

#ifndef BUILD_FOR_CPU
	if (isHtoD)
	{
		cudaFree(input_points);
		cudaFree(input_points_flags);
	}
#endif

	//SaveVoxelValues(pSettings->GetResourcesFolderPath() + "\\Debug\\VoxelValues_ApplyEditResult.ply", alloc_, st);
}

__global__ void Kernel_RemapVoxelStartPatchID(
	MarchingCubes::ExecutionInfo exeInfo,
	VoxelExtraAttrib * voxelExtraAttribs,
	const ushort * devicePatchIndexMap, size_t patchIndexMapSize) {

	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid < exeInfo.globalHashInfo->HashTableCapacity)
	{
		HashEntry iVoxel = exeInfo.globalHash[threadid];
		if (iVoxel.Exists() && exeInfo.globalHash[threadid].value != kEmpty8)
		{
			unsigned short startPatchID = voxelExtraAttribs[threadid].startPatchID;
			if (startPatchID < patchIndexMapSize)
				voxelExtraAttribs[threadid].startPatchID = devicePatchIndexMap[startPatchID];
		}
	}
}

void MarchingCubes::RemapVoxelStartPatchID(const ushort * devicePatchIndexMap, size_t patchIndexMapSize, CUstream_st * st) {
	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_RemapVoxelStartPatchID, 0, 0));
	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	Kernel_RemapVoxelStartPatchID << <gridsize, threadblocksize, 0, st >> > (
		exeInfo,
		thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data()),
		devicePatchIndexMap, patchIndexMapSize);

	checkCudaErrors(cudaGetLastError());
}

void MarchingCubes::SetVoxelExtraAttribs(const Eigen::Vector3f * points, const VoxelExtraAttrib * extraAttribs, size_t pointCount, float pointMag, cached_allocator * alloc_, CUstream_st * st)
{
	if (pointCount) {
		thrust::device_vector<Eigen::Vector3f> device_points(points, points + pointCount);
		thrust::device_vector<VoxelExtraAttrib> device_extraAttribs(extraAttribs, extraAttribs + pointCount);

#ifndef BUILD_FOR_CPU
		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_ApplyEditResult, 0, 0));
		int gridsize = (pointCount + threadblocksize - 1) / threadblocksize;
		Kernel_SetVoxelExtraAttrib << <gridsize, threadblocksize, 0, st >> > (
#else
		Kernel_SetVoxelExtraAttrib(
#endif
			exeInfo,
			thrust::raw_pointer_cast(device_points.data()),
			thrust::raw_pointer_cast(device_extraAttribs.data()),
			pointCount, pointMag,
			thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data()));
		checkCudaErrors(cudaGetLastError());
		checkCudaSync(st);
	}
}

#ifndef BUILD_FOR_CPU
__global__ void Kernel_SaveVolume(
	MarchingCubes::ExecutionInfo exeInfo,
	const Eigen::Vector3f * normals,
	const Eigen::Vector3b * colors,
	const voxel_value_t * voxelValues,
	const unsigned short* voxelValueCounts,
	const float* voxelColorScores,
	const char* voxelSegmentations,
	const VoxelExtraAttrib * voxelExtraAttribs,

	Eigen::Vector3f * output_points,
	Eigen::Vector3f * output_normals,
	Eigen::Vector3b * output_colors,
	voxel_value_t * output_voxelValues,
	unsigned short* output_voxelValueCounts,
	float* output_voxelColorScores,
	char* output_voxelSegmentations,
	VoxelExtraAttrib * output_voxelExtraAttribs,
	size_t * outputCount)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;

	auto key = exeInfo.globalHash[threadid];
	auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
	//auto i = threadid;
	if (i != kEmpty32)
	{
		auto vv = voxelValues[i];
		if (VOXEL_INVALID != vv)
		{
			auto x_idx = key.x;
			auto y_idx = key.y;
			auto z_idx = key.z;

			auto x = (float)x_idx * exeInfo.global.voxelSize + exeInfo.global.voxelSize * 0.5f + exeInfo.global.globalMin.x();
			auto y = (float)y_idx * exeInfo.global.voxelSize + exeInfo.global.voxelSize * 0.5f + exeInfo.global.globalMin.y();
			auto z = (float)z_idx * exeInfo.global.voxelSize + exeInfo.global.voxelSize * 0.5f + exeInfo.global.globalMin.z();

			auto index = atomicAdd(outputCount, 1);

			output_points[index] = Eigen::Vector3f(x, y, z);
			output_normals[index] = normals[i];
			output_colors[index] = colors[i];
			output_voxelValues[index] = voxelValues[i];
			output_voxelValueCounts[index] = voxelValueCounts[i];
			output_voxelColorScores[index] = voxelColorScores[i];
			output_voxelSegmentations[index] = voxelSegmentations[i];
			output_voxelExtraAttribs[index] = voxelExtraAttribs[i];
		}
	}
}

void MarchingCubes::SerializeVolume(const string & filename, cached_allocator * alloc_, CUstream_st * st)
{
	auto HashTableCapacity = exeInfo.globalHashInfo_host->HashTableCapacity;

	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	Eigen::Vector3f* output_points;
	cudaMalloc(&output_points, sizeof(Eigen::Vector3f) * HashTableCapacity);

	Eigen::Vector3f* output_normals;
	cudaMalloc(&output_normals, sizeof(Eigen::Vector3f) * HashTableCapacity);

	Eigen::Vector3b* output_colors;
	cudaMalloc(&output_colors, sizeof(Eigen::Vector3b) * HashTableCapacity);

	voxel_value_t* output_voxelValues;
	cudaMalloc(&output_voxelValues, sizeof(voxel_value_t) * HashTableCapacity);

	unsigned short* output_voxelValueCounts;
	cudaMalloc(&output_voxelValueCounts, sizeof(unsigned short) * HashTableCapacity);

	float* output_voxelColorScores;
	cudaMalloc(&output_voxelColorScores, sizeof(float) * HashTableCapacity);

	char* output_voxelSegmentations;
	cudaMalloc(&output_voxelSegmentations, sizeof(char) * HashTableCapacity);

	VoxelExtraAttrib* output_voxelExtraAttribs;
	cudaMalloc(&output_voxelExtraAttribs, sizeof(VoxelExtraAttrib) * HashTableCapacity);

	size_t* outputCount;
	cudaMalloc(&outputCount, sizeof(size_t));
	cudaMemsetAsync(outputCount, 0, sizeof(size_t), st);

	checkCudaSync(st);

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_SaveVolume, 0, 0));
	int gridsize = (exeInfo.globalHashInfo_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;
	Kernel_SaveVolume << <gridsize, threadblocksize, 0, st >> > (
		exeInfo,
		thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
		thrust::raw_pointer_cast(m_MC_voxelColors.data()),
		thrust::raw_pointer_cast(m_MC_voxelValues.data()),
		thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
		thrust::raw_pointer_cast(m_MC_voxelColorScores.data()),
		thrust::raw_pointer_cast(m_MC_voxelSegmentations.data()),
		thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data()),
		output_points,
		output_normals,
		output_colors,
		output_voxelValues,
		output_voxelValueCounts,
		output_voxelColorScores,
		output_voxelSegmentations,
		output_voxelExtraAttribs,
		outputCount);

	checkCudaSync(st);

	size_t host_outputCount;
	cudaMemcpyAsync(&host_outputCount, outputCount, sizeof(size_t), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	Eigen::Vector3f* points = new Eigen::Vector3f[host_outputCount];
	cudaMemcpyAsync(points, output_points, sizeof(Eigen::Vector3f) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	//Eigen::AlignedBox3f aabb(Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX), Eigen::Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX));
	Eigen::AlignedBox3f aabb;
	for (size_t i = 0; i < host_outputCount; i++)
	{
		aabb.extend(points[i]);
	}

	Eigen::Vector3f* normals = new Eigen::Vector3f[host_outputCount];
	cudaMemcpyAsync(normals, output_normals, sizeof(Eigen::Vector3f) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* colors = new Eigen::Vector3b[host_outputCount];
	cudaMemcpyAsync(colors, output_colors, sizeof(Eigen::Vector3b) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	voxel_value_t* voxelValues = new voxel_value_t[host_outputCount];
	cudaMemcpyAsync(voxelValues, output_voxelValues, sizeof(voxel_value_t) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	unsigned short* voxelValueCounts = new unsigned short[host_outputCount];
	cudaMemcpyAsync(voxelValueCounts, output_voxelValueCounts, sizeof(unsigned short) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	float* voxelColorScores = new float[host_outputCount];
	cudaMemcpyAsync(
		voxelColorScores, output_voxelColorScores, sizeof(float) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	VoxelExtraAttrib* voxelExtraAttribs = new VoxelExtraAttrib[host_outputCount];
	cudaMemcpyAsync(
		voxelExtraAttribs, output_voxelExtraAttribs, sizeof(VoxelExtraAttrib) * host_outputCount,
		cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	FILE* fp = nullptr;
	if (0 != fopen_s(&fp, filename.c_str(), "wb"))
	{
		qDebug("Error occured while opening %s.", filename.c_str());
		return;
	}

	fwrite(&host_outputCount, sizeof(uint32_t), 1, fp);
	fwrite(points, sizeof(Eigen::Vector3f) * host_outputCount, 1, fp);
	fwrite(normals, sizeof(Eigen::Vector3f) * host_outputCount, 1, fp);
	fwrite(colors, sizeof(Eigen::Vector3b) * host_outputCount, 1, fp);
	fwrite(voxelValues, sizeof(voxel_value_t) * host_outputCount, 1, fp);
	fwrite(voxelValueCounts, sizeof(unsigned short) * host_outputCount, 1, fp);
	fwrite(voxelColorScores, sizeof(float) * host_outputCount, 1, fp);
	fwrite(voxelExtraAttribs, sizeof(VoxelExtraAttrib) * host_outputCount, 1, fp);

	Eigen::Vector3f min_corner = aabb.min();
	Eigen::Vector3f max_corner = aabb.max();
	float corners[6];
	corners[0] = min_corner.x();
	corners[1] = min_corner.y();
	corners[2] = min_corner.z();
	corners[3] = max_corner.x();
	corners[4] = max_corner.y();
	corners[5] = max_corner.z();

	fwrite(corners, sizeof(float) * 6, 1, fp);

	fclose(fp);

	delete[] points;
	delete[] normals;
	delete[] colors;
	delete[] voxelValues;
	delete[] voxelValueCounts;
	delete[] voxelColorScores;
	delete[] voxelExtraAttribs;

	cudaFree(output_points);
	cudaFree(output_normals);
	cudaFree(output_colors);
	cudaFree(output_voxelValues);
	cudaFree(output_voxelValueCounts);
	cudaFree(output_voxelColorScores);
	cudaFree(output_voxelExtraAttribs);
	cudaFree(outputCount);
}

__global__ void Kernel_LoadVolume(
	MarchingCubes::ExecutionInfo exeInfo,
	Eigen::AlignedBox3f globalScanAreaAABB,
	Eigen::Vector3f * input_points,
	Eigen::Vector3f * input_normals,
	Eigen::Vector3b * input_colors,
	voxel_value_t * input_voxelValues,
	unsigned short* input_voxelValueCounts,
	float* input_voxelColorScores,
	char* input_voxelSegmentations,
	VoxelExtraAttrib * input_voxelExtraAttribs,

	Eigen::Vector3f * points,
	Eigen::Vector3f * normals,
	Eigen::Vector3b * colors,
	voxel_value_t * voxelValues,
	unsigned short* voxelValueCounts,
	float* voxelColorScores,
	char* voxelSegmentations,
	VoxelExtraAttrib * voxelExtraAttribs)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->Count_HashTableUsed - 1) return;

	size_t index = threadid;
	auto point = input_points[index];

	auto xGlobalIndex = (size_t)floorf(point.x() / exeInfo.global.voxelSize - globalScanAreaAABB.min().x() / exeInfo.global.voxelSize);
	auto yGlobalIndex = (size_t)floorf(point.y() / exeInfo.global.voxelSize - globalScanAreaAABB.min().y() / exeInfo.global.voxelSize);
	auto zGlobalIndex = (size_t)floorf(point.z() / exeInfo.global.voxelSize - globalScanAreaAABB.min().z() / exeInfo.global.voxelSize);

	HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
	auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
	if (i == kEmpty32)
	{
		i = exeInfo.globalHashInfo->get_insert_idx_func64_v4(key);

		points[i] = point;
		normals[i] = input_normals[index];
		colors[i] = input_colors[index];
		voxelValues[i] = input_voxelValues[index];
		voxelValueCounts[i] = input_voxelValueCounts[index];
		voxelColorScores[i] = input_voxelColorScores[index];
		voxelSegmentations[i] = input_voxelSegmentations[index];
		voxelExtraAttribs[i] = input_voxelExtraAttribs[index];
	}
}

void MarchingCubes::DeserializeVolume(const string & filename, cached_allocator * alloc_, CUstream_st * st)
{
	FILE* fp = nullptr;
	if (0 != fopen_s(&fp, filename.c_str(), "rb"))
	{
		qDebug("Error occured while opening %s.", filename.c_str());
		return;
	}

	uint32_t host_Count_HashTableUsed = 0;
	fread(&host_Count_HashTableUsed, sizeof(uint32_t), 1, fp);

	Eigen::Vector3f* points = new Eigen::Vector3f[host_Count_HashTableUsed];
	fread(points, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed, 1, fp);

	Eigen::Vector3f* normals = new Eigen::Vector3f[host_Count_HashTableUsed];
	fread(normals, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed, 1, fp);

	Eigen::Vector3b* colors = new Eigen::Vector3b[host_Count_HashTableUsed];
	fread(colors, sizeof(Eigen::Vector3b) * host_Count_HashTableUsed, 1, fp);

	voxel_value_t* voxelValues = new voxel_value_t[host_Count_HashTableUsed];
	fread(voxelValues, sizeof(voxel_value_t) * host_Count_HashTableUsed, 1, fp);

	unsigned short* voxelValueCounts = new unsigned short[host_Count_HashTableUsed];
	fread(voxelValueCounts, sizeof(unsigned short) * host_Count_HashTableUsed, 1, fp);

	float* voxelColorScores = new float[host_Count_HashTableUsed];
	fread(voxelColorScores, sizeof(float) * host_Count_HashTableUsed, 1, fp);

	char* voxelSegmentations = new char[host_Count_HashTableUsed];
	fread(voxelSegmentations, sizeof(char) * host_Count_HashTableUsed, 1, fp);
	VoxelExtraAttrib* voxelExtraAttribs = new VoxelExtraAttrib[host_Count_HashTableUsed];
	fread(voxelExtraAttribs, sizeof(VoxelExtraAttrib) * host_Count_HashTableUsed, 1, fp);

	float corners[6];
	fread(corners, sizeof(float) * 6, 1, fp);

	loadedVolumeAABB = Eigen::AlignedBox3f(
		Eigen::Vector3f(corners[0], corners[1], corners[2]),
		Eigen::Vector3f(corners[3], corners[4], corners[5]));

	qDebug("Volume\n - min : %f, %f, %f\n - max : %f, %f, %f",
		corners[0], corners[1], corners[2], corners[3], corners[4], corners[5]);

	//PLYFormat ply;
	//for (size_t i = 0; i < host_Count_HashTableUsed; i++)
	//{
	//	auto& p = points[i];
	//	auto& n = normals[i];
	//	auto& c = colors[i];

	//	ply.AddPointFloat3(p.data());
	//	ply.AddNormalFloat3(n.data());
	//	ply.AddColor(
	//		(float)c.x() / 255.0f,
	//		(float)c.y() / 255.0f,
	//		(float)c.z() / 255.0f);
	//}
	//ply.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\Capture\\Temp.ply");

	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);
	//	mscho	@20240530
	//	cudaMallocAsync => cudaMalloc
	Eigen::Vector3f* input_points;
	cudaMalloc(&input_points, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_points, points, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	Eigen::Vector3f* input_normals;
	cudaMalloc(&input_normals, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_normals, normals, sizeof(Eigen::Vector3f) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	Eigen::Vector3b* input_colors;
	cudaMalloc(&input_colors, sizeof(Eigen::Vector3b) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_colors, colors, sizeof(Eigen::Vector3b) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	voxel_value_t* input_voxelValues;
	cudaMalloc(&input_voxelValues, sizeof(voxel_value_t) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_voxelValues, voxelValues, sizeof(voxel_value_t) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	unsigned short* input_voxelValueCounts;
	cudaMalloc(&input_voxelValueCounts, sizeof(unsigned short) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_voxelValueCounts, voxelValueCounts, sizeof(unsigned short) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	float* input_voxelColorScores;
	cudaMalloc(&input_voxelColorScores, sizeof(float) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_voxelColorScores, voxelColorScores, sizeof(float) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	char* input_voxelSegmentations;
	cudaMalloc(&input_voxelSegmentations, sizeof(char) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_voxelSegmentations, voxelSegmentations, sizeof(char) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	VoxelExtraAttrib* input_voxelExtraAttribs;
	cudaMalloc(&input_voxelExtraAttribs, sizeof(VoxelExtraAttrib) * host_Count_HashTableUsed);
	cudaMemcpyAsync(input_voxelExtraAttribs, voxelExtraAttribs, sizeof(VoxelExtraAttrib) * host_Count_HashTableUsed, cudaMemcpyHostToDevice, st);

	checkCudaSync(st);

	InitGlobalVoxelValues();
	initialize_ExecutionInfo(alloc_, st);

	if (0 == m_MC_voxelPositions.size())
	{
		assert(false);
		//pRegistration->ResetVoxel(); // registration에 접근할 수 없으므로 일단 주석처리함
	}

	pHashManager->reset_hashKeyCountValue64(globalHash_info, st);

	qDebug("pRegistration->m_MC_voxelPositions.size() : %llu", m_MC_voxelPositions.size());

	int mingridsize;
	int threadblocksize;
	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_LoadVolume, 0, 0));
	int gridsize = (host_Count_HashTableUsed + threadblocksize - 1) / threadblocksize;
	Kernel_LoadVolume << <gridsize, threadblocksize, 0, st >> > (
		exeInfo,
		globalScanAreaAABB,
		input_points,
		input_normals,
		input_colors,
		input_voxelValues,
		input_voxelValueCounts,
		input_voxelColorScores,
		input_voxelSegmentations,
		input_voxelExtraAttribs,
		thrust::raw_pointer_cast(m_MC_voxelPositions.data()),
		thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
		thrust::raw_pointer_cast(m_MC_voxelColors.data()),
		thrust::raw_pointer_cast(m_MC_voxelValues.data()),
		thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
		thrust::raw_pointer_cast(m_MC_voxelColorScores.data()),
		thrust::raw_pointer_cast(m_MC_voxelSegmentations.data()),
		thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data()));

	checkCudaSync(st);

	cudaMemcpyAsync(&exeInfo.globalHashInfo->Count_HashTableUsed, &host_Count_HashTableUsed, sizeof(unsigned int), cudaMemcpyHostToDevice, st);

	fclose(fp);

	delete[] points;
	delete[] normals;
	delete[] colors;
	delete[] voxelValues;
	delete[] voxelValueCounts;
	delete[] voxelColorScores;
	delete[] voxelSegmentations;
	delete[] voxelExtraAttribs;
}

struct ExtractionEdge
{
	uint32_t startVoxelIndex = UINT32_MAX;
	uint32_t endVoxelIndex = UINT32_MAX;
	uint8_t edgeDirection = 0;
	bool zeroCrossing = false;
	//Eigen::Vector3f zeroCrossingPoint = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	uint32_t zeroCrossingPointIndex = UINT32_MAX;
	int neighborCount = 0;
};

struct ExtractionTriangle
{
	uint32_t edgeIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
	uint32_t vertexIndices[3] = { UINT32_MAX, UINT32_MAX, UINT32_MAX };
};

struct ExtractionVoxel
{
	uint32_t globalIndexX = UINT32_MAX;
	uint32_t globalIndexY = UINT32_MAX;
	uint32_t globalIndexZ = UINT32_MAX;

	Eigen::Vector3f position = Eigen::Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	Eigen::Vector3f normal = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	Eigen::Vector3b color = Eigen::Vector3b(0, 0, 0);
	float value = FLT_MAX;

	uint32_t edgeIndexX = UINT32_MAX;
	uint32_t edgeIndexY = UINT32_MAX;
	uint32_t edgeIndexZ = UINT32_MAX;

	ExtractionTriangle triangles[4] = { ExtractionTriangle(), ExtractionTriangle(), ExtractionTriangle(), ExtractionTriangle() };
	uint32_t numberOfTriangles = 0;
};

__global__ void Kernel_ExtractVolume(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	Eigen::Vector3f * resultPositions, Eigen::Vector3f * resultNormals, Eigen::Vector4f * resultColors,
	unsigned int* numberOfVoxelPositions, ExtractionVoxel * extractionVoxels, ExtractionEdge * extractionEdges)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity - 1) return;

	auto key = globalHash_info->hashtable[threadid];
	if ((key.Exists()) && (kEmpty8 != globalHash_info->hashtable[threadid].value))
	{
		auto vv = voxelValues[threadid];
		if (VOXEL_INVALID != vv)
		{
			auto v = VV2D(vv);

			auto x_idx = key.x;
			auto y_idx = key.y;
			auto z_idx = key.z;

			if (x_idx < exeInfo.local.GetGlobalIndexX(0)) return;
			if (x_idx > exeInfo.local.GetGlobalIndexX(exeInfo.local.voxelCountX)) return;
			if (y_idx < exeInfo.local.GetGlobalIndexY(0)) return;
			if (y_idx > exeInfo.local.GetGlobalIndexY(exeInfo.local.voxelCountY)) return;
			if (z_idx < exeInfo.local.GetGlobalIndexZ(0)) return;
			if (z_idx > exeInfo.local.GetGlobalIndexZ(exeInfo.local.voxelCountZ)) return;

			auto x = (float)x_idx * exeInfo.global.voxelSize
				+ exeInfo.global.voxelSize * 0.5f
				- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountX * 0.5f;
			auto y = (float)y_idx * exeInfo.global.voxelSize
				+ exeInfo.global.voxelSize * 0.5f
				- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountY * 0.5f;
			auto z = (float)z_idx * exeInfo.global.voxelSize
				+ exeInfo.global.voxelSize * 0.5f
				- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountZ * 0.5f;

			auto nx = x + exeInfo.global.voxelSize;
			auto ny = y + exeInfo.global.voxelSize;
			auto nz = z + exeInfo.global.voxelSize;

			MarchingCubes::GRIDCELL gridcell;
			gridcell.p[0] = Eigen::Vector3f(x, y, z);
			gridcell.p[1] = Eigen::Vector3f(nx, y, z);
			gridcell.p[2] = Eigen::Vector3f(nx, y, nz);
			gridcell.p[3] = Eigen::Vector3f(x, y, nz);
			gridcell.p[4] = Eigen::Vector3f(x, ny, z);
			gridcell.p[5] = Eigen::Vector3f(nx, ny, z);
			gridcell.p[6] = Eigen::Vector3f(nx, ny, nz);
			gridcell.p[7] = Eigen::Vector3f(x, ny, nz);

			uint32_t  hs_voxel_hashIdx_core;

			for (size_t idx = 0; idx < 8; idx++)
			{


				//	mscho	@20240214
				//auto xGlobalIndex = (size_t)(floorf((gridcell.p[idx].x() - globalMinX) / voxelSize));
				//auto yGlobalIndex = (size_t)(floorf((gridcell.p[idx].y() - globalMinY) / voxelSize));
				//auto zGlobalIndex = (size_t)(floorf((gridcell.p[idx].z() - globalMinZ) / voxelSize));

				//printf("----- %llu %llu %llu\n", xGlobalIndex, yGlobalIndex, zGlobalIndex);

				auto voxel_ = 10.f;

				auto xGlobalIndex = (size_t)floorf((gridcell.p[idx].x()) * voxel_ + 2500.f);
				auto yGlobalIndex = (size_t)floorf((gridcell.p[idx].y()) * voxel_ + 2500.f);
				auto zGlobalIndex = (size_t)floorf((gridcell.p[idx].z()) * voxel_ + 2500.f);



				HashKey key(xGlobalIndex, yGlobalIndex, zGlobalIndex);
				auto i = globalHash_info->get_lookup_idx_func64_v4(key);

				if (idx == 0)
				{
					hs_voxel_hashIdx_core = i;
					if (hs_voxel_hashIdx_core == kEmpty32)	return;
				}

				if (i != kEmpty32)
				{
					auto voxelValueCount = voxelValueCounts[i];

					//gridcell.val[idx] = VV2D(values[i]);
					gridcell.val[idx] = VV2D(voxelValues[i]) / (float)voxelValueCount;
					// 
					//printf("values[%d] : %d\tgridcell.val[%llu] : %f\n", i, values[i], idx, gridcell.val[idx]);
					//printf("GlobalIndex : %llu, %llu, %llu\tvalues[%d] : %d\t VV2D[%d] : %f\n", xGlobalIndex, yGlobalIndex, zGlobalIndex, i, values[i], values[i], VV2D(values[i]));
				}
				else
				{
					gridcell.val[idx] = FLT_MAX;
				}
			}

			auto oldIndex = atomicAdd(numberOfVoxelPositions, 1);
			resultPositions[oldIndex] = Eigen::Vector3f(x, y, z);

			resultNormals[oldIndex] = voxelNormals[threadid];

			auto r = (float)(voxelColors[threadid].x()) / 255.0f;
			auto g = (float)(voxelColors[threadid].y()) / 255.0f;
			auto b = (float)(voxelColors[threadid].z()) / 255.0f;
			auto a = v;
			if (a > 1.0f) a = 1.0f;
			if (a < -1.0f) a = -1.0f;
			a = a + 1.0f;
			a = (a / 2) * 255.0f;

			//r = a;
			//g = a;
			//b = a;

			if (v > 1.0f) v = 1.0f;
			if (v < -1.0f) v = -1.0f;

			r = r * (v + 1.0f) * 0.5f;
			g = 0.5 * g;
			b = b * (1.0f - (v + 1.0f) * 0.5f);
			a = 1.0f;

			resultColors[oldIndex] = Eigen::Vector4f(r, g, b, a);




			float isoValue = 0.0f;
			int cubeindex = 0;
			float isolevel = isoValue;
			Eigen::Vector3f vertlist[12];

			if (FLT_VALID(gridcell.val[0]) && gridcell.val[0] < isolevel) cubeindex |= 1;
			if (FLT_VALID(gridcell.val[1]) && gridcell.val[1] < isolevel) cubeindex |= 2;
			if (FLT_VALID(gridcell.val[2]) && gridcell.val[2] < isolevel) cubeindex |= 4;
			if (FLT_VALID(gridcell.val[3]) && gridcell.val[3] < isolevel) cubeindex |= 8;
			if (FLT_VALID(gridcell.val[4]) && gridcell.val[4] < isolevel) cubeindex |= 16;
			if (FLT_VALID(gridcell.val[5]) && gridcell.val[5] < isolevel) cubeindex |= 32;
			if (FLT_VALID(gridcell.val[6]) && gridcell.val[6] < isolevel) cubeindex |= 64;
			if (FLT_VALID(gridcell.val[7]) && gridcell.val[7] < isolevel) cubeindex |= 128;

			if (edgeTable[cubeindex] == 0)
			{
				return;
			}

			if (edgeTable[cubeindex] & 1)
				vertlist[0] =
				VertexInterp(isolevel, gridcell.p[0], gridcell.p[1], gridcell.val[0], gridcell.val[1]);
			if (edgeTable[cubeindex] & 2)
				vertlist[1] =
				VertexInterp(isolevel, gridcell.p[1], gridcell.p[2], gridcell.val[1], gridcell.val[2]);
			if (edgeTable[cubeindex] & 4)
				vertlist[2] =
				VertexInterp(isolevel, gridcell.p[2], gridcell.p[3], gridcell.val[2], gridcell.val[3]);
			if (edgeTable[cubeindex] & 8)
				vertlist[3] =
				VertexInterp(isolevel, gridcell.p[3], gridcell.p[0], gridcell.val[3], gridcell.val[0]);
			if (edgeTable[cubeindex] & 16)
				vertlist[4] =
				VertexInterp(isolevel, gridcell.p[4], gridcell.p[5], gridcell.val[4], gridcell.val[5]);
			if (edgeTable[cubeindex] & 32)
				vertlist[5] =
				VertexInterp(isolevel, gridcell.p[5], gridcell.p[6], gridcell.val[5], gridcell.val[6]);
			if (edgeTable[cubeindex] & 64)
				vertlist[6] =
				VertexInterp(isolevel, gridcell.p[6], gridcell.p[7], gridcell.val[6], gridcell.val[7]);
			if (edgeTable[cubeindex] & 128)
				vertlist[7] =
				VertexInterp(isolevel, gridcell.p[7], gridcell.p[4], gridcell.val[7], gridcell.val[4]);
			if (edgeTable[cubeindex] & 256)
				vertlist[8] =
				VertexInterp(isolevel, gridcell.p[0], gridcell.p[4], gridcell.val[0], gridcell.val[4]);
			if (edgeTable[cubeindex] & 512)
				vertlist[9] =
				VertexInterp(isolevel, gridcell.p[1], gridcell.p[5], gridcell.val[1], gridcell.val[5]);
			if (edgeTable[cubeindex] & 1024)
				vertlist[10] =
				VertexInterp(isolevel, gridcell.p[2], gridcell.p[6], gridcell.val[2], gridcell.val[6]);
			if (edgeTable[cubeindex] & 2048)
				vertlist[11] =
				VertexInterp(isolevel, gridcell.p[3], gridcell.p[7], gridcell.val[3], gridcell.val[7]);

			MarchingCubes::TRIANGLE tris[4];
			Eigen::Vector3f nm;
			int ntriang = 0;
			for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
				auto v0 = vertlist[triTable[cubeindex][i]];
				auto v1 = vertlist[triTable[cubeindex][i + 1]];
				auto v2 = vertlist[triTable[cubeindex][i + 2]];

				tris[ntriang].p[0] = v0;
				tris[ntriang].p[1] = v1;
				tris[ntriang].p[2] = v2;
				ntriang++;
			}

		}
	}
}

__global__ void Kernel_PopulateExtractionVoxels(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
	ExtractionVoxel * extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge * extractionEdges, unsigned int* numberOfExtractionEdges)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;

	auto key = exeInfo.globalHash[threadid];
	auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
	if (i == kEmpty32) return;

	auto vv = voxelValues[i];
	if (VOXEL_INVALID == vv) return;

	extractionVoxels[i].value = VV2D(vv);

	auto globalIndexX = key.x;
	auto globalIndexY = key.y;
	auto globalIndexZ = key.z;

	extractionVoxels[i].globalIndexX = globalIndexX;
	extractionVoxels[i].globalIndexY = globalIndexY;
	extractionVoxels[i].globalIndexZ = globalIndexZ;

	auto x = (float)globalIndexX * exeInfo.global.voxelSize
		+ exeInfo.global.voxelSize * 0.5f
		- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountX * 0.5f;
	auto y = (float)globalIndexY * exeInfo.global.voxelSize
		+ exeInfo.global.voxelSize * 0.5f
		- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountY * 0.5f;
	auto z = (float)globalIndexZ * exeInfo.global.voxelSize
		+ exeInfo.global.voxelSize * 0.5f
		- exeInfo.global.voxelSize * (float)exeInfo.global.voxelCountZ * 0.5f;

	extractionVoxels[i].position = Eigen::Vector3f(x, y, z);
	extractionVoxels[i].normal = voxelNormals[i];
	extractionVoxels[i].color = voxelColors[i];

	extractionVoxels[i].edgeIndexX = i * 3;
	extractionVoxels[i].edgeIndexY = i * 3 + 1;
	extractionVoxels[i].edgeIndexZ = i * 3 + 2;

	extractionVoxels[i].numberOfTriangles = 0;

	extractionVoxels[i].triangles[0].edgeIndices[0] = UINT32_MAX;
	extractionVoxels[i].triangles[0].edgeIndices[1] = UINT32_MAX;
	extractionVoxels[i].triangles[0].edgeIndices[2] = UINT32_MAX;

	extractionVoxels[i].triangles[1].edgeIndices[0] = UINT32_MAX;
	extractionVoxels[i].triangles[1].edgeIndices[1] = UINT32_MAX;
	extractionVoxels[i].triangles[1].edgeIndices[2] = UINT32_MAX;

	extractionVoxels[i].triangles[2].edgeIndices[0] = UINT32_MAX;
	extractionVoxels[i].triangles[2].edgeIndices[1] = UINT32_MAX;
	extractionVoxels[i].triangles[2].edgeIndices[2] = UINT32_MAX;

	extractionVoxels[i].triangles[3].edgeIndices[0] = UINT32_MAX;
	extractionVoxels[i].triangles[3].edgeIndices[1] = UINT32_MAX;
	extractionVoxels[i].triangles[3].edgeIndices[2] = UINT32_MAX;

	extractionEdges[i * 3 + 0].startVoxelIndex = i;
	extractionEdges[i * 3 + 0].edgeDirection = 0;
	extractionEdges[i * 3 + 0].zeroCrossing = false;
	extractionEdges[i * 3 + 0].zeroCrossingPointIndex = UINT32_MAX;
	extractionEdges[i * 3 + 1].startVoxelIndex = i;
	extractionEdges[i * 3 + 1].edgeDirection = 1;
	extractionEdges[i * 3 + 1].zeroCrossing = false;
	extractionEdges[i * 3 + 1].zeroCrossingPointIndex = UINT32_MAX;
	extractionEdges[i * 3 + 2].startVoxelIndex = i;
	extractionEdges[i * 3 + 2].edgeDirection = 2;
	extractionEdges[i * 3 + 2].zeroCrossing = false;
	extractionEdges[i * 3 + 2].zeroCrossingPointIndex = UINT32_MAX;

	// Next X
	{
		HashKey voxel_key(globalIndexX + 1, globalIndexY, globalIndexZ);
		uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(voxel_key);
		if (hashSlot_idx != kEmpty32)
		{
			extractionEdges[i * 3 + 0].endVoxelIndex = hashSlot_idx;
		}
		else
		{
			extractionEdges[i * 3 + 0].endVoxelIndex = UINT32_MAX;
		}
	}

	// Next Y
	{
		HashKey voxel_key(globalIndexX, globalIndexY + 1, globalIndexZ);
		uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(voxel_key);
		if (hashSlot_idx != kEmpty32)
		{
			extractionEdges[i * 3 + 1].endVoxelIndex = hashSlot_idx;
		}
		else
		{
			extractionEdges[i * 3 + 1].endVoxelIndex = UINT32_MAX;
		}
	}

	// Next Z
	{
		HashKey voxel_key(globalIndexX, globalIndexY, globalIndexZ + 1);
		uint32_t hashSlot_idx = globalHash_info->get_lookup_idx_func64_v4(voxel_key);
		if (hashSlot_idx != kEmpty32)
		{
			extractionEdges[i * 3 + 2].endVoxelIndex = hashSlot_idx;
		}
		else
		{
			extractionEdges[i * 3 + 2].endVoxelIndex = UINT32_MAX;
		}
	}
}

__global__ void Kernel_CalculateZeroCrossingPoints(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
	ExtractionVoxel * extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge * extractionEdges, unsigned int* numberOfExtractionEdges,
	Eigen::Vector3f * zeroCrossingPositions, Eigen::Vector3f * zeroCrossingNormals, Eigen::Vector3b * zeroCrossingColors, unsigned int* numberOfZeroCrossingPositions)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity * 3 - 1) return;

	auto& extractionEdge = extractionEdges[threadid];
	if (UINT32_MAX == extractionEdge.startVoxelIndex || UINT32_MAX == extractionEdge.endVoxelIndex) return;

	auto startVoxel = extractionVoxels[extractionEdge.startVoxelIndex];
	auto endVoxel = extractionVoxels[extractionEdge.endVoxelIndex];
	if ((FLT_MAX == startVoxel.value) || (FLT_MAX == endVoxel.value)) return;

	if ((startVoxel.value > 0 && endVoxel.value < 0) || (startVoxel.value < 0 && endVoxel.value > 0)) {
		extractionEdge.zeroCrossing = true;

		float ratio = startVoxel.value / (startVoxel.value - endVoxel.value);
		auto zeroCrossingPoint = startVoxel.position + ratio * (endVoxel.position - startVoxel.position);
		auto zeroCrossingNormal = startVoxel.normal + ratio * (endVoxel.normal - startVoxel.normal);
		auto r = (float)startVoxel.color.x() + ratio * ((float)endVoxel.color.x() - (float)startVoxel.color.x());
		auto g = (float)startVoxel.color.y() + ratio * ((float)endVoxel.color.y() - (float)startVoxel.color.y());
		auto b = (float)startVoxel.color.z() + ratio * ((float)endVoxel.color.z() - (float)startVoxel.color.z());

		auto zeroCrossingIndex = atomicAdd(numberOfZeroCrossingPositions, 1);
		zeroCrossingPositions[zeroCrossingIndex] = zeroCrossingPoint;
		zeroCrossingNormals[zeroCrossingIndex] = zeroCrossingNormal;
		zeroCrossingColors[zeroCrossingIndex] = Eigen::Vector3b((unsigned char)r, (unsigned char)g, (unsigned char)b);
		extractionEdge.zeroCrossingPointIndex = zeroCrossingIndex;
	}
}

__device__ ExtractionVoxel* GetVoxel(MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues,
	ExtractionVoxel * extractionVoxels, uint32_t globalIndexX, uint32_t globalIndexY, uint32_t globalIndexZ)
{
	HashKey key(globalIndexX, globalIndexY, globalIndexZ);
	auto i = globalHash_info->get_lookup_idx_func64_v4(key);
	if (i != kEmpty32)
	{
		return extractionVoxels + i;
	}
	else
	{
		return nullptr;
	}
}

__device__ int calcCubeIndex(MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info, voxel_value_t * voxelValues,
	ExtractionVoxel * extractionVoxels, ExtractionEdge * extractionEdges, size_t voxelIndex, ExtractionVoxel * *voxels)
{
	int cubeIndex = 0;
	float isolevel = 0.0f;

	auto currentVoxel = extractionVoxels + voxelIndex;
	voxels[0] = currentVoxel;
	voxels[1] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY, currentVoxel->globalIndexZ);
	voxels[2] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY, currentVoxel->globalIndexZ + 1);
	voxels[3] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX, currentVoxel->globalIndexY, currentVoxel->globalIndexZ + 1);
	voxels[4] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ);
	voxels[5] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ);
	voxels[6] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX + 1, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ + 1);
	voxels[7] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
		currentVoxel->globalIndexX, currentVoxel->globalIndexY + 1, currentVoxel->globalIndexZ + 1);

	if (nullptr != voxels[0]) if (FLT_VALID(voxels[0]->value)) if (voxels[0]->value < isolevel) cubeIndex |= 1;
	if (nullptr != voxels[1]) if (FLT_VALID(voxels[1]->value)) if (voxels[1]->value < isolevel) cubeIndex |= 2;
	if (nullptr != voxels[2]) if (FLT_VALID(voxels[2]->value)) if (voxels[2]->value < isolevel) cubeIndex |= 4;
	if (nullptr != voxels[3]) if (FLT_VALID(voxels[3]->value)) if (voxels[3]->value < isolevel) cubeIndex |= 8;
	if (nullptr != voxels[4]) if (FLT_VALID(voxels[4]->value)) if (voxels[4]->value < isolevel) cubeIndex |= 16;
	if (nullptr != voxels[5]) if (FLT_VALID(voxels[5]->value)) if (voxels[5]->value < isolevel) cubeIndex |= 32;
	if (nullptr != voxels[6]) if (FLT_VALID(voxels[6]->value)) if (voxels[6]->value < isolevel) cubeIndex |= 64;
	if (nullptr != voxels[7]) if (FLT_VALID(voxels[7]->value)) if (voxels[7]->value < isolevel) cubeIndex |= 128;

	return cubeIndex;
}

__global__ void Kernel_MarchingCubes(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
	ExtractionVoxel * extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge * extractionEdges, unsigned int* numberOfExtractionEdges)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > exeInfo.globalHashInfo->HashTableCapacity - 1) return;

	auto key = exeInfo.globalHash[threadid];
	auto i = exeInfo.globalHashInfo->get_lookup_idx_func64_v4(key);
	if (i == kEmpty32) return;

	auto vv = voxelValues[i];
	if (VOXEL_INVALID == vv) return;

	float isolevel = 0.0f;
	ExtractionVoxel* voxels[8] = { nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr };

	int cubeIndex = calcCubeIndex(exeInfo, globalHash_info, voxelValues,
		extractionVoxels, extractionEdges, i, (ExtractionVoxel**)voxels);

	uint32_t edgelist[12]
		= { UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX };

	uint32_t vertlist[12]
		= { UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX,
			UINT32_MAX, UINT32_MAX, UINT32_MAX };

	if (edgeTable[cubeIndex] == 0)
	{
		return;
	}

	if (edgeTable[cubeIndex] & 1)
	{
		if (nullptr != voxels[0])
		{
			auto edge = extractionEdges[voxels[0]->edgeIndexX];
			edgelist[0] = voxels[0]->edgeIndexX;
			vertlist[0] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 2)
	{
		if (nullptr != voxels[1])
		{
			auto edge = extractionEdges[voxels[1]->edgeIndexZ];
			edgelist[1] = voxels[1]->edgeIndexZ;
			vertlist[1] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 4)
	{
		if (nullptr != voxels[3])
		{
			auto edge = extractionEdges[voxels[3]->edgeIndexX];
			edgelist[2] = voxels[3]->edgeIndexX;
			vertlist[2] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 8)
	{
		if (nullptr != voxels[0])
		{
			auto edge = extractionEdges[voxels[0]->edgeIndexZ];
			edgelist[3] = voxels[0]->edgeIndexZ;
			vertlist[3] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 16)
	{
		if (nullptr != voxels[4])
		{
			auto edge = extractionEdges[voxels[4]->edgeIndexX];
			edgelist[4] = voxels[4]->edgeIndexX;
			vertlist[4] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 32)
	{
		if (nullptr != voxels[5])
		{
			auto edge = extractionEdges[voxels[5]->edgeIndexZ];
			edgelist[5] = voxels[5]->edgeIndexZ;
			vertlist[5] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 64)
	{
		if (nullptr != voxels[7])
		{
			auto edge = extractionEdges[voxels[7]->edgeIndexX];
			edgelist[6] = voxels[7]->edgeIndexX;
			vertlist[6] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 128)
	{
		if (nullptr != voxels[4])
		{
			auto edge = extractionEdges[voxels[4]->edgeIndexZ];
			edgelist[7] = voxels[4]->edgeIndexZ;
			vertlist[7] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 256)
	{
		if (nullptr != voxels[0])
		{
			auto edge = extractionEdges[voxels[0]->edgeIndexY];
			edgelist[8] = voxels[0]->edgeIndexY;
			vertlist[8] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 512)
	{
		if (nullptr != voxels[1])
		{
			auto edge = extractionEdges[voxels[1]->edgeIndexY];
			edgelist[9] = voxels[1]->edgeIndexY;
			vertlist[9] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 1024)
	{
		if (nullptr != voxels[2])
		{
			auto edge = extractionEdges[voxels[2]->edgeIndexY];
			edgelist[10] = voxels[2]->edgeIndexY;
			vertlist[10] = edge.zeroCrossingPointIndex;
		}
	}
	if (edgeTable[cubeIndex] & 2048)
	{
		if (nullptr != voxels[3])
		{
			auto edge = extractionEdges[voxels[3]->edgeIndexY];
			edgelist[11] = voxels[3]->edgeIndexY;
			vertlist[11] = edge.zeroCrossingPointIndex;
		}
	}

	for (int ti = 0; ti < 4; ti++) {
		auto ti0 = triTable[cubeIndex][ti * 3];
		auto ti1 = triTable[cubeIndex][ti * 3 + 1];
		auto ti2 = triTable[cubeIndex][ti * 3 + 2];

		if (-1 == ti0 || -1 == ti1 || -1 == ti2) break;

		auto ei0 = edgelist[ti0];
		auto ei1 = edgelist[ti1];
		auto ei2 = edgelist[ti2];

		auto vi0 = vertlist[ti0];
		auto vi1 = vertlist[ti1];
		auto vi2 = vertlist[ti2];

		if (UINT32_MAX == ei0 || UINT32_MAX == ei1 || UINT32_MAX == ei2) break;
		if (0 == ei0 || 0 == ei1 || 0 == ei2) break;
		if (ei0 == ei1 || ei0 == ei2 || ei1 == ei2) break;

		if (UINT32_MAX == vi0 || UINT32_MAX == vi1 || UINT32_MAX == vi2) break;
		if (0 == vi0 || 0 == vi1 || 0 == vi2) break;
		if (vi0 == vi1 || vi0 == vi2 || vi1 == vi2) break;

		voxels[0]->triangles[ti].edgeIndices[0] = ei0;
		voxels[0]->triangles[ti].edgeIndices[1] = ei1;
		voxels[0]->triangles[ti].edgeIndices[2] = ei2;

		voxels[0]->triangles[ti].vertexIndices[0] = vi0;
		voxels[0]->triangles[ti].vertexIndices[1] = vi1;
		voxels[0]->triangles[ti].vertexIndices[2] = vi2;

		if (extractionEdges[ei0].zeroCrossingPointIndex != vi0)
		{
			printf("vi0 != ei0");
		}
		if (extractionEdges[ei1].zeroCrossingPointIndex != vi1)
		{
			printf("vi1 != ei1");
		}
		if (extractionEdges[ei2].zeroCrossingPointIndex != vi2)
		{
			printf("vi2 != ei2");
		}

		voxels[0]->numberOfTriangles++;
	}
}

__global__ void Kernel_MarchingCubes_Verify(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info, uint64_t * globalHash,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	//Eigen::Vector3f* resultPositions, Eigen::Vector3f* resultNormals, Eigen::Vector4f* resultColors, unsigned int* numberOfVoxelPositions,
	ExtractionVoxel * extractionVoxels, unsigned int* numberOfExtractionVoxels, ExtractionEdge * extractionEdges, unsigned int* numberOfExtractionEdges)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity) return;

	auto& voxel = extractionVoxels[threadid];
	for (int i = 0; i < voxel.numberOfTriangles; i++)
	{
		auto& triangle = voxel.triangles[i];

		auto ei0 = triangle.edgeIndices[0];
		auto ei1 = triangle.edgeIndices[1];
		auto ei2 = triangle.edgeIndices[2];

		auto e0 = extractionEdges[ei0];
		auto e1 = extractionEdges[ei1];
		auto e2 = extractionEdges[ei2];

		auto v0 = triangle.vertexIndices[0];
		auto v1 = triangle.vertexIndices[1];
		auto v2 = triangle.vertexIndices[2];

		if (e0.zeroCrossingPointIndex != v0)
		{
			printf("e0.zeroCrossingPointIndex != v0\n");
		}
		if (e1.zeroCrossingPointIndex != v1)
		{
			printf("e1.zeroCrossingPointIndex != v1\n");
		}
		if (e2.zeroCrossingPointIndex != v2)
		{
			printf("e2.zeroCrossingPointIndex != v2\n");
		}
	}
}

__global__ void Kernel_MeshSmooth(MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info, voxel_value_t * voxelValues,
	ExtractionVoxel * extractionVoxels, unsigned int* numberOfExtractionVoxels,
	ExtractionEdge * extractionEdges, unsigned int* numberOfExtractionEdges,
	Eigen::Vector3f * zeroCrossingPositions, unsigned int* numberOfZeroCrossingPositions,
	Eigen::Vector3f * resultZeroCrossingPositions)
{
	unsigned int edgeIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (edgeIndex > *numberOfExtractionEdges - 1) return;

	ExtractionEdge* edge = extractionEdges + edgeIndex;
	if (false == edge->zeroCrossing) return;

	if (UINT32_MAX == edge->zeroCrossingPointIndex) return;

	ExtractionVoxel* voxel = extractionVoxels + edge->startVoxelIndex;

	ExtractionVoxel* voxels[4] = { nullptr, nullptr, nullptr, nullptr };
	if (0 == edge->edgeDirection)
	{
		voxels[0] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ - 1);

		voxels[1] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ);

		voxels[2] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX, voxel->globalIndexY, voxel->globalIndexZ - 1);

		voxels[3] = voxel;
	}
	else if (1 == edge->edgeDirection)
	{
		voxels[0] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ - 1);

		voxels[1] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ);

		voxels[2] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX, voxel->globalIndexY, voxel->globalIndexZ - 1);

		voxels[3] = voxel;
	}
	else if (2 == edge->edgeDirection)
	{
		voxels[0] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX - 1, voxel->globalIndexY - 1, voxel->globalIndexZ);

		voxels[1] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX, voxel->globalIndexY - 1, voxel->globalIndexZ);

		voxels[2] = GetVoxel(exeInfo, globalHash_info, voxelValues, extractionVoxels,
			voxel->globalIndexX - 1, voxel->globalIndexY, voxel->globalIndexZ);

		voxels[3] = voxel;
	}

	Eigen::Vector3f accumulatedPosition = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
	int positionCount = 0;
	for (int voxelIndex = 0; voxelIndex < 4; voxelIndex++)
	{
		auto incidentVoxel = voxels[voxelIndex];
		if (nullptr == incidentVoxel) continue;

		for (int i = 0; i < incidentVoxel->numberOfTriangles; i++)
		{
			auto triangle = incidentVoxel->triangles[i];
			if (*numberOfExtractionEdges <= triangle.edgeIndices[0] ||
				*numberOfExtractionEdges <= triangle.edgeIndices[1] ||
				*numberOfExtractionEdges <= triangle.edgeIndices[2])
				continue;

			if (edgeIndex == triangle.edgeIndices[0])
			{
				auto& ea = extractionEdges[triangle.edgeIndices[1]];
				auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pa))
				{
					accumulatedPosition += pa;
					positionCount++;
				}

				auto& eb = extractionEdges[triangle.edgeIndices[2]];
				auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pb))
				{
					accumulatedPosition += pb;
					positionCount++;
				}
			}
			else if (edgeIndex == triangle.edgeIndices[1])
			{
				auto& ea = extractionEdges[triangle.edgeIndices[0]];
				auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pa))
				{
					accumulatedPosition += pa;
					positionCount++;
				}

				auto& eb = extractionEdges[triangle.edgeIndices[2]];
				auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pb))
				{
					accumulatedPosition += pb;
					positionCount++;
				}
			}
			else if (edgeIndex == triangle.edgeIndices[2])
			{
				auto& ea = extractionEdges[triangle.edgeIndices[0]];
				auto& pa = zeroCrossingPositions[ea.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pa))
				{
					accumulatedPosition += pa;
					positionCount++;
				}

				auto& eb = extractionEdges[triangle.edgeIndices[1]];
				auto& pb = zeroCrossingPositions[eb.zeroCrossingPointIndex];
				if (VECTOR3F_VALID_(pb))
				{
					accumulatedPosition += pb;
					positionCount++;
				}
			}
		}
	}

	if (positionCount > 0)
	{
		resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = (Eigen::Vector3f)((accumulatedPosition / (float)positionCount));// +Eigen::Vector3f(0.0f, 1.0f, 0.0f);

		//printf("%f, %f, %f\n", meanPosition.x(), meanPosition.y(), meanPosition.z());
	}

	//if (positionCount > 0)
	//{
	//	resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = meanPosition / (float)positionCount;
	//}
	//else
	//{
	//	resultZeroCrossingPositions[edge->zeroCrossingPointIndex] = zeroCrossingPositions[edge->zeroCrossingPointIndex];
	//}
}

__global__ void Kernel_CountNeighborPoints(
	MarchingCubes::ExecutionInfo exeInfo,
	HashKey64 * globalHash_info, uint64_t * globalHash,
	voxel_value_t * voxelValues, unsigned short* voxelValueCounts, Eigen::Vector3f * voxelNormals, Eigen::Vector3b * voxelColors,
	Eigen::Vector3f * resultPositions, Eigen::Vector3f * resultNormals, Eigen::Vector4f * resultColors,
	unsigned int* numberOfVoxelPositions, ExtractionVoxel * extractionVoxels, ExtractionEdge * extractionEdges,
	Eigen::Vector3f * tempPositions, unsigned int* numberOfTemp)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > globalHash_info->HashTableCapacity - 1) return;

	auto extractionEdge = extractionEdges[threadid];

	auto startVoxel = extractionVoxels[extractionEdge.startVoxelIndex];
	auto endVoxel = extractionVoxels[extractionEdge.endVoxelIndex];
}

#ifndef VOXEL_GRID_MODULE
void MarchingCubes::ExtractVolume(const std::string & filename, Eigen::Vector3f & localMin, Eigen::Vector3f & localMax, cached_allocator * alloc_, CUstream_st * st)
{
	if (globalHash_info_host == nullptr) return;

	qDebug("ExtractVolume()");

	Eigen::Vector3f* zeroCrossingPositions = nullptr;
	cudaMalloc(&zeroCrossingPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);

	Eigen::Vector3f* zeroCrossingNormals = nullptr;
	cudaMalloc(&zeroCrossingNormals, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);

	Eigen::Vector3b* zeroCrossingColors = nullptr;
	cudaMalloc(&zeroCrossingColors, sizeof(Eigen::Vector3b) * globalHash_info_host->HashTableCapacity * 3);

	unsigned int* numberOfzeroCrossingPositions = nullptr;
	cudaMalloc(&numberOfzeroCrossingPositions, sizeof(unsigned int));
	cudaMemset(numberOfzeroCrossingPositions, 0, sizeof(unsigned int));

	Eigen::Vector3f* resultZeroCrossingPositions = nullptr;
	cudaMalloc(&resultZeroCrossingPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);



	//Eigen::Vector3f* voxelPositions = nullptr;
	//cudaMalloc(&voxelPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

	//Eigen::Vector3f* voxelNormals = nullptr;
	//cudaMalloc(&voxelNormals, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity);

	//Eigen::Vector4f* voxelColors = nullptr;
	//cudaMalloc(&voxelColors, sizeof(Eigen::Vector4f) * globalHash_info_host->HashTableCapacity);

	//unsigned int* numberOfVoxelPositions = nullptr;
	//cudaMalloc(&numberOfVoxelPositions, sizeof(unsigned int));
	//cudaMemset(numberOfVoxelPositions, 0, sizeof(unsigned int));

	checkCudaSync(st);

	exeInfo.local.SetLocalMinMax(localMin, localMax);
	//exeInfo.local.SetLocalMinMax(exeInfo.global.globalMin, exeInfo.global.globalMax);

	ExtractionVoxel* extractionVoxels = nullptr;
	cudaMalloc(&extractionVoxels, sizeof(ExtractionVoxel) * globalHash_info_host->HashTableCapacity);

	unsigned int* numberOfExtractionVoxels = nullptr;
	cudaMalloc(&numberOfExtractionVoxels, sizeof(unsigned int));
	cudaMemset(numberOfExtractionVoxels, 0, sizeof(unsigned int));

	ExtractionEdge* extractionEdges = nullptr;
	cudaMalloc(&extractionEdges, sizeof(ExtractionEdge) * globalHash_info_host->HashTableCapacity * 3);

	unsigned int* numberOfExtractionEdges = nullptr;
	cudaMalloc(&numberOfExtractionEdges, sizeof(unsigned int));
	cudaMemset(numberOfExtractionEdges, 0xFFFFFFFF, sizeof(unsigned int));

	Eigen::Vector3f* tempPositions = nullptr;
	cudaMalloc(&tempPositions, sizeof(Eigen::Vector3f) * globalHash_info_host->HashTableCapacity * 3);

	unsigned int* numberOfTemp = nullptr;
	cudaMalloc(&numberOfTemp, sizeof(unsigned int));
	cudaMemset(numberOfTemp, 0, sizeof(unsigned int));

	checkCudaSync(st);

	{
		NvtxRangeCuda nvtxPrint("Kernel_PopulateExtractionVoxels");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_PopulateExtractionVoxels, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		Kernel_PopulateExtractionVoxels << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info,
			thrust::raw_pointer_cast(m_MC_voxelValues.data()),
			thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
			thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
			thrust::raw_pointer_cast(m_MC_voxelColors.data()),
			//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
			extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

		checkCudaSync(st);
	}

	{
		NvtxRangeCuda nvtxPrint("Kernel_CalculateZeroCrossingPoints");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_CalculateZeroCrossingPoints, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity * 3 + threadblocksize - 1) / threadblocksize;

		Kernel_CalculateZeroCrossingPoints << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info,
			thrust::raw_pointer_cast(m_MC_voxelValues.data()),
			thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
			thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
			thrust::raw_pointer_cast(m_MC_voxelColors.data()),
			//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
			extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges,
			zeroCrossingPositions, zeroCrossingNormals, zeroCrossingColors, numberOfzeroCrossingPositions);

		checkCudaSync(st);
	}

	//{
	//	auto chash = Utilities::Device::Clustering::ClusteringHash(input->m_nowSize, &alloc, m_stream);

	//	auto d_compondPoints = thrust::raw_pointer_cast(input->points_.data());
	//	auto d_compondNormals = thrust::raw_pointer_cast(input->normals_.data());
	//	auto d_compondColors = thrust::raw_pointer_cast(input->colors_.data());
	//	auto d_compondExtraAttribs = thrust::raw_pointer_cast(input->extraAttribs_.data());

	//	chash.DoClustering(d_compondPoints, d_compondNormals, d_compondColors, d_compondExtraAttribs, m_stream, &alloc);

	//}

	{
		NvtxRangeCuda nvtxPrint("Kernel_MarchingCubes");

		int mingridsize;
		int threadblocksize;
		checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MarchingCubes, 0, 0));
		int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

		Kernel_MarchingCubes << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info,
			thrust::raw_pointer_cast(m_MC_voxelValues.data()),
			thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
			thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
			thrust::raw_pointer_cast(m_MC_voxelColors.data()),
			//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
			extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

		checkCudaSync(st);
	}

	//unsigned int host_numberOfExtractionEdges = 0;
	//cudaMemcpyAsync(&host_numberOfExtractionEdges, numberOfExtractionEdges, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

	//checkCudaErrors(cudaStreamSynchronize(st));

	//unsigned int* numberOfResultIndices;
	//cudaMalloc(&numberOfResultIndices, sizeof(unsigned int));

	//unsigned int* resultIndices;
	//cudaMalloc(&resultIndices, sizeof(unsigned int)* host_numberOfExtractionEdges * 3 * 4);


	//for (size_t count = 0; count < 3; count++)
	{
		//{
		//	NvtxRangeCuda nvtxPrint("Kernel_PopulateExtractionVoxels");

		//	unsigned int h_numberOfzeroCrossingPositions = 0;
		//	cudaMemcpyAsync(&h_numberOfzeroCrossingPositions, numberOfzeroCrossingPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);
		//	cudaStreamSynchronize(st);

		//	int mingridsize;
		//	int threadblocksize;
		//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MeshSmooth, 0, 0));
		//	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity * 3 + threadblocksize - 1) / threadblocksize;

		//	Kernel_MeshSmooth << <gridsize, threadblocksize, 0, st >> > (
		//		exeInfo, globalHash_info,
		//		thrust::raw_pointer_cast(m_MC_voxelValues.data()),
		//		extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges,
		//		zeroCrossingPositions, numberOfzeroCrossingPositions, resultZeroCrossingPositions);

		//	checkCudaErrors(cudaStreamSynchronize(st));

		//	cudaMemcpyAsync(
		//		zeroCrossingPositions,
		//		resultZeroCrossingPositions,
		//		sizeof(Eigen::Vector3f) * h_numberOfzeroCrossingPositions,
		//		cudaMemcpyDeviceToDevice, st);

		//	cudaStreamSynchronize(st);
		//}
	}

	//nvtxRangePushA("Kernel_MarchingCubes_Verify");
	//{
	//	int mingridsize;
	//	int threadblocksize;
	//	checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, Kernel_MarchingCubes, 0, 0));
	//	int gridsize = ((uint32_t)globalHash_info_host->HashTableCapacity + threadblocksize - 1) / threadblocksize;

	//	Kernel_MarchingCubes << <gridsize, threadblocksize, 0, st >> > (exeInfo, globalHash_info,
	//		thrust::raw_pointer_cast(m_MC_voxelValues.data()),
	//		thrust::raw_pointer_cast(m_MC_voxelValueCounts.data()),
	//		thrust::raw_pointer_cast(m_MC_voxelNormals.data()),
	//		thrust::raw_pointer_cast(m_MC_voxelColors.data()),
	//		//voxelPositions, voxelNormals, voxelColors, numberOfVoxelPositions,
	//		extractionVoxels, numberOfExtractionVoxels, extractionEdges, numberOfExtractionEdges);

	//	checkCudaErrors(cudaStreamSynchronize(st));
	//}
	//nvtxRangePop();

	unsigned int host_numberOfZeroCrossingValues = 0;
	cudaMemcpyAsync(&host_numberOfZeroCrossingValues, numberOfzeroCrossingPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_ZeroCrossingPositions = new Eigen::Vector3f[host_numberOfZeroCrossingValues];
	cudaMemcpyAsync(host_ZeroCrossingPositions, zeroCrossingPositions,
		sizeof(Eigen::Vector3f) * host_numberOfZeroCrossingValues, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* host_ZeroCrossingNormals = new Eigen::Vector3f[host_numberOfZeroCrossingValues];
	cudaMemcpyAsync(host_ZeroCrossingNormals, zeroCrossingNormals,
		sizeof(Eigen::Vector3f) * host_numberOfZeroCrossingValues, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* host_ZeroCrossingColors = new Eigen::Vector3b[host_numberOfZeroCrossingValues];
	cudaMemcpyAsync(host_ZeroCrossingColors, zeroCrossingColors,
		sizeof(Eigen::Vector3b) * host_numberOfZeroCrossingValues, cudaMemcpyDeviceToHost, st);

	printf("host_numberOfZeroCrossingValues : %d\n", host_numberOfZeroCrossingValues);

	//unsigned int host_numberOfVoxelValues = 0;
	//cudaMemcpyAsync(&host_numberOfVoxelValues, numberOfVoxelPositions, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

	//Eigen::Vector3f* host_voxelPositions = new Eigen::Vector3f[host_numberOfVoxelValues];
	//cudaMemcpyAsync(host_voxelPositions, voxelPositions,
	//	sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

	//Eigen::Vector3f* host_voxelNormals = new Eigen::Vector3f[host_numberOfVoxelValues];
	//cudaMemcpyAsync(host_voxelNormals, voxelNormals,
		//	sizeof(Eigen::Vector3f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

		//Eigen::Vector4f* host_voxelColors = new Eigen::Vector4f[host_numberOfVoxelValues];
		//cudaMemcpyAsync(host_voxelColors, voxelColors,
		//	sizeof(Eigen::Vector4f) * host_numberOfVoxelValues, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	qDebug("host_numberOfZeroCrossingValues : %d", host_numberOfZeroCrossingValues);

	{
		HVETM::Mesh mesh;

		ExtractionVoxel* host_extractionVoxels = new ExtractionVoxel[globalHash_info_host->HashTableCapacity];
		cudaMemcpy(host_extractionVoxels, extractionVoxels, sizeof(ExtractionVoxel) * globalHash_info_host->HashTableCapacity, cudaMemcpyDeviceToHost);

		ExtractionEdge* host_extractionEdges = new ExtractionEdge[globalHash_info_host->HashTableCapacity * 3];
		cudaMemcpy(host_extractionEdges, extractionEdges, sizeof(ExtractionEdge) * globalHash_info_host->HashTableCapacity * 3, cudaMemcpyDeviceToHost);

		//PLYFormat ply;

		map<unsigned int, HVETM::Vertex*> mapping;
		for (size_t i = 0; i < host_numberOfZeroCrossingValues; i++)
		{
			//ply.AddPointFloat3(host_ZeroCrossingPositions[i].data());
			//ply.AddNormalFloat3(host_ZeroCrossingNormals[i].data());
			//ply.AddColor(host_ZeroCrossingColors[i]);

			auto r = (float)host_ZeroCrossingColors[i].x() / 255.0f;
			auto g = (float)host_ZeroCrossingColors[i].y() / 255.0f;
			auto b = (float)host_ZeroCrossingColors[i].z() / 255.0f;

			auto vertex = mesh.AddVertex(
				host_ZeroCrossingPositions[i].data(),
				host_ZeroCrossingNormals[i].data(),
				{ r, g, b });
			mapping[i] = vertex;
		}

		for (size_t i = 0; i < globalHash_info_host->HashTableCapacity; i++)
		{
			auto voxel = host_extractionVoxels[i];
			for (size_t j = 0; j < voxel.numberOfTriangles; j++)
			{
				auto& triangle = voxel.triangles[j];
				auto e0 = triangle.vertexIndices[0];
				auto e1 = triangle.vertexIndices[1];
				auto e2 = triangle.vertexIndices[2];

				if (e0 == e1 || e0 == e2 || e1 == e2) continue;
				if (UINT32_MAX == e0 || UINT32_MAX == e1 || UINT32_MAX == e2) continue;
				//if (0 == e0 || 0 == e1 || 0 == e2) continue;
				if (e0 >= host_numberOfZeroCrossingValues ||
					e1 >= host_numberOfZeroCrossingValues ||
					e2 >= host_numberOfZeroCrossingValues) continue;

				//ply.AddTriangleIndex(e0);
				//ply.AddTriangleIndex(e1);
				//ply.AddTriangleIndex(e2);

				mesh.AddTriangle(mapping[e0], mapping[e1], mapping[e2]);
			}
		}
		//ply.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\MarchingCubes.ply");

		mesh.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\MarchingCubes_Mesh.ply");

		delete[] host_extractionVoxels;
		delete[] host_extractionEdges;
	}

	PLYFormat plyZeroCrossing;
	for (size_t i = 0; i < host_numberOfZeroCrossingValues; i++)
	{
		plyZeroCrossing.AddPointFloat3(host_ZeroCrossingPositions[i].data());
		plyZeroCrossing.AddNormalFloat3(host_ZeroCrossingNormals[i].data());
		plyZeroCrossing.AddColor(host_ZeroCrossingColors[i]);
	}
	plyZeroCrossing.Serialize(pSettings->GetResourcesFolderPath() + "\\Debug\\ZeroCrossingPoints.ply");

#pragma region Save VoxelValues
	//{
//	PLYFormat ply;

//	for (size_t i = 0; i < host_numberOfVoxelValues; i++)
//	{
//		auto& v = host_voxelPositions[i];
		//		if (VECTOR3F_VALID_(v))
		//		{
		//			ply.AddPointFloat3(v.data());
		//			auto& normal = host_voxelNormals[i];
		//			normal.normalize();
		//			ply.AddNormalFloat3(normal.data());
		//			auto& color = host_voxelColors[i];
		//			ply.AddColorFloat4(color.data());
		//		}
		//	}
		//	ply.Serialize(filename);
		//}

		//{
		//	unsigned int host_numberOfTemp = 0;
		//	cudaMemcpyAsync(&host_numberOfTemp, numberOfTemp, sizeof(unsigned int), cudaMemcpyDeviceToHost, st);

		//	Eigen::Vector3f* host_tempPositions = new Eigen::Vector3f[host_numberOfTemp];
		//	cudaMemcpyAsync(host_tempPositions, tempPositions,
		//		sizeof(Eigen::Vector3f) * host_numberOfTemp, cudaMemcpyDeviceToHost, st);

		//	PLYFormat ply;

		//	for (size_t i = 0; i < host_numberOfTemp; i++)
		//	{
		//		auto& v = host_tempPositions[i];
		//		if (VECTOR3F_VALID_(v))
		//		{
		//			ply.AddPointFloat3(v.data());
		//		}
		//	}
		//	ply.Serialize(filename);

		//	delete[] host_tempPositions;
		//}

//delete[] host_voxelPositions;
//delete[] host_voxelNormals;
//delete[] host_voxelColors;  
#pragma endregion

	cudaFree(zeroCrossingPositions);
	cudaFree(zeroCrossingNormals);
	cudaFree(zeroCrossingColors);
	cudaFree(numberOfzeroCrossingPositions);

	cudaFree(resultZeroCrossingPositions);

	//cudaFree(voxelPositions);
	//cudaFree(voxelNormals);
	//cudaFree(voxelColors);
	//cudaFree(numberOfVoxelPositions);

	cudaFree(extractionVoxels);
	cudaFree(extractionEdges);
}
#endif

void MarchingCubes::SaveFrame(
	int patchID,
	unsigned char* current_img_0,
	unsigned char* current_img_45,
	std::shared_ptr<pointcloud_Hios> mesh_0,
	std::shared_ptr<pointcloud_Hios> mesh_45,
	const Eigen::Matrix4f & transform_0,
	const Eigen::Matrix4f & transform_45,
	const Eigen::Vector3f & cameraPosition,
	cached_allocator * alloc_, CUstream_st * st)
{
	unsigned char image_0[CU_CX_SIZE_MAX * CU_CY_SIZE_MAX];
	unsigned char image_45[CU_CX_SIZE_MAX * CU_CY_SIZE_MAX];

	cudaMemcpyAsync(image_0, current_img_0, sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX, cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(image_45, current_img_45, sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX, cudaMemcpyDeviceToHost, st);

	auto size_0 = mesh_0->m_nowSize;
	auto size_45 = mesh_45->m_nowSize;

	Eigen::Vector3f* points_0 = new Eigen::Vector3f[size_0];
	Eigen::Vector3f* points_45 = new Eigen::Vector3f[size_45];

	cudaMemcpyAsync(points_0, thrust::raw_pointer_cast(mesh_0->points_.data()), sizeof(Eigen::Vector3f) * size_0, cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(points_45, thrust::raw_pointer_cast(mesh_45->points_.data()), sizeof(Eigen::Vector3f) * size_45, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3f* normals_0 = new Eigen::Vector3f[size_0];
	Eigen::Vector3f* normals_45 = new Eigen::Vector3f[size_45];

	cudaMemcpyAsync(normals_0, thrust::raw_pointer_cast(mesh_0->normals_.data()), sizeof(Eigen::Vector3f) * size_0, cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(normals_45, thrust::raw_pointer_cast(mesh_45->normals_.data()), sizeof(Eigen::Vector3f) * size_45, cudaMemcpyDeviceToHost, st);

	Eigen::Vector3b* colors_0 = new Eigen::Vector3b[size_0];
	Eigen::Vector3b* colors_45 = new Eigen::Vector3b[size_45];

	cudaMemcpyAsync(colors_0, thrust::raw_pointer_cast(mesh_0->colors_.data()), sizeof(Eigen::Vector3b) * size_0, cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(colors_45, thrust::raw_pointer_cast(mesh_45->colors_.data()), sizeof(Eigen::Vector3b) * size_45, cudaMemcpyDeviceToHost, st);

	checkCudaSync(st);

	stringstream ss;
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	ss << "C:\\Debug\\Patches\\";
	//ss << std::put_time(&tm, "%Y_%m_%d %H_%M_%S\\");
	CreateDirectoryA(ss.str().c_str(), nullptr);
	ss << "patch_";
	ss << patchID << ".pat";

	ofstream ofs;
	ofs.open(ss.str(), ios::out | ios::binary);

	size_t size = 0;
	ofs.write((char*)&patchID, sizeof(int)); size += sizeof(int);
	ofs.write((char*)&size_0, sizeof(size_t));  size += sizeof(size_t);
	ofs.write((char*)&size_45, sizeof(size_t)); size += sizeof(size_t);

	ofs.write((char*)transform_0.data(), sizeof(float) * 16); size += sizeof(float) * 16;
	ofs.write((char*)transform_45.data(), sizeof(float) * 16); size += sizeof(float) * 16;

	ofs.write((char*)cameraPosition.data(), sizeof(float) * 3); size += sizeof(float) * 3;

	ofs.write((char*)image_0, sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX); size += sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX;
	ofs.write((char*)image_45, sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX); size += sizeof(unsigned char) * CU_CX_SIZE_MAX * CU_CY_SIZE_MAX;

	ofs.write((char*)points_0, sizeof(Eigen::Vector3f) * size_0); size += sizeof(Eigen::Vector3f) * size_0;
	ofs.write((char*)points_45, sizeof(Eigen::Vector3f) * size_45); size += sizeof(Eigen::Vector3f) * size_45;

	ofs.write((char*)normals_0, sizeof(Eigen::Vector3f) * size_0); size += sizeof(Eigen::Vector3f) * size_0;
	ofs.write((char*)normals_45, sizeof(Eigen::Vector3f) * size_45); size += sizeof(Eigen::Vector3f) * size_45;

	ofs.write((char*)colors_0, sizeof(Eigen::Vector3b) * size_0); size += sizeof(Eigen::Vector3b) * size_0;
	ofs.write((char*)colors_45, sizeof(Eigen::Vector3b) * size_45); size += sizeof(Eigen::Vector3b) * size_45;

	qDebug("Written : %llu", size);

	ofs.close();

	delete[] points_0;
	delete[] points_45;

	delete[] normals_0;
	delete[] normals_45;

	delete[] colors_0;
	delete[] colors_45;
}

#endif

#ifndef BUILD_FOR_CPU
#pragma region CheckForFilter
__global__ void Kernel_CheckFilterOperation(
	unsigned short* deeplearningMap,
	unsigned char* liveViewMask,
	int img_size_x, int img_size_y,
	unsigned int* numberOfToothPixels, unsigned int* numberOfMaskedPixels)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > img_size_x * img_size_y - 1) return;

	unsigned int xDLIndex = img_size_x - 1 - (threadid % img_size_x);
	unsigned int yDLIndex = threadid / img_size_x;

	if (xDLIndex < img_size_x / 10 || (img_size_x - img_size_x / 10 < xDLIndex && xDLIndex < img_size_x)) return;

	unsigned int xMaskIndex = threadid % img_size_x;
	unsigned int yMaskIndex = threadid / img_size_x;

	auto liveViewMaskValue = liveViewMask[((img_size_y - 1 - yMaskIndex) * img_size_x + xMaskIndex) * 3];
	auto materialID = deeplearningMap[yDLIndex * img_size_x + xDLIndex];

	if (materialID == DL_TOOTH ||
		materialID == DL_DENTIFORM_TOOTH ||
		materialID == DL_ABUTMENT ||
		materialID == DL_SCANBODY ||
		materialID == DL_METAL ||
		materialID == DL_PLASTER)
	{
#ifdef AARON_TEST
		//Utilities::Device::Debugging::AddPointPC(8888, { (float)xDLIndex, (float)yDLIndex, 0.0f }, { 1.0f, 1.0f, 1.0f });
#endif
		atomicAdd(numberOfToothPixels, 1);
		if (0 == liveViewMaskValue)
		{
			atomicAdd(numberOfMaskedPixels, 1);
		}
	}
	else
	{
#ifdef AARON_TEST
		//Utilities::Device::Debugging::AddPointPC(8888, { (float)xDLIndex, (float)yDLIndex, 0.0f }, { 0.0f, 0.0f, 0.0f });
#endif
	}

	if (0 < liveViewMaskValue)
	{
#ifdef AARON_TEST
		//Utilities::Device::Debugging::AddPointPC(9999, { (float)xMaskIndex, (float)yMaskIndex, 0.0f }, { 1.0f, 1.0f, 1.0f });
#endif
	}
	else
	{
#ifdef AARON_TEST
		//Utilities::Device::Debugging::AddPointPC(9999, { (float)xMaskIndex, (float)yMaskIndex, 0.0f }, { 0.0f, 0.0f, 0.0f });
#endif
	}
}
#pragma endregion

#pragma region Fetch Patch
__host__ __device__
Eigen::Vector2f ComputeUV(
	float width, float height,
	const Eigen::Vector3f & refPoint,
	const Eigen::Quaternionf & rotation,
	const Eigen::Vector3f & point,
	const Eigen::Matrix4f & dev_camRT,
	const Eigen::Matrix3f & dev_cam_tilt)
{
	auto camPos = (dev_camRT * Eigen::Vector4f(point.x(), point.y(), point.z(), 1.0f)).head(3);

	//float rx = camPos[0] / camPos[2];
	//float ry = camPos[1] / camPos[2];

	//Eigen::Vector3f CamPos3f(rx, ry, 1);

	//const Eigen::Vector3f tiltcam = dev_cam_tilt * CamPos3f;
	//float u = tiltcam.z() ? tiltcam.x() / tiltcam.z() : tiltcam.x();
	//float v = tiltcam.z() ? tiltcam.y() / tiltcam.z() : tiltcam.y();




	//auto camPos = (dev_camRT * Eigen::Vector4f(point.x(), point.y(), point.z(), 1.0f)).head(3);
	//float u = camPos[0];
	//float v = camPos[1];
	//if (0.0f != camPos[2])
	//{
	//	u = u / camPos[2];
	//	v = v / camPos[2];
	//}

	//if (0.0f > u) u = 0.0f;
	//if (1.0f < u) u = 1.0f;
	//if (0.0f > v) v = 0.0f;
	//if (1.0f < v) v = 1.0f;


	Eigen::Vector3f tp = rotation.inverse() * (point - refPoint);
	float camZ = refPoint.norm();
	float perspectiveDiv = (camZ - tp.z());
	if (fabsf(perspectiveDiv) < 1e-6f) perspectiveDiv = 1e-6f;

	float u = (tp.x() / (0.5f * width * perspectiveDiv / camZ)) * 0.5f + 0.5f;
	float v = (tp.y() / (0.5f * height * perspectiveDiv / camZ)) * 0.5f + 0.5f;

	//alog("u: %f, v: %f\n", u, v);
	return Eigen::Vector2f(u, v);
}

__global__
void Kernel_FetchPatchToFilterCache2D(
	const Eigen::Matrix4f transform,
	const Eigen::Quaternionf rotation,
	Eigen::Vector3f refPoint,
	cudaSurfaceObject_t surfaceObject2D, cudaSurfaceObject_t surfaceObject2DMID,
	dim3 dimensions2D, dim3 dimensions3D, int zOffset,
	float voxelSize, float* patchMaxZ,
	Eigen::Vector3f * points, VoxelExtraAttrib * attribs, unsigned int numberOfPoints,
	unsigned short* materialIDs, unsigned int materialMapWidth, unsigned int materialMapHeight,
	const Eigen::Matrix4f & dev_camRT,
	const Eigen::Matrix3f & dev_cam_tilt)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > numberOfPoints - 1) return;

	auto lp = points[threadid];

	if (false == VECTOR3F_VALID_(lp)) return;

	if (*patchMaxZ < lp.z()) *patchMaxZ = lp.z();

	auto uv = ComputeUV((float)dimensions2D.x * voxelSize, (float)dimensions2D.y * voxelSize, refPoint, rotation, lp, dev_camRT, dev_cam_tilt);
	if ((0.0f > uv.x() || uv.x() > 1.0f) || (0.0f > uv.y() || uv.y() > 1.0f)) return;

	auto xLocalIndex = (int)(uv.x() * (float)dimensions2D.x);
	auto yLocalIndex = (int)(uv.y() * (float)dimensions2D.y);

	if (dimensions2D.x - 1 < xLocalIndex || dimensions2D.y - 1 < yLocalIndex) return;

	auto zLocalIndex = (unsigned int)floorf((lp - refPoint).norm() / voxelSize);

	float data = surf2Dread<float>(surfaceObject2D, xLocalIndex * sizeof(float), yLocalIndex);

	if (FLT_MAX == data)
	{
		surf2Dwrite((lp - refPoint).norm(), surfaceObject2D, xLocalIndex * sizeof(float), yLocalIndex);
	}
	else
	{
		surf2Dwrite(data > (lp - refPoint).norm() ? data : (lp - refPoint).norm(), surfaceObject2D, xLocalIndex * sizeof(float), yLocalIndex);
	}


	unsigned char mid = attribs[threadid].materialID;
	if (32 < mid)
	{
		surf2Dwrite<unsigned char>(1, surfaceObject2DMID, xLocalIndex * sizeof(unsigned char), yLocalIndex);
	}
	else
	{
		surf2Dwrite<unsigned char>(0, surfaceObject2DMID, xLocalIndex * sizeof(unsigned char), yLocalIndex);
	}

	{
		//auto muv = ComputeUV((float)materialMapWidth, (float)materialMapHeight, refPoint, rotation, lp / voxelSize, dev_camRT, dev_cam_tilt);
		auto muv = ComputeUV(25.6f / 2.0f, 48.0f / 3.0f, refPoint, rotation, lp / voxelSize, dev_camRT, dev_cam_tilt);
		if ((0.0f > muv.x() || muv.x() > 1.0f) || (0.0f > muv.y() || muv.y() > 1.0f)) return;

		//auto xIndex = materialMapWidth - 1 - (unsigned int)(uv.x() * (float)materialMapWidth);
		auto xIndex = (unsigned int)(muv.x() * (float)materialMapWidth);
		auto yIndex = (unsigned int)(muv.y() * (float)materialMapHeight);
		auto materialMapIndex = yIndex * materialMapWidth + xIndex;

		if (DL_TOOTH == materialIDs[materialMapIndex] ||
			DL_DENTIFORM_TOOTH == materialIDs[materialMapIndex] ||
			DL_METAL == materialIDs[materialMapIndex] ||
			DL_ABUTMENT == materialIDs[materialMapIndex] ||
			DL_SCANBODY == materialIDs[materialMapIndex])
		{
			surf2Dwrite<unsigned char>(1, surfaceObject2DMID, xLocalIndex * sizeof(unsigned char), yLocalIndex);
		}
	}
}
#pragma endregion


#pragma region Filter Compound
__global__
void Kernel_FilterCompound(
	Eigen::Vector3f * compoundPoints, unsigned int numberOfCompoundPoints,
	VoxelExtraAttrib * voxelExtraAttribs,
	const Eigen::Quaternionf rotation, Eigen::Vector3f refPoint,
	const Eigen::Matrix4f transform,
	cudaSurfaceObject_t surfaceObject2D, cudaSurfaceObject_t surfaceObject2DMID,
	cudaSurfaceObject_t surfaceObject3D, float voxelSize,
	dim3 dimensions2D, dim3 dimensions3D, int zOffset,
	Eigen::Vector3f * filteringSeedPoints, unsigned int* numberOfFilteringSeedPoints,
	float filterDistance,
	const Eigen::Matrix4f & dev_camRT,
	const Eigen::Matrix3f & dev_cam_tilt)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > numberOfCompoundPoints - 1) return;


	Eigen::Vector3f gp = compoundPoints[threadid];
	Eigen::Vector3f lp = (transform.inverse() * Eigen::Vector4f(gp.x(), gp.y(), gp.z(), 1.0f)).head(3);

	{
		auto muv = ComputeUV(25.6f / 2.0f, 48.0f / 3.0f, refPoint, rotation, lp, dev_camRT, dev_cam_tilt);
		if ((0.0f > muv.x() || muv.x() > 1.0f) || (0.0f > muv.y() || muv.y() > 1.0f)) return;

		//auto xIndex = materialMapWidth - 1 - (unsigned int)(uv.x() * (float)materialMapWidth);
		auto xIndex = (unsigned int)(muv.x() * (float)dimensions2D.x);
		auto yIndex = (unsigned int)(muv.y() * (float)dimensions2D.y);
		if (xIndex >= dimensions2D.x || yIndex >= dimensions2D.y) return;
		auto materialMapIndex = yIndex * dimensions2D.x + xIndex;

		auto mid = surf2Dread<unsigned char>(surfaceObject2DMID, xIndex * sizeof(unsigned char), yIndex);
		if (0 < mid) return;
	}

	auto uv = ComputeUV((float)dimensions2D.x * voxelSize, (float)dimensions2D.y * voxelSize, refPoint, rotation, lp, dev_camRT, dev_cam_tilt);
	if ((0.0f > uv.x() || uv.x() > 1.0f) || (0.0f > uv.y() || uv.y() > 1.0f)) return;

	auto xLocalIndex = (int)(uv.x() * (float)dimensions2D.x);
	auto yLocalIndex = (int)(uv.y() * (float)dimensions2D.y);

	if (dimensions2D.x - 1 < xLocalIndex || dimensions2D.y - 1 < yLocalIndex) return;

	float data2D = surf2Dread<float>(surfaceObject2D, xLocalIndex * sizeof(float), yLocalIndex);
	if (FLT_MAX == data2D) return;
	if (data2D - filterDistance < (lp - refPoint).norm()) return;

	auto direction = (lp - refPoint).normalized();
	auto point = refPoint + direction * data2D;

	auto cp = lp;
	cp.x() += (float)dimensions3D.x * voxelSize * 0.5f;
	cp.y() += (float)dimensions3D.y * voxelSize * 0.5f;
	cp.z() -= (float)zOffset * voxelSize;

	if (0 > cp.x() || 0 > cp.y() || 0 > cp.z()) return;

	auto xCacheIndex = (unsigned int)floorf(cp.x() / voxelSize);
	auto yCacheIndex = (unsigned int)floorf(cp.y() / voxelSize);
	auto zCacheIndex = (unsigned int)floorf(cp.z() / voxelSize);

	if (dimensions3D.x - 1 < xCacheIndex ||
		dimensions3D.y - 1 < yCacheIndex ||
		dimensions3D.z - 1 < zCacheIndex) return;

	auto mid = voxelExtraAttribs[threadid].materialID;
	if (32 < mid)
	{
		surf3Dwrite<uint8_t>(7, surfaceObject3D, xCacheIndex * sizeof(uint8_t), yCacheIndex, zCacheIndex);
		return;
	}

	surf3Dwrite<uint8_t>(0, surfaceObject3D, xCacheIndex * sizeof(uint8_t), yCacheIndex, zCacheIndex);

	if (dimensions3D.y - yCacheIndex < zCacheIndex - (int)floorf(-(float)zOffset + filterDistance / voxelSize)) return;

	auto pointIndex = atomicAdd(numberOfFilteringSeedPoints, 1);
	filteringSeedPoints[pointIndex] = gp;

	//if (filteringSeedPoints[0].z() < lp.z() - (float)zOffset * voxelSize)
	//{
	//	filteringSeedPoints[0].z() = lp.z() - (float)zOffset * voxelSize;
	//	filteringSeedPoints[1] = gp;
	//}

	surf3Dwrite<uint8_t>(1, surfaceObject3D, xCacheIndex * sizeof(uint8_t), yCacheIndex, zCacheIndex);
}

__global__
void Kernel_FilterCompound_Test(
	unsigned int numberOfCompoundPoints,
	const Eigen::Quaternionf rotation, Eigen::Vector3f refPoint,
	const Eigen::Matrix4f transform,
	cudaSurfaceObject_t surfaceObject2D, cudaSurfaceObject_t surfaceObject3D, float voxelSize,
	dim3 dimensions2D, dim3 dimensions3D, int zOffset,
	Eigen::Vector3f * filteringSeedPoints, unsigned int* numberOfFilteringSeedPoints,
	float filterDistance,
	const Eigen::Matrix4f & dev_camRT,
	const Eigen::Matrix3f & dev_cam_tilt)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > dimensions3D.x * dimensions3D.y * dimensions3D.z - 1) return;

	auto zIndex = threadid / (dimensions3D.x * dimensions3D.y);
	auto yIndex = (threadid % (dimensions3D.x * dimensions3D.y)) / dimensions3D.x;
	auto xIndex = (threadid % (dimensions3D.x * dimensions3D.y)) % dimensions3D.x;

	auto x = (float)xIndex * voxelSize - (float)dimensions3D.x * 0.5f * voxelSize;
	auto y = (float)yIndex * voxelSize - (float)dimensions3D.y * 0.5f * voxelSize;
	auto z = (float)zIndex * voxelSize;// +(float)zOffset * voxelSize;

	Eigen::Vector3f gp(x, y, z);
	Eigen::Vector3f lp = (transform.inverse() * Eigen::Vector4f(gp.x(), gp.y(), gp.z(), 1.0f)).head(3);

	auto uv = ComputeUV((float)dimensions2D.x * voxelSize, (float)dimensions2D.y * voxelSize, refPoint, rotation, lp, dev_camRT, dev_cam_tilt);
	if ((0.0f > uv.x() || uv.x() > 1.0f) || (0.0f > uv.y() || uv.y() > 1.0f)) return;

	auto xLocalIndex = (int)(uv.x() * (float)dimensions2D.x);
	auto yLocalIndex = (int)(uv.y() * (float)dimensions2D.y);
	auto zLocalIndex = (unsigned int)floorf(lp.z() / voxelSize);

	if (dimensions3D.x - 1 < xLocalIndex ||
		dimensions3D.y - 1 < yLocalIndex ||
		dimensions3D.z - 1 < zLocalIndex) return;

	float data2D = surf2Dread<float>(surfaceObject2D, xLocalIndex * sizeof(float), yLocalIndex);

	if (FLT_MAX == data2D) return;
	if (data2D + filterDistance > (float)zLocalIndex * voxelSize + (float)zOffset * voxelSize) return;
	if (dimensions3D.y - yLocalIndex < zLocalIndex - (int)floorf(-(float)zOffset + filterDistance / voxelSize)) return;
}
#pragma endregion

#endif

#pragma region Do Noise Filter
void MarchingCubes::DoNoiseFilter(
	std::shared_ptr<pointcloud_Hios> patch,
	std::shared_ptr<pointcloud_Hios> compound,
	const Eigen::Matrix4f & transform,
	pointcloud_Hios & tmp_PC1,
	CUstream_st * stream, cached_allocator * alloc, bool async)
{
	NvtxRangeCuda nvtxPrint("DoNoiseFilter");

	auto noiseFilterCenter = (Eigen::Vector3f)((exeInfo.local.localMin + exeInfo.local.localMax) * 0.5f);

	noiseFilter->SetData(patch, compound, transform, noiseFilterCenter, m_voxelSize, stream, alloc);

	auto _voxelValues = thrust::raw_pointer_cast(m_MC_voxelValues.data());
	auto _voxelValueCounts = thrust::raw_pointer_cast(m_MC_voxelValueCounts.data());
	auto _voxelNormals = thrust::raw_pointer_cast(m_MC_voxelNormals.data());
	auto _voxelColors = thrust::raw_pointer_cast(m_MC_voxelColors.data());
	auto _voxelColorScores = thrust::raw_pointer_cast(m_MC_voxelColorScores.data());
	auto _voxelSegmentations = thrust::raw_pointer_cast(m_MC_voxelSegmentations.data());
	auto _voxelExtraAttribs = thrust::raw_pointer_cast(m_MC_voxelExtraAttribs.data());

	Eigen::AlignedBox3f globalScanAreaAABB(exeInfo.global.globalMin, exeInfo.global.globalMax);

	noiseFilter->RaycastFromPatch(
		exeInfo.globalHashInfo,
		exeInfo.globalHashInfo_host,
		globalScanAreaAABB,
		_voxelValues,
		_voxelValueCounts,
		_voxelNormals,
		_voxelColors,
		_voxelColorScores,
		_voxelSegmentations,
		_voxelExtraAttribs,
		stream, alloc);

	//cudaStreamSynchronize(stream);

	{
		NvtxRangeCuda nvtxPrint("Update Compound", true, true, 0xFF0000FF);

		auto begin_vtx = compound->get_TupleIter_Main(0);
		auto res_vtx = tmp_PC1.get_TupleIter_Main(0);
		auto end_vtx = thrust::copy_if(
			thrust::cuda::par_nosync(*alloc).on(stream),
			begin_vtx, begin_vtx + compound->m_nowSize, res_vtx,
			[] __device__(const auto & x) {
			const Eigen::Vector3f& point = thrust::get<0>(x);
			bool is_nan = isnan(point(0)) || isnan(point(1)) || isnan(point(2));
			bool is_max = FLT_VALID(point(0)) && FLT_VALID(point(1)) && FLT_VALID(point(2));
			bool is_infinite = isinf(point(0)) || isinf(point(1)) || isinf(point(2));
			return !is_nan && !is_infinite && is_max;
		}
		);

		//cudaStreamSynchronize(stream);

		tmp_PC1.m_nowSize = thrust::distance(res_vtx, end_vtx);

		__memCpyPC2PC_v2(tmp_PC1, *compound, PC_ALL, stream, true);

		//	mscho	@20250709
		noiseFilter->setPcdFiltered(true);
		//compound->rm_PC_nanf(filterCache.allocator, filterCache.stream);
		//	mscho	@20250520
		//cudaStreamSynchronize(stream);
	}
}

//	mscho	@20250709
void MarchingCubes::DoNoiseFilterClear(CUstream_st * *clear_stream, CUstream_st * stream, cached_allocator * alloc, bool bPreClear, bool async)
{
	if (bPreClear)
	{
		if (!noiseFilter->isCacheCleared())
		{
			noiseFilter->Clear(stream, alloc);
			noiseFilter->setCacheClear(true);
		}
		else if (!async)
		{
			for (int i = 0; i < 11; i++)
			{
				checkCudaErrors(cudaStreamSynchronize(clear_stream[i]));
			}
		}
	}
	else
	{
		if (noiseFilter->isPcdFiltered())
		{
			noiseFilter->ClearBlock(clear_stream, alloc);
			noiseFilter->setCacheClear(true);
			noiseFilter->setPcdFiltered(false);
		}
	}

}
#pragma endregion

void MarchingCubes::SaveClusterCacheVoxels(const string & filename)
{
	noiseFilter->SaveClusterCacheVoxels(filename);
}

void MarchingCubes::SaveCompound(const string & filename)
{
	noiseFilter->SaveCompound(filename);
}

void MarchingCubes::SavePatch(const string & filename)
{
	noiseFilter->SavePatch(filename);
}

void MarchingCubes::SaveLocalCache2D(const string & filename)
{
	noiseFilter->SaveLocalCache2D(filename);
}
#pragma endregion

void MarchingCubes::SaveMCVoxelsAsOffFormat(const std::string & filename, CUstream_st * st)
{
	qDebug("SaveVoxelValues()");

	auto voxelPositions = new Eigen::Vector3f[m_MC_voxelValues.size()];
	auto voxelValues = new voxel_value_t[m_MC_voxelValues.size()];

	cudaMemcpyAsync(voxelPositions, thrust::raw_pointer_cast(m_MC_voxelPositions.data()), sizeof(Eigen::Vector3f) * m_MC_voxelValues.size(), cudaMemcpyDeviceToHost, st);
	cudaMemcpyAsync(voxelValues, thrust::raw_pointer_cast(m_MC_voxelValues.data()), sizeof(voxel_value_t) * m_MC_voxelValues.size(), cudaMemcpyDeviceToHost, st);
	checkCudaSync(st);

	OFFFormat off;
	for (size_t i = 0; i < m_MC_voxelValues.size(); i++)
	{
		auto& v = voxelPositions[i];
		auto& vv = voxelValues[i];
		if (SHORT_VALID(vv) && VECTOR3F_VALID_(v))
		{
			off.AddPointFloat3(v.data());
		}
	}
	off.Serialize(filename);

	delete[] voxelPositions;
	delete[] voxelValues;
}

// 이미지 좌표계로 변환
void TransformToImageSpace(const thrust::device_vector<Eigen::Vector3f>&localPoints, thrust::device_vector<Eigen::Vector2i>&imagePoints,
	const Eigen::Matrix4f & dev_camRT, const Eigen::Matrix3f & dev_cam_tilt, CUstream_st * stream) {
	thrust::transform(localPoints.begin(), localPoints.end(), imagePoints.begin(),
		[dev_camRT, dev_cam_tilt] __device__(const Eigen::Vector3f & localPoint) {
		Eigen::Vector2i imagePoint(0, 0);
		if (FLT_VALID(localPoint.z())) {
			if (!ColorUtil::getPixelCoord_pos(localPoint, imagePoint, dev_camRT, dev_cam_tilt))
				imagePoint = Eigen::Vector2i(-1, -1);
		}
		else
			imagePoint = Eigen::Vector2i(-1, -1);
		return imagePoint;
	});

}