# VoxelSG

Voxel-based Signed Distance Function (SDF) reconstruction and ray-casting rendering system using CUDA.

## Overview

VoxelSG는 CUDA를 활용한 실시간 3D 복원 및 렌더링 시스템입니다. 깊이 맵 데이터를 해시 테이블 기반 TSDF (Truncated Signed Distance Function)로 통합하고, 레이 캐스팅을 통해 가상 뷰를 생성합니다.

## Features

- **TSDF Integration**: 깊이 맵을 해시 테이블 기반 볼륨으로 통합
- **Ray Cast Rendering**: GPU 기반 레이 캐스팅 렌더링
- **Custom Depth Map Generation**: 포인트 클라우드로부터 깊이 맵 생성
- **Point Cloud Visualization**: Open3D 기반 3D 시각화
- **Scan Data Management**: 스캔 데이터 로딩 및 저장

## Requirements

- **CUDA** 12.9+ (또는 호환 버전)
- **OpenCV** 4.5.0+
- **Open3D** (포인트 클라우드 시각화용)
- **Visual Studio** 2019+ (MSVC)
- **Windows** (현재 Windows 전용 빌드)

## Build

1. Visual Studio에서 `VoxelSG.sln` 열기
2. CUDA Toolkit 경로 설정 확인
3. Release 모드로 빌드

## Project Structure

```
VoxelSG/
├── Source/              # 레거시 소스 코드
├── VoxelSG/
│   ├── include/        # 헤더 파일
│   │   ├── VoxelScene.h
│   │   ├── RayCastRender.h
│   │   ├── scanData.h
│   │   └── ...
│   ├── src/            # 소스 파일
│   │   ├── VoxelScene.cpp/.cu
│   │   ├── RayCastRender.cpp/.cu
│   │   ├── voxelSG.cpp
│   │   └── ...
│   └── output/         # 출력 파일 (gitignore)
└── VoxelSG.sln         # Visual Studio 솔루션
```

## Usage

기본 사용 예시:

```cpp
// VoxelScene 초기화
GlobalParamsConfig::get().initialize();
VoxelScene scene(params);

// 스캔 데이터 로드 및 통합
scene.integrateFromScanData(d_depthmap, d_colormap, d_normalmap,
                            width, height, truncation, transform,
                            fx, fy, cx, cy);

// 레이 캐스팅 렌더링
RayCastRender renderer;
renderer.initialize(width, height);
renderer.render(&scene, cameraPos, transform, fx, fy, cx, cy,
                minDepth, maxDepth);
```

## Configuration

주요 파라미터는 `globalParamasConfig.cpp`에서 설정:

- `g_SDFVoxelSize`: 보셀 크기 (기본: 0.004m = 4mm)
- `g_SDFTruncation`: TSDF 절단 거리 (기본: 0.02m)
- `g_hashNumSlots`: 해시 테이블 슬롯 수 (기본: 200000)

## Output

- 렌더된 깊이 맵 (OpenCV imshow)
- 포인트 클라우드 (.ply)
- 커스텀 깊이 맵 (bin, xyz)

## License

[License 정보 추가 필요]

## Author

[작성자 정보]

