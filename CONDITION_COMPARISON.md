# Hubitz 코드 vs 현재 코드 조건 비교

## 1. Voxel 유효성 체크

### Hubitz 코드 (H_MarchingCubes.cu)
```cpp
// 1. SDF 값 유효성
if (!FLT_VALID(centerVoxelValue)) return;

// 2. Normal 유효성
if (!FLT_VALID(centerVoxelNormal.x())) return;
centerVoxelNormal.normalize();

// 3. Integration count 체크
auto centerVoxelValueCount = VOXELCNT_VALUE(voxelValueCounts[slotIndex]);
if (centerVoxelValueCount < 1 || (filtering && centerVoxelValueCount < 2))
    return;
```

### 현재 코드 (RayCastRender.cu)
```cpp
// 1. Weight 체크만 수행
if (voxel.weight == 0) continue;

// 2. SDF 범위 체크
if (voxel.sdf < minSDF || voxel.sdf > maxSDF) continue;
```

**차이점:**
- ❌ Hubitz: Normal 유효성 체크 + Integration count 체크
- ✅ 현재: Weight 체크만 (Normal은 나중에 계산)
- ❌ Hubitz: SDF 범위 체크 없음 (모든 voxel 처리)
- ✅ 현재: SDF 범위 체크 있음 (-0.05 ~ 0.05)

---

## 2. Iso-Surface 위치 계산

### Hubitz 코드
```cpp
// Normal을 사용한 계산
auto iso_pos = pc - centerVoxelNormal * centerVoxelValue;

// Iso-surface가 voxel 경계 안에 있는지 체크
if ((voxelMin.x() > iso_pos.x() || iso_pos.x() > voxelMax.x()) ||
    (voxelMin.y() > iso_pos.y() || iso_pos.y() > voxelMax.y()) ||
    (voxelMin.z() > iso_pos.z() || iso_pos.z() > voxelMax.z()))
    return;
```

### 현재 코드
```cpp
// Zero-crossing을 찾아서 linear interpolation
// 또는 SDF가 0에 가까우면 voxel center 사용
if (validNeighbors == 0) {
    if (fabsf(voxel.sdf) <= fmaxf(fabsf(minSDF), fabsf(maxSDF))) {
        isoPos = voxelCenter;
    } else {
        continue;
    }
} else {
    isoPos = accumulatedPos / validNeighbors;  // Zero-crossing 평균
}
```

**차이점:**
- ✅ Hubitz: Normal × SDF로 직접 계산
- ✅ 현재: Zero-crossing 찾아서 interpolation (더 정확할 수 있음)
- ✅ Hubitz: Iso-surface가 voxel 경계 밖이면 제외
- ❌ 현재: 경계 체크 없음

---

## 3. Iso-Surface 판별

### Hubitz 코드
```cpp
const bool bIsoSurface = fabsf(centerVoxelValue) <= 0.001;
```

### 현재 코드
```cpp
// 명시적인 bIsoSurface 변수 없음
// 단, zero-crossing이 없고 SDF가 범위 내면 voxel center 사용
```

**차이점:**
- ✅ Hubitz: `|SDF| <= 0.001`이면 이미 표면으로 간주
- ❌ 현재: 명시적인 iso-surface 판별 없음

---

## 4. 인접 Voxel 체크

### Hubitz 코드
```cpp
// 6방향 또는 26방향 체크
if (false == use26Direction) {
    ForEachNeighbor_v5(-1, 0, 0);  // 6방향
    ForEachNeighbor_v5(1, 0, 0);
    // ...
} else {
    ForEachNeighbor_v5(-1, -1, -1);  // 26방향
    // ...
}

// minCount = 2 (최소 2개 이상의 valid neighbor 필요)
const uint32_t minCount = 2;
if (countOfAccumulation > minCount) {
    // Point 생성
}
```

### 현재 코드
```cpp
// 6방향만 체크
int offsets[6][3] = {{1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}};

// Zero-crossing이 있으면 point 생성 (minCount 체크 없음)
if (validNeighbors > 0) {
    isoPos = accumulatedPos / validNeighbors;
    // Point 생성
} else if (fabsf(voxel.sdf) <= threshold) {
    isoPos = voxelCenter;
    // Point 생성
}
```

**차이점:**
- ✅ Hubitz: 6방향 또는 26방향 선택 가능
- ❌ 현재: 6방향만
- ✅ Hubitz: minCount = 2 (최소 2개 neighbor 필요)
- ❌ 현재: minCount 체크 없음 (1개만 있어도 생성)

---

## 5. Point 생성 조건

### Hubitz 코드
```cpp
// 조건 1: countOfAccumulation > minCount (2)
if (countOfAccumulation > minCount) {
    if (bIsoSurface) {
        pointPositions[prevTableIndex] = pc;  // Voxel center
    } else {
        pointPositions[prevTableIndex] = accumulatedPoint / countOfAccumulation;
    }
    // Point 생성
}
```

### 현재 코드
```cpp
// 조건 1: Zero-crossing이 있으면 항상 생성
// 조건 2: Zero-crossing이 없어도 SDF가 범위 내면 생성
if (validNeighbors > 0 || fabsf(voxel.sdf) <= threshold) {
    // Point 생성
}
```

**차이점:**
- ✅ Hubitz: 최소 2개 neighbor 필요 (더 엄격)
- ❌ 현재: 1개만 있어도 생성 (더 관대)
- ✅ Hubitz: bIsoSurface면 voxel center, 아니면 neighbor 평균
- ✅ 현재: Zero-crossing이 있으면 interpolation, 없으면 center

---

## 요약: 주요 차이점

| 조건 | Hubitz 코드 | 현재 코드 |
|------|------------|----------|
| **SDF 범위 필터** | 없음 (모든 voxel 처리) | 있음 (-0.05 ~ 0.05) |
| **Normal 체크** | 필수 (미리 저장됨) | 없음 (나중에 계산) |
| **Integration count** | 필수 (>= 1 또는 >= 2) | 없음 |
| **Iso-surface 계산** | `pc - normal × SDF` | Zero-crossing interpolation |
| **Voxel 경계 체크** | 있음 | 없음 |
| **인접 방향** | 6 또는 26 | 6만 |
| **Min neighbor count** | 2개 필요 | 없음 (1개만 있어도) |
| **Iso-surface 판별** | `\|SDF\| <= 0.001` | 없음 |

---

## 개선 제안

1. **Voxel 경계 체크 추가**: Iso-surface 위치가 voxel 경계 밖이면 제외
2. **Min neighbor count 추가**: 최소 2개 이상의 valid neighbor 필요
3. **26방향 옵션 추가**: 더 정확한 neighbor 체크
4. **Integration count 체크**: Weight 대신 integration count도 체크

