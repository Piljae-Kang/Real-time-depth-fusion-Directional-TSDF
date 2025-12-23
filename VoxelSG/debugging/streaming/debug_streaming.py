import numpy as np

# 카메라 위치
camera_pos = np.array([2.979228, -1.340333, 16.123268])
radius = 30

# 구면 좌표계를 사용하여 구의 표면 점 생성
# theta: azimuth angle (0 to 2π)
# phi: polar angle (0 to π)
num_points_theta = 50  # 수평 방향 점 개수
num_points_phi = 50    # 수직 방향 점 개수

points = []

for i in range(num_points_phi):
    phi = np.pi * i / (num_points_phi - 1)  # 0 to π
    for j in range(num_points_theta):
        theta = 2 * np.pi * j / num_points_theta  # 0 to 2π
        
        # 구면 좌표를 직교 좌표로 변환
        x = camera_pos[0] + radius * np.sin(phi) * np.cos(theta)
        y = camera_pos[1] + radius * np.sin(phi) * np.sin(theta)
        z = camera_pos[2] + radius * np.cos(phi)
        
        points.append([x, y, z])

# .xyz 파일로 저장
output_file = "sphere_boundary.xyz"
with open(output_file, 'w') as f:
    for point in points:
        f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

print(f"구의 경계 점 {len(points)}개가 {output_file}에 저장되었습니다.")
print(f"카메라 위치: ({camera_pos[0]:.6f}, {camera_pos[1]:.6f}, {camera_pos[2]:.6f})")
print(f"반경: {radius}")

