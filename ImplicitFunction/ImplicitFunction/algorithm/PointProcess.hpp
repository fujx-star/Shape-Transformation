#ifndef __POINT_PROCESS_HPP__
#define __POINT_PROCESS_HPP__

#define POINT_SIZE 1.0f
#define ALPHA 100.0f
#include <iostream>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>

bool isEqual(float x, float y) {
	return fabs(x - y) < 1e-10;
}

void pointConvertTriangle(const std::vector<Eigen::Vector3f>& points, std::vector<Eigen::Vector3f>& actualPoints) {
	for (const auto& point : points) {
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));

		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));

		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));

		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));

		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() - POINT_SIZE, point.z() - POINT_SIZE));

		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() + POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() + POINT_SIZE));
		actualPoints.push_back(Eigen::Vector3f(point.x() - POINT_SIZE, point.y() + POINT_SIZE, point.z() - POINT_SIZE));
	}
}

// 判断target在start和end的直线的哪边
// 右边返回1，左边返回-1，在线上返回0
int pointSide(Eigen::Vector3f start, Eigen::Vector3f end, Eigen::Vector3f target) {
	// 忽略第三维度
	float res = (end[1] - start[1]) * target[0] + (start[0] - end[0]) * target[1] + end[0] * start[1] - start[0] * end[1];
	if (isEqual(res, 0)) {
		return 0;
	}
	else if (res < 0) {
		return -1;
	}
	else {
		return 1;
	}
}

// 所有的点在凸包边界的右边，暴力遍历
void SlowConvexHull(std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f>& retPoints) {
	int n = points.size();
	if (n <= 2) {
		retPoints = points;
		return;
	}
	std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> lines;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i != j) {
				bool isBorder = true;
				for (int k = 0; k < n; k++) {
					if (k != i && k != j) {
						if (pointSide(points[i], points[j], points[k]) <= 0) {
							isBorder = false;
							break;
						}
					}
				}
				if (isBorder) {
					lines.push_back({ points[i], points[j]});
				}
			}
		}
	}
	retPoints.push_back(lines[0].first);
	retPoints.push_back(lines[0].second);
	int linesSize = lines.size();
	int cnt = 1;
	while (cnt < linesSize) {
		auto endPoint = retPoints.back();
		for (auto& line : lines) {
			if (line.first == endPoint) {
				retPoints.push_back(line.second);
				break;
			}
		}
		cnt++;
	}
	//std::cout << points.size() << ret.size();
}

// 迭代法，看后三个点朝左拐还是朝右拐，凸包边界肯定是朝右拐
void ConvexHull(std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f>& retPoints) {
	int n = points.size();
	if (n <= 2) {
		retPoints = points;
		return;
	}
	std::vector<Eigen::Vector3f> upperHull, lowerHull;
	// 按照横坐标大小排序
	sort(points.begin(), points.end(), [](const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
		return p1.x() < p2.x();
	});

	upperHull = { points[0], points[1] };
	for (int i = 2; i < n; i++) {
		// 如果能形成转折，如果是朝左转的话就舍弃中间节点
		Eigen::Vector3f start, middle, end;
		end = points[i];
		while (upperHull.size() >= 2) {
			start = upperHull[upperHull.size() - 2];
			middle = upperHull.back();
			if (pointSide(start, middle, end) < 0) {
				upperHull.pop_back();
			}
			else {
				break;
			}
		}
		upperHull.push_back(points[i]);
	}

	lowerHull = { points[n - 1], points[n - 2] };
	for (int i = n - 3; i >= 0; i--) {
		Eigen::Vector3f start, middle, end;
		end = points[i];
		while (lowerHull.size() >= 2) {
			start = lowerHull[lowerHull.size() - 2];
			middle = lowerHull.back();
			if (pointSide(start, middle, end) < 0) {
				lowerHull.pop_back();
			}
			else {
				break;
			}
		}
		lowerHull.push_back(points[i]);
	}

	// 首尾相接，横坐标最小和最大的点肯定在凸包边界上
	retPoints = upperHull;
	for (int i = 0; i < lowerHull.size(); i++) {
		if (i != 0 && i != lowerHull.size() - 1) {
			retPoints.push_back(lowerHull[i]);
		}
	}
	//std::cout << points.size() << retPoints.size();
}

float eh(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2) {
	return sqrt(pow(ALPHA, 2) / (p1 - p2).squaredNorm() - 0.25f);
}

void circleCenterCalc(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, Eigen::Vector3f& c1, Eigen::Vector3f& c2) {
	float ehValue = eh(p1, p2);
	c1 = { (p1[0] + p2[0]) / 2 - ehValue * (p1[1] - p2[1]), (p1[1] + p2[1]) / 2 - ehValue * (p2[0] - p1[0]), 0.0f };
	c2 = { (p1[0] + p2[0]) / 2 + ehValue * (p1[1] - p2[1]), (p1[1] + p2[1]) / 2 + ehValue * (p2[0] - p1[0]), 0.0f };
}

void ConcaveHull(const std::vector<Eigen::Vector3f>& points, std::vector<int>& retPointIndexes, std::vector<std::pair<int, int>>& retEdgeIndexes) {
	int n = points.size();
	std::vector<std::vector<int>> candPoints(n);
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			float squaredDistance = (points[i] - points[j]).squaredNorm();
			if (squaredDistance < ALPHA * 2) {
				candPoints[i].push_back(j);
				candPoints[j].push_back(i);
			}
		}
	}
	for (int i = 0; i < points.size(); i++) {
		for (int j = 0; j < candPoints[i].size(); j++) {
			if (candPoints[i][j] <= i) {
				continue;
			}
			Eigen::Vector3f c1, c2;
			circleCenterCalc(points[i], points[candPoints[i][j]], c1, c2);
			bool flag1 = true, flag2 = true;
			for (int k = 0; k < candPoints[i].size(); k++) {
				if (candPoints[i][k] != candPoints[i][j]) {
					if ((c1 - points[candPoints[i][k]]).norm() <= ALPHA) {
						flag1 = false;
					}
					if ((c2 - points[candPoints[i][k]]).norm() <= ALPHA) {
						flag2 = false;
					}
				}
			}
			// 此时points[i]和points[candPoints[i][j]]构成凹包边界
			if (flag1 || flag2) {
				retPointIndexes.emplace_back(i);
				retEdgeIndexes.emplace_back(std::make_pair(i, candPoints[i][j]));
			}
		}
	}
}

#endif
