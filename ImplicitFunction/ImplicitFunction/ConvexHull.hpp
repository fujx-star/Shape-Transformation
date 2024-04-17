#ifndef __CONVEXHULL_HPP__
#define __CONVEXHULL_HPP__

#include <iostream>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Eigen>

#define POINT_SIZE 0.003

bool isEqual(float x, float y) {
	return fabs(x - y) < 1e-10;
}

void point2DConvertTriangle(const std::vector<Eigen::VectorXf>& points, std::vector<float>& glPoints) {
	for (auto& point : points) {
		glPoints.push_back(point.x() - POINT_SIZE);
		glPoints.push_back(point.y() + POINT_SIZE);
		glPoints.push_back(point.x() - POINT_SIZE);
		glPoints.push_back(point.y() - POINT_SIZE);
		glPoints.push_back(point.x() + POINT_SIZE);
		glPoints.push_back(point.y() + POINT_SIZE);
		glPoints.push_back(point.x() - POINT_SIZE);
		glPoints.push_back(point.y() - POINT_SIZE);
		glPoints.push_back(point.x() + POINT_SIZE);
		glPoints.push_back(point.y() + POINT_SIZE);
		glPoints.push_back(point.x() + POINT_SIZE);
		glPoints.push_back(point.y() - POINT_SIZE);
	}
}

// �ж�target��start��end��ֱ�ߵ��ı�
// �ұ߷���1����߷���-1�������Ϸ���0
float pointSide(glm::vec2 start, glm::vec2 end, glm::vec2 target) {
	float res = (end.y - start.y) * target.x + (start.x - end.x) * target.y + end.x * start.y - start.x * end.y;
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

// ���еĵ���͹���߽���ұߣ���������
void SlowConvexHull(std::vector<glm::vec2> points, std::vector<glm::vec2>& ret) {
	int n = points.size();
	if (n <= 2) {
		ret = points;
		return;
	}
	std::vector<std::pair<glm::vec2, glm::vec2>> lines;
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
					lines.push_back({ points[i], points[j] });
				}
			}
		}
	}
	ret.push_back(lines[0].first);
	ret.push_back(lines[0].second);
	int linesSize = lines.size();
	int cnt = 1;
	while (cnt < linesSize) {
		auto endPoint = ret.back();
		for (auto& line : lines) {
			if (line.first == endPoint) {
				ret.push_back(line.second);
				break;
			}
		}
		cnt++;
	}
	//std::cout << points.size() << ret.size();
}

bool cmp(glm::vec2 v1, glm::vec2 v2) {
	return v1.x < v2.x;
}

// �����������������㳯��ջ��ǳ��ҹգ�͹���߽�϶��ǳ��ҹ�
void ConvexHull(std::vector<glm::vec2> points, std::vector<glm::vec2>& ret) {
	int n = points.size();
	if (n <= 2) {
		ret = points;
		return;
	}
	std::vector<glm::vec2> upperHull, lowerHull;
	// ���պ������С����
	sort(points.begin(), points.end(), cmp);

	upperHull = { points[0], points[1] };
	for (int i = 2; i < n; i++) {
		// ������γ�ת�ۣ�����ǳ���ת�Ļ��������м�ڵ�
		glm::vec2 start, middle, end;
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
		glm::vec2 start, middle, end;
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

	// ��β��ӣ���������С�����ĵ�϶���͹���߽���
	ret = upperHull;
	for (int i = 0; i < lowerHull.size(); i++) {
		if (i != 0 && i != lowerHull.size() - 1) {
			ret.push_back(lowerHull[i]);
		}
	}
	//std::cout << points.size() << ret.size();
}

#endif
