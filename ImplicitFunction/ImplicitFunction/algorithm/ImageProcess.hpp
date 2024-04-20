#ifndef __IMAGE_PROCESS_HPP__
#define __IMAGE_PROCESS_HPP__

#include "../algorithm/ConvexHull.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#define AREA_LIMIT 1000
#define EPSILON 0.003
#define LOW_THRESHOLD 100
#define HIGH_THRESHOLD 200
#define APERTURE_SIZE 3
#define SAMPLE_NUM 30

#define X_MIN -3.0
#define X_MAX 3.0
#define X_SPAN (X_MAX - X_MIN)
#define Y_MIN -3.0
#define Y_MAX 3.0
#define Y_SPAN (Y_MAX - Y_MIN)
#define Z_MIN 0.0
#define Z_MAX 0.0
#define Z_SPAN (Z_MAX - Z_MIN)

#define IMAGE_DEBUG


void scalePoints(const std::vector<cv::Point>& cvPoints, std::vector<Eigen::Vector2f>& eigenPoints, int rows, int cols) {
	for (const auto& cvPoint : cvPoints) {
		float x = static_cast<float>(cvPoint.x);
		float y = static_cast<float>(cvPoint.y);
		eigenPoints.emplace_back(Eigen::Vector2f(x / cols * X_SPAN + X_MIN, y / rows * Y_SPAN + Y_MIN));
	}
}


// 图像处理函数1：用opencv提取外轮廓和内轮廓并逼近，分别得到边界约束点和法向约束点
void processImage1(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints) 
{
	cv::Mat srcImage = cv::imread(imagePath);
	cv::Mat grayImage, threshImage;
	cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);

#ifdef IMAGE_DEBUG
	cv::imshow("gray_image", grayImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	threshold(grayImage, threshImage, 127, 255, cv::THRESH_BINARY);

#ifdef IMAGE_DEBUG
	cv::imshow("thresh_image", threshImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	int rows = srcImage.rows;
	int cols = srcImage.cols;

	cv::Mat cannyImage;
	cv::Canny(threshImage, cannyImage, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE_SIZE);

#ifdef IMAGE_DEBUG
	cv::imshow("canny_image", cannyImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(cannyImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<double> areas;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > AREA_LIMIT) {
			areas.emplace_back(area);
		}
	}
	int maxPos = max_element(areas.begin(), areas.end()) - areas.begin();
	int minPos = min_element(areas.begin(), areas.end()) - areas.begin();

	std::vector<cv::Point> externalContour = contours[maxPos];
	std::vector<cv::Point> internalContour = contours[minPos];

	std::vector<cv::Point> externalContourProx, internalContourProx;
	cv::approxPolyDP(externalContour, externalContourProx, EPSILON * cv::arcLength(externalContour, true), true);
	cv::approxPolyDP(internalContour, internalContourProx, EPSILON * cv::arcLength(internalContour, true), true);

#ifdef IMAGE_DEBUG
	std::vector<std::vector<cv::Point>> contourProxs = { externalContourProx, internalContourProx };
	cv::Mat outputImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::drawContours(outputImage, contourProxs, 0, cv::Scalar(255, 0, 0));
	cv::drawContours(outputImage, contourProxs, 1, cv::Scalar(0, 255, 0));
	cv::imshow("prox_image", outputImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	scalePoints(externalContourProx, boundaryPoints, rows, cols);
	scalePoints(internalContourProx, normalPoints, rows, cols);
}

// 角平分线法求法向量
void normalWithoutWeights(const cv::Point& pre, const cv::Point& cur, const cv::Point& post, cv::Vec2f& normal) {
	cv::Vec2f v1 = cv::Vec2f(post.x - cur.x, post.y - cur.y);
	cv::Vec2f v2 = cv::Vec2f(pre.x - cur.x, pre.y - cur.y);
	normal = cv::normalize(cv::normalize(v1) + cv::normalize(v2));
}

// 加权平均法求法向量
void normalWithWeights(const cv::Point& pre, const cv::Point& cur, const cv::Point& post, cv::Vec2f& normal) {
	cv::Vec2f v1 = cv::Vec2f(post.x - cur.x, post.y - cur.y);
	cv::Vec2f v2 = cv::Vec2f(pre.x - cur.x, pre.y - cur.y);
	normal = cv::normalize(v1 + v2);
}

// 求法向约束点
void normalPointCalc(const std::vector<cv::Point>& boundaryPoints, std::vector<cv::Point>& normalPoints) {
	int n = boundaryPoints.size();
	auto previousIndex = [=](int i) { return (i - 1 + n) % n; };
	auto postIndex = [=](int i) { return (i + 1) % n; };
	cv::Point candidiates[2];
	bool candidateInsideContour[2];
	for (int i = 0; i < n; i++) {
		cv::Point cur = boundaryPoints[i];
		cv::Point prev = boundaryPoints[previousIndex(i)];
		cv::Point post = boundaryPoints[postIndex(i)];
		cv::Vec2f normal;
		normalWithWeights(prev, cur, post, normal);
		float offset = 1.0f;
		bool findPoint = true;
		while (findPoint) {
			candidiates[0] = cv::Point(cur.x + normal[0] * offset, cur.y + normal[1] * offset);
			candidiates[1] = cv::Point(cur.x - normal[0] * offset, cur.y - normal[1] * offset);
			candidateInsideContour[0] = pointPolygonTest(boundaryPoints, candidiates[0], true);
			candidateInsideContour[1] = pointPolygonTest(boundaryPoints, candidiates[1], true);
			findPoint = candidateInsideContour[0] ^ candidateInsideContour[1];
			offset += 1.0f;
		}
		if (candidateInsideContour[0]) {
			normalPoints.emplace_back(candidiates[0]);
		}
		else {
			normalPoints.emplace_back(candidiates[1]);
		}
	}
}


// 图像处理函数2：用opencv提取外轮廓得到边界约束点并逼近，根据边界约束点计算法向约束点
void processImage2(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints)
{
	cv::Mat srcImage = cv::imread(imagePath);
	int rows = srcImage.rows;
	int cols = srcImage.cols;

	cv::Mat cannyImage;
	cv::Canny(srcImage, cannyImage, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE_SIZE);

#ifdef IMAGE_DEBUG
	cv::imshow("canny_image", cannyImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(cannyImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<double> areas;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > AREA_LIMIT) {
			areas.emplace_back(area);
		}
	}
	int maxPos = max_element(areas.begin(), areas.end()) - areas.begin();
	std::vector<cv::Point> externalContour = contours[maxPos];
	std::vector<cv::Point> externalContourProx, internalContourProx;
	cv::approxPolyDP(externalContour, externalContourProx, EPSILON * cv::arcLength(externalContour, true), true);
	normalPointCalc(externalContourProx, internalContourProx);

#ifdef IMAGE_DEBUG
	std::vector<std::vector<cv::Point>> contourProxs = { externalContourProx, internalContourProx };
	cv::Mat outputImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::drawContours(outputImage, contourProxs, 0, cv::Scalar(255, 0, 0));
	cv::drawContours(outputImage, contourProxs, 1, cv::Scalar(0, 255, 0));
	cv::imshow("prox_image", outputImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	scalePoints(externalContourProx, boundaryPoints, rows, cols);
	scalePoints(internalContourProx, normalPoints, rows, cols);
}

// 图像处理函数3：用opencv提取外轮廓得到边界约束点并手动逼近，根据边界约束点计算法向约束点
void processImage3(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints)
{
	cv::Mat srcImage = cv::imread(imagePath);
	int rows = srcImage.rows;
	int cols = srcImage.cols;

	cv::Mat cannyImage;
	cv::Canny(srcImage, cannyImage, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE_SIZE);

#ifdef IMAGE_DEBUG
	cv::imshow("canny_image", cannyImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(cannyImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<double> areas;
	for (const auto& contour : contours) {
		double area = cv::contourArea(contour);
		if (area > AREA_LIMIT) {
			areas.emplace_back(area);
		}
	}
	int maxPos = max_element(areas.begin(), areas.end()) - areas.begin();
	std::vector<cv::Point> externalContour = contours[maxPos];
	std::vector<cv::Point> externalContourProx, internalContourProx;
	double contourLength = cv::arcLength(externalContour, true);
	double segmentLength = contourLength / SAMPLE_NUM;
	int contourPointsNum = externalContour.size();
	double curLength = cv::norm(externalContour[1] - externalContour[0]), preLength = 0.0;
	int cnt = 1;
	for (int i = 1; i < contourPointsNum; i++) {
		curLength += cv::norm(externalContour[i] - externalContour[i - 1]);
		if (curLength > segmentLength * cnt) {
			if (curLength - segmentLength * cnt > segmentLength * cnt - preLength) {
				externalContourProx.emplace_back(externalContour[i - 1]);
			}
			else {
				externalContourProx.emplace_back(externalContour[i]);
			}
			cnt++;
		}
		preLength = curLength;
	}
	normalPointCalc(externalContourProx, internalContourProx);

#ifdef IMAGE_DEBUG
	std::vector<std::vector<cv::Point>> contourProxs = { externalContourProx, internalContourProx };
	cv::Mat outputImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::drawContours(outputImage, contourProxs, 0, cv::Scalar(255, 0, 0));
	cv::drawContours(outputImage, contourProxs, 1, cv::Scalar(0, 255, 0));
	cv::imshow("prox_image", outputImage);
	cv::waitKey(0);
#endif // IMAGE_DEBUG

	scalePoints(externalContourProx, boundaryPoints, rows, cols);
	scalePoints(internalContourProx, normalPoints, rows, cols);
}

// 图像处理函数4：用自己的边界检测算法得到边界约束点，根据边界约束点计算法向约束点
void processImage4(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints)
{

}

#endif