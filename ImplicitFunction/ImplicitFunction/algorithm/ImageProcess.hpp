#ifndef __IMAGE_PROCESS_HPP__
#define __IMAGE_PROCESS_HPP__

#include "../algorithm/ConvexHull.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>
#define AREA_LIMIT 1000
#define EPSILON 0.005
#define LOW_THRESHOLD 100
#define HIGH_THRESHOLD 200
#define APERTURE_SIZE 3

#define X_MIN -3.0
#define X_MAX 3.0
#define X_SPAN (X_MAX - X_MIN)
#define Y_MIN -3.0
#define Y_MAX 3.0
#define Y_SPAN (Y_MAX - Y_MIN)
#define Z_MIN 0.0
#define Z_MAX 0.0
#define Z_SPAN (Z_MAX - Z_MIN)

#define DEBUG


// 用opencv提取外轮廓和内轮廓，分别得到边界约束点和法向约束点
void processImage1(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints) 
{
	cv::Mat srcImage = cv::imread(imagePath);
	cv::Mat grayImage, threshImage;
	cv::cvtColor(srcImage, grayImage, cv::COLOR_BGR2GRAY);

#ifdef DEBUG
	cv::imshow("gray_image", grayImage);
	cv::waitKey(0);
#endif // DEBUG

	threshold(grayImage, threshImage, 127, 255, cv::THRESH_BINARY);

#ifdef DEBUG
	cv::imshow("thresh_image", threshImage);
	cv::waitKey(0);
#endif // DEBUG

	int rows = srcImage.rows;
	int cols = srcImage.cols;

	cv::Mat cannyImage;
	cv::Canny(threshImage, cannyImage, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE_SIZE);

#ifdef DEBUG
	cv::imshow("canny_image", cannyImage);
	cv::waitKey(0);
#endif // DEBUG

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

#ifdef DEBUG
	std::vector<std::vector<cv::Point>> contourProxs = { externalContourProx, internalContourProx };
	cv::Mat outputImage = cv::Mat::zeros(srcImage.size(), CV_8UC3);
	cv::drawContours(outputImage, contourProxs, 0, cv::Scalar(255, 0, 0));
	cv::drawContours(outputImage, contourProxs, 1, cv::Scalar(0, 255, 0));
	cv::imshow("prox_image", outputImage);
	cv::waitKey(0);
#endif // DEBUG

	for (int i = 0; i < externalContourProx.size(); i++) {
		float x = static_cast<float>(externalContourProx[i].x);
		float y = static_cast<float>(externalContourProx[i].y);
		boundaryPoints.emplace_back(Eigen::Vector2f(x / cols * X_SPAN + X_MIN, y / rows * Y_SPAN + Y_MIN));
	}
	for (int i = 0; i < internalContourProx.size(); i++) {
		float x = static_cast<float>(internalContourProx[i].x);
		float y = static_cast<float>(internalContourProx[i].y);
		normalPoints.emplace_back(Eigen::Vector2f(x / cols * X_SPAN + X_MIN, y / rows * Y_SPAN + Y_MIN));
	}
	//for (int i = 0; i < externalContour.size(); i++) {
	//	float x = static_cast<float>(externalContour[i].x);
	//	float y = static_cast<float>(externalContour[i].y);
	//	boundaryPoints.emplace_back(Eigen::Vector2f(x / cols * X_SPAN + X_MIN, y / rows * Y_SPAN + Y_MIN));
	//}
	//for (int i = 0; i < internalContour.size(); i++) {
	//	float x = static_cast<float>(internalContour[i].x);
	//	float y = static_cast<float>(internalContour[i].y);
	//	normalPoints.emplace_back(Eigen::Vector2f(x / cols * X_SPAN + X_MIN, y / rows * Y_SPAN + Y_MIN));
	//}
}


// 用opencv提取外轮廓得到边界约束点，根据边界约束点计算法向约束点
void processImage2(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints)
{
	cv::Mat srcImage = cv::imread(imagePath);
	int rows = srcImage.rows;
	int cols = srcImage.cols;

	cv::Mat cannyImage;
	cv::Canny(srcImage, cannyImage, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE_SIZE);

#ifdef DEBUG
	cv::imshow("canny_image", cannyImage);
	cv::waitKey(0);
#endif // DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(cannyImage, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	std::vector<cv::Point> externalContour = contours[0];


}

// 用自己的边界检测算法得到边界约束点，根据边界约束点计算法向约束点
void processImage3(const char* imagePath,
	std::vector<Eigen::Vector2f>& boundaryPoints,
	std::vector<Eigen::Vector2f>& normalPoints)
{

}

#endif