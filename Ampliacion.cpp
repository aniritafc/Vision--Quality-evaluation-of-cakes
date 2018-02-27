#include "stdafx.h" 
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <windows.h>
#include "opencv2/opencv.hpp" 
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#pragma comment (lib, "opencv_world320d.lib") 

using namespace std;
using namespace cv;

Mat img_threshold, img_dilated, imgOrig, imgOrig_gray;
RNG rng(12345);

int main(int argc, char** argv) {
	Mat img_threshold_aguj, img_dilate_aguj;

	imgOrig = imread("C:/Users/Rita Carvalho/Desktop/Projeto/Tortas Ampliación/Amp5.BMP", 1);

	if (imgOrig.empty()) {
		cout << "Error : Image cannot be loaded..!!" << endl;
		return 1;
	}
	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imgOrig);
	
	Mat imgROrig = imgOrig + Scalar(30, 30, 30);

	Mat brightLAB, brightHSV;
	cvtColor(imgROrig, brightHSV, CV_RGB2HSV);
	Vec3b rgb_normal(115, 111, 25);
	Mat3b rgb(rgb_normal);
	cvtColor(imgROrig, brightLAB, COLOR_RGB2Lab);

	int thresh = 40;

	Scalar minBGR = Scalar(rgb_normal.val[0] - thresh, rgb_normal.val[1] - thresh, rgb_normal.val[2] - thresh);
	Scalar maxBGR = Scalar(rgb_normal.val[0] + thresh, rgb_normal.val[1] + thresh, rgb_normal.val[2] + thresh);

	Mat maskBGR, resultBGR;
	inRange(imgROrig, minBGR, maxBGR, maskBGR);
	bitwise_and(imgROrig, imgROrig, resultBGR, maskBGR);
	//cvNamedWindow("Result BGR", CV_WINDOW_AUTOSIZE);
	//imshow("Result BGR", resultBGR);
	//cvWaitKey();
	cvtColor(resultBGR, imgOrig_gray, CV_RGB2GRAY);
	blur(imgOrig_gray, imgOrig_gray, Size(11, 11));
	//cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	//imshow("Result", imgOrig_gray);
	//cvWaitKey();
	threshold(imgOrig_gray, img_threshold, 5, 255, CV_THRESH_BINARY_INV);
	//cvNamedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	//imshow("Threshold", img_threshold);
	erode(img_threshold, img_threshold, Mat(), Point(-1, -1), 8);
	//dilate(img_threshold, img_dilated, Mat(), Point(-1, -1), 8);

	cvNamedWindow("Dilated", CV_WINDOW_AUTOSIZE);
	imshow("Dilated", img_threshold);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(img_threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	Moments mu = moments(img_threshold, true);
	Point center;

	char resultado[20];
	for (size_t i = 0; i < contours.size(); i++) {
		Moments mu = moments(contours[i], true);
		center.x = mu.m10 / mu.m00;
		center.y = mu.m01 / mu.m00;
		circle(img_threshold, center, 2, Scalar(0, 0, 255));
		cout << "x=" << center.x << "." << "y=" << center.y << endl;

	}

	cvNamedWindow("Circles", CV_WINDOW_AUTOSIZE);
	imshow("Circles", img_threshold);

	//imshow("Result", img_dilated);
	//sprintf(resultado, "X = %d Y =%d", center.x, center.y);
	//putText(imgOrig, resultado, Point(100, 500), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
	//cvNamedWindow("Ampliación", CV_WINDOW_AUTOSIZE);
	//imshow("Ampliación", imgOrig);

	cvWaitKey(0);
	return(0);

}