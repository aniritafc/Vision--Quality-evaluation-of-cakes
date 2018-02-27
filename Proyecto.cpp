#include "stdafx.h" 
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <chrono>
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
using namespace std::chrono;

Mat img_threshold, img_dilated, imgOrig, imgOrig_gray;
RNG rng(12345);

int agujereadas(high_resolution_clock::time_point t1) {
	Vec3b rgb_normal(115, 111, 25);
	Mat3b rgb(rgb_normal), ycb;
	Mat maskYCB, resultYCB, brightYCB, img_eroded_aguj, img_threshold_aguj;
	int thresh = 40;

	cvtColor(rgb, ycb, COLOR_RGB2YCrCb);

	//Get back the vector from Mat
	Vec3b ycbPixel(ycb.at<Vec3b>(0, 0));

	Scalar minYCB = Scalar(ycbPixel.val[0] - thresh, ycbPixel.val[1] - thresh, ycbPixel.val[2] - thresh);
	Scalar maxYCB = Scalar(ycbPixel.val[0] + thresh, ycbPixel.val[1] + thresh, ycbPixel.val[2] + thresh);

	cvtColor(imgOrig, brightYCB, COLOR_RGB2YCrCb);

	inRange(brightYCB, minYCB, maxYCB, maskYCB);
	bitwise_and(brightYCB, brightYCB, resultYCB, maskYCB);
	//cvNamedWindow("Result YCB", CV_WINDOW_AUTOSIZE);
	//imshow("Result YCB", resultYCB);
	//cvWaitKey();
	//Rect myROI(110, 70, 450, 450);

	//resultYCB = resultYCB(myROI);

	blur(resultYCB, resultYCB, Size(9,9));
	threshold(resultYCB, img_threshold_aguj, 40, 255, CV_THRESH_BINARY_INV);
	//cvNamedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	//imshow("Threshold", img_threshold_aguj);
	//cvWaitKey();

	erode(img_threshold_aguj, img_eroded_aguj, Mat(), Point(-1, -1), 3);
	//cvNamedWindow("erode", CV_WINDOW_AUTOSIZE);
	//imshow("erode", img_dilate_aguj);
	//cvWaitKey();

	SimpleBlobDetector::Params params_aguj;
	params_aguj.minDistBetweenBlobs = 50.0f;
	params_aguj.filterByInertia = false;
	params_aguj.minInertiaRatio = 0.7;
	params_aguj.filterByConvexity = false;
	params_aguj.filterByColor =  true;
	params_aguj.blobColor = 0;
	params_aguj.filterByCircularity = false;
	////params_aguj.minCircularity = 0.1;
	params_aguj.filterByArea = true;
	params_aguj.minArea = 37.5f;
	params_aguj.maxArea = 700.0f;

	Ptr<SimpleBlobDetector> detector_aguj = SimpleBlobDetector::create(params_aguj);
	std::vector<KeyPoint> keypoints_aguj;
	detector_aguj->detect(img_eroded_aguj, keypoints_aguj);
	Mat img_with_keypoints_aguj;
	drawKeypoints(img_eroded_aguj, keypoints_aguj, img_with_keypoints_aguj, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cvNamedWindow("CircAguj", CV_WINDOW_AUTOSIZE);
	//imshow("CircAguj", img_with_keypoints_aguj);
	//cvWaitKey();
	if (keypoints_aguj.size() != 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Tiempo: " << duration << " " << endl;
		putText(imgOrig, "Agujereada", Point(80, 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
		cvNamedWindow("Agujereada", CV_WINDOW_AUTOSIZE);
		imshow("Agujereada", imgOrig);
		cvWaitKey();
		return -1;
	}


}

int crudas(high_resolution_clock::time_point t1) {

	Mat img_eroded_crudas, img_with_keypoints_crudas;
	erode(img_dilated, img_eroded_crudas, Mat(), Point(-1, -1), 3);
	//cvNamedWindow("Eroded", CV_WINDOW_AUTOSIZE);
	//imshow("Eroded", img_eroded_crudas);

	vector<vector<Point> > contours3;
	vector<Vec4i> hierarchy3;

	findContours(img_eroded_crudas, contours3, hierarchy3, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > hull3(contours3.size());
	for (size_t i = 0; i < contours3.size(); i++) {
		convexHull(Mat(contours3[i]), hull3[i], false);
	}
	Mat drawing_crudas = Mat::zeros(img_eroded_crudas.size(), CV_8UC3);
	for (size_t i = 0; i< contours3.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing_crudas, contours3, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing_crudas, hull3, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}

	std::vector<KeyPoint> keypoints_crudas;
	SimpleBlobDetector::Params params_crudas;
	params_crudas.minDistBetweenBlobs = 50.0f;
	params_crudas.filterByInertia = false;
	params_crudas.filterByConvexity = false;
	params_crudas.minConvexity = 0.7;
	params_crudas.maxConvexity = 0.95;
	params_crudas.filterByColor = false;
	params_crudas.blobColor = 255;
	params_crudas.filterByCircularity = false;
	params_crudas.filterByArea = true;
	params_crudas.minArea = 200.0f;
	params_crudas.maxArea = 5000.0f;

	Ptr<SimpleBlobDetector> detector_cudas = SimpleBlobDetector::create(params_crudas);
	detector_cudas->detect(drawing_crudas, keypoints_crudas);
	drawKeypoints(drawing_crudas, keypoints_crudas, img_with_keypoints_crudas, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cvNamedWindow("Circulos_Crudas", CV_WINDOW_AUTOSIZE);
	//imshow("Circulos_Crudas", img_with_keypoints_crudas);
	//cvWaitKey();
	if (keypoints_crudas.size() == 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Tiempo: " << duration << " " << endl;
		putText(imgOrig, "Cruda", Point(80, 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
		cvNamedWindow("Cruda", CV_WINDOW_AUTOSIZE);
		imshow("Cruda", imgOrig);
		cvWaitKey();
		return -1;
	}

}

int deformadas(Mat imgOrig_gray, high_resolution_clock::time_point t1) {
	Mat imgDOrig = imgOrig + Scalar(30, 30, 30);
	Vec3b rgb_normal(115, 111, 25);
	Mat3b rgb(rgb_normal);
	int thresh = 40;
	Mat img_threshold_deformada, resultRGB_gray, maskRGB, resultRGB, img_with_keypoints_deformada, imgDOrig_gray;

	Scalar minBGR = Scalar(rgb_normal.val[0] - thresh, rgb_normal.val[1] - thresh, rgb_normal.val[2] - thresh);
	Scalar maxBGR = Scalar(rgb_normal.val[0] + thresh, rgb_normal.val[1] + thresh, rgb_normal.val[2] + thresh);
	inRange(imgDOrig, minBGR, maxBGR, maskRGB);
	bitwise_and(imgDOrig, imgDOrig, resultRGB, maskRGB);
	//cvNamedWindow("Result RGB", CV_WINDOW_AUTOSIZE);
	//imshow("Result RGB", resultBGR);
	//cvWaitKey();

	// Setup a rectangle to define your region of interest
	Rect myROI(110, 70, 450, 450);

	Mat croppedImage = resultRGB(myROI);
	//cvNamedWindow("Img Cortada", CV_WINDOW_AUTOSIZE);
	//imshow("Img Cortada", croppedImage);
	//cvWaitKey();

	cvtColor(croppedImage, resultRGB_gray, CV_RGB2GRAY);
	blur(resultRGB_gray, resultRGB_gray, Size(9, 9));
	threshold(resultRGB_gray, img_threshold_deformada, 5, 255, CV_THRESH_BINARY_INV);
	//cvNamedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	//imshow("Threshold", img_threshold_deformada);
	//cvWaitKey();

	//Filtrar por área minima
	SimpleBlobDetector::Params params;
	params.minDistBetweenBlobs = 50.0f;
	params.filterByInertia = false;
	params.filterByConvexity = false;
	params.filterByColor = false;
	params.filterByCircularity = false;
	params.filterByArea = true;
	params.minArea = 36000.0f;//34000
	params.maxArea = 100000.0f; //1000000.0f
	params.blobColor = 255;

	dilate(img_dilated, img_dilated, Mat(), Point(-1, -1),3);
	// Detect blobs
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	std::vector<KeyPoint> keypoints;
	detector->detect(img_threshold_deformada, keypoints);
	Mat img_with_keypoints;

	drawKeypoints(img_threshold_deformada, keypoints, img_with_keypoints_deformada, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cvNamedWindow("Drawn", CV_WINDOW_AUTOSIZE);
	//imshow("Drawn", img_with_keypoints_deformada);
	//cvWaitKey();

	if (keypoints.size() == 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Tiempo: " << duration << " " << endl;
		putText(imgOrig, "Deformada", Point(80, 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
		cvNamedWindow("KeyPoints", CV_WINDOW_AUTOSIZE);
		imshow("KeyPoints", imgOrig);
		cvWaitKey();
		return -1;
	}

}

int rotas(high_resolution_clock::time_point t1) {
	Mat imgROrig = imgOrig + Scalar(30, 30, 30);
	Vec3b rgb_normal(115, 111, 25);
	Mat3b rgb(rgb_normal);
	int thresh = 40;
	Mat img_threshold_rota, resultRGB_gray, maskRGB, resultRGB, img_with_keypoints_rota, imgROrig_gray;

	Scalar minBGR = Scalar(rgb_normal.val[0] - thresh, rgb_normal.val[1] - thresh, rgb_normal.val[2] - thresh);
	Scalar maxBGR = Scalar(rgb_normal.val[0] + thresh, rgb_normal.val[1] + thresh, rgb_normal.val[2] + thresh);
	inRange(imgROrig, minBGR, maxBGR, maskRGB);
	bitwise_and(imgROrig, imgROrig, resultRGB, maskRGB);
	//cvNamedWindow("Result RGB", CV_WINDOW_AUTOSIZE);
	//imshow("Result RGB", resultBGR);
	//cvWaitKey();
	
	// Setup a rectangle to define your region of interest
	Rect myROI(110, 70, 450, 450);

	Mat croppedImage = resultRGB(myROI);
	//cvNamedWindow("Img Cortada", CV_WINDOW_AUTOSIZE);
	//imshow("Img Cortada", croppedImage);
	//cvWaitKey();

	cvtColor(croppedImage, resultRGB_gray, CV_RGB2GRAY);
	blur(resultRGB_gray, resultRGB_gray, Size(9, 9));
	threshold(resultRGB_gray, img_threshold_rota, 5, 255, CV_THRESH_BINARY_INV);
	//cvNamedWindow("Threshold", CV_WINDOW_AUTOSIZE);
	//imshow("Threshold", img_threshold_rota);
	//cvWaitKey();

	dilate(img_threshold_rota, img_threshold_rota, Mat(), Point(-1, -1), 3);
	erode(img_threshold_rota, img_threshold_rota, Mat(), Point(-1, -1), 3); //NAO HAVIA ERODE
	vector<vector<Point> > contours2;
	vector<Vec4i> hierarchy2;

	findContours(img_threshold_rota, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > hull2(contours2.size());
	for (size_t i = 0; i < contours2.size(); i++) {
		convexHull(Mat(contours2[i]), hull2[i], false);
	}
	Mat drawing2 = Mat::zeros(img_threshold_rota.size(), CV_8UC3);
	for (size_t i = 0; i< contours2.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing2, contours2, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing2, hull2, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}
	//cvNamedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
	//imshow("Hull demo", drawing2);
	//cvWaitKey();

	std::vector<KeyPoint> keypoints_rota;
	SimpleBlobDetector::Params params_rota;
	params_rota.minDistBetweenBlobs = 50.0f;
	params_rota.filterByInertia = false;
	params_rota.filterByConvexity = false; //true
	params_rota.minConvexity = 0.7;
	params_rota.maxConvexity = 0.95;
	params_rota.filterByColor = false;
	params_rota.blobColor = 255;
	params_rota.filterByCircularity = false;
	params_rota.filterByArea = true;
	params_rota.minArea = 55.0f; //45
	params_rota.maxArea = 30000.0f; //3000
	Ptr<SimpleBlobDetector> detector_rota = SimpleBlobDetector::create(params_rota);
	detector_rota->detect(drawing2, keypoints_rota);
	drawKeypoints(drawing2, keypoints_rota, img_with_keypoints_rota, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cvNamedWindow("CircRota", CV_WINDOW_AUTOSIZE);
	//imshow("CircRota", img_with_keypoints_rota);
	//cvWaitKey();
	if (keypoints_rota.size() != 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Tiempo: " << duration << " " << endl;
		putText(imgOrig, "Rota", Point(80, 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
		cvNamedWindow("Rota", CV_WINDOW_AUTOSIZE);
		imshow("Rota", imgOrig);
		cvWaitKey();
		return -1;
	}

}

int quemadas(high_resolution_clock::time_point t1) {
	Mat imgQOrig_gray, img_threshold_quemada, img_dilated_quemada, img_with_keypoints_quemadas;
	
	Mat imgQOrig = imgOrig + Scalar(40, 40, 40);
	//cvNamedWindow("Img Mas Clara", CV_WINDOW_AUTOSIZE);
	//imshow("Img Mas Clara", imgQOrig);

	cvtColor(imgQOrig, imgQOrig_gray, CV_RGB2GRAY);
	blur(imgQOrig_gray, imgQOrig_gray, Size(9, 9));
	threshold(imgQOrig_gray, img_threshold_quemada, 150, 255, CV_THRESH_BINARY);
	//cvNamedWindow("Threshold Quemada", CV_WINDOW_AUTOSIZE);
	//imshow("Threshold Quemada", img_threshold_quemada);

	dilate(img_threshold_quemada, img_dilated_quemada, Mat(), Point(-1, -1), 2);
	//cvNamedWindow("Dilated Quemada", CV_WINDOW_AUTOSIZE);
	//imshow("Dilated Quemada", img_dilated_quemada);

	vector<vector<Point> > contours1;
	vector<Vec4i> hierarchy1;

	findContours(img_dilated_quemada, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point> > hull1(contours1.size());
	for (size_t i = 0; i < contours1.size(); i++) {
		convexHull(Mat(contours1[i]), hull1[i], false);
	}
	Mat drawing_quemadas = Mat::zeros(img_dilated_quemada.size(), CV_8UC3);
	for (size_t i = 0; i< contours1.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing_quemadas, contours1, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
		drawContours(drawing_quemadas, hull1, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point());
	}

	std::vector<KeyPoint> keypoints_quemadas;
	SimpleBlobDetector::Params params_quemadas;
	params_quemadas.minDistBetweenBlobs = 50.0f;
	params_quemadas.filterByInertia = false;
	params_quemadas.filterByConvexity = false;
	params_quemadas.minConvexity = 0.7;
	params_quemadas.maxConvexity = 0.95;
	params_quemadas.filterByColor = false;
	params_quemadas.blobColor = 255;
	params_quemadas.filterByCircularity = false;
	params_quemadas.filterByArea = true;
	params_quemadas.minArea = 900.0f;
	params_quemadas.maxArea = 5000.0f;
	Ptr<SimpleBlobDetector> detector_quemadas = SimpleBlobDetector::create(params_quemadas);
	detector_quemadas->detect(drawing_quemadas, keypoints_quemadas);
	drawKeypoints(drawing_quemadas, keypoints_quemadas, img_with_keypoints_quemadas, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//cvNamedWindow("CircQuemadas", CV_WINDOW_AUTOSIZE);
	//imshow("CircQuemadas", img_with_keypoints_quemadas);
	//cvWaitKey();
	if (keypoints_quemadas.size() != 0) {
		high_resolution_clock::time_point t2 = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(t2 - t1).count();
		cout << "Tiempo: " << duration << " " << endl;
		putText(imgOrig, "Quemada", Point(80, 100), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
		cvNamedWindow("Quemada", CV_WINDOW_AUTOSIZE);
		imshow("Quemada", imgOrig);
		cvWaitKey();
		return -1;
	}
}


int main(int argc, char** argv) {
	Mat3b res;
	char resultado[20];
	Point center;

	imgOrig = imread("C:/Users/Rita Carvalho/Desktop/Projeto/Tortas Unitarias/Tortas indeterminadas/I2.BMP",1);

	if (imgOrig.empty()) {
		cout << "Error : Image cannot be loaded..!!" << endl;
		return 1;
	}
	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
	imshow("Original", imgOrig);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	if (agujereadas(t1) == -1) {
		return -1;
	}
	cvtColor(imgOrig, imgOrig_gray, CV_RGB2GRAY);
	blur(imgOrig_gray, imgOrig_gray, Size(3, 3));
	threshold(imgOrig_gray, img_threshold, 120, 255, THRESH_BINARY);
	dilate(img_threshold, img_dilated, Mat(), Point(-1, -1), 3);
	//cvNamedWindow("Dilated", CV_WINDOW_AUTOSIZE);
	//imshow("Dilated", img_dilated);

	if (crudas(t1) == -1) {
		return -1;
	}

	if (deformadas(imgOrig_gray, t1) == -1) {
		return -1;
	}

	if (rotas(t1) == -1) {
		return -1;
	}

	if (quemadas(t1) == -1) {
		return -1;
	}


	threshold(imgOrig_gray, img_threshold, 120, 255, THRESH_BINARY);

	//Tortas buenas
	Moments mu = moments(img_threshold, true);
	center.x = mu.m10 / mu.m00;
	center.y = mu.m01 / mu.m00;

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	cout << "Tiempo: " << duration << " " << endl;

	cvtColor(imgOrig_gray, res, CV_GRAY2BGR);

	circle(res, center, 2, Scalar(0, 0, 255));
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	imshow("Result", res);
	sprintf(resultado, "X = %d Y =%d", center.x, center.y);
	putText(imgOrig, resultado, Point(100, 500), FONT_HERSHEY_PLAIN, 2, Scalar(0, 250, 200), 2);
	cvNamedWindow("Buena", CV_WINDOW_AUTOSIZE);
	imshow("Buena", imgOrig);
	
	cout << "x=" << center.x << "." << "y=" << center.y << endl;

	cvWaitKey(0);
	return(0);

}