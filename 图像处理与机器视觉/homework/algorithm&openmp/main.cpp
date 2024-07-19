/*
 *  main.cpp
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */

#include "sgbm.h"
#include "val.h"
#include "d_type.h"
#include "sad_ssd_ncc.h"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <stdio.h>
#include <sstream>
#include <iostream>

using namespace cv;
using namespace std;


int main(int argc, char** argv)
{

	// ---init ��ʼ����������
	std::string img1_filename = "../left_0.jpg";
	std::string img2_filename = "../right_0.jpg";
	std::string dispg_filename = "../disp.pgm";
	std::string disparity_filename = "../disparity.png";

	const int color_display = 1;
	const int block_size = 3;		//block_size����Ϊ����
	const int min_disparity = 0;	//��С�Ӳ�
	const int max_disparity = 128;	//����Ӳ�
	const int radio = 4;			//΢��ϵ��
	enum { STEREO_SAD = 0, STEREO_SSD = 1, STEREO_NCC = 2, STEREO_SGBM = 3 };
	int alg = STEREO_SGBM;

	// input -- read in grayscale
	Mat img1 = imread(img1_filename, IMREAD_GRAYSCALE);
	Mat img2 = imread(img2_filename, IMREAD_GRAYSCALE);

	if (img1.empty() || img2.empty())
	{
		printf("error: could not load the input image file\n");
		return -1;
	}
	if (img1.rows != img2.rows || img1.cols != img2.cols)
	{
		printf("error: img sizes don't match\n");
		return -1;
	}
	if (alg < 0)
	{
		printf("error: Unknown stereo algorithm\n\n");
		return -1;
	}
	if (block_size % 2 == 0) {
		printf("block_size ����Ϊ����\n");
		return -1;
	}

	// --- �Ӳ�ƥ�䣬ѡ���㷨
	Mat disp = Mat::zeros(img1.rows, img2.cols, CV_32F);

	const uint32 width = static_cast<uint32>(img1.cols);
	const uint32 height = static_cast<uint32>(img2.rows);
	auto disparity = new float32[width * height]();
	int64 t = getTickCount();
	if (alg == STEREO_SAD)
	{
		S_D(img1, img2, disp, block_size, 0);
	}
	else if (alg == STEREO_SSD)
	{
		S_D(img1, img2, disp, block_size, 1);
	}
	else if (alg == STEREO_NCC)
	{
		NCC(img1, img2, disp, block_size);
	}
	else if (alg == STEREO_SGBM)
	{
		// ��ʼ��
		sgb_matching sgm;
		sgb_matching::SGM_arg sgm_arg;
		sgm_arg.block_size = block_size;
		sgm_arg.min_disparity = min_disparity;
		sgm_arg.max_disparity = max_disparity;
		sgm_arg.p1 = 10;
		sgm_arg.p2_int = 150;
		if (!sgm.Initialize(width, height, sgm_arg)) {
			std::cout << "SGM��ʼ��ʧ�ܣ�" << std::endl;
			return -2;
		}

		// ƥ��
		if (!sgm.Match(img1.data, img2.data, disparity)) {
			std::cout << "SGMƥ��ʧ�ܣ�" << std::endl;
			return -2;
		}
	}

	t = getTickCount() - t;

	// --- ��ʾ�Ӳ�ͼ
	Mat disp8_3c, disp8;
	
	disp.convertTo(disp8, CV_8U);
	disp8 = disp8 * radio;   //΢��

	if (alg == STEREO_SGBM) {
		for (uint32 i = 0; i < height; i++) {
			for (uint32 j = 0; j < width; j++) {
				const float32 disp = disparity[i * width + j];
				if (disp == invalid_float) {
					disp8.data[i * width + j] = 0;
				}
				else {
					disp8.data[i * width + j] = static_cast<uchar>((disp - min_disparity) / (max_disparity - min_disparity) * 255);
					//disp8.data[i * width + j] = radio * static_cast<uchar>(disp);
				}
			}
		}
	}

	if (color_display)
		cv::applyColorMap(disp8, disp8_3c, COLORMAP_JET);
	//imwrite(disparity_filename, color_display ? disp8_3c : disp8);
	imwrite(disparity_filename,  disp8);

	// --- ��������
	printf("\nTime elapsed: %f ms\n", t * 1000 / getTickFrequency());
	Mat dispg = imread(dispg_filename, IMREAD_GRAYSCALE);
	// double e_RMS = eRMS(disp8, dispg);
	// printf("eRMS : %lf \n", e_RMS);
	// double e_PEP = ePEP(disp8, dispg);
	// printf("ePEP : %.2f %% \n", e_PEP);
	// double Mde_s = width * height * max_disparity / (t / getTickFrequency()) * 0.000001;
	// printf("Mde/s : %.2f \n", Mde_s);


	//Mat depth(disp8.rows, disp8.cols, CV_16S);
	Mat depth(disp8.rows, disp8.cols, CV_16U);
	//Mat depth_3c(disp8.rows, disp8.cols, CV_16U);
	disp2Depth(disp8, depth);
	//cv::applyColorMap(depth, depth_3c, COLORMAP_JET);


	std::ostringstream oss;
	oss << "disparity  " << (alg == STEREO_SAD ? "sad" :
		alg == STEREO_SSD ? "ssd" :
		alg == STEREO_NCC ? "ncc" :
		alg == STEREO_SGBM ? "sgbm" : "");
	oss << "  blocksize:" << (block_size);
	std::string disp_name = oss.str();

	/*namedWindow("depth", cv::WINDOW_NORMAL);
	imshow("depth", depth);
	namedWindow("depth_3c", cv::WINDOW_NORMAL);
	imshow("depth_3c", depth_3c);*/
	namedWindow("left", cv::WINDOW_NORMAL);
	imshow("left", img1);
	namedWindow("right", cv::WINDOW_NORMAL);
	imshow("right", img2);
	imshow(disp_name, color_display ? disp8_3c : disp8);

	printf("press ESC key or CTRL+C to close...");
	fflush(stdout);
	printf("\n");
	while (1)
	{
		if (waitKey() == 27) //ESC (prevents closing on actions like taking screenshots)
			break;
	}

	delete[] disparity;
	disparity = nullptr;

	return 0;
}
