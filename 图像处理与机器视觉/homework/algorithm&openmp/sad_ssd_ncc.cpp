/*
 *  hm_match.cpp
 *  stereo_camera
 *
 *  Created by pp,yy on 28/5/22.
 *  Copyright 2010 Argus Corp. All rights reserved.
 *
 */

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

const uint16_t min_disparity = 0;//��С�Ӳ�


template<typename __T>
float subpixel(const int idx_min_cost, const int best_disp, const vector<__T>& cost_vec)
{
	if (idx_min_cost < 2)
		return best_disp;
	const int idx_1 = idx_min_cost - 1 - min_disparity;
	const int idx_2 = idx_min_cost + 1 - min_disparity;
	const uint16_t min_cost = cost_vec[idx_min_cost];
	const uint16_t cost_1 = cost_vec[idx_1];
	const uint16_t cost_2 = cost_vec[idx_2];
	// ��һԪ�������߼�ֵ
	const float denorm = max(1, cost_1 + cost_2 - 2 * min_cost);

	return (best_disp + (cost_1 - cost_2) / (denorm * 2.0f));
}


int mult_region_pixel(const Mat& img1, const Mat& img2)
{
	const int rows = img1.rows;
	const int cols = img1.cols;

	int sum_pixel = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum_pixel += img1.at<uchar>(i, j) * img2.at<uchar>(i, j);
		}
	}
	return sum_pixel;
}

int sq_region_pixel(const Mat& img)
{
	const int rows = img.rows;
	const int cols = img.cols;

	int sum_pixel = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum_pixel += img.at<uchar>(i, j) * img.at<uchar>(i, j);
		}
	}
	return sum_pixel;
}

int region_pixel(const Mat& img)
{
	const int rows = img.rows;
	const int cols = img.cols;

	int sum_pixel = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			sum_pixel += img.at<uchar>(i, j);
		}
	}
	return sum_pixel;
}

int cost_min_index(vector<uint16_t>& cost_vec)
{
	uint16_t min = UINT16_MAX;;
	size_t len = cost_vec.size();
	int min_index = 0;

	//�ҵ���С��cost���ڵ�λ��
	for (int i = 0; i < len; i++) {
		int num = cost_vec[i];

		if (num && (num < min)) { //ֻ�д��������Ƚ�
			min = num;
			min_index = i;
		}
	}
	return min_index;
}

int cost_max_index(vector<double>& cost_vec)
{
	double max = 0;
	double num = 0;
	size_t len = cost_vec.size();
	int max_index = 0;

	//�ҵ�����cost���ڵ�λ��
	for (int i = 0; i < len; i++) {
		num = abs(cost_vec[i]);

		if (num > max) {
			max = num;
			max_index = i;
		}
	}
	return max_index;
}

double ncc_mean(const Mat& img)
{
	const int n = img.rows * img.cols;//���ص����
	int sum_pixel = region_pixel(img);
	double mean = sum_pixel / n;
	return mean;
}

double ncc_standard_deviations(const Mat& img)
{
	const int rows = img.rows;
	const int cols = img.cols;
	const int n = img.rows * img.cols;//���ص����
	double mean = ncc_mean(img);
	double st_dev = 0;
	double sum = 0;
	int i, j;
	for (i = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			sum += ((img.at<uchar>(i, j)) - mean) * ((img.at<uchar>(i, j)) - mean);
		}
	}
	st_dev = sqrt(sum / n);
	return st_dev;
}

double ncc_cost(const Mat& img1, const Mat& img2)
{
	const int rows = img1.rows;
	const int cols = img1.cols;

	const int n = img1.rows * img1.cols;//���ص����
	int mean1 = ncc_mean(img1);
	double st_dev1 = ncc_standard_deviations(img1);
	int mean2 = ncc_mean(img2);
	double st_dev2 = ncc_standard_deviations(img2);

	int sum = 0;
	double cost = 0;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			sum += (img1.at<uchar>(i, j) - mean1) * (img2.at<uchar>(i, j) - mean2);
		}
	}
	cost = sum / (n * st_dev1 * st_dev2);
	return cost;
}


void NCC(const Mat& img1, const Mat& img2, Mat& disp, const int size)
{
	const int rows = img1.rows;
	const int cols = img1.cols;

	const int r = size / 2;  // ����뾶����3x3�ķ���뾶Ϊ1
	const int N = size * size;//�����ڵ����ص����
	int u_start, v_start, u_end, v_end;  //�������ĵ���ʼ����ֹ������λ��
	u_start = v_start = r;
	u_end = rows - r - 1;
	v_end = cols - r - 1;

	//����ͼΪ��׼����ͼ����ƥ��
	for (int u = u_start; u <= u_end; u = u + 1) {
		for (int v = v_start; v <= v_end; v = v + 1) {
			//����region�ķ�Χ,Rect()������ҿ�������endҪ+1
			int i_start = u - r;
			int j_start = v - r;

			//��ͼ����������󻬶������㣬���cost
			//int vec_len = v - v_start + 1;
			int vec_len = cols;
			vector<double> cost_vec(vec_len); //��ʼ��cost����,����Ϊcols
			for (int k = v; k >= v_start; k = k - 1) {

				int k_start = k - r;



				double cost = ncc_cost(img1(Rect(j_start, i_start, size, size)), img2(Rect(k_start, i_start, size, size)));

				cost_vec[k] = cost;
			}

			//��������Ӳ�
			int idx_min_cost = cost_max_index(cost_vec);
			int best_disp = v - idx_min_cost;

			//��������ǿ
			//disp.at<uchar>(u, v) = best_disp;
			disp.at<float>(u, v) = subpixel(idx_min_cost, best_disp, cost_vec);
		}
	}
}


void S_D(const Mat& img1, const Mat& img2, Mat& disp, const int size, int SSD)
{
	// ��������ͼƬ�ѵõ�У�����Ҵ�Сһ��
	const int rows = img1.rows;
	const int cols = img1.cols;

	const int r = size / 2;  // ����뾶����3x3�ķ���뾶Ϊ1
	int u_start, v_start, u_end, v_end;  //�������ĵ���ʼ����ֹ������λ��
	u_start = v_start = r;
	u_end = rows - r - 1;
	v_end = cols - r - 1;

	//����ͼΪ��׼����ͼ����ƥ��
	for (int u = u_start; u <= u_end; u++) {
		for (int v = v_start; v <= v_end; v++) {
			//����region�ķ�Χ,Rect()������ҿ�������endҪ+1
			int i_start = u - r;
			int j_start = v - r;
			//�����ͼ�������С����С�Ӳ��һ������
			if (j_start < min_disparity)
				continue;

			// ��ͼ����������ƽ���ͣ���Rect()�������µ�Mat,�����ǹ����ڴ��
			Mat region_1 = img1(Rect(j_start, i_start, size, size));
			int sq_img1_pixel = sq_region_pixel(region_1);//ƽ����
			int img1_pixel = region_pixel(region_1);
			//int img1_pixel = region_pixel(img1(Rect(j_start, i_start, size, size)));

			//��ͼ����������󻬶����������غͣ����cost
			//int vec_len = v - v_start + 1;
			int vec_len = cols;
			vector<uint16_t> cost_vec(vec_len); //��ʼ��cost����,����Ϊcols

			for (int k = v; k >= v_start; k--) {

				int disp = v - k;
				//����Ӳ�С����С�Ӳ��һ������
				if (disp < min_disparity)
					continue;

				int k_start = k - r;
				Mat region_2 = img2(Rect(k_start, i_start, size, size));
				int mult_img1_img2_pixel = 2 * mult_region_pixel(region_1, region_2);
				int sq_img2_pixel = sq_region_pixel(region_2);//ƽ����
				//int img2_pixel = region_pixel(img2(Rect(k_start,  i_start, size, size)));
				int cost = 0;
				if (SSD) {  //SSD
					cost = sq_img1_pixel - mult_img1_img2_pixel + sq_img2_pixel;
					//cost = diff * diff;
				}
				else {   //SAD
					img1_pixel = region_pixel(region_1);
					int img2_pixel = region_pixel(region_2);
					cost = abs(img1_pixel - img2_pixel);
				}
				cost_vec[k] = cost;
			}
			//��������Ӳ�
			int idx_min_cost = cost_min_index(cost_vec);
			int best_disp = v - idx_min_cost;


			//disp.at<uchar>(u, v) = best_disp;
			//��������ǿ
			disp.at<float>(u, v) = subpixel(idx_min_cost, best_disp, cost_vec);


		}
	}
}