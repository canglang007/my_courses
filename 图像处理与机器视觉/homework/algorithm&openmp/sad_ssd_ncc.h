/*
 *  ssd_ncc_matching.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */
#pragma once

#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;
const uint16_t min_disparity = 0;//��С�Ӳ�

/** @brief SSD/SAD�����㷨����
* @param[in]  img1    ��ͼ
* @param[in]  img2    ��ͼ
* @param[in]  disp    �Ӳ�ͼ
* @param[in]  size    ����Ĵ�С
* @param[in]  SSD    SSDΪ1��SSD,Ϊ0��SAD
* @return  ��
* @example S_D(img1, img2, disp, 3,1);
* @attention
* @note
*/
template<typename __T>
float subpixel(const int idx_min_cost, const int best_disp, const vector<__T>& cost_vec);

/** @brief �����������㺯��
* @param[in]  disp_e    ���Ƶõ����Ӳ�ͼ
* @param[in]  disp_g    �ṩ�ı�׼�Ӳ�ͼ
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief �������������س˻�
* @param[in]  img       С����
* @return     sum_pixel �����������ƽ����
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int mult_region_pixel(const Mat& img1, const Mat& img2);

/** @brief ��������������ƽ����
* @param[in]  img       С����
* @return     sum_pixel �����������ƽ����
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int sq_region_pixel(const Mat& img);

/** @brief �������������غ�
* @param[in]  img       С����
* @return     sum_pixel ����������غ�
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int region_pixel(const Mat& img);

/** @brief ����cost��������С��һ��
* @param[in]  cost_vec    cost����
* @return  ��С��cost������
* @example  cost_min_index(cost_vec);
* @attention
* @note
*/
int cost_min_index(vector<uint16_t>& cost_vec);

/** @brief ����cost����������һ��
* @param[in]  cost_vec    cost����
* @return  ����cost������
* @example
* @attention
* @note
*/
int cost_max_index(vector<double>& cost_vec);

/** @brief ����ncc�Ŀ�������ؾ�ֵ
* @param[in]  const Mat& img, const int size, const int posi_u, const int posi_v ͼƬ�����������ĵ��λ��
* @return  mean��������ؾ�ֵ
* @example ncc_mean(img);
* @attention
* @note
*/
double ncc_mean(const Mat& img);

/** @brief ����ncc�Ŀ�������ر�׼��
* @param[in]  const Mat& img
* @return  ��������ر�׼��
* @example
* @attention
* @note
*/
double ncc_standard_deviations(const Mat& img);

/** @brief ����ncc�Ĵ���
* @param[in]  const Mat& img
* @return  ����
* @example
* @attention
* @note
*/
double ncc_cost(const Mat& img1, const Mat& img2);

/** @brief NCC�㷨
* @param[in]  img1    ��ͼ
* @param[in]  img2    ��ͼ
* @param[in]  disp    �Ӳ�ͼ
* @param[in]  size    ����Ĵ�С
* @return  ��
* @example S_D(img1, img2, disp, size=3,2);
* @attention
* @note
*/
void NCC(const Mat& img1, const Mat& img2, Mat& disp, const int size);

/** @brief SSD/SAD�����㷨����
* @param[in]  img1    ��ͼ
* @param[in]  img2    ��ͼ
* @param[in]  disp    �Ӳ�ͼ
* @param[in]  size    ����Ĵ�С
* @param[in]  SSD    SSDΪ1��SSD,Ϊ0��SAD
* @return  ��
* @example S_D(img1, img2, disp, 3,1);
* @attention
* @note
*/
void S_D(const Mat& img1, const Mat& img2, Mat& disp, const int size, int SSD);


