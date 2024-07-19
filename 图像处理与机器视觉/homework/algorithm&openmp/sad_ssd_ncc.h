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
const uint16_t min_disparity = 0;//最小视差

/** @brief SSD/SAD误差差算法函数
* @param[in]  img1    左图
* @param[in]  img2    右图
* @param[in]  disp    视差图
* @param[in]  size    方框的大小
* @param[in]  SSD    SSD为1则SSD,为0则SAD
* @return  无
* @example S_D(img1, img2, disp, 3,1);
* @attention
* @note
*/
template<typename __T>
float subpixel(const int idx_min_cost, const int best_disp, const vector<__T>& cost_vec);

/** @brief 均方根误差计算函数
* @param[in]  disp_e    估计得到的视差图
* @param[in]  disp_g    提供的标准视差图
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief 计算该区域的像素乘积
* @param[in]  img       小区域
* @return     sum_pixel 该区域的像素平方和
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int mult_region_pixel(const Mat& img1, const Mat& img2);

/** @brief 计算该区域的像素平方和
* @param[in]  img       小区域
* @return     sum_pixel 该区域的像素平方和
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int sq_region_pixel(const Mat& img);

/** @brief 计算该区域的像素和
* @param[in]  img       小区域
* @return     sum_pixel 该区域的像素和
* @example  region_pixel(img1(Rect(j_start, i_start, size, size)));
* @attention
* @note
*/
int region_pixel(const Mat& img);

/** @brief 计算cost数组里最小的一个
* @param[in]  cost_vec    cost数组
* @return  最小的cost的索引
* @example  cost_min_index(cost_vec);
* @attention
* @note
*/
int cost_min_index(vector<uint16_t>& cost_vec);

/** @brief 计算cost数组里最大的一个
* @param[in]  cost_vec    cost数组
* @return  最大的cost的索引
* @example
* @attention
* @note
*/
int cost_max_index(vector<double>& cost_vec);

/** @brief 计算ncc的框里的像素均值
* @param[in]  const Mat& img, const int size, const int posi_u, const int posi_v 图片，框宽，框的中心点的位置
* @return  mean框里的像素均值
* @example ncc_mean(img);
* @attention
* @note
*/
double ncc_mean(const Mat& img);

/** @brief 计算ncc的框里的像素标准差
* @param[in]  const Mat& img
* @return  框里的像素标准差
* @example
* @attention
* @note
*/
double ncc_standard_deviations(const Mat& img);

/** @brief 计算ncc的代价
* @param[in]  const Mat& img
* @return  代价
* @example
* @attention
* @note
*/
double ncc_cost(const Mat& img1, const Mat& img2);

/** @brief NCC算法
* @param[in]  img1    左图
* @param[in]  img2    右图
* @param[in]  disp    视差图
* @param[in]  size    方框的大小
* @return  无
* @example S_D(img1, img2, disp, size=3,2);
* @attention
* @note
*/
void NCC(const Mat& img1, const Mat& img2, Mat& disp, const int size);

/** @brief SSD/SAD误差差算法函数
* @param[in]  img1    左图
* @param[in]  img2    右图
* @param[in]  disp    视差图
* @param[in]  size    方框的大小
* @param[in]  SSD    SSD为1则SSD,为0则SAD
* @return  无
* @example S_D(img1, img2, disp, 3,1);
* @attention
* @note
*/
void S_D(const Mat& img1, const Mat& img2, Mat& disp, const int size, int SSD);


