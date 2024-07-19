/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */
#pragma once
#include "d_type.h"
#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
using namespace std;

/** @brief �����������㺯��
* @param[in]  disp_e    ���Ƶõ����Ӳ�ͼ
* @param[in]  disp_g    �ṩ�ı�׼�Ӳ�ͼ
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief Percentage of error pixels(PEP)
 * @param[in]  disp_e    ���Ƶõ����Ӳ�ͼ
 * @param[in]  disp_g    �ṩ�ı�׼�Ӳ�ͼ
 * @return  eRMS
 * @example double e_PEP = ePEP(disp8, dispg);
 * @attention
 * @note
 */
double ePEP(const Mat& disp_e, const Mat& disp_g);

/** @brief �����������㺯��
* @param[in]  disp_e    ���Ƶõ����Ӳ�ͼ
* @param[in]  disp_g    �ṩ�ı�׼�Ӳ�ͼ
* @return  eRMS
* @example double e_RMS = eRMS(disp8, dispg);
* @attention
* @note
*/
double eRMS(const Mat& disp_e, const Mat& disp_g);

/** @brief �Ӳ�ͼת���ͼ
* @param[in]   dispMap     �Ӳ�ͼ��8λ��ͨ����CV_8UC1
* @param[out]  depthMap    ���ͼ��16λ�޷��ŵ�ͨ����CV_16UC1
* @example                 disp2Depth(disp8, depth);
* @attention
* @note
*/

void disp2Depth(cv::Mat dispMap, cv::Mat& depthMap);


/** @brief census�任
 * @param[in]  source     ͼ��
 * @param[out] census     censusֵ����
 * @param[in]  width      ͼ���
 * @param[in]  height     ͼ���
 * @example
 */
void census_transform_rxr(const uint8* source, uint32* census, const sint32& width, const sint32& height, const sint32& size);
// Hamming����
uint8 Hamming32(const uint32& x, const uint32& y);


/**
	 * \brief ����·���ۺ� �� ��
	 * \param img_data			���룬Ӱ������
	 * \param width				���룬Ӱ���
	 * \param height			���룬Ӱ���
	 * \param min_disparity		���룬��С�Ӳ�
	 * \param max_disparity		���룬����Ӳ�
	 * \param p1				���룬�ͷ���P1
	 * \param p2_init			���룬�ͷ���P2_Init
	 * \param cost_init			���룬��ʼ��������
	 * \param cost_aggr			�����·���ۺϴ�������
	 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ�����ң�������Ϊ���ҵ���
	 */
void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);

/**
 * \brief ����·���ۺ� �� ��
 * \param img_data			���룬Ӱ������
 * \param width				���룬Ӱ���
 * \param height			���룬Ӱ���
 * \param min_disparity		���룬��С�Ӳ�
 * \param max_disparity		���룬����Ӳ�
 * \param p1				���룬�ͷ���P1
 * \param p2_init			���룬�ͷ���P2_Init
 * \param cost_init			���룬��ʼ��������
 * \param cost_aggr			�����·���ۺϴ�������
 * \param is_forward		���룬�Ƿ�Ϊ������������Ϊ���ϵ��£�������Ϊ���µ��ϣ�
 */
void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward = true);



/**
	 * \brief �޳�С��ͨ��
	 * \param disparity_map		���룬�Ӳ�ͼ
	 * \param width				���룬����
	 * \param height			���룬�߶�
	 * \param diff_insame		���룬ͬһ��ͨ���ڵľֲ����ز���
	 * \param min_speckle_aera	���룬��С��ͨ�����
	 * \param invalid_val		���룬��Чֵ
	 */
void RemoveSpeckles(float32* disparity_map, const sint32& width, const sint32& height, const sint32& diff_insame, const uint32& min_speckle_aera, const float32& invalid_val);


/**
 * \brief ��ֵ�˲�
 * \param in				���룬Դ����
 * \param out				�����Ŀ������
 * \param width				���룬����
 * \param height			���룬�߶�
 * \param wnd_size			���룬���ڿ���
 */
void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height, const sint32 wnd_size);
