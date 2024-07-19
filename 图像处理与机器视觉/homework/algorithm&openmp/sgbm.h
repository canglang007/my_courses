/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */

#include "d_type.h"
#include <omp.h>
#include <vector>
#pragma once

class sgb_matching
{
public:
	sgb_matching();
	~sgb_matching();


	/** @brief SGM�����ṹ�� */
	struct SGM_arg {
		uint8	num_paths;		// �ۺ�·����
		sint32  min_disparity;	// ��С�Ӳ�
		sint32	max_disparity;	// ����Ӳ�
		sint32	block_size;  	// �����С

		// P1,P2 
		// P2 = P2_int / (Ip-Iq)
		sint32  p1;				// �ͷ������P1
		sint32  p2_int;			// �ͷ������P2

		//�Ż�
		bool	is_check_unique;	// �Ƿ���Ψһ��
		float32	uniqueness_ratio;	// Ψһ��Լ����ֵ ����С����-����С����)/��С���� > ��ֵ Ϊ��Ч����

		bool	is_check_lr;		// �Ƿ�������һ����
		float32	lrcheck_thres;		// ����һ����Լ����ֵ

		bool	is_remove_speckles;	// �Ƿ��Ƴ�С����ͨ��
		int		min_speckle_aera;	// ��С����ͨ���������������

		bool	is_fill_holes;		// �Ƿ�����Ӳ�ն�

		bool	is_MedianFilter;     //�Ƿ���ֵ�˲�

		SGM_arg() : num_paths(4), min_disparity(0), max_disparity(64), 
			block_size(5),is_check_unique(true), uniqueness_ratio(0.95f),
			is_check_lr(1), lrcheck_thres(1.0f),
			is_remove_speckles(1), min_speckle_aera(40),
			is_fill_holes(1), is_MedianFilter(0), p1(10), p2_int(150) {
		}

	};

public:
	/** @brief ��ĳ�ʼ�������һЩ�ڴ��Ԥ���䡢������Ԥ���õ�
	* @param[in]  width     ͼ�����
	* @param[in]  height    ͼ��߶�
	* @param[in]  arg       sgb_matching����
	* @example S_D(img1, img2, disp, 3,1);
	*/
	bool Initialize(const uint32& width, const uint32& height, const SGM_arg& arg);

	/** @brief ƥ�亯��
	* @param[in]  img_left     ��ͼ��ָ��
	* @param[in]  img_right    ��ͼ��ָ��
	* @param[out]  disp_left   ��Ӱ���Ӳ�ͼָ��
	* @example S_D(img1, img2, disp, 3,1);
	*/
	bool Match(const uint8* img_left, const uint8* img_right, float32* disp_left);

private:
	/** @brief Census�任 */
	void census_transform() const;

	/** @brief ���ۼ���	 */
	void compute_cost() const;

	/** @brief ���۾ۺ�	 */
	void cost_aggregation() const;

	/** @brief �Ӳ����	 */
	void compute_disparity() const;
	void compute_disparity_right() const;
	
	/** @brief һ���Լ�� */
	void LR_check() ;

	/** \brief �Ӳ�ͼ��� */
	void FillHolesInDispMap();


private:
	/** @brief SGM����	 */
	SGM_arg arg_;

	/** @brief ͼ���	 */
	sint32 width_;

	/** @brief ͼ���	 */
	sint32 height_;

	/** @brief �����С	 */
	sint32 block_size_;

	/** @brief ��ͼ������	 */
	const uint8* img_left_;

	/** @brief ��ͼ������	 */
	const uint8* img_right_;

	/** @brief ��ͼ��censusֵ	*/
	uint32* census_left_;

	/** @brief ��ͼ��censusֵ	*/
	uint32* census_right_;

	/** @brief ��ʼƥ�����	*/
	uint8* cost_init_;

	/** @brief �ۺ�ƥ�����	*/

	uint16* cost_aggr_;

	// �K �� �L   5  3  7
	// ��    ��	 1    2
	// �J �� �I   8  4  6
	/** \brief �ۺ�ƥ�����-����1	*/
	uint8* cost_aggr_1_;
	/** \brief �ۺ�ƥ�����-����2	*/
	uint8* cost_aggr_2_;
	/** \brief �ۺ�ƥ�����-����3	*/
	uint8* cost_aggr_3_;
	/** \brief �ۺ�ƥ�����-����4	*/
	uint8* cost_aggr_4_;
	/** \brief �ۺ�ƥ�����-����5	*/
	uint8* cost_aggr_5_;
	/** \brief �ۺ�ƥ�����-����6	*/
	uint8* cost_aggr_6_;
	/** \brief �ۺ�ƥ�����-����7	*/
	uint8* cost_aggr_7_;
	/** \brief �ۺ�ƥ�����-����8	*/
	uint8* cost_aggr_8_;


	/** @brief ��ͼ���Ӳ�ͼ	*/
	float32* disp_left_;

	/** @brief ��ͼ���Ӳ�ͼ	*/
	float32* disp_right_;

	/** @brief �Ƿ��ʼ����־	*/
	bool is_initialized_;



	/** @brief �ڵ������ؼ�	*/
	std::vector<std::pair<int, int>> occlusions_;
	/** @brief ��ƥ�������ؼ�	*/
	std::vector<std::pair<int, int>> mismatches_;

};
