/*
 *  sgbm_types.h
 *  stereo_camera
 *
 *  Created by pp,yy on 1/6/22.
 *  Copyright 2022 Argus Corp. All rights reserved.
 *
 */
 //#include "stdafx.h"
#include <iostream>
#include "sgbm.h"
#include "val.h"
#include <cassert>
#include <vector>
#include <algorithm>
#include <queue>
#include <opencv2/core/utility.hpp>

using namespace cv;
//#include "sgbm_types.h"


sgb_matching::sgb_matching()
{

}

sgb_matching::~sgb_matching()
{

	if (census_left_ != nullptr) {
		delete[] census_left_;
		census_left_ = nullptr;
	}
	if (census_right_ != nullptr) {
		delete[] census_right_;
		census_right_ = nullptr;
	}
	if (cost_init_ != nullptr) {
		delete[] cost_init_;
		cost_init_ = nullptr;
	}
	if (cost_aggr_ != nullptr) {
		delete[] cost_aggr_;
		cost_aggr_ = nullptr;
	}
	if (disp_left_ != nullptr) {
		delete[] disp_left_;
		disp_left_ = nullptr;
	}
	is_initialized_ = false;


}

bool sgb_matching::Initialize(const uint32& width, const uint32& height, const SGM_arg& arg)
{
	// ������ ��ֵ

	// Ӱ��ߴ�
	width_ = width;
	height_ = height;
	// SGM����
	arg_ = arg;

	if (width == 0 || height == 0) {
		return false;
	}

	//������ �����ڴ�ռ�

	// censusֵ������ͼ��
	census_left_ = new uint32[width * height]();
	census_right_ = new uint32[width * height]();

	// ƥ����ۣ���ʼ/�ۺϣ�
	const sint32 disp_range = arg.max_disparity - arg.min_disparity;
	if (disp_range <= 0) {
		return false;
	}
	cost_init_ = new uint8[width * height * disp_range]();
	cost_aggr_ = new uint16[width * height * disp_range]();

	cost_aggr_1_ = new uint8[width * height * disp_range]();
	cost_aggr_2_ = new uint8[width * height * disp_range]();
	cost_aggr_3_ = new uint8[width * height * disp_range]();
	cost_aggr_4_ = new uint8[width * height * disp_range]();
	cost_aggr_5_ = new uint8[width * height * disp_range]();
	cost_aggr_6_ = new uint8[width * height * disp_range]();
	cost_aggr_7_ = new uint8[width * height * disp_range]();
	cost_aggr_8_ = new uint8[width * height * disp_range]();

	// �Ӳ�ͼ
	disp_left_ = new float32[width * height]();
	disp_right_ = new float32[width * height]();

	is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

	return is_initialized_;
}

bool sgb_matching::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
	if (!is_initialized_) {
		return false;
	}
	if (img_left == nullptr || img_right == nullptr) {
		return false;
	}

	img_left_ = img_left;
	img_right_ = img_right;
	block_size_ = arg_.block_size;

	// census�任
	census_transform();

	// ���ۼ���
	int64 t = getTickCount();
	compute_cost();
	t = getTickCount() - t;
	printf("Hamming cost %lf ms\n", t * 1000 / getTickFrequency());

	t = getTickCount();
	// ���۾ۺ�
	cost_aggregation();

	t = getTickCount() - t;
	printf("cost aggregating time:  %lf ms\n", t * 1000 / getTickFrequency());

	t = getTickCount();
	// �Ӳ����
	compute_disparity();
	t = getTickCount() - t;
	printf("compute_disparity time:  %lf ms\n", t * 1000 / getTickFrequency());

	
	// ����һ���Լ��
	if (arg_.is_check_lr) {
		// �Ӳ���㣨��Ӱ��
		t = getTickCount();
		compute_disparity_right();
		// һ���Լ��
		LR_check();
		t = getTickCount() - t;
		printf("LR_check time:  %lf ms\n", t * 1000 / getTickFrequency());
	}
	

	//�Ƴ�С��ͨ��
	if (arg_.is_remove_speckles) {
		t = getTickCount();
		RemoveSpeckles(disp_left_, width_, height_, 1, arg_.min_speckle_aera, invalid_float);
		t = getTickCount() - t;
		printf("RemoveSpeckles time:  %lf ms\n", t * 1000 / getTickFrequency());
	}

	

	// �Ӳ����
	if (arg_.is_fill_holes) {
		t = getTickCount();
		FillHolesInDispMap();
		t = getTickCount() - t;
		printf("FillHolesInDispMap time:  %lf ms\n", t * 1000 / getTickFrequency());
	}

	//��ֵ�˲�
	if (arg_.is_MedianFilter) {
		t = getTickCount();
		MedianFilter(disp_left_, disp_left_, width_, height_, 3);
		t = getTickCount() - t;
		printf("MedianFilter time:  %lf ms\n\n\n", t * 1000 / getTickFrequency());
	}
	// ����Ӳ�ͼ
	memcpy(disp_left, disp_left_, width_ * height_ * sizeof(float32));

	return true;
}

void sgb_matching::census_transform() const
{
	// ����ͼ��census�任
	census_transform_rxr(img_left_, census_left_, width_, height_, block_size_);
	census_transform_rxr(img_right_, census_right_, width_, height_, block_size_);
}

void sgb_matching::compute_cost() const
{

	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;

	// ������ۣ�����Hamming���룩
#pragma omp parallel for 
	for (sint32 i = 0; i < height_; i++) {
		for (sint32 j = 0; j < width_; j++) {

			// ��ͼcensusֵ
			const uint32 census_val_l = census_left_[i * width_ + j];

			// ���Ӳ�������ֵ

			for (sint32 d = min_disparity; d < max_disparity; d++) {
				auto& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
				if (j - d < 0 || j - d >= width_) {
					cost = UINT8_MAX / 2;
					continue;
				}
				// ��ͼ��Ӧ����censusֵ
				const uint32 census_val_r = census_right_[i * width_ + j - d];

				// ����ƥ�����

				cost = Hamming32(census_val_l, census_val_r);

			}
		}
	}
}
void sgb_matching::compute_disparity() const
{
	// ��С����Ӳ�
	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;

	if (disp_range <= 0) {
		return;
	}

	// ��Ӱ���Ӳ�ͼ
	const auto disparity = disp_left_;
	// ��Ӱ��ۺϴ�������
	const auto cost_ptr = cost_aggr_;
	// δʵ�־ۺϲ��裬���ó�ʼ����ֵ������
    //auto cost_ptr = cost_init_;

	const sint32 width = width_;
	const sint32 height = height_;

	// Ϊ�˼ӿ��ȡЧ�ʣ��ѵ������ص����д���ֵ�洢���ֲ�������
	std::vector<uint16> cost_local(disp_range);

	// �����ؼ��������Ӳ�
	for (sint32 i = 0; i < height_; i++) {
		for (sint32 j = 0; j < width_; j++) {

			uint16 min_cost = UINT16_MAX;
			uint16 sec_min_cost = UINT16_MAX;
			uint16 max_cost = 0;
			sint32 best_disparity = 0;

			// �����ӲΧ�ڵ����д���ֵ�������С����ֵ����Ӧ���Ӳ�ֵ
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				const sint32 d_idx = d - min_disparity;
				const auto& cost = cost_ptr[i * width * disp_range + j * disp_range + d_idx];
				if (min_cost > cost) {
					min_cost = cost;
					best_disparity = d;
				}
				max_cost = std::max(max_cost, static_cast<uint16>(cost));
			}


			//��С����ֵ��Ӧ���Ӳ�ֵ��Ϊ���ص������Ӳ�
			if (max_cost != min_cost) {
				disp_left_[i * width_ + j] = static_cast<float>(best_disparity);
			}
			else {
				// ��������Ӳ��µĴ���ֵ��һ�������������Ч
				disp_left_[i * width_ + j] = invalid_float;
			}


			// ---���������
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = invalid_float;
				continue;
			}
			// �����Ӳ�ǰһ���Ӳ�Ĵ���ֵc_1����һ���Ӳ�Ĵ���ֵc_2
			const sint32 idx_1 = best_disparity - 1 - min_disparity;
			const sint32 idx_2 = best_disparity + 1 - min_disparity;
			const uint16 c_1 = cost_local[idx_1];
			const uint16 c_2 = cost_local[idx_2];
			// ��һԪ�������߼�ֵ
			const uint16 den = std::max(1, c_1 + c_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float32>(best_disparity) + static_cast<float32>(c_1 - c_2) / (den * 2.0f);
		}
	}
}


void sgb_matching::compute_disparity_right() const
{
	const sint32& min_disparity = arg_.min_disparity;
	const sint32& max_disparity = arg_.max_disparity;
	const sint32 disp_range = max_disparity - min_disparity;
	if (disp_range <= 0) {
		return;
	}

	// ��Ӱ���Ӳ�ͼ
	const auto disparity = disp_right_;
	// ��Ӱ��ۺϴ�������
	const auto cost_ptr = cost_aggr_;

	const sint32 width = width_;
	const sint32 height = height_;


	// Ϊ�˼ӿ��ȡЧ�ʣ��ѵ������ص����д���ֵ�洢���ֲ�������
	std::vector<uint16> cost_local(disp_range);

	// ---�����ؼ��������Ӳ�
	// ͨ����Ӱ��Ĵ��ۣ���ȡ��Ӱ��Ĵ���
	// ��cost(xr,yr,d) = ��cost(xr+d,yl,d)
	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			uint16 min_cost = UINT16_MAX;
			uint16 max_cost = 0;
			sint32 best_disparity = 0;

			// ---ͳ�ƺ�ѡ�Ӳ��µĴ���ֵ
			for (sint32 d = min_disparity; d < max_disparity; d++) {
				const sint32 d_idx = d - min_disparity;
				const sint32 col_left = j + d;
				if (col_left >= 0 && col_left < width) {
					const auto& cost = cost_local[d_idx] = cost_ptr[i * width * disp_range + col_left * disp_range + d_idx];
					if (min_cost > cost) {
						min_cost = cost;
						best_disparity = d;
					}
				}
				else {
					cost_local[d_idx] = UINT16_MAX;
				}
			}

			// ��С����ֵ��Ӧ���Ӳ�ֵ��Ϊ���ص������Ӳ�
			if (max_cost != min_cost) {
				disp_right_[i * width_ + j] = static_cast<float>(best_disparity);
			}
			else {
				// ��������Ӳ��µĴ���ֵ��һ�������������Ч
				disp_right_[i * width_ + j] = invalid_float;
			}

			// ---���������
			if (best_disparity == min_disparity || best_disparity == max_disparity - 1) {
				disparity[i * width + j] = invalid_float;
				continue;
			}
			// �����Ӳ�ǰһ���Ӳ�Ĵ���ֵcost_1����һ���Ӳ�Ĵ���ֵcost_2
			const sint32 idx_1 = best_disparity - 1 - min_disparity;
			const sint32 idx_2 = best_disparity + 1 - min_disparity;
			const uint16 cost_1 = cost_local[idx_1];
			const uint16 cost_2 = cost_local[idx_2];
			// ��һԪ�������߼�ֵ
			const uint16 denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
			disparity[i * width + j] = static_cast<float32>(best_disparity) + static_cast<float32>(cost_1 - cost_2) / (denom * 2.0f);
		}
	}
}



void sgb_matching::cost_aggregation() const
{
	// ·���ۺ�
	// 1����->��/��->��
	// 2����->��/��->��
	// 3������->����/����->����
	// 4������->����/����->����
	//
	// �K �� �L   5  3  7
	// ��    ��	 1    2
	// �J �� �I   8  4  6
	//
	const auto& min_disparity = arg_.min_disparity;
	const auto& max_disparity = arg_.max_disparity;
	assert(max_disparity > min_disparity);

	const sint32 size = width_ * height_ * (max_disparity - min_disparity);
	if (size <= 0) {
		return;
	}

	const auto& P1 = arg_.p1;
	const auto& P2_Int = arg_.p2_int;

	if (arg_.num_paths == 4 || arg_.num_paths == 8) {
		// ���Ҿۺ�
		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
		CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
		// ���¾ۺ�
		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
		CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);
	}

	if (arg_.num_paths == 8) {
		// �Խ��߾ۺ�
		//#TODO:��ɶԽ��߾ۺϴ���

	}

	// ��4/8�����������
	for (sint32 i = 0; i < size; i++) {
		if (arg_.num_paths == 4 || arg_.num_paths == 8) {
			cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
		}
		if (arg_.num_paths == 8) {
			cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
		}
	}
}


void sgb_matching::LR_check()
{
	const sint32 width = width_;
	const sint32 height = height_;

	const float32& threshold = arg_.lrcheck_thres;

	// �ڵ������غ���ƥ��������
	auto& occlusions = occlusions_;
	auto& mismatches = mismatches_;
	occlusions.clear();
	mismatches.clear();

	// ---����һ���Լ��

	for (sint32 i = 0; i < height; i++) {
		for (sint32 j = 0; j < width; j++) {
			// ��Ӱ���Ӳ�ֵ
			auto& disp = disp_left_[i * width + j];
			if (disp == invalid_float) {
				mismatches.emplace_back(i, j);
				continue;
			}

			// �����Ӳ�ֵ�ҵ���Ӱ���϶�Ӧ��ͬ������
			const auto col_right = static_cast<sint32>(j - disp + 0.5);//��ȥ�Ӳ�ֵ

			if (col_right >= 0 && col_right < width) {
				// ��Ӱ����ͬ�����ص��Ӳ�ֵ
				const auto& disp_r = disp_right_[i * width + col_right];

				// �ж������Ӳ�ֵ�Ƿ�һ�£���ֵ����ֵ�ڣ�
				if (abs(disp - disp_r) > threshold) {
					// �����ڵ�������ƥ����
					// ͨ����Ӱ���Ӳ��������Ӱ���ƥ�����أ�����ȡ�Ӳ�disp_rl
					// if(disp_rl > disp) 
					//		pixel in occlusions
					// else 
					//		pixel in mismatches
					const sint32 col_rl = static_cast<sint32>(col_right + disp_r + 0.5);
					if (col_rl > 0 && col_rl < width) {
						const auto& disp_l = disp_left_[i * width + col_rl];
						if (disp_l > disp) {
							occlusions.emplace_back(i, j);
						}
						else {
							mismatches.emplace_back(i, j);
						}
					}
					else {
						mismatches.emplace_back(i, j);
					}

					// ���Ӳ�ֵ��Ч
					disp = invalid_float;
				}
			}
			else {
				// ͨ���Ӳ�ֵ����Ӱ�����Ҳ���ͬ�����أ�����Ӱ��Χ��
				disp = invalid_float;
				mismatches.emplace_back(i, j);
			}
		}
	}
}




void sgb_matching::FillHolesInDispMap()
{
	const sint32 width = width_;
	const sint32 height = height_;

	std::vector<float32> disp_collects;

	// ����8������
	const float32 pi = 3.1415926f;
	float32 angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
	float32 angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
	float32* angle = angle1;
	// ��������г̣�û�б�Ҫ������Զ������
	const sint32 max_search_length = 1.0 * std::max(abs(arg_.max_disparity), abs(arg_.min_disparity));

	float32* disp_ptr = disp_left_;

	for (sint32 k = 0; k < 3; k++) {
		// ��һ��ѭ�������ڵ������ڶ���ѭ��������ƥ����
		auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;
		if (trg_pixels.empty()) {
			continue;
		}
		std::vector<float32> fill_disps(trg_pixels.size());
		std::vector<std::pair<sint32, sint32>> inv_pixels;
		if (k == 2) {
			//  ������ѭ������ǰ����û�д����ɾ�������
			for (sint32 i = 0; i < height; i++) {
				for (sint32 j = 0; j < width; j++) {
					if (disp_ptr[i * width + j] == invalid_float) {
						inv_pixels.emplace_back(i, j);
					}
				}
			}
			trg_pixels = inv_pixels;
		}

		// ��������������

		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 y = pix.first;
			const sint32 x = pix.second;

			if (y == height / 2) {
				angle = angle2;
			}

			// �ռ�8���������������׸���Ч�Ӳ�ֵ
			disp_collects.clear();
			for (sint32 s = 0; s < 8; s++) {
				const float32 ang = angle[s];
				const float32 sina = float32(sin(ang));
				const float32 cosa = float32(cos(ang));
				for (sint32 m = 1; m < max_search_length; m++) {
					const sint32 yy = lround(y + m * sina);
					const sint32 xx = lround(x + m * cosa);
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
						break;
					}
					const auto& disp = *(disp_ptr + yy * width + xx);
					if (disp != invalid_float) {
						disp_collects.push_back(disp);
						break;
					}
				}
			}
			if (disp_collects.empty()) {
				continue;
			}

			std::sort(disp_collects.begin(), disp_collects.end());

			// ������ڵ�������ѡ��ڶ�С���Ӳ�ֵ
			// �������ƥ��������ѡ����ֵ
			if (k == 0) {
				if (disp_collects.size() > 1) {
					fill_disps[n] = disp_collects[1];
				}
				else {
					fill_disps[n] = disp_collects[0];
				}
			}
			else {
				fill_disps[n] = disp_collects[disp_collects.size() / 2];
			}
		}
		for (auto n = 0u; n < trg_pixels.size(); n++) {
			auto& pix = trg_pixels[n];
			const sint32 y = pix.first;
			const sint32 x = pix.second;
			disp_ptr[y * width + x] = fill_disps[n];
		}
	}
}
