#pragma once
#ifndef MAP_H
#define MAP_H

#include<opencv2\opencv.hpp>
#include<iostream>
#include"config.h"

namespace mAP {
	float rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
	{
		float intersection = (rectA&rectB).area();

		float Union = (rectA | rectB).area();

		float IOU = intersection / Union;
		//std::cout << "IOU: " << IOU << std::endl;

		return IOU;
	}

	bool isOverIOU(cv::Rect &det_rect, std::vector<cv::Rect> &gt) {


		for (auto &gt_rect : gt) {
			if (rectA_intersect_rectB(det_rect, gt_rect) >= 0.5) {
				return true;
			}
		}
		return false;
	}

	void arange(std::vector<float> &threshold, float begin, float end, float step) {
		if (step > 0) {
			for (float b = begin; b <= end; b += step) {
				threshold.push_back(b);
			}
		}
		else
		{
			for (float b = begin; b >= end; b += step) {
				threshold.push_back(b);
			}
		}

	}

	void measure_mAP(std::vector<std::pair<DetBBOXs, GTBBOXs>> &result) {
		double gt_sum = 0;
		for (auto &r : result) {
			gt_sum += r.second.size();
		}
		//std::cout << gt_sum << std::endl;

		std::vector<float> threshold;
		arange(threshold, 4.0, 0.0, -0.1);
		std::vector<std::pair<double, double>> PR_score;

		for (int i = 0; i < threshold.size(); i++) {
			double Td = threshold[i];
			double det_sum = 0;
			double tp = 0;
			for (auto &r : result) {
				for (auto &dets : r.first) {
					if (dets.second >= Td) {
						det_sum += 1;

						if (isOverIOU(dets.first, r.second)) {
							tp += 1;
						}
					}
				}

			}
			std::cout << "threshold: " << Td << "\t det_sum: " << det_sum << "\t tp: " << tp << "\t precision: " << tp / det_sum << "\t recall: " << tp / gt_sum << std::endl;

			double precision = tp / det_sum;
			double recall = tp / gt_sum;


			PR_score.push_back(std::make_pair(precision, recall));
		}


		float max_s=0;
		std::map<float, float> PR_curve;
		std::vector<float > recall_threshold;
		arange(recall_threshold, 0.0, 1.1, 0.1);

		for (int i = 0; i < recall_threshold.size(); i++) {
			if (PR_curve.find(recall_threshold[i]) == PR_curve.end()) {
				PR_curve[recall_threshold[i]] = 0.0;
			}
		}
		for (auto &score : PR_score) {
			//std::cout << "precision: " << score.first << "\t recall:" << score.second << std::endl;
			for (int i = 0; i < recall_threshold.size();i++) {
				
				if (score.second>= recall_threshold[i] &&score.first > PR_curve[recall_threshold[i]]) {
					PR_curve[recall_threshold[i]] = score.first;
				}

			}
		}
		float mAP = 0;
		//std::cout << PR_curve.size()<< std::endl;
		for (auto iter = PR_curve.begin(); iter != PR_curve.end(); iter++) {
			mAP+= iter->second;
			std::cout << "(recall, precision)= " << "(" << iter->first << ", " << iter->second << ")" << std::endl;
		}
		mAP /= PR_curve.size();
		std::cout << "mAP: " << mAP*100<<"%" << std::endl;
		
	}

	void show_result(std::vector<std::pair<DetBBOXs, GTBBOXs>> &result){
		for (int i = 0; i < result.size();i++) {
			auto dets = result[i].first;
			auto gts = result[i].second;
			std::cout << i << " --> dets&gts" << std::endl;

			for (auto &d : dets) {
				std::cout <<"dets:\t"<< d.first.tl() << " " << d.first.br() <<" "<<d.second<< std::endl;
			}
			std::cout << std::endl;

			for (auto &gt : gts) {
				std::cout << "gts:\t" << gt.tl() << " " << gt.br() << std::endl;
			}
			std::cout << std::endl;

		}
	}
}







#endif // !MAP_H
