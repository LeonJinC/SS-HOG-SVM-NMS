#pragma once
#ifndef _NMS_H
#define _NMS_H

#include<opencv2\opencv.hpp>
#include<iostream>
#include"config.h"
namespace nms {
	//升序
	bool comp(const BBOX &value1, const BBOX &value2)
	{
		return value1.second > value2.second;
	}
	//计算两个边框的iou值
	float get_iou_value(cv::Rect rect1, cv::Rect rect2) {

		float intersection = (rect1&rect2).area();

		float Union = (rect1 | rect2).area();

		float IOU = intersection / Union;
		//std::cout << "IOU: " << IOU << std::endl;

		return IOU;

	}
	
	/**brief 该函数用于利用非极大值抑制算法，去除检测框中冗余的边框
	@param dets 带有预测置信度的检测边框集合
	@param nmsThreshold iou大于nmsThreshold的边框会被去除
	*/
	void nms_boxes(DetBBOXs &dets,float nmsThreshold) {

		sort(dets.begin(), dets.end(), comp);

		int updated_size = dets.size();
		for (int i = 0; i < updated_size; i++)
		{
			for (int j = i + 1; j < updated_size; j++)
			{
				float iou = get_iou_value(dets[i].first, dets[j].first);
				if (iou > nmsThreshold)
				{
					dets.erase(dets.begin() + j);
					//std::cout << "0k" << std::endl;
					updated_size = dets.size();
				}
			}
		}

	}
	
}






#endif // !_NMS_H
