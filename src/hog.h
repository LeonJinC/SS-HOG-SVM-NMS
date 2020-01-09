#pragma once
#ifndef HOG_H
#define HOG_H
#include<opencv2\opencv.hpp>
#include"config.h"

class MYHOG {
public:
	MYHOG(Config config)
		: height(config.height), width(config.width), cell_size(config.cell_size), 
		scaleBlock(config.scaleBlock), stride(config.stride),bin_size(config.bin_size){

		blocksize = scaleBlock*cell_size;
		x_dim = height / cell_size - (scaleBlock - 2) - stride;
		y_dim = width / cell_size - (scaleBlock - 2) - stride;
		z_dim = pow(scaleBlock, 2)*bin_size;
		length = x_dim*y_dim*z_dim;

		hog = new cv::HOGDescriptor(cv::Size(width, height), //winSize
			cv::Size(blocksize, blocksize), //blockSize
			cv::Size(stride*cell_size, stride*cell_size), //blockStride
			cv::Size(cell_size, cell_size), //cellSize
			bin_size);//nbins
	};
	~MYHOG() {};


	/** @brief 对于给定图像Img计算HOG描述子
	@param Img 图像
	@param descriptor 待计算的描述子，std::vector<float>
	*/
	void compute(cv::Mat &Img, std::vector<float> &descriptor) {
		hog->compute(Img, descriptor);

	};

	int height;// = 64;
	int width;// = 32;
	int length;//特征向量的长度


private:
	int cell_size;// = 8;//图像分块,行像素点数量
	int scaleBlock;// = 2;//行cell数量
	int stride;// = 1;//block，滑动步长，单位是cell_size，即滑动一步的步长为一个cell_size

	int bin_size;// = 8;//角度分区，最小单元特征向量维度数

	int blocksize;
	int x_dim;
	int y_dim;
	int z_dim;


	cv::HOGDescriptor *hog;


};




#endif // !HOG_H

