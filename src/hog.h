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


	/** @brief ���ڸ���ͼ��Img����HOG������
	@param Img ͼ��
	@param descriptor ������������ӣ�std::vector<float>
	*/
	void compute(cv::Mat &Img, std::vector<float> &descriptor) {
		hog->compute(Img, descriptor);

	};

	int height;// = 64;
	int width;// = 32;
	int length;//���������ĳ���


private:
	int cell_size;// = 8;//ͼ��ֿ�,�����ص�����
	int scaleBlock;// = 2;//��cell����
	int stride;// = 1;//block��������������λ��cell_size��������һ���Ĳ���Ϊһ��cell_size

	int bin_size;// = 8;//�Ƕȷ�������С��Ԫ��������ά����

	int blocksize;
	int x_dim;
	int y_dim;
	int z_dim;


	cv::HOGDescriptor *hog;


};




#endif // !HOG_H

