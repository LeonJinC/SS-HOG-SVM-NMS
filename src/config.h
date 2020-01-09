#pragma once
#ifndef CONFIG_H
#define CONFIG_H
#include<direct.h>
#include<io.h>
#include<iostream>
using BBOX = std::pair<cv::Rect, float>;
using DetBBOXs = std::vector<BBOX>;
using GTBBOXs = std::vector<cv::Rect>;

class Config {
public:
	Config() {

		if (_access(DataPath.c_str(), 0) == -1)
		{
			int i = _mkdir(DataPath.c_str());
		}

		if (_access(ModelPath.c_str(), 0) == -1)
		{
			int i = _mkdir(ModelPath.c_str());
		}
	}
	~Config() {

	}

public:
	//基本框架参数
	std::string RootPath = "./path_to_dataset/";
	std::string DetectionPath = RootPath + std::string("Detection/");
	std::string ClassificationPath = RootPath + std::string("Classification/");
	std::string DataPath = std::string("./data/");
	std::string ModelPath = std::string("./model/");

	//目标检测器的测试参数
	//std::string testDetectionTXT = "Train/pos1.txt";
	//std::string testAnnotationsPath = DetectionPath + std::string("Train/annotations/");
	std::string testDetectionTXT = "Test/pos_06_part1.txt";
	std::string testAnnotationsPath = DetectionPath + std::string("Test/annotations/");
	float confidence_threshold = 1.5;
	float iou_threshold = 0.1;

	//目标检测器的pipe训练参数
	std::string pipetrainDetectionTXT = "Train/pos.txt";
	std::string pipetrainAnnotationsPath = DetectionPath + std::string("Train/annotations/");
	std::string pipeTrainModel = ModelPath + "pipeTrainModel.xml";//pipe训练阶段的SVM模型
	std::string pipeTrainData = DataPath + "pipeTrainData.xml";//pipe训练阶段的train数据集

	//HOG特征提取器的提取参数
	int height = 64; //图像高度
	int width = 32; //图像宽度
	int cell_size = 8; //cell的尺寸
	int scaleBlock = 2; //单行Block包含cell的个数，比如一行有2个cell
	int stride = 1; //block的滑行步幅
	int bin_size = 8;//梯度直方图的分区数量，比如0~180分为8个区间

	//分类器SVM的pre训练参数
	std::string TrainPath = ClassificationPath + std::string("Train/");
	std::string TestPath = ClassificationPath + std::string("Test/");
	std::string preTrainModel = ModelPath+"preTrainModel.xml";//预训练SVM模型
	std::string preTestData = DataPath+"preTestData.xml";//预训练阶段的test数据集
	std::string preTrainData = DataPath+"preTrainData.xml";//预训练阶段的train数据集
	cv::ml::SVM::Types svmtype = cv::ml::SVM::Types::C_SVC;//SVM的类型，比如C_SVC表示C类支撑向量分类机。n类分组（n≥2），容许用异常值处罚因子C进行不完全分类。
	cv::ml::SVM::KernelTypes svmkernel = cv::ml::SVM::LINEAR;//SVM核函数的类型，比如LINEAR表示线性函数
	int maxCount = 1e6;//迭代次数
};

#endif // !CONFIG_H
