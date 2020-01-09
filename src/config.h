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
	//������ܲ���
	std::string RootPath = "./path_to_dataset/";
	std::string DetectionPath = RootPath + std::string("Detection/");
	std::string ClassificationPath = RootPath + std::string("Classification/");
	std::string DataPath = std::string("./data/");
	std::string ModelPath = std::string("./model/");

	//Ŀ�������Ĳ��Բ���
	//std::string testDetectionTXT = "Train/pos1.txt";
	//std::string testAnnotationsPath = DetectionPath + std::string("Train/annotations/");
	std::string testDetectionTXT = "Test/pos_06_part1.txt";
	std::string testAnnotationsPath = DetectionPath + std::string("Test/annotations/");
	float confidence_threshold = 1.5;
	float iou_threshold = 0.1;

	//Ŀ��������pipeѵ������
	std::string pipetrainDetectionTXT = "Train/pos.txt";
	std::string pipetrainAnnotationsPath = DetectionPath + std::string("Train/annotations/");
	std::string pipeTrainModel = ModelPath + "pipeTrainModel.xml";//pipeѵ���׶ε�SVMģ��
	std::string pipeTrainData = DataPath + "pipeTrainData.xml";//pipeѵ���׶ε�train���ݼ�

	//HOG������ȡ������ȡ����
	int height = 64; //ͼ��߶�
	int width = 32; //ͼ����
	int cell_size = 8; //cell�ĳߴ�
	int scaleBlock = 2; //����Block����cell�ĸ���������һ����2��cell
	int stride = 1; //block�Ļ��в���
	int bin_size = 8;//�ݶ�ֱ��ͼ�ķ�������������0~180��Ϊ8������

	//������SVM��preѵ������
	std::string TrainPath = ClassificationPath + std::string("Train/");
	std::string TestPath = ClassificationPath + std::string("Test/");
	std::string preTrainModel = ModelPath+"preTrainModel.xml";//Ԥѵ��SVMģ��
	std::string preTestData = DataPath+"preTestData.xml";//Ԥѵ���׶ε�test���ݼ�
	std::string preTrainData = DataPath+"preTrainData.xml";//Ԥѵ���׶ε�train���ݼ�
	cv::ml::SVM::Types svmtype = cv::ml::SVM::Types::C_SVC;//SVM�����ͣ�����C_SVC��ʾC��֧�������������n����飨n��2�����������쳣ֵ��������C���в���ȫ���ࡣ
	cv::ml::SVM::KernelTypes svmkernel = cv::ml::SVM::LINEAR;//SVM�˺��������ͣ�����LINEAR��ʾ���Ժ���
	int maxCount = 1e6;//��������
};

#endif // !CONFIG_H
