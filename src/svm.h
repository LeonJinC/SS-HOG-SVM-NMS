#pragma once
#ifndef SVM_H
#define SVM_H
#include"config.h"
#include"opencv2\opencv.hpp"

class MYSVM {

public:
	MYSVM(Config &config,std::string modelpath,int &istrain):_config(config){
		_file.open(modelpath, std::ios::in);
		if (istrain==1&& !_file)
		{	
			std::cout << modelpath << " is not exit! " << std::endl;
			std::cout << modelpath << " start training! " << std::endl;
			svm = cv::ml::SVM::create();
			svm->setType(_config.svmtype);
			svm->setKernel(_config.svmkernel);//RBF//LINEAR
			svm->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_ITER, _config.maxCount, FLT_EPSILON));
		}
		else {
			std::cout << modelpath << " is loaded! " << std::endl;
			svm = cv::Algorithm::load<cv::ml::SVM>(modelpath);
		}


		
	}


	~MYSVM() {

	}
public:
	/** @brief ���ڸ�����X��Y�����������SVM����׼ȷ��
	@param X ����ͼ��
	@param Y ͼ��ı�ǩ
	*/
	void test(cv::Mat &X, cv::Mat &Y) {
		int totalNum = X.rows;
		int featuresize = X.cols;
		float TrueNum = 0;
		for (int i = 0; i < totalNum; i++)
		{
			cv::Mat testFeatureMat = cv::Mat::zeros(1, featuresize, CV_32FC1);
			for (int j = 0; j<featuresize; j++)
			{
				testFeatureMat.at<float>(0, j) = X.at<float>(i, j);
			}
			testFeatureMat.convertTo(testFeatureMat, CV_32F);

			//float confidence = svm->predict(testFeatureMat, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
			//confidence = confidence < 0 ? -confidence / 10 : confidence / 10;

			float prey = svm->predict(testFeatureMat);

			if (prey == Y.at<int>(i, 0))
			{
				TrueNum++;
			}
		}
		std::cout << "TrueNum: " << TrueNum << std::endl;
		std::cout << "TotalNum: " << totalNum << std::endl;
		std::cout << "Classification Accuracy: " << TrueNum / float(totalNum) * 100 << "%" << std::endl;
	}

	/** @brief ���ڸ�����X��Y��ѵ��SVM
	@param X ����ͼ��
	@param Y ͼ��ı�ǩ
	@param savepath ����·��
	*/
	void train(cv::Mat &X, cv::Mat &Y,std::string savepath) {
		std::cout << "��ʼѵ����������" << std::endl;


		svm->train(X, cv::ml::ROW_SAMPLE, Y);
		svm->save(savepath);

		std::cout << "������ѵ��������" << std::endl;
	}

	/** @brief ���ڸ�����descriptor���õ�Ԥ���������Ŷ�
	@param descriptor ����ͼ���HOG������
	@return std::pair<float,float> ����Ԥ�����prey�����Ŷ�confidence
	*/
	std::pair<float,float> predict(std::vector<float> &descriptor) {

		cv::Mat testFeatureMat = cv::Mat(descriptor);
		testFeatureMat.convertTo(testFeatureMat, CV_32F);
		testFeatureMat = testFeatureMat.t();
		
		float confidence = svm->predict(testFeatureMat, cv::noArray(), cv::ml::StatModel::Flags::RAW_OUTPUT);
		float prey = svm->predict(testFeatureMat);

		return std::make_pair(prey, confidence);
	}


private:
	cv::Ptr<cv::ml::SVM> svm;
	Config _config;
	std::fstream _file;
};



#endif // !SVM_H
