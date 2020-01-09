#include<opencv2\opencv.hpp>
#include<iostream>
#include"ss.h"
#include"hog.h"
#include"config.h"
#include"nms.h"
#include"map.h"
#include"svm.h"
#include"utils.h"


/*
�ú������ڶԷ�����SVM����Ԥѵ��pretrain
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void preTrainSVM(Config &config, int istrain=1) {
	cv::Mat trainX, trainY, testX, testY;
	std::fstream _file;
	std::cout << "loading Data ...... " << std::endl;
	_file.open(config.preTrainData, std::ios::in);
	if (!_file)
	{
		std::cout << config.preTrainData << " û�б�����" << std::endl;

		utils::loadData(config, config.TrainPath, config.preTrainData, trainX, trainY);
	}
	else {
		std::cout << config.preTrainData << " �Ѿ�����" << std::endl;
		utils::readData(config.preTrainData, trainX, trainY);
	}
	_file.close();
	std::cout << "trainX�ߴ� --> " << trainX.rows << " x " << trainX.cols << std::endl;
	std::cout << "trainY�ߴ� --> " << trainY.rows << " x " << trainY.cols << std::endl;

	_file.open(config.preTestData, std::ios::in);
	if (!_file)
	{
		std::cout << config.preTestData << " û�б�����" << std::endl;

		utils::loadData(config, config.TestPath, config.preTestData, testX, testY);
	}
	else {
		std::cout << config.preTestData << " �Ѿ�����" << std::endl;
		utils::readData(config.preTestData, testX, testY);

	}
	_file.close();
	std::cout << "testX�ߴ� --> " << testX.rows << " x " << testX.cols << std::endl;
	std::cout << "testY�ߴ� --> " << testY.rows << " x " << testY.cols << std::endl;


	MYSVM svm(config, config.preTrainModel, istrain);
	
	if(istrain==1)svm.train(trainX,trainY,config.preTrainModel);

	svm.test(testX,testY);

}
/*
�ú������ڶԷ�����SVM����pipelineѵ��
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void pipeLineTrain(Config &config, int istrain = 1)
{
	std::ifstream f;// ��ȡ����ͼƬ�ļ�·��

	f.open(config.DetectionPath + config.pipetrainDetectionTXT, std::ios::in);//����ͼ��

	if (f.fail())
	{
		std::fprintf(stderr, "ERROR: the specified file could not be loaded\n");
		return;
	}

	std::string buffer; 
	std::vector<std::vector<float>> vectorX;
	std::vector<int> vectorY;
	MYHOG hog(config);

	while (getline(f, buffer, '\n')) {
		
		
		cv::Mat img = cv::imread(config.DetectionPath + buffer);
		if (!img.data)continue;

		std::string annotation_name = buffer.substr(buffer.find("/") + 1, 2) + "_" + buffer.substr(buffer.rfind("/") + 1, 5) + ".txt";

		GTBBOXs GTs;
		if (!utils::get_GTset(config.pipetrainAnnotationsPath + annotation_name, GTs))continue;

		//std::cout << buffer << " " << std::endl;

		
		//auto proposals = ss::selectiveSearch(img, 10.0, 1.0, 50.0, 1, 1e5, 1.3);

		auto proposals = ss::denseSearch(img, 16, 8);


		std::vector<BBOX> pos_map;
		std::vector<BBOX> neg_map;
		for (auto &p_rect : proposals) {
			for (auto &gt_rect : GTs) {
				float iou = utils::calIOU(p_rect, gt_rect);
				if (iou < 0.1)
				{
					neg_map.push_back(std::make_pair(p_rect,iou));
				}

				if (iou >= 0.7)
				{
					pos_map.push_back(std::make_pair(p_rect, iou));
				}


			}

		}
		
		for (auto &rect : pos_map) {
			cv::Mat roi = img(rect.first).clone();
			cv::resize(roi, roi, cv::Size(hog.width, hog.height));

			std::vector<float> descriptor;
			hog.compute(roi, descriptor);
			vectorX.push_back(descriptor);
			vectorY.push_back(1);
		}

		std::sort(neg_map.begin(), neg_map.end(), utils::cmp);
		for (int i = 0; i < pos_map.size()*4;i++) {
			//std::cout << neg_map[i] << std::endl;
			cv::Mat roi = img(neg_map[i].first).clone();
			cv::resize(roi, roi, cv::Size(hog.width, hog.height));

			std::vector<float> descriptor;
			hog.compute(roi, descriptor);
			vectorX.push_back(descriptor);
			vectorY.push_back(-1);
		}
	}

	std::cout << vectorX.size() << " " << vectorY.size() << std::endl;
	cv::Mat X = cv::Mat::zeros(vectorX.size(), hog.length, CV_32FC1);
	cv::Mat Y = cv::Mat::zeros(vectorY.size(), 1, CV_32SC1);

	for (int i = 0; i < vectorX.size(); i++) {
		Y.at<int>(i, 0) = vectorY[i];
		for (int j = 0; j < hog.length; j++) {
			X.at<float>(i, j) = vectorX[i][j];
		}
	}


	std::ifstream file;
	file.open(config.pipeTrainData, std::ios::in);
	if (!file) {
		utils::downloadData(config.pipeTrainData, X, Y);
	}

	//������Խ�preTrainModel�ĳ�pipeTrainModel������model�ļ�����ɾȥpipeTrainModel.xml
	//�����Ͳ���loadԤѵ����SVMģ�ͣ����Ǵ��¿�ʼѵ����
	//������������loadԤѵ����SVMģ��preTrainModel���������¿�ʼѵ��pipeTrainModel
	//�Խ��mAP�����κ�ʵ��ָ�궼û���κ�Ӱ��
	MYSVM svm(config,config.preTrainModel, istrain);
	
	if(istrain==1)svm.train(X, Y,config.pipeTrainModel);


}

/*
�ú������ڶ�Ŀ���ⷽ����������Ĳ��ԣ�����mAP
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void pipeLineTest(Config &config, int istrain = 0) {
	std::ifstream f;// ��ȡ����ͼƬ�ļ�·��

	f.open(config.DetectionPath + config.testDetectionTXT, std::ios::in);//����ͼ��

	if (f.fail())
	{
		std::fprintf(stderr, "ERROR: the specified file could not be loaded\n");
		return;
	}

	std::string buffer;
	std::vector<std::vector<float>> vectorX;
	std::vector<int> vectorY;
	MYHOG hog(config);

	MYSVM svm(config, config.pipeTrainModel, istrain);

	std::vector<std::pair<DetBBOXs, GTBBOXs>> result;
	while (getline(f, buffer, '\n')) {

		cv::Mat img = cv::imread(config.DetectionPath + buffer);

		if (!img.data)continue;

		std::string annotation_name = buffer.substr(buffer.find("/") + 1, 2) 
										+ "_" + buffer.substr(buffer.rfind("/") + 1, 5) 
										+ ".txt";

		GTBBOXs GTs;
		if (!utils::get_GTset(config.testAnnotationsPath + annotation_name, GTs))continue;


		DetBBOXs DETs;
		int rows = img.rows;
		int cols = img.cols;
		//std::cout << rows << " " << cols << std::endl;
		//auto proposals = ss::selectiveSearch(img, 10.0, 1.0, 50.0, 1, 1e5, 1.3);//cout << proposals.size() << endl;
		auto proposals = ss::denseSearch(img, 16, 8);

		for (auto &rect : proposals) {

			//cv::rectangle(img, rect, Scalar(0, 255, 0));
			//std::cout << rect.tl() << " " << rect.br() << std::endl;

			cv::Mat roi = img(rect).clone();
			cv::resize(roi, roi, cv::Size(config.width, config.height));

			std::vector<float> descriptor;
			hog.compute(roi, descriptor);
			std::pair<float, float> r=svm.predict(descriptor);

			float prey = r.first;
			float confidence = r.second;

			
			confidence = prey > 0 ? -confidence : confidence;

			if (prey == 1 && confidence >= config.confidence_threshold) {
				DETs.push_back(std::make_pair(rect, confidence));
			}

			nms::nms_boxes(DETs, config.iou_threshold);
		}

		result.push_back(std::make_pair(DETs, GTs));

		utils::visualise(img, DETs, GTs, 10);
	}
	
	mAP::measure_mAP(result);
}

int main()
{

	Config config;
	
	preTrainSVM(config);

	pipeLineTrain(config);

    pipeLineTest(config);

    return 0;
}

