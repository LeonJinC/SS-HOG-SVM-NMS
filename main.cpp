#include<opencv2\opencv.hpp>
#include<iostream>
#include"ss.h"
#include"hog.h"
#include"config.h"
#include"nms.h"
#include"map.h"
#include"svm.h"
#include"utils.h"


/*brief 该函数用于对分类器SVM进行预训练pretrain
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为1
*/
void preTrainSVM(Config &config, int istrain=1) {
	cv::Mat trainX, trainY, testX, testY;
	std::fstream _file;
	std::cout << "loading Data ...... " << std::endl;
	_file.open(config.preTrainData, std::ios::in);
	if (!_file)
	{
		std::cout << config.preTrainData << " 没有被创建" << std::endl;

		utils::loadData(config, config.TrainPath, config.preTrainData, trainX, trainY);
	}
	else {
		std::cout << config.preTrainData << " 已经存在" << std::endl;
		utils::readData(config.preTrainData, trainX, trainY);
	}
	_file.close();
	std::cout << "trainX尺寸 --> " << trainX.rows << " x " << trainX.cols << std::endl;
	std::cout << "trainY尺寸 --> " << trainY.rows << " x " << trainY.cols << std::endl;

	_file.open(config.preTestData, std::ios::in);
	if (!_file)
	{
		std::cout << config.preTestData << " 没有被创建" << std::endl;

		utils::loadData(config, config.TestPath, config.preTestData, testX, testY);
	}
	else {
		std::cout << config.preTestData << " 已经存在" << std::endl;
		utils::readData(config.preTestData, testX, testY);

	}
	_file.close();
	std::cout << "testX尺寸 --> " << testX.rows << " x " << testX.cols << std::endl;
	std::cout << "testY尺寸 --> " << testY.rows << " x " << testY.cols << std::endl;


	MYSVM svm(config, config.preTrainModel, istrain);
	
	if(istrain==1)svm.train(trainX,trainY,config.preTrainModel);

	svm.test(testX,testY);

}
/*brief 该函数用于对分类器SVM进行预训练pretrain
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为1
*/
void pipeLineTrain(Config &config, int istrain = 1)
{
	std::ifstream f;// 获取测试图片文件路径

	f.open(config.DetectionPath + config.pipetrainDetectionTXT, std::ios::in);//读入图像

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

	//这里可以将preTrainModel改成pipeTrainModel，并在model文件夹中删去pipeTrainModel.xml
	//这样就不是load预训练的SVM模型，而是从新开始训练，
	//不过，不管是load预训练的SVM模型preTrainModel，还是重新开始训练pipeTrainModel
	//对结果mAP或者任何实验指标都没有任何影响
	MYSVM svm(config,config.preTrainModel, istrain);
	
	if(istrain==1)svm.train(X, Y,config.pipeTrainModel);


}

/*brief 该函数用于对目标检测方法进行整体的测试，计算mAP
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为0,因为这条函数用于pipeLine测试，所以不用训练
*/
void pipeLineTest(Config &config, int istrain = 0) {
	std::ifstream f;// 获取测试图片文件路径

	f.open(config.DetectionPath + config.testDetectionTXT, std::ios::in);//读入图像

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

