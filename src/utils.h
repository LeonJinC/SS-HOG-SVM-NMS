#pragma once
#ifndef UTILS_H
#define UTILS_H
#include<opencv2\opencv.hpp>
#include"config.h"
#include"hog.h"

namespace utils {
	//获取ground truth边框集合
	bool get_GTset(std::string annotation_name, GTBBOXs &GTs) {
		std::ifstream file;

		file.open(annotation_name, std::ios::in);
		if (file.fail())
		{
			return false;
		}

		std::string buffer2;


		while (getline(file, buffer2, '\n')) {
			if (buffer2.find("Bounding box") != std::string::npos) {
				//std::cout << buffer2 << std::endl;

				std::string temp = buffer2.substr(buffer2.find(":") + 2);
				int Xmin = std::stoi(temp.substr(temp.find("(") + 1, temp.find(",") - 1));

				std::string temp2 = temp.substr(temp.find(",") + 2);
				int Ymin = std::stoi(temp2.substr(0, temp2.find(")")));

				std::string temp3 = temp2.substr(temp2.find("(") + 1);
				int Xmax = std::stoi(temp3.substr(0, temp3.find(",")));

				std::string temp4 = temp3.substr(temp3.find(",") + 2);
				int Ymax = std::stoi(temp4.substr(0, temp.find(")")));

				//std::cout << "(Xmin, Ymin) - (Xmax, Ymax) : " << "(" << Xmin << ", " << Ymin << ") - (" << Xmax<<", "<< Ymax<<")" <<std::endl;
				GTs.push_back(cv::Rect(cv::Point(Xmin, Ymin), cv::Point(Xmax, Ymax)));
			}
		}
		return true;
	}

	//计算iou
	float calIOU(cv::Rect rectA, cv::Rect rectB)
	{
		float intersection = (rectA&rectB).area();

		float Union = (rectA | rectB).area();

		float IOU = intersection / Union;
		//std::cout << "IOU: " << IOU << std::endl;

		return IOU;
	}
	//升序比较函数
	bool cmp(std::pair<cv::Rect, float>&p1, std::pair<cv::Rect, float>&p2) {
		return p1.second > p2.second;
	}
	//重命名，比如数字10转变为字符串00010，数字2转变为字符串00002
	std::string rename(int count) {
		int n = 5 - std::to_string(count).size();
		std::string prex;
		for (int i = 0; i < n; i++) {
			prex += std::to_string(0);
		}
		return prex;
	}
	
	//将X和Y保存到指定文件路径savepath
	void downloadData(std::string savepath, cv::Mat &X, cv::Mat &Y) {
		cv::FileStorage fs(savepath, cv::FileStorage::WRITE);
		fs << "X" << X;
		fs << "Y" << Y;
		fs.release();
	}

	/**brief 将检测结果和GroundTruth可视化
	@param img 待检测的图像
	@param dets 检测结果
	@param GTs GroundTruth边框集合
	@param wtime cv::waitKey(wtime),等待时间，单位ms
	@param resize_scale 图像缩放尺度，比如resize_scale=5就是将图像放大5倍
	*/
	void visualise(cv::Mat &img, DetBBOXs& dets, GTBBOXs& GTs, int wtime = 10, float resize_scale = 5) {
		//float resize_scale = 5;
		cv::resize(img, img, cv::Size(0, 0), resize_scale, resize_scale);
		for (auto &d : dets) {
			cv::Rect o = d.first;
			cv::Rect r = cv::Rect(cv::Point(o.tl().x * resize_scale, o.tl().y * resize_scale), cv::Point(o.br().x * resize_scale, o.br().y * resize_scale));
			float p = d.second;
			cv::rectangle(img, r, cv::Scalar(0, 0, 255), 2);
			cv::putText(img, std::to_string(p), r.tl(), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
			//std::cout << "confience: " << p << "\t r.tl(): " << r.tl() << "\t r.br(): " << r.br() << std::endl;
		}

		for (auto &rect : GTs) {
			cv::Rect r = cv::Rect(cv::Point(rect.tl().x * resize_scale, rect.tl().y * resize_scale), cv::Point(rect.br().x * resize_scale, rect.br().y * resize_scale));
			cv::rectangle(img, r, cv::Scalar(255, 0, 0), 2);
		}

		cv::imshow("pipeLineTest", img);
		int c = cv::waitKey(wtime);
		if (c == 'p' || c == 'P') {
			c = cv::waitKey(0);
		}

	}

	//根据图像文件路径，输出数据集X和Y
	void loadData(Config config, std::string DataSetPath, std::string savefile, cv::Mat &X, cv::Mat &Y) {
		MYHOG hog(config);
		std::ifstream filestream;
		std::vector<std::vector<float>> descriptors;
		std::vector<int> labels;
		std::string buffer;
		filestream.open(DataSetPath + std::string("pos.txt"), std::ios::in);
		if (filestream.fail()) {
			std::cout << "open fail!" << std::endl;
		}
		while (std::getline(filestream, buffer, '\n'))
		{
			cv::Mat Img = cv::imread(config.ClassificationPath + buffer);
			cv::resize(Img, Img, cv::Size(hog.width, hog.height));

			std::vector<float> descriptor;
			hog.compute(Img, descriptor);
			descriptors.push_back(descriptor);
			labels.push_back(1);
		}
		filestream.close();

		filestream.open(DataSetPath + std::string("neg.txt"), std::ios::in);
		if (filestream.fail()) {
			std::cout << "open fail!" << std::endl;
		}
		while (std::getline(filestream, buffer, '\n'))
		{
			cv::Mat Img = cv::imread(config.ClassificationPath + buffer);
			cv::resize(Img, Img, cv::Size(hog.width, hog.height));

			std::vector<float> descriptor;
			hog.compute(Img, descriptor);
			descriptors.push_back(descriptor);
			labels.push_back(-1);
		}
		filestream.close();

		X = cv::Mat::zeros(descriptors.size(), hog.length, CV_32FC1);
		Y = cv::Mat::zeros(descriptors.size(), 1, CV_32SC1);


		for (int i = 0; i < descriptors.size(); i++) {
			Y.at<int>(i, 0) = labels[i];
			for (int j = 0; j < hog.length; j++) {
				X.at<float>(i, j) = descriptors[i][j];
			}
		}

		//std::cout << "X尺寸 --> " << X.rows << " x " << X.cols << std::endl;
		//std::cout << "Y尺寸 --> " << Y.rows << " x " << Y.cols << std::endl;

		downloadData(savefile, X, Y);

	}

	//读取数据集X和Y
	void readData(std::string savefile, cv::Mat &X, cv::Mat &Y) {
		cv::FileStorage fs(savefile, cv::FileStorage::READ);
		fs["X"] >> X;
		fs["Y"] >> Y;
		//std::cout << "X尺寸 --> " << X.rows << " x " << X.cols << std::endl;
		//std::cout << "Y尺寸 --> " << Y.rows << " x " << Y.cols << std::endl;
	};

}



#endif // !UTILS_H
