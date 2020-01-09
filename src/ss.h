#pragma once
#ifndef SS_H
#define SS_H

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include<opencv2\opencv.hpp>

#define CV_VERSION_STR CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define CV_EXT_STR "d.lib"
#else
#define CV_EXT_STR ".lib"
#endif

#pragma comment(lib, "opencv_world"	CV_VERSION_STR CV_EXT_STR)

namespace std
{
	template<>
	class hash<std::pair<int, int>>
	{
	public:
		std::size_t operator()(const std::pair<int, int> &x) const
		{
			return hash<int>()(x.first) ^ hash<int>()(x.second);
		}
	};
}


namespace ss
{

	inline double square(double a)
	{
		return a*a;
	}

	//�����(x1,y1)�͵�(x2,y2)��ͼ������ͨ���ϵ��ۼӲ�ĸ���Ҳ���Ǽ�������������ɫ�ռ��ŷ�Ͼ���
	inline double diff(const cv::Mat &img, int x1, int y1, int x2, int y2)
	{
		return sqrt(square(img.at<cv::Vec3f>(y1, x1)[0] - img.at<cv::Vec3f>(y2, x2)[0]) +
			   square(img.at<cv::Vec3f>(y1, x1)[1] - img.at<cv::Vec3f>(y2, x2)[1]) +
			   square(img.at<cv::Vec3f>(y1, x1)[2] - img.at<cv::Vec3f>(y2, x2)[2]));
	}


	struct UniverseElement
	{
		int rank;//����ĵȼ����ȼ�Խ�ߴ�����Խ��
		int p;//�ö���ĸ��ڵ���
		int size;//�ö���Ķȣ�Ҳ���Ǻ͸ö������ӵĽڵ���

		UniverseElement() : rank(0), size(1), p(0) {}
		UniverseElement(int rank, int size, int p) : rank(rank), size(size), p(p) {}
	};


	class Universe
	{
	private:
		std::vector<UniverseElement> elements;
		int num;

	public:
		Universe(int num) : num(num)
		{
			elements.reserve(num);

			for (int i = 0; i < num; i++)
			{
				elements.emplace_back(0, 1, i);
			}
		}

		~Universe() {}

		//����x��xΪͼ��ĳһ�����㣬����ö���������һ������Ķ�����
		int find(int x)
		{
			int y = x;
			while (y != elements[y].p)
			{
				y = elements[y].p;
			}
			elements[x].p = y;

			return y;
		}
		//����x��y�����x�����rank>y�����rank,��y���������x��x�������������+1
		//���x�����rank<=y�����rank,��x���������y��y�������������+1,�����x�����rank==y�����rank����y�����rank+1
		void join(int x, int y)
		{
			if (elements[x].rank > elements[y].rank)
			{
				elements[y].p = x;
				elements[x].size += elements[y].size;
			}
			else
			{
				elements[x].p = y;
				elements[y].size += elements[x].size;
				if (elements[x].rank == elements[y].rank)
				{
					elements[y].rank++;
				}
			}
			num--;
		}

		int size(int x) const { return elements[x].size; }
		int numSets() const { return num; }
	};


	struct edge
	{
		int a;//��ʼ���y1*width+x1
		int b;//�յ��y2*width+x2
		double w;//(x1,y1)��(x2,y2)����ɫ�ռ��ϵ�ŷ�Ͼ���
	};


	bool operator<(const edge &a, const edge &b)
	{
		return a.w < b.w;
	}


	inline double calThreshold(int size, double scale)
	{
		return scale / size;
	}


	std::shared_ptr<Universe> segmentGraph(int numVertices, int numEdges, std::vector<edge> &edges, double scale)
	{
		std::sort(edges.begin(), edges.end());//���������㷨����sort��Ҫ���ء�С�ں���������� operator< ��Ĭ������Ҳ���Ǵ�С����

		auto universe = std::make_shared<Universe>(numVertices);

		std::vector<double> threshold(numVertices, scale);//���ж�������ڲ����ƶ���ֵ

		for (auto &pedge : edges)
		{
			int a = universe->find(pedge.a);
			int b = universe->find(pedge.b);

			if (a != b)
			{
				if ((pedge.w <= threshold[a]) && (pedge.w <= threshold[b]))
				{//�����������ľ���С�ڸ����������Ĳ����ƶ���ֵ��Ҳ����˵��������㹻�Ľӽ���
					universe->join(a, b);
					a = universe->find(a);
					threshold[a] = pedge.w + calThreshold(universe->size(a), scale);
					//���¸���Ĳ����ƶ���ֵ����������edges�Ǹ���w��С��������ģ����Ը���Ĳ����ƶ���ֵʵ���������
					//scale�ܿ�����Ĳ����ƶ���ֵ��scaleԽ������Ĳ����ƶ���ֵ��Ծ�Խ��Ҳ����˵������𶼸������ں���һ��
				}
			}
		}

		return universe;
	}

	// image segmentation using "Efficient Graph-Based Image Segmentation"
	std::shared_ptr<Universe> segmentation(const cv::Mat &img, double scale, double sigma, int minSize)
	{
		const int width = img.cols;
		const int height = img.rows;

		cv::Mat imgF;
		img.convertTo(imgF, CV_32FC3);
		cv::cvtColor(imgF, imgF, CV_RGB2HSV);//H��ɫ����S�����Ͷȣ�V������//Զ��rgbЧ������
		//cv::cvtColor(imgF, imgF, CV_RGB2YCrCb);

		cv::Mat blurred;
		cv::GaussianBlur(imgF, blurred, cv::Size(5, 5), sigma);

		std::vector<edge> edges(width*height * 4);//ͼ���ÿ�������4����ɫ�ռ�ŷ�Ͼ���

		int num = 0;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
				{
					edges[num].a = y * width + x;//��ʼ����ı��
					edges[num].b = y * width + (x + 1);//�յ㶥��ı��
					edges[num].w = diff(blurred, x, y, x + 1, y);//������������ɫ�ռ��е�ŷ�Ͼ���
					num++;
				}

				if (y < height - 1)
				{
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + x;
					edges[num].w = diff(blurred, x, y, x, y + 1);
					num++;
				}

				if ((x < width - 1) && (y < height - 1))
				{
					edges[num].a = y * width + x;
					edges[num].b = (y + 1) * width + (x + 1);
					edges[num].w = diff(blurred, x, y, x + 1, y + 1);
					num++;
				}

				if ((x < width - 1) && (y > 0))
				{
					edges[num].a = y * width + x;
					edges[num].b = (y - 1) * width + (x + 1);
					edges[num].w = diff(blurred, x, y, x + 1, y - 1);
					num++;
				}
			}
		}

		auto universe = segmentGraph(width*height, num, edges, scale);


		for (int i = 0; i < num; i++)
		{
			int a = universe->find(edges[i].a);
			int b = universe->find(edges[i].b);
			if ((a != b) && ((universe->size(a) < minSize) || (universe->size(b) < minSize)))
			{//�����������size��С��С��minSize�����������ϲ�
				universe->join(a, b);
			}
		}

		return universe;
	}


	void visualize(const cv::Mat &img, std::shared_ptr<Universe> universe)
	{
		const int height = img.rows;
		const int width = img.cols;
		std::vector<cv::Vec3b> colors;

		cv::Mat segmentated(height, width, CV_8UC3);

		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::uniform_int_distribution<> rand256(0, 255);

		for (int i = 0; i < height*width; i++)
		{
			cv::Vec3b color(rand256(mt), rand256(mt), rand256(mt));
			colors.push_back(color);
		}

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				segmentated.at<cv::Vec3b>(y, x) = colors[universe->find(y*width + x)];
			}
		}
		cv::resize(segmentated, segmentated, cv::Size(0, 0), 5, 5);
		cv::imshow( "Initial Segmentation Result", segmentated );
		//cv::waitKey(0);
	}


	struct Region
	{
		int size;
		cv::Rect rect;
		std::vector<int> labels;
		std::vector<float> colourHist;
		std::vector<float> textureHist;

		Region() {}

		Region(const cv::Rect &rect, int label) : rect(rect)
		{
			labels.push_back(label);
		}

		Region(
			const cv::Rect &rect, int size,
			const std::vector<float> &&colourHist,
			const std::vector<float> &&textureHist,
			const std::vector<int> &&labels
		)
			: rect(rect), size(size), colourHist(std::move(colourHist)), textureHist(std::move(textureHist)), labels(std::move(labels))
		{}

		Region& operator=(const Region& region) = default;

		Region& operator=(Region&& region) noexcept
		{
			if (this != &region)
			{
				this->size = region.size;
				this->rect = region.rect;
				this->labels = std::move(region.labels);
				this->colourHist = std::move(region.colourHist);
				this->textureHist = std::move(region.textureHist);
			}

			return *this;
		}

		Region(Region&& region) noexcept
		{
			*this = std::move(region);
		}
	};


	std::shared_ptr<Universe> generateSegments(const cv::Mat &img, double scale, double sigma, int minSize)
	{
		auto universe = segmentation(img, scale, sigma, minSize);

		//visualize(img, universe);

		return universe;
	}

	//������������֮�����ɫ���ƶ�
	double calcSimOfColour(const Region &r1, const Region &r2)
	{
		assert(r1.colourHist.size() == r2.colourHist.size());

		float sum = 0.0;

		for (auto i1 = r1.colourHist.cbegin(), i2 = r2.colourHist.cbegin(); i1 != r1.colourHist.cend(); i1++, i2++)
		{
			sum += std::min(*i1, *i2);
		}

		return sum;
	}

	//������������֮����������ƶ�
	double calcSimOfTexture(const Region &r1, const Region &r2)
	{
		assert(r1.colourHist.size() == r2.colourHist.size());

		double sum = 0.0;

		for (auto i1 = r1.textureHist.cbegin(), i2 = r2.textureHist.cbegin(); i1 != r1.textureHist.cend(); i1++, i2++)
		{
			sum += std::min(*i1, *i2);
		}

		return sum;
	}

	//������������֮��Ĵ�С���ƶȣ�
	//���������������С�����ƶȺܸߣ�һ��һС���߶��ܴ������ƶȵ�
	inline double calcSimOfSize(const Region &r1, const Region &r2, int imSize)
	{
		return (1.0 - (double)(r1.size + r2.size) / imSize);
	}

	//������������֮��Ŀռ佻�����ƶȣ�
	//Ҳ����������������໥Χ�Ƶúܽ��ܣ������ƶȺܸ�
	inline double calcSimOfRect(const Region &r1, const Region &r2, int imSize)
	{
		return (1.0 - (double)((r1.rect | r2.rect).area() - r1.size - r2.size) / imSize);
	}

	//��������֮����������ƶ�
	inline double calcSimilarity(const Region &r1, const Region &r2, int imSize)
	{
		return (calcSimOfColour(r1, r2) + calcSimOfTexture(r1, r2) + 10*calcSimOfSize(r1, r2, imSize) + calcSimOfRect(r1, r2, imSize));
	}

	//����label��Ӧ�������ɫֱ��ͼ
	std::vector<float> calcColourHist(const cv::Mat &img, std::shared_ptr<Universe> universe, int label)
	{
		std::array<std::vector<unsigned char>, 3> hsv;

		for (auto &e : hsv)
		{
			e.reserve(img.total());
		}

		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				if (universe->find(y*img.cols + x) != label)
				{
					continue;
				}

				for (int channel = 0; channel < 3; channel++)
				{
					hsv[channel].push_back(img.at<cv::Vec3b>(y, x)[channel]);
				}
			}
		}

		int channels[] = { 0 };
		const int bins = 25;
		int histSize[] = { bins };
		float range[] = { 0, 256 };
		const float *ranges[] = { range };

		std::vector<float> features;

		for (int channel = 0; channel < 3; channel++)
		{
			cv::Mat hist;

			cv::Mat input(hsv[channel]);

			cv::calcHist(&input, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

			cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

			std::vector<float> histogram;
			hist.copyTo(histogram);

			if (features.empty())
			{
				features = std::move(histogram);
			}
			else
			{
				std::copy(histogram.begin(), histogram.end(), std::back_inserter(features));
			}
		}

		return features;
	}

	//�����label��Ӧ����������ĸ�������size��Ҳ���Ǹ�����������С
	int calcSize(const cv::Mat &img, std::shared_ptr<Universe> universe, int label)
	{
		int num = 0;

		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				if (universe->find(y * img.cols + x) == label)
				{//�鿴���ص���������Ƿ����label��������������+1
					num++;
				}
			}
		}

		return num;
	}

	//����ͼ����ݶ�ͼ
	cv::Mat calcTextureGradient(const cv::Mat &img)
	{
		cv::Mat sobelX, sobelY;

		cv::Sobel(img, sobelX, CV_32F, 1, 0);
		cv::Sobel(img, sobelY, CV_32F, 0, 1);

		cv::Mat magnitude, angle;
		cv::cartToPolar(sobelX, sobelY, magnitude, angle, true);

		return angle;
	}

	//����label��Ӧ������ݶ�ֱ��ͼ
	std::vector<float> calcTextureHist(const cv::Mat &img, const cv::Mat &gradient, std::shared_ptr<Universe> universe, int label)
	{
		const int orientations = 8;

		std::array<std::array<std::vector<unsigned char>, orientations>, 3> intensity;

		for (auto &e : intensity)
		{
			for (auto &ee : e)
			{
				ee.reserve(img.total());
			}
		}

		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				if (universe->find(y * img.cols + x) != label)
				{
					continue;
				}

				for (int channel = 0; channel < 3; channel++)
				{
					int angle = (int)(gradient.at<cv::Vec3f>(y, x)[channel] / 22.5) % orientations;
					intensity[channel][angle].push_back(img.at<cv::Vec3b>(y, x)[channel]);
				}
			}
		}

		int channels[] = { 0 };
		const int bins = 10;
		int histSize[] = { bins };
		float range[] = { 0, 256 };
		const float *ranges[] = { range };

		std::vector<float> features;

		for (int channel = 0; channel < 3; channel++)
		{
			for (int angle = 0; angle < orientations; angle++)
			{
				cv::Mat hist;

				cv::Mat input(intensity[channel][angle]);

				cv::calcHist(&input, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

				cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L1);

				std::vector<float> histogram;
				hist.copyTo(histogram);

				if (features.empty())
				{
					features = std::move(histogram);
				}
				else
				{
					std::copy(histogram.begin(), histogram.end(), std::back_inserter(features));
				}

			}
		}

		return features;
	}

	//��universe�м�¼�����ص㼰����Ϣ������ȼ�rank���������p���������ĸ�������size�������������ϣ�
	//�ó��������õ�Region����
	std::map<int, Region> extractRegions(const cv::Mat &img, std::shared_ptr<Universe> universe)
	{
		std::map<int, Region> R;

		for (int y = 0; y < img.rows; y++)
		{
			for (int x = 0; x < img.cols; x++)
			{
				int label = universe->find(y*img.cols + x);

				//Ϊÿһ��region����һ��label�����û�и�label��region�򴴽���
				if (R.find(label) == R.end())
				{
					R[label] = Region(cv::Rect(100000, 100000, 0, 0), label);
				}

				//����region��rect�߿򣬸�region��Ӧ�ı߿���ǡ�ý�����Χregion�ı߿�
				if (R[label].rect.x > x)//rect.xΪ�߿����Ͻǵ�x�����Ͻǵ�xԽ��ԽС
				{
					R[label].rect.x = x;
				}

				if (R[label].rect.y > y)//rect.yΪ�߿����Ͻǵ�y�����Ͻǵ�yԽ��ԽС
				{
					R[label].rect.y = y;
				}

				if (R[label].rect.br().x < x)//rect.br().xΪ�߿����½ǵ�x�����½ǵ�xԽ��Խ��
				{
					R[label].rect.width = x - R[label].rect.x + 1;
				}

				if (R[label].rect.br().y < y)//rect.br().yΪ�߿����½ǵ�y�����½ǵ�yԽ��Խ��
				{
					R[label].rect.height = y - R[label].rect.y + 1;
				}
			}
		}

		cv::Mat gradient = calcTextureGradient(img);//��ֵ��0~180֮��

		cv::Mat hsv;
		cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

		for (auto &labelRegion : R)
		{
			labelRegion.second.size = calcSize(img, universe, labelRegion.first);
			labelRegion.second.colourHist = calcColourHist(hsv, universe, labelRegion.first);
			labelRegion.second.textureHist = calcTextureHist(img, gradient, universe, labelRegion.first);
		}

		return R;
	}

	//�ж������Ƿ��ཻ
	inline bool isIntersecting(const Region &a, const Region &b)
	{
		return ((a.rect & b.rect).area() != 0);
	}


	using LabelRegion = std::pair<int, Region>;

	using Neighbour = std::pair<int, int>;

	//ȷ���ָ�����֮��Ĺ�ϵ�����ڻ��߲�����
	std::vector<Neighbour> extractNeighbours(const std::map<int, Region> &R)
	{
		std::vector<Neighbour> neighbours;
		neighbours.reserve(R.size()*(R.size() - 1) / 2);

		for (auto a = R.cbegin(); a != R.cend(); a++)
		{
			auto tmp = a;
			tmp++;

			for (auto b = tmp; b != R.cend(); b++)
			{
				if (isIntersecting(a->second, b->second))
				{
					neighbours.push_back(std::make_pair(std::min(a->first, b->first), std::max(a->first, b->first)));
				}
			}
		}

		return neighbours;
	}

	//�����������ֱ��ͼ�����ں�
	std::vector<float> merge(const std::vector<float> &a, const std::vector<float> &b, int asize, int bsize)
	{
		std::vector<float> newVector;
		newVector.reserve(a.size());

		for (auto ai = a.begin(), bi = b.begin(); ai != a.end(); ai++, bi++)
		{
			newVector.push_back(((*ai) * asize + (*bi) * bsize) / (asize + bsize));
		}

		return newVector;
	};

	//�ں���������
	Region mergeRegions(const Region &r1, const Region &r2)
	{
		assert(r1.colourHist.size() == r2.colourHist.size());
		assert(r1.textureHist.size() == r2.textureHist.size());

		int newSize = r1.size + r2.size;

		std::vector<int> newLabels(r1.labels);
		std::copy(r2.labels.begin(), r2.labels.end(), std::back_inserter(newLabels));

		return Region(r1.rect | r2.rect,
			newSize,
			std::move(merge(r1.colourHist, r2.colourHist, r1.size, r2.size)),
			std::move(merge(r1.textureHist, r2.textureHist, r1.size, r2.size)),
			std::move(newLabels)
		);
	}

	/*
	@param img:������ͼ��
	@param scale��ͼ��ָ��㷨�е�̰��ָ����Խ����ָ������������������٣�������ѡ���������������
	@param sigma����˹������sigma����
	@param minSize���ָ����кܶ�С���򣬵��������ص�ĸ���С��minʱ��ѡ�����������С������ϲ�
	@param smallest��������С���
	@param largest������������
	@param distorted��h/w����С������ֵ
	@return std::vector<cv::Rect> Ԥѡ�߿�
	*/
	std::vector<cv::Rect> selectiveSearch(const cv::Mat &img, double scale = 1.0, double sigma = 0.8, int minSize = 50, int smallest = 1000, int largest = 270000, double distorted = 5.0)
	{
		assert(img.channels() == 3);

		auto universe = generateSegments(img, scale, sigma, minSize);

		int imgSize = img.total();

		auto R = extractRegions(img, universe);

		auto neighbours = extractNeighbours(R);

		std::unordered_map<std::pair<int, int>, double> S;

		for (auto &n : neighbours)
		{
			S[n] = calcSimilarity(R[n.first], R[n.second], imgSize);
		}

		using NeighbourSim = std::pair<std::pair<int, int>, double >;

		while (!S.empty())
		{
			auto cmp = [](const NeighbourSim &a, const NeighbourSim &b) { return a.second < b.second; };

			auto m = std::max_element(S.begin(), S.end(), cmp);

			int i = m->first.first;
			int j = m->first.second;
			auto ij = std::make_pair(i, j);

			int t = R.rbegin()->first + 1;

			R[t] = mergeRegions(R[i], R[j]);

			std::vector<std::pair<int, int>> keyToDelete;

			for (auto &s : S)
			{
				auto key = s.first;

				if ((i == key.first) || (i == key.second) || (j == key.first) || (j == key.second))
				{
					keyToDelete.push_back(key);
				}
			}

			for (auto &key : keyToDelete)
			{
				S.erase(key);

				if (key == ij)
				{
					continue;
				}

				int n = (key.first == i || key.first == j) ? key.second : key.first;
				S[std::make_pair(n, t)] = calcSimilarity(R[n], R[t], imgSize);
			}
		}

		std::vector<cv::Rect> proposals;
		proposals.reserve(R.size());

		for (auto &r : R)
		{
			// exclude same rectangle (with different segments)
			if (std::find(proposals.begin(), proposals.end(), r.second.rect) != proposals.end())
			{
				continue;
			}

			// exclude regions that is smaller/larger than assigned size
			if (r.second.size < smallest || r.second.size > largest)
			{
				continue;
			}

			double w = r.second.rect.width;
			double h = r.second.rect.height;


			if (h / w > distorted&&h / w <= 2.0)
			{
				proposals.push_back(r.second.rect);
			}

		}

		int rows = img.rows;
		int cols = img.cols;
		std::vector<cv::Rect> denseproposals;
		for (auto &rect : proposals) {
			//std::cout << rect << " " << rect.tl() << " " << rect.br() << std::endl;
			float w = rect.width;
			float h = rect.height;
			float delta = w / 5;
			float alpha = h / w;
			for (int i = -10; i <= 20; i++) {
				float delta_w = i*delta;
				float delta_h = 0.5*(alpha*(w + 2 * delta_w) - h);
				float tl_x = rect.tl().x - delta_w;
				float tl_y = rect.tl().y - delta_h;
				float br_x = rect.br().x + delta_w;
				float br_y = rect.br().y + delta_h;
				if (tl_x<0 || tl_y<0 || br_x>cols || br_y>rows || br_x - tl_x <= 0 || br_y - tl_y <= 0) {
					continue;
				}
				denseproposals.push_back(cv::Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y));
			}

		}

		return denseproposals;
	}

	std::vector<cv::Rect> denseSearch(const cv::Mat &img, int mini_box_h=16, int mini_box_w = 8) {
		int rows = img.rows;
		int cols = img.cols;


		std::vector<cv::Rect> denseproposals;
		std::vector<cv::Rect> proposals;

		//int mini_box_h = 16;
		//int mini_box_w = 8;

		for (int i = 0; i < cols / mini_box_w; i++) {
			for (int j = 0; j <rows / mini_box_h; j++) {
				proposals.push_back(cv::Rect(i * mini_box_w, j * mini_box_h, mini_box_w, mini_box_h));
			}
		}
		for (auto &rect : proposals) {
			//std::cout << rect << " " << rect.tl() << " " << rect.br() << std::endl;
			float w = rect.width;
			float h = rect.height;
			float delta = w / 8;
			float alpha = h / w;
			for (int i = 0; i <= 20; i++) {
				float delta_w = i*delta;
				float delta_h = 0.5*(alpha*(w + 2 * delta_w) - h);
				float tl_x = rect.tl().x - delta_w;
				float tl_y = rect.tl().y - delta_h;
				float br_x = rect.br().x + delta_w;
				float br_y = rect.br().y + delta_h;
				if (tl_x<0 || tl_y<0 || br_x>cols || br_y>rows || br_x - tl_x <= 0 || br_y - tl_y <= 0) {
					continue;
				}
				denseproposals.push_back(cv::Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y));
			}

		}
		return denseproposals;

	}

}

#endif // !SS_H
