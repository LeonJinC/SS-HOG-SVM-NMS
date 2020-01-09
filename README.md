笔者在研究过程中，尝试结合特征提取器HOG、分类器SVM、区域边框提取方法SelectiveSearch和非极大值抑制方法NMS，用于解决红外行人检测问题

# 在VS中的设置
## 1. 添加源代码路径
   项目 -> 属性 -> C/C++ -> 常规 -> 附加包含目录 ：添加 ./src
## 2. 配置好OpenCV的环境
   属性管理器 -> 右键（添加现有属性表）-> 选择（Opencv3_4_1.props）
   这个属性表是笔者自己在用，需要相对应的修改里面opencv的路径（比如，AdditionalIncludeDirectories、AdditionalLibraryDirectories、AdditionalDependencies）
   
   
# 代码中的修改
## 1. config.h文件中修改RootPath="./path_to_dataset/"
   修改这个路径到数据集的根目录，数据集文件夹包含两个子文件（Detection、Classification）
```
数据集文件可以在下面的链接处下载：
https://pan.baidu.com/s/1a81Mx-JWAu0ALcP963kCXA
```
```
数据集来源：http://portal.uc3m.es/portal/page/portal/dpto_ing_sistemas_automatica/investigacion/IntelligentSystemsLab/research/InfraredDataset
```

# 代码的运行

main.cpp中包含3个函数，分别为preTrainSVM、pipeLineTrain、pipeLineTest
```
/*brief 该函数用于对分类器SVM进行预训练pretrain
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为1
*/
void preTrainSVM(Config &config, int istrain=1)
 ```
 
 ```
/*brief 该函数用于对分类器SVM进行pipeline训练
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为1
*/ 
void pipeLineTrain(Config &config, int istrain = 1)
```

```
/*brief 该函数用于对目标检测方法进行整体的测试，计算mAP
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
@param istrain 是否对SVM训练的标识符，默认为0,因为这条函数用于pipeLine测试，所以不用训练
*/  
void pipeLineTest(Config &config, int istrain = 0)
```
编译成功之后，直接运行就可以了！
enjoy it！

# 训练和测试结果展示
蓝色框为Ground Truth,红色框为检测框，红色数字为置信度
训练结果的mAP如下所示，71.4%
![](https://github.com/LeonJinC/SS-HOG-SVM-NMS/blob/master/traindataset_mAP.jpg)

训练数据集的检测结果，如下
![](https://github.com/LeonJinC/SS-HOG-SVM-NMS/blob/master/train_detection.jpg)

测试结果的mAP如下所示，61.8%
![](https://github.com/LeonJinC/SS-HOG-SVM-NMS/blob/master/testdataset_mAP.jpg)

测试数据集的检测结果，如下
![](https://github.com/LeonJinC/SS-HOG-SVM-NMS/blob/master/test_detection.jpg)

# 实现细节
## 训练阶段的SVM训练
所有的预选边框与对应的GroundTruth计算iou值，iou≥0.7为正样本，iou<0.1为负样本，从而构建正负样本数据集用于训练SVM分类器。

## 测试阶段mAP计算
iou阈值≥0.5为tp，iou<0.5为fp
