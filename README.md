##在VS中的设置
1. 添加源代码路径
   项目 -> 属性 -> C/C++ -> 常规 -> 附加包含目录 ：添加 ./src
2. 配置好OpenCV的环境
   属性管理器 -> 右键（添加现有属性表）-> 选择（Opencv3_4_1.props）
   这个属性表是笔者自己在用，需要相对应的修改里面opencv的路径（比如，AdditionalIncludeDirectories、AdditionalLibraryDirectories、AdditionalDependencies）
   
   
##代码中的修改
1. config.h文件中修改RootPath="./path_to_dataset/"
   修改这个路径到数据集的根目录，数据集文件夹包含两个子文件（Detection、Classification）
   
   数据集文件可以在下面的链接处下载：
   
   数据集来源：http://portal.uc3m.es/portal/page/portal/dpto_ing_sistemas_automatica/investigacion/IntelligentSystemsLab/research/InfraredDataset


##代码的运行

main.cpp中包含3个函数，分别为
/*
该函数用于对分类器SVM进行预训练pretrain
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
*/
void preTrainSVM(Config &config, int istrain=1)
 
/*
该函数用于对分类器SVM进行pipeline训练
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
*/
void pipeLineTrain(Config &config, int istrain = 1)
   
 /*
该函数用于对目标检测方法进行整体的测试，计算mAP
@param config 目标检测方法的配置参数变量，是一个结构体，定义在"config.h"中
*/
void pipeLineTest(Config &config, int istrain = 0)

编译成功之后，直接运行就可以了！
enjoy it！
