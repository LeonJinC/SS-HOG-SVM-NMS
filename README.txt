#####��VS�е�����
1. ���Դ����·��
   ��Ŀ -> ���� -> C/C++ -> ���� -> ���Ӱ���Ŀ¼ ����� ./src
2. ���ú�OpenCV�Ļ���
   ���Թ����� -> �Ҽ�������������Ա�-> ѡ��Opencv3_4_1.props��
   ������Ա��Ǳ����Լ����ã���Ҫ���Ӧ���޸�����opencv��·�������磬AdditionalIncludeDirectories��AdditionalLibraryDirectories��AdditionalDependencies��
#####�����е��޸�
1. config.h�ļ����޸�RootPath="./path_to_dataset/"
   �޸����·�������ݼ��ĸ�Ŀ¼�����ݼ��ļ��а����������ļ���Detection��Classification��
#####���������
main.cpp�а���3���������ֱ�Ϊ
/*
�ú������ڶԷ�����SVM����Ԥѵ��pretrain
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void preTrainSVM(Config &config, int istrain=1)
 
/*
�ú������ڶԷ�����SVM����pipelineѵ��
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void pipeLineTrain(Config &config, int istrain = 1)
   
 /*
�ú������ڶ�Ŀ���ⷽ����������Ĳ��ԣ�����mAP
@param config Ŀ���ⷽ�������ò�����������һ���ṹ�壬������"config.h"��
*/
void pipeLineTest(Config &config, int istrain = 0)

����ɹ�֮��ֱ�����оͿ����ˣ�
enjoy it��