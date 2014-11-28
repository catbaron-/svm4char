#ifdef _DEBUG
#pragma comment(lib, "opencv_core2410d.lib")
#pragma comment(lib, "opencv_imgproc2410d.lib")
#pragma comment(lib, "opencv_highgui2410d.lib")
#pragma comment(lib, "opencv_video2410d.lib")
#pragma comment(lib, "opencv_objdetect2410d.lib")
#pragma comment(lib, "opencv_legacy2410d.lib")
#pragma comment(lib, "opencv_calib3d2410d.lib")
#pragma comment(lib, "opencv_features2d2410d.lib")
#pragma comment(lib, "opencv_flann2410d.lib")
#pragma comment(lib, "opencv_ml2410d.lib")
#pragma comment(lib, "opencv_gpu2410d.lib")
#pragma comment(lib, "opencv_nonfree2410d.lib")
#pragma comment(lib, "opencv_photo2410d.lib")
#pragma comment(lib, "opencv_stitching2410d.lib")
#pragma comment(lib, "opencv_ts2410d.lib")
#pragma comment(lib, "opencv_videostab2410d.lib")
#pragma comment(lib, "opencv_contrib2410d.lib")
#else
#pragma comment(lib, "opencv_core2410.lib")
#pragma comment(lib, "opencv_imgproc2410.lib")
#pragma comment(lib, "opencv_highgui2410.lib")
#pragma comment(lib, "opencv_video2410.lib")
#pragma comment(lib, "opencv_objdetect2410.lib")
#pragma comment(lib, "opencv_legacy2410.lib")
#pragma comment(lib, "opencv_calib3d2410.lib")
#pragma comment(lib, "opencv_features2d2410.lib")
#pragma comment(lib, "opencv_flann2410.lib")
#pragma comment(lib, "opencv_ml2410.lib")
#pragma comment(lib, "opencv_gpu2410.lib")
#pragma comment(lib, "opencv_nonfree2410.lib")
#pragma comment(lib, "opencv_photo2410.lib")
#pragma comment(lib, "opencv_stitching2410.lib")
#pragma comment(lib, "opencv_ts2410.lib")
#pragma comment(lib, "opencv_videostab2410.lib")
#pragma comment(lib, "opencv_contrib2410.lib")
#endif

#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/video.hpp>
#include <cmath>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <sstream>
#include <string>
using namespace cv;
using namespace std;

const int NUM_POINTS = 16;
const int MAX_DIM_DESCRIPTOR = 256;
const int THRESHOLD_NEIGHBORHOOD = 5.0;
const int SAMPLE_POINT_P = 1;
const int STEP = 16;
const int SAMPLE_POINT_Q = 9;
const int FILE_NAME_LENGTH = 100;
typedef char Cfname[FILE_NAME_LENGTH];
typedef float Dcount[MAX_DIM_DESCRIPTOR];
const int CLASS_NUM = 62;
const int TEST_LOC = CLASS_NUM;
const int HULL_LIST_MAX = 20;
float classLabels[CLASS_NUM];
char classDir[CLASS_NUM + 1][10] = {
	"A0", "B0", "C0", "D0",
	"E0", "F0", "G0", "H0",
	"I0", "J0", "K0", "L0",
	"M0", "N0", "O0", "P0",
	"Q0", "R0", "S0", "T0",
	"U0", "V0", "W0", "X0",
	"Y0", "Z0", "A1", "B1",
	"C1", "D1", "E1", "F1",
	"G1", "H1", "I1", "J1",
	"K1", "L1", "M1", "N1",
	"O1", "P1", "Q1", "R1",
	"S1", "T1", "U1", "V1",
	"W1", "X1", "Y1", "Z1",
	"0", "1", "2", "3",
	"4", "5", "6", "7",
	"8", "9", "TEST"
};
char testDir[] = "TEST";
int getDescriptorFromImage(char* imageFile, float* descriptor)
{
	cv::Mat input_img = cv::imread(imageFile, 1);
	if (!input_img.data) {
		cout << "open image file:" << imageFile << " failed!" << endl;
		return -1;
	}

	cv::Mat resized_img;
	cv::Mat gray_img;

	//// resizing image
	cv::resize(input_img, resized_img, cv::Size(48, 48), 0.0, 0.0, 1);
	//// resizing end


	//// Generate Gray images
	cv::cvtColor(resized_img, gray_img, CV_BGR2GRAY);
	//// Generate Gray end

	////weight points////

	Mat w_map(resized_img.cols, resized_img.rows, CV_32FC1);
	int w_cn = 0;
	for (int i = 0; i < resized_img.cols; ++i)
	{
		for (int j = 0; j < resized_img.rows; ++j)
		{
			w_cn = 0;
			if (resized_img.at<uchar>(i, j) < 50)
			{
				for (int l = i - 1; l < i + 1; ++l)
				{
					if (l < 0)
					{
						continue;
					}
					for (int m = j - 1; m < j + 1; ++m)
					{
						if (m<0){
							continue;
						}
						if (resized_img.at<uchar>(l, m) < 50){
							w_cn++;
						}
					}
				}
			}
			//cout << "w_cn "<<w_cn<< endl;
			if (w_cn == 0)
			{
				w_map.at<float>(i, j) = 0.0;
			}
			else if (w_cn != 0)
			{
				w_map.at<float>(i, j) = w_cn;
			}
		}
	}

	///// Initialize
	int d_ptr, d_ptr02; // pointer for descriptor[ ]
	for (d_ptr = 0; d_ptr < MAX_DIM_DESCRIPTOR; ++d_ptr)
	{
		descriptor[d_ptr] = 0;
	}

	d_ptr = 0;
	d_ptr02 = 0;
	///// Get hit points
	float cn = 0.0;
	float hit_rate = 0.0;

	for (int i = 0; i < STEP; i++)
	{
		for (int j = 0; j< STEP; j++)
		{
			cn = 0;
			for (int l = 0; l < resized_img.cols / STEP; ++l)
			{
				for (int m = 0; m < resized_img.rows / STEP; ++m)
				{
					if (resized_img.at<uchar>(((i*resized_img.cols / STEP) + l), ((j*resized_img.rows / STEP) + m)) <50)
					{
						cn += w_map.at<float>((i*resized_img.cols / STEP) + l, (j*resized_img.rows / STEP) + m);
					}
				}
			}

			hit_rate = cn / ((resized_img.cols / STEP)*(resized_img.rows / STEP));
			descriptor[d_ptr] = hit_rate;
			//cout << "descriptor[ " << d_ptr <<"] = "<<descriptor[d_ptr]<< endl;
			d_ptr++;
		}
	}

	return 0;
}

int _getDescriptorFromImage(char* imageFile, float* descriptor)
{
	cv::Mat input_img = cv::imread(imageFile, 1);
	if (!input_img.data) {
		cout << "open image file:" << imageFile << " failed!" << endl;
		return -1;
	}

	cv::Mat gray_img;
	cv::Mat canny_img;

	//// resizing image
	int width, height;
	width = input_img.cols;
	height = input_img.rows;
	cv::resize(input_img, input_img, cv::Size(width, height), 0.0, 0.0, 1);
	//// resizing end


	//// Generate Canny images
	cv::cvtColor(input_img, gray_img, CV_BGR2GRAY);
	cv::Canny(gray_img, canny_img, 50.0, 200.0, 3, true);
	//// Generate Canny end


	///// Get abstract points of convex hulls
	cv::vector<cv::vector<cv::Point>> contours;
	cv::vector<cv::Vec4i> hierarchy;

	cv::Mat canny_img_working = canny_img.clone();
	cv::findContours(canny_img_working, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	cv::vector<cv::vector<cv::Point>> hull01(contours.size());

	for (int i = 0; i < contours.size(); ++i)
	{
		cv::convexHull(cv::Mat(contours[i]), hull01[i], false);
	}

	cv::Mat convexhull_img = input_img.clone();
	for (int i = 0; i<contours.size(); ++i)
	{
		cv::Scalar color01 = cv::Scalar(0, 0, 255); // color01 = "red"
		cv::drawContours(convexhull_img, hull01, i, color01, 3, 8, cv::vector<cv::Vec4i>(), 0, cv::Point());
	}
	///// Get abstract end


	///// Get all pixels on each convex hull
	int convexhull_list_x[HULL_LIST_MAX][2000];
	int convexhull_list_y[HULL_LIST_MAX][2000];
	int convexhull_list_num[HULL_LIST_MAX];
	int length;
	int xb = 0, yb = 0, xe = 0, ye = 0; // b: begin, e:end
	int current_k;

	for (int i = 0; i<HULL_LIST_MAX; ++i)
	{
		for (int j = 0; j<1000; ++j)
		{
			convexhull_list_x[i][j] = -1;
			convexhull_list_y[i][j] = -1;
			convexhull_list_x[i][j] = -1;
			convexhull_list_y[i][j] = -1;
		}
	}

	// ********************************************************************
	// ****"i < HULL_LIST_MAX" in for statement is add by wtq, 
	// ****to avoid illegal visitation to convexhull_list_x(y) 
	// *******************************************************************
	for (int i = 0; i < contours.size() && i < HULL_LIST_MAX; ++i)
	{
		current_k = 0;

		for (int j = 0; j < hull01[i].size(); ++j)
		{
			if (j != (hull01[i].size() - 1))
			{
				xb = hull01[i].at(j).x;
				yb = hull01[i].at(j).y;
				xe = hull01[i].at(j + 1).x;
				ye = hull01[i].at(j + 1).y;
			}
			else if (j == (hull01[i].size() - 1))
			{
				xb = hull01[i].at(j).x;
				yb = hull01[i].at(j).y;
				xe = hull01[i].at(0).x;
				ye = hull01[i].at(0).y;
			}

			length = (int)std::sqrt((double)(xe - xb)*(xe - xb) + (ye - yb)*(ye - yb));
			for (int k = 0; k<length; k = k + STEP)
			{
				convexhull_list_x[i][current_k + k] = (int)((1.0 - (double)k / length)*xb + (double)k / length*xe);
				convexhull_list_y[i][current_k + k] = (int)((1.0 - (double)k / length)*yb + (double)k / length*ye);
				if (k == (length - 1))
					current_k += (k + 1);
			}
		}

		convexhull_list_num[i] = current_k;
	}


	///// Select the largest convex hull
	int max_i = 0;
	int max_convexhull_list_num = -1;

	for (int i = 0; i<contours.size(); ++i)
	{
		if (i < HULL_LIST_MAX && convexhull_list_num[i] > max_convexhull_list_num)
		{
			max_i = i;
			max_convexhull_list_num = convexhull_list_num[i];
		}
	}

	int interval;

	interval = (int)max_convexhull_list_num / NUM_POINTS;


	for (int p = 0; p < NUM_POINTS; ++p)
	{
		if (p == 0)
			cv::circle(convexhull_img, cv::Point(convexhull_list_x[max_i][p*interval], convexhull_list_y[max_i][p*interval]), 3, cv::Scalar(255, 255, 255), 3, 8, 0);
		else
			cv::circle(convexhull_img, cv::Point(convexhull_list_x[max_i][p*interval], convexhull_list_y[max_i][p*interval]), 3, cv::Scalar(0, 0, 0), 3, 8, 0);
	}


	///////// Main process (Generate descriptors)

	int xi, yi, xj, yj;
	int xq, yq;
	int previous_xq, previous_yq;
	double length_previous_current;
	double length_current_next;
	double cn;
	//double descriptor[MAX_DIM_DESCRIPTOR];
	int d_ptr, d_ptr02; // pointer for descriptor[ ]
	int length_ij;
	double value_a, value_b;
	int count_q;
	int count_edge;

	///// Initialize

	for (d_ptr = 0; d_ptr < MAX_DIM_DESCRIPTOR; ++d_ptr)
	{
		descriptor[d_ptr] = 0;
	}

	d_ptr = 0;
	d_ptr02 = 0;

	///// Calculate the Intervals between Pi's (or Pj's)

	interval = (int)max_convexhull_list_num / NUM_POINTS;

	for (int p = 0; p < NUM_POINTS; ++p)
	{
		for (int q = 0; q < NUM_POINTS; ++q)
		{
			if (p < q)
			{
				///// Choose two points (Pi and Pj) for input_img

				xi = convexhull_list_x[max_i][p*interval];
				yi = convexhull_list_y[max_i][p*interval];
				xj = convexhull_list_x[max_i][q*interval];
				yj = convexhull_list_y[max_i][q*interval];
				///// Calculate CNs for input_img

				length_ij = (int)std::sqrt((double)(xj - xi)*(xj - xi) + (yj - yi)*(yj - yi));

				cn = 1.0;
				count_q = 0;

				previous_xq = xi;
				previous_yq = yi;

				for (int k = 1; k<length_ij; k = k + STEP)
				{
					xq = (int)((1.0 - (float)k / length_ij)*xi + (float)k / length_ij*xj);
					yq = (int)((1.0 - (float)k / length_ij)*yi + (float)k / length_ij*yj);

					///// Check exceptional intersection for input_img

					count_edge = 0;

					if (canny_img.at<uchar>(yq + 1, xq + 1) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq + 1, xq) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq + 1, xq - 1) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq, xq - 1) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq - 1, xq - 1) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq - 1, xq) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq - 1, xq + 1) > 50)
						count_edge++;
					if (canny_img.at<uchar>(yq, xq + 1) > 50)
						count_edge++;

					///// Detect intersection for input_img

					if ((canny_img.at<uchar>(yq, xq) > 50) || count_edge >= 2)
					{
						length_previous_current = std::sqrt((double)(xq - previous_xq)*(xq - previous_xq) + (yq - previous_yq)*(yq - previous_yq));
						length_current_next = std::sqrt((double)(xj - xq)*(xj - xq) + (yj - yq)*(yj - yq));

						if ((length_previous_current > THRESHOLD_NEIGHBORHOOD)
							&& (length_current_next > THRESHOLD_NEIGHBORHOOD))
						{
							count_q++;

							value_b = (double)k / length_ij;
							value_a = 1.0 - value_b;

							if (k % 2 == 0)
							{
								cn *= value_b / value_a;
								//cn *= value_a/value_b;
							}
							else if (k % 2 == 1)
							{
								cn *= value_a / value_b;
							}
						}

						previous_xq = xq;
						previous_yq = yq;
					}
				}

				///// Check exception rules for input_img

				if (((double)count_q / length_ij > 0.3) || (count_q == 0))
				{
					cn = 0.0;
				}

				///// Generate descriptors for input_img

				descriptor[d_ptr] = cn;
				d_ptr += 1;
			}
		}
	}

	///////// Display descriptors

	return 0;
}



int getClassFilesNum(char *dir)
{
	FILE *fp;
	int num = 0;
	char cmd[100] = "dir /B ";
	char line[100];
	strcat_s(cmd, dir);
	strcat_s(cmd, " 2>null");
	fp = _popen(cmd, "r");
	while (fgets(line, 100, fp) != NULL)
		num++;
	_pclose(fp);
	return num;
}
int getTrainFilesNum(int trainCount[])
{

	int trainCountAll = 0;
	for (int i = 0; i < CLASS_NUM; i++)
	{
		trainCount[i] = getClassFilesNum(classDir[i]);
		trainCountAll += trainCount[i];
	}
	return trainCountAll;
}

void readFiles(int classLoc, Cfname *imgFiles, float *labels)
{
	FILE *fp;
	int num = 0;
	char cmd[100] = "dir /B ";
	char line[100] = "";
	strcat_s(cmd, classDir[classLoc]);
	strcat_s(cmd, " 2>null");

	fp = _popen(cmd, "r");
	while (fgets(line, 100, fp) != NULL)
	{
		imgFiles[num][0] = '\0';
		size_t name_len = strlen(line);
		if (line[name_len - 1] == '\r' || line[name_len - 1] == '\n')
		{
			line[name_len - 1] = '\0';
		}
		strcat_s(imgFiles[num], classDir[classLoc]);
		strcat_s(imgFiles[num], "\\");
		strcat_s(imgFiles[num], line);
		if (labels)
			*(labels + num) = classLabels[classLoc];
		num++;
	}

	_pclose(fp);
}



void init(){
	for (int i = 0; i < CLASS_NUM; i++){
		classLabels[i] = (float)i;
	}
}
int readTrainFiles(Cfname *trainFiles, int trainClassCount[], int trainFilesCount, float *labels)
{
	for (int i = 0, cur = 0; i < CLASS_NUM; cur += trainClassCount[i], i++)
	{
		readFiles(i, trainFiles + cur, labels + cur);
	}
	for (int i = 0; i < trainFilesCount; i++){
		cout << labels[i] << ":" << trainFiles[i] << endl;
	}
	return trainFilesCount;
}
int readTestFiles(Cfname *testFiles, int testFilesCount)
{
	readFiles(TEST_LOC, testFiles, NULL);

	for (int i = 0; i < testFilesCount; i++){
		cout << testFiles[i] << endl;
	}
	return testFilesCount;
}
int main(int argc, char **argv)
{
	init();
	int trainCount[CLASS_NUM];
	int classDirs[CLASS_NUM];
	int testFilesCount = getClassFilesNum(testDir);
	int trainFilesCount = getTrainFilesNum(classDirs);

	Cfname *imgFiles = new Cfname[trainFilesCount];
	Cfname *testFiles = new Cfname[testFilesCount];
	float *labels = new float[trainFilesCount];
	int rows_for_train = readTrainFiles(imgFiles, classDirs, trainFilesCount, labels);
	int rows_for_test = readTestFiles(testFiles, testFilesCount);


	///// 训练用descriptor
	Dcount *descriptors = new Dcount[rows_for_train];
	Dcount *descriptorsTest = new Dcount[rows_for_test];
	//float descriptors[rows_for_train][MAX_DIM_DESCRIPTOR];
	//// 测试用用descriptor



	///// 提取训练用图片descriptor

	for (int i = 0; i < rows_for_train; i++) {
		try{

			getDescriptorFromImage(imgFiles[i], descriptors[i]);
		}
		catch (Exception &e)
		{
			cout << "seems something wrong when try to get descriptor from " << imgFiles[i] << endl;
		}
	}


	///// 提取测试用图片descriptor
	for (int i = 0; i < rows_for_test; i++) {
		string fn = testFiles[i];
		getDescriptorFromImage(testFiles[i], descriptorsTest[i]);
	}
	Mat labelsMat(rows_for_train, 1, CV_32FC1, labels);
	//Mat labelsMatTest(2, 1, CV_32FC1, lableTest);

	Mat trainDataMat(rows_for_train, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < rows_for_train; i++) {
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++) {
			trainDataMat.at<float>(i, j) = descriptors[i][j];
		}
	}

	///// 初始化SVM参数
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;

	///// 使用带参数的SVM进行训练
	for (int i = 0; i < rows_for_train; i++)
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++)
			if (descriptors[i][j] != 0)
				;//cout << "des_for_train[" << i << "][" << j << "]" << descriptors[i][j] << endl;
	SVM.train(trainDataMat, labelsMat, Mat(), Mat(), params);
	///// 使用默认参数的SVM进行训练
	//SVM.train(trainDataMat, labelsMat);


	float response; //预测结果
	float right = 0; //预测正确的数量，用于计算正确率

	///// 根据每张图片的descriptor生成测试用数据，并进行测试

	Mat testDesMat(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < rows_for_test; i++) {
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++) {
			///// 生成测试数据
			testDesMat.at<float>(0, j) = descriptorsTest[i][j];
		}
		///// 进行预测
		response = SVM.predict(testDesMat);
		std::cout << " RESULT" << i << "---" << response << "---" << testFiles[i] << endl;
	}


	///// kmeans
	Mat testKmeansData(rows_for_test, MAX_DIM_DESCRIPTOR, CV_32FC1);
	Mat kmeansLabels;
	for (int i = 0; i < rows_for_test; i++)
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++)
		{
		testKmeansData.at<float>(i, j) = descriptorsTest[i][j];
		}
	cv::kmeans(testKmeansData, 2, kmeansLabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 1.0), 10, KMEANS_PP_CENTERS);
	///// result of kmeans:
	for (int i = 0; i < rows_for_test; i++)
		cout << "K-Means--" << testFiles[i] << ":" << kmeansLabels.at<int>(i) << endl;
	system("pause");

	return 0;
}
