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
#include "tinyxml.h"

using namespace cv;
using namespace std;

/*mser regions 合并条件
used by
Region()
*/
#define THR_MERGE_IN 0.2
#define THR_MERGE_AREA 0.2

/*descriptor 维度
used by
_G_getDescriptorFromImage(),
_STROKE_getDescriptorFromImage(),
*/
#define MAX_DIM_DESCRIPTOR 10

/*descriptor中有多种描述子时使用
used by
-_STROKE_getDescriptorFromImage(),
*/
#define MAX_DIM_DESCRIPTOR_G 10
#define MAX_DIM_DESCRIPTOR_STROKE 50
#define MAX_DIM_DESCRIPTOR_COLOR 1

/*SRC大图resize用
used by
-getTextBox(),
*/
#define SRC_WIDTH 800
#define SRC_HEIGHT 600

/*threshout for filters */
#define THR_FILTER_SIZE 250000
#define THR_FILTER_ASPECT 5
#define THR_FILTER_FILLFACTOR 0.8
#define FILTER_EDGE_PERCENT 7

/****************SWITCH***************/
#define DIRTY_IMG "dirtycar.jpg"
#define _DIV_IMG_	//是否重新切割cc。
#define _SESG_			//选择descriptor函数[STROKE/SG]
#define _SVM_		//选择kmeans[_KM_]或者svm[_SVM_]
//#define _PIPE_	//使用管道。开启后会在0TEST和1TEST两个目录下保存分类后的图片
#define _SHOW_HIST_ 0
#define K_MEANS_K 2
#define _DETECT_
#define _FIGHT_
#define _BATTLE_

/*选择descriptor*/
#ifdef _SG_
#define getDescriptorFromImage _superG_getDescriptorFromImage
#define MAX_DIM_DESCRIPTOR 80
#endif

#ifdef _SESG_
#define MAX_DIM_SE 20
#define MAX_DIM_SG 20
#define getDescriptorFromImage _sesg_getDescriptorFromImage
#define MAX_DIM_DESCRIPTOR 40
#endif

#ifdef _SE_
#define getDescriptorFromImage _se_getDescriptorFromImage
#define MAX_DIM_DESCRIPTOR 15
#endif

#ifdef _STROKE_
#define getDescriptorFromImage _stroke_getDescriptorFromImage
#define MAX_DIM_DESCRIPTOR 20
#endif // _g_

/*descriptor选择结束*/
/****************SWITCH***************/


/*33*/
const double threshold_box_size_min01 = 0.0001; // approximately 15 x 15 pixel for 1600 x 1200 image
const double threshold_box_size_max01 = 0.1;  // approximately 150 x 150 pixel for 1600 x 1200 image
const double threshold_box_size_min02 = 0.0001; // approximately 15 x 15 pixel for 1600 x 1200 image
const double threshold_box_size_max02 = 0.1;  // approximately 150 x 150 pixel for 1600 x 1200 image
const double threshold_similarity = 0.4;
const double threshold_length_connected_box_min = 0.005; // approximately 100 pixel for 1600 x 1200 image
const double threshold_length_connected_box_max = 0.5; // approximately 1000 pixel for 1600 x 1200 image
const double threshold_aspect_ratio = 0.2;
const double threshold_relative_distance = 1.0;
const int threshold_count_label = 7;
const int num_bilateral_filtering01 = 12;
const int num_bilateral_filtering02 = 3;
const double threshold_score = 0.5;

const int NUM_POINTS = 16;

const int THRESHOLD_NEIGHBORHOOD = 5.0;
const int SAMPLE_POINT_P = 1;
const int SAMPLE_POINT_Q = 9;
const int FILE_NAME_LENGTH = 100;
/*33*/


typedef char Cfname[FILE_NAME_LENGTH];
typedef float Dcount[MAX_DIM_DESCRIPTOR];
vector<vector<float>> MEANRGB;
/*用于SVM训练*/
const int CLASS_NUM = 2;
const int TEST_LOC = CLASS_NUM;
const int HULL_LIST_MAX = 20;


float classLabels[CLASS_NUM];
char classDir[CLASS_NUM + 1][10] = {
	"Dirty", "Text", "TEST"
};

/*用于预测*/
char testDir[] = "TEST";


class SuperPoint
{
	//location.
	//loc = 1: on the left-top point; 2: on the top edge;
	//3: on top-right point;4:on the right edge;
	//5: on the right-bottom point; 6: on the bottom edge;
	//7: onthe left-bottom point; 8:on the left edge
private:
	int x, y;
	int value;			//
	float l, r, t, b;	//value besides
	int flag;			//flag,0 means has not been visited, 1 means has been visited, -1 means dropt
	int grad;			//gradient. start from up to right, values is 1,2,3,4,5,6,7,8
	float grad_map[3][3]; /*= {
						  { 4, 3, 2 },
						  { 5, 0, 1 },
						  { 6, 7, 8 },
						  };*/
	int stroke = 0;

public:
	SuperPoint findNext(vector<vector<SuperPoint>> &grad_map)
	{
		SuperPoint sp;
		SuperPoint np;
		int find = 0;
		int sub_grad_min = 8;
		for (int x = -1; x < 2; x++)
		{
			for (int y = -1; y < 2; y++)
			{
				//在图片边界的情况
				if ((x == 0 && y == 0) || y + this->y < 0 || x + this->x < 0 || y + this->y >= grad_map.size() || x + this->x >= grad_map[0].size())
					continue;
				sp = grad_map[y + this->y][x + this->x];
				if (sp.grad != 0 && sp.flag == 0)
				{
					find = 1;
					int sub_grad = abs(sp.grad - this->grad);
					if (sub_grad < sub_grad_min)
					{
						sub_grad_min = sub_grad;
						np = sp;
					}
				}
			}
		}
		if (find)
		{
			grad_map[np.y][np.x].flag = 1;
			return np;
		}
		//找不到点，返回空点
		SuperPoint sp0;
		return sp0;
	}
	SuperPoint()
	{
		x = y = value = l = r = t = b = flag = grad = -1;
	}
	SuperPoint(int row, int col, Mat img)
	{

		grad_map[0][0] = 4;
		grad_map[0][1] = 3;
		grad_map[0][2] = 2;
		grad_map[1][0] = 5;
		grad_map[1][1] = 0;
		grad_map[1][2] = 1;
		grad_map[2][0] = 6;
		grad_map[2][1] = 7;
		grad_map[2][2] = 8;
		x = col;
		y = row;
		flag = 0;
		value = img.at<uchar>(y, x);
		if (value > 200)
		{
			l = (0 == x) ? value : (int)img.at<uchar>(y, x - 1);
			t = (0 == y) ? value : (int)img.at<uchar>(y - 1, x);
			r = (img.cols == x + 1) ? value : (int)img.at<uchar>(y, x + 1);
			b = (img.rows == y + 1) ? value : (int)img.at<uchar>(y + 1, x);
			int g_t = t - value;
			int g_r = r - value;
			int g_b = b - value;
			int g_l = l - value;
			int lr = (g_l - g_r == 0) ? 0 : (g_l - g_r) / abs(g_l - g_r);
			int tb = (g_t - g_b == 0) ? 0 : (g_t - g_b) / abs(g_t - g_b);
			grad = grad_map[lr + 1][tb + 1];
		}
		else
		{
			grad = 0;
		}
		if (grad == 0)
		{
			flag = -1;
		}
	}

	int getX()
	{
		return x;
	}
	int getY()
	{
		return y;
	}
	void setPoint(int ix, int iy)
	{
		x = ix;
		y = iy;
	}
	void setX(int ix)
	{
		x = ix;
	}
	void setY(int iy)
	{
		y = iy;
	}

	int getValue()
	{
		return value;
	}
	void setValue(int v)
	{
		value = v;
	}

	int getFlag()
	{
		return flag;
	}
	void setFlag(int f)
	{
		flag = f;
	}

	int getStroke()
	{
		return stroke;
	}
	void setStroke(int s)
	{
		stroke = s;
	}

	float getValueOfLeft()
	{
		return l;
	}
	void setValueOfLeft(int left)
	{
		l = left;
	}

	float getValueOfRight()
	{
		return r;
	}
	void SetValueOfRight(int right)
	{
		r = right;
	}

	float getValueOfUp()
	{
		return t;
	}
	void setValueOfUp(int up)
	{
		t = up;
	}

	float getValueOfDonw()
	{
		return b;
	}
	void SetValueOfDonw(int down)
	{
		b = down;
	}

	int getGradient()
	{
		return grad;
	}
	void setGradient(int gradient)
	{
		grad = gradient;
	}

};

vector<Point> rectRegion(vector<Point> region)
{
	//返回region的lt,rb点
	Point lt = region[0];
	Point rb = region[0];
	vector<Point> re;
	re.push_back(lt);
	re.push_back(rb);

	lt.x = 10000;
	lt.y = 10000;
	rb.x = 0;
	rb.y = 0;

	Point cp;
	for (int i = 0; i < region.size(); i++)
	{
		cp = region[i];
		lt.x = lt.x < cp.x ? lt.x : cp.x;
		lt.y = lt.y < cp.y ? lt.y : cp.y;
		rb.x = rb.x > cp.x ? rb.x : cp.x;
		rb.y = rb.y > cp.y ? rb.y : cp.y;
	}
	re[0].x = lt.x;
	re[0].y = lt.y;
	re[1].x = rb.x;
	re[1].y = rb.y;
	return re;
}


class Region
{
private:
	Point tl, br, ct;		//区域左上右下顶点和中心
	vector<Point> points;	//区域包含的点集
	int parent;			// 此区域被合并到的区域下标
	int seq;			// 在Region集合中的下标
	float area = 0;		// 对应rect面积
	float ff = 0;		// fillfactor
	float height;		// 对应rect高
	float width;		// 对应rect 宽
	float is_hidden = 0;//是否显示（大于0表示被过滤或被合并，不显示）
	float is_char = 0;	//区域内是否为文字
	int label = 0;
	vector<float> m_color;	//mean of color
	vector<float> v_color;	//variance of color
	vector<Mat> rgb;
	vector<Mat> img;
	Rect rect;


	void setCent()
	{
		ct.x = (tl.x + br.x) / 2;
		ct.y = (tl.y + br.y) / 2;
	}
public:
	void updateTLBR()
	{
		//返回region的lt,rb点
		vector<Point> re = rectRegion(points);
		Point lt = re[0];
		Point rb = re[1];
		setTL(lt.x, lt.y);
		setBR(rb.x, rb.y);
	}
	Region()
	{
		;
	}
	Region(vector<Point> region, int seq_in)
	{
		//用点集初始化
		//初始化点集
		points = region;
		//初始化顶点
		updateTLBR();
		//parent默认seq表示没有被合并
		parent = seq_in;
		this->seq = seq_in;
	}

	Rect getRect()
	{
		return rect;
	}

	void addPoint(Point p)
	{
		points.push_back(p);
		//updateTLBR();
	}

	vector<Point> getPoints()
	{
		return points;
	}
	int getParent()
	{
		return parent;
	}
	void setParent(int par)
	{
		parent = par;
	}

	int getSeq()
	{
		return seq;
	}
	void setSeq(int se)
	{
		seq = se;
	}

	float getArea()
	{
		return area;
	}
	void setArea(int are)
	{
		area = are;
	}

	int getLabel()
	{
		return label;
	}
	void setLabel(int l)
	{
		label = l;
	}

	int isHidden()
	{
		return is_hidden;
	}
	void hide()
	{
		is_hidden = 1;
	}
	void appear()
	{
		is_hidden = 0;
	}

	int isChar(){
		return is_char;
	}
	void setChar(int ch)
	{
		is_char = ch;
	}
	void setNotChar()
	{
		is_char = 0;
	}


	Point getTL()
	{
		return this->tl;
	}
	Point getBR()
	{
		return this->br;
	}
	void setTL(int x, int y)
	{
		tl.x = x;
		tl.y = y;
		width = br.x - tl.x;
		height = br.y - tl.y;
		setCent();
		area = height * width;
		rect = Rect(tl, br);

	}
	void setBR(int x, int y)
	{
		br.x = x;
		br.y = y;
		width = br.x - tl.x;
		height = br.y - tl.y;
		setCent();
		area = height * width;
		rect = Rect(tl, br);

	}
	int getWidth()
	{
		return this->width;
	}
	int getHeight()
	{
		return this->height;
	}
	Point getCent()
	{
		return this->ct;
	}

	void setWidth(int w)
	{
		setBR(br.x + w, br.y);
	}
	void setHeight(int h)
	{
		setBR(br.x, br.y + h);
	}

	//看一个Region在另一个Region中的部分有多少
	float inRect(Rect rect)
	{
		float sum = area;
		float in = 0;
		for (int x = tl.x; x < br.x; x++)
		{
			for (int y = tl.y; y< br.y; y++)
			{
				Point tl = Point(rect.x, rect.y);
				Point br = Point(rect.x + rect.width, rect.y + rect.height);
				if (
					x > tl.x
					&& y > tl.y
					&& x < br.x
					&& y < br.y
					)
					in++;
			}
		}
		in = in / sum;
		//cout << "inRegion:" << in << endl;
		return in;
	}

	//根据宽长比例过滤
	void filterWH()
	{
		float width = this->width;
		float height = this->height;
		if ((float)width / height > 1.5)
			this->is_hidden = 1;
	}
};

int merge2Text(Region &r1, Region &r2, int &label)
{
	//label
	int l1 = r1.getLabel();
	int l2 = r2.getLabel();
	int l = 0;

	if (0 == l1 && 0 == l2)
		l = label + 1;
	if (l2 != 0)
		l = l2;
	r1.setLabel(l);
	r2.setLabel(l);
	label = 1>label ? l : label;
	//r1.is_hidden = r2.is_hidden = 1;
	return 1;
}
int isTextsNear(Region r1, Region r2)
{
	//center
	Point c1, c2;
	c1 = r1.getCent();
	c2 = r2.getCent();
	//width
	int w1 = r1.getWidth();
	int w2 = r2.getWidth();
	int wmax = w1 > w2 ? w1 : w2;
	int wmin = w1 < w2 ? w1 : w2;
	//height
	int h1 = r1.getHeight();
	int h2 = r2.getHeight();
	int hmin = h1 < h2 ? h1 : h2;
	//center水平距
	int dh = abs(c1.x - c2.x);
	//center竖直距
	int dv = abs(c1.y - c2.y);
	//size差
	int ds = abs(h1 - h2);

	if (dh <= 2 * wmin && dv <= 0.5*wmax && ds <= 1 * hmin)
	{
		return 1;
	}
	return  0;
}

Region mergeTextLabels(vector<Region> rs, Mat img)
{
	Region r = rs[0];
	rectangle(img, r.getRect(), 255, 2, 8, 0);
	for (int i = 1; i < rs.size(); i++)
	{
		Scalar color = (255, 0, 0);
		rectangle(img, rs[i].getRect(), 255, 2, 8, 0);

		rs[i].isHidden();
		Region rt = rs[i];
		for (int p = 0; p < rs[i].getPoints().size(); p++)
			r.addPoint(rs[i].getPoints()[p]);
		r.updateTLBR();
		if (r.getTL().x > rs[i].getTL().x)
			r.setTL(rs[i].getTL().x, r.getTL().y);
		if (r.getTL().y > rs[i].getTL().y)
			r.setTL(r.getTL().x, rs[i].getTL().y);
		if (r.getBR().x < rs[i].getBR().x)
			r.setBR(rs[i].getBR().x, r.getBR().y);
		if (r.getBR().y < rs[i].getBR().y)
			r.setBR(r.getBR().x, rs[i].getBR().y);
	}
	//imshow("label", img);
	//waitKey(0);
	return r;
}
//判断每一个region在同一水平线上是否有临近的其他region，没有则删除
void removeSingleRegion(vector<Region> &r)
{
	for (int i = 0; i < r.size(); i++)
	{
		if (r[i].isHidden())
			continue;
		int mindis = -1;//同一水平线的最小距离
		for (int j = 0; j < r.size(); j++)
		{
			if (i == j || r[j].isHidden())
				continue;
			Point ci, cj;//centdroid
			ci = r[i].getCent();
			cj = r[j].getCent();

			int hi = r[i].getHeight();
			int hj = r[j].getHeight();
			float dh = abs(hj - hi);
			float dsize = abs(r[i].getHeight()*r[i].getWidth() - r[j].getHeight()*r[j].getWidth());
			int sizei = r[i].getHeight()*r[i].getWidth();
			if (dh / hi >= 0.3)
				continue;
			if (dsize / sizei >= 0.8)
				continue;
			//if (ci.x == 171 && ci.y == 68 && cj.x == 170 && cj.y == 68)
			//	system("pause");
			int wij = abs(ci.x - cj.x);
			int hij = abs(ci.y - cj.y);
			Region ri = r[i];
			//判断是否在同一水平线上
			if (hij <= r[i].getHeight())
			{
				//在同一水平线上(垂直距离足够小),找最小水平距离
				if (mindis < 0 || mindis >= wij)
					mindis = wij;
			}
		}
		if (mindis < 0 || mindis > 2 * r[i].getWidth())
		{

			r[i].hide();//mindis<0说明没有找到同一水平，>4倍宽度说明最小距离太远
		}
	}
}
class ImageResult
{
public:
	string imgName;
	vector<Rect> rects;
	int width;
	int height;
};
int mergeTextRegions(vector<Region> &regions, Mat img)
{
	int size = regions.size();
	int maxlabel = 0;
	for (int i = 0; i < size; i++)
	{
		/*已经隐藏的(过滤掉或被合并)不参与合并*/
		if (regions[i].isHidden())
			continue;
		int nolabel = -1;
		for (int j = 0; j < i; j++)
		{
			/*已经隐藏的(过滤掉或被合并)不参与合并*/
			if (regions[j].isHidden())
				continue;
			if (isTextsNear(regions[i], regions[j]) && regions[j].getLabel() != 0)
			{
				//找到有标签的region直接合并
				nolabel = -1;
				merge2Text(regions[i], regions[j], maxlabel);
				break;
			}
			if (isTextsNear(regions[i], regions[j]) && regions[i].getLabel() == 0)
			{
				//找到的是没有标签的region，记录
				nolabel = j;
			}
		}
		if (nolabel >= 0)
			merge2Text(regions[i], regions[nolabel], maxlabel);
		if (regions[i].getLabel() == 0)
		{
			maxlabel++;
			regions[i].setLabel(maxlabel);
		}
	}
	vector<vector<Region>> texts;
	for (int i = 1; i < maxlabel + 1; i++)
	{
		vector<Region> text;
		for (int j = 0; j < regions.size(); j++)
		{
			if (regions[j].getLabel() == i)
			{
				cout << "labels: " << i << endl;
				text.push_back(regions[j]);

			}
		}

		if (text.size() > 0)
			texts.push_back(text);
	}
	for (int i = 0; i < texts.size(); i++)
	{
		Mat im = img.clone();
		Region r = mergeTextLabels(texts[i], im);
		r.appear();

		regions.push_back(r);
	}
	return 0;
}

/*合并区域集合。可选参数is_char大于零时，只有is_char属性相等的Region才会被合并*/
int mergeRegions(vector<Region> &regions, int is_char = -1)
{
	int size = regions.size();
	cout << "start 1" << endl;
	for (int i = 0; i < size; i++)
	{
		/*已经隐藏的(过滤掉或被合并)不参与合并*/
		if (regions[i].isHidden())
			continue;
		cout << "start 2" << endl;
		for (int j = 0; j < i; j++)
		{
			/*已经隐藏的(过滤掉或被合并)不参与合并*/
			if (regions[j].isHidden())
				continue;
			float in_rect = regions[i].inRect(regions[j].getRect());
			float in_area = (float)regions[i].getArea() / regions[j].getArea();
			/*merge*/
			if (i != j && in_rect>THR_MERGE_IN && in_area > THR_MERGE_AREA)
			{
				if (is_char > 0 && regions[i].isChar() != regions[j].isChar())
					continue;
				int parent = 0;
				//合并到j中，i不再显示
				regions[i].hide();
				Region rj = regions[j]; Region ri = regions[i];
				regions[i].setParent(regions[j].getParent());
				parent = regions[j].getParent();
				vector<Point> rpsi = regions[i].getPoints();
				cout << "start 3" << endl;
				for (int k = 0; k < rpsi.size(); k++)
				{
					regions[parent].addPoint(rpsi[k]);
				}
				cout << "end 3" << endl;
				regions[parent].updateTLBR();
				Region rg = Region(regions[parent].getPoints(), regions[parent].getSeq());
			}
		}
		cout << "end 2" << endl;
	}
	cout << "end1" << endl;
	return 0;
}
/*返回大值*/
float biggerThanbigger(float f1, float f2)
{
	return f1 > f2 ? f1 : f2;
}
/////排序/////
void sortArry(float a[], int len)
{
	int flag = 1;
	while (flag)
	{
		flag = 0;
		for (int i = 0; i < len - 1; i++)
		{
			if (a[i] < a[i + 1])
			{
				float temp = a[i];
				a[i] = a[i + 1];
				a[i + 1] = temp;
				flag = 1;
			}
		}
	}
}

/*求矩阵平均值/方差*/
float avgMatF(Mat input)
{
	float mean = 0;
	int count = 0;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (input.at<float>(i, j) != 0)
			{
				mean += input.at<float>(i, j);
				count++;
			}
		}
	}
	if (count != 0)
		mean = (float)mean / count;
	return mean;
}
float avgMatI(Mat input)
{
	float mean = 0;
	int count = 0;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			if (input.at<uchar>(i, j) != 0)
			{
				mean += input.at<uchar>(i, j);
				count++;
			}
		}
	}
	if (count != 0)
		mean = mean / count;
	return mean;
}
float varMatF(Mat input)
{
	float mean = avgMatF(input);
	float var = 0;
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			int v = input.at<float>(i, j);
			var += (v - mean)*(v - mean);
		}
	}
	return var;
}
/*求stroke width*/
Mat stroke(Mat input, int flag)
{
	Mat bw8u;
	cvtColor(input, bw8u, CV_RGB2GRAY);
	Mat bw32f, swt32f, kernel;
	double min, max, mean1, mean2;
	int strokeRadius;
	Point minp, maxp;
	float var1 = 0, var2 = 0;

	for (int i = 0; i < bw8u.rows; i++)
	{
		for (int j = 0; j < bw8u.cols; j++)
		{
			if (flag == 1)
			{
				if (bw8u.at<uchar>(i, j) > 50)
				{
					bw8u.at<uchar>(i, j) = 1;
				}
				else
				{
					bw8u.at<uchar>(i, j) = 0;
				}
			}
			else
			{
				if (bw8u.at<uchar>(i, j) > 50)
				{
					bw8u.at<uchar>(i, j) = 0;
				}
				else
				{
					bw8u.at<uchar>(i, j) = 1;
				}
			}

		}
	}

	bw8u.convertTo(bw32f, CV_32F);  // format conversion for multiplication

	distanceTransform(bw8u, swt32f, CV_DIST_L2, 5); // distance transform

	minMaxLoc(swt32f, NULL, &max, NULL, &maxp);  // find max
	//对于全黑全白的图，max会是一个很大的数？为了避免这种情况，对max进行判断，
	//如果max大于原图尺寸，则重新对swt32f赋值，每个点取值为点x,y坐标中的大者
	if (max > biggerThanbigger(swt32f.rows, swt32f.cols))
	{
		for (int i = 0; i < swt32f.rows; i++)
		{
			for (int j = 0; j < swt32f.cols; j++)
			{
				swt32f.at<float>(i, j) = biggerThanbigger(i, j);
			}
		}
		minMaxLoc(swt32f, NULL, &max, NULL, &maxp);  // find max
	}
	circle(bw8u, maxp, 10, Scalar(255, 255, 255));
	strokeRadius = (int)ceil(max);  // half the max stroke width
	kernel = getStructuringElement(MORPH_RECT, Size(3, 3)); // 3x3 kernel used to select 8-connected neighbors
	for (int j = 0; j < strokeRadius; j++)
	{
		dilate(swt32f, swt32f, kernel); // assign the max in 3x3 neighborhood to each center pixel
		swt32f = swt32f.mul(bw32f); // apply mask to restore original shape and to avoid unnecessary max propogation
	}
	waitKey();
	return swt32f;
}
Mat skel(Mat input)
{

	Mat input1 = stroke(input, 1);
	Mat input2 = stroke(input, 0);
	float var1 = varMatF(input1);
	float var2 = varMatF(input2);
	if (var1 > var2)
		return input2;
	else
		return input1;
}
/*根据文件名获得矩形顶点坐标*/
Rect readRectFromImageName(char *fname)
{
	Mat img = imread(fname);
	int pos[4];
	char fname_cp[50];
	strcpy_s(fname_cp, fname);
	char * split = "_";
	char *p;
	char *token;
	token = strtok_s(fname_cp, split, &p);
	for (int i = 0; i < 4; i++)
	{
		token = strtok_s(NULL, split, &p);
		pos[i] = atoi(token);
	}
	Point tl, br;
	vector<Point> points;
	tl.x = pos[0];
	tl.y = pos[1];
	br.x = pos[2];
	br.y = pos[3];
	Rect rect(tl, br);
	return rect;
}

/*绘制直方图，默认最高200，单个直方宽度5, 默认flat=0时不自动显示*/
Mat drawHist(Mat input, int flag = 0, int width = 5, int height = 200)
{
	int winHeight = (int)(height / 0.8);
	int winWidth = width*(input.cols);
	Mat hist = Mat::zeros(winHeight, winWidth, CV_8UC3);
	double max;
	minMaxLoc(input, NULL, &max);
	int count = input.cols;
	float max_height = 0.8 * winHeight;
	if (!max)
		max = 1;
	Scalar color;
	for (int i = 0; i < count; i++)
	{
		float height = input.at<float>(i) / max * max_height;
		Point tl;
		Point br;
		tl.x = i*width;
		br.x = tl.x + width;
		tl.y = winHeight - 1;
		br.y = tl.y - height;
		if (i % 2)
			color = Scalar(200, 200, 200);
		else
			color = Scalar(100, 100, 100);
		rectangle(hist, tl, br, color, -1, 8);
	}
	if (flag != 0)
	{
		imshow("hist", hist);
		waitKey();
	}
	return hist;
}
void runEdge(SuperPoint &sp, float *hist, vector<vector<SuperPoint>> &grad_map, int len)
{
	float bin = len / MAX_DIM_DESCRIPTOR;
	if (bin == 0)
		bin = len;
	sp.setFlag(1);
	int count5 = 1;
	SuperPoint np, pp;
	SuperPoint cp = sp;
	np = pp = cp;
	int find = 1;
	while (find)
	{
		cp.setFlag(1);
		find = 0;
		if (sp.getGradient() == 0)
			break;
		np = cp.findNext(grad_map);
		if (np.getX() < 0)
		{
			int index = count5 / bin;
			hist[index]++;
			count5 = 1;
		}
		if (np.getX() >= 0)
		{
			find = 1;
			int sub_grad = abs(np.getGradient() - cp.getGradient());
			if (sub_grad <= 1 || sub_grad >= 7)
				count5++;
			else
			{
				int index = count5 / bin;
				hist[index]++;
				count5 = 1;
			}
			cp = np;
		}
	}
}
void strokeEdge(float *hist, vector<vector<SuperPoint>> &grad_map, Mat bw, int len)
{
	vector<int> ss; //所有的sw，用来计算variance
	float ssum = 0.0;		//sw的和，同上

	vector<Point> skel;

	int dx[9] = { 0, 0, 1, 1, 1, 0, -1, -1, -1 };
	int dy[9] = { 0, -1, -1, 0, 1, 1, 1, 0, -1 };
	int is_visited = 1;
	int max_stroke = 0;
	for (int i = 0; i < grad_map.size(); i++)
	{
		for (int j = 0; j < grad_map[i].size(); j++)
		{
			SuperPoint sp = grad_map[i][j];
			if (sp.getFlag() == is_visited || sp.getGradient() <= 0)
				continue;
			int ddx = dx[sp.getGradient()];
			int ddy = dy[sp.getGradient()];
			int x = sp.getX();
			int y = sp.getY();
			int stroke = 0;
			while (1)
			{
				//find the opposite point
				x = x + ddx;
				y = y + ddy;
				if (x < 0 || y < 0 || x >= grad_map[i].size() || y >= grad_map.size())
					break;
				if (grad_map[y][x].getGradient() > 0)
					break;
				stroke++;
			}
			int skel_x = sp.getX() + ddx *(stroke / 2);
			int skel_y = sp.getY() + ddy *(stroke / 2);
			Point skel_p = Point(skel_x, skel_y);
			skel.push_back(skel_p);
			if (stroke > max_stroke)
				max_stroke = stroke;
			grad_map[i][j].setStroke(stroke);
		}
	}

	//bw组偶distanceTransform
	Mat dst;
	distanceTransform(bw, dst, CV_DIST_L2, 5); // distance transform
	Mat stroke_map = cv::Mat::zeros(bw.rows, bw.cols, CV_8UC1);
	for (int i = 0; i < skel.size(); i++)
	{
		int stroke_row = skel[i].y;
		int stroke_col = skel[i].x;

		stroke_map.at<uchar>(stroke_row, stroke_col) = (float)dst.at<float>(stroke_row, stroke_col);
	}
	/*for (int i = 0; i < dst.rows; i++)
	{
	for (int j = 0; j < dst.cols; j++)
	{
	printf("%d ", (int)dst.at<float>(i, j));
	}
	cout << endl;
	}

	for (int i = 0; i < stroke_map.rows; i++)
	{
	for (int j = 0; j < stroke_map.cols; j++)
	{
	printf("%d ", stroke_map.at<uchar>(i, j));
	}
	cout << endl;
	}

	*/
	//imshow("bw", bw);

	// Sekl map
	Mat skel_map = Mat::zeros(bw.rows, bw.cols, CV_8UC3);
	for (int i = 0; i<skel_map.rows; i++)
	{
		for (int j = 0; j<skel_map.cols; j = j++)
		{
			if (stroke_map.at<uchar>(i, j) > 0)
			{
				int stro = stroke_map.at<uchar>(i, j);
				int green = 255 - 10 * stro;
				int red = 255;
				if (green < 0)
					green = 0;
				skel_map.at<cv::Vec3b>(i, j)[0] = 0;
				skel_map.at<cv::Vec3b>(i, j)[1] = green;
				skel_map.at<cv::Vec3b>(i, j)[2] = red;
			}
		}
	}
	//imshow("skel_map", skel_map);
	//waitKey(0);
	int bin = max_stroke / MAX_DIM_DESCRIPTOR;
	if (bin == 0)
		bin = 1;


	//Stroke Edge map
	Mat edge_map = Mat::zeros(bw.rows, bw.cols, CV_8UC3);
	for (int i = 0; i < grad_map.size(); i++)
	{
		for (int j = 0; j < grad_map[i].size(); j++)
		{
			SuperPoint sp = grad_map[i][j];
			if (sp.getGradient() <= 0)
				continue;
			int s = grad_map[i][j].getStroke();
			if (s > 0)
			{
				int green = 255 - 10 * s;
				int red = 255;
				if (green < 0)
					green = 0;
				edge_map.at<cv::Vec3b>(i, j)[0] = 0;
				edge_map.at<cv::Vec3b>(i, j)[1] = green;
				edge_map.at<cv::Vec3b>(i, j)[2] = red;
			}
		}
	}
	//imshow("edge_map", edge_map);

	for (int i = 0; i < stroke_map.rows; i++)
	{
		for (int j = 0; j < stroke_map.cols; j++)
		{
			int s = (int)stroke_map.at<uchar>(i, j);
			ss.push_back(s);
			ssum = ssum + s;
			int h = s / bin;
			hist[h]++;
		}
	}

	//计算sw的variance
	float mean = ssum / ss.size();
	float v = 0.0;	//variance
	for (int i = 0; i < ss.size(); i++)
	{
		v = v + pow(ss[i] - mean, 2);
	}
	v = v / ss.size();
	cout << "VVVVVVVVVVVVVVVVVVVVVVVVVarianceis: " << v;
}
void normalizeHist(float *hist, int len)
{
	float sum = 0;
	for (int i = 0; i < len; i++)
	{
		sum = sum + hist[i];
	}
	if (sum == 0)
		sum = 1;
	for (int i = 0; i < len; i++)
	{
		hist[i] = hist[i] / sum;
	}
}
int filterAspect(Region region, float th_aspect)
{
	int w = region.getWidth();
	int h = region.getHeight();
	float fa = (float)h / w;
	if (fa > th_aspect || 1 / fa > th_aspect)
		return 0;
	else
		return 1;
}
int filterFillFactor(Region region, float th_ff)
{
	int w = region.getWidth();
	int h = region.getHeight();
	int square = w*h;
	int fill = region.getPoints().size();
	float ff = (float)fill / square;
	if (ff > th_ff) {
		return 0;
	}
	else
		return 1;
}
int filterIntensity(Region region, Mat img, float th_ins)
{
	Rect rect = region.getRect();
	Mat input_img = img(rect);
	float avg_intensity = 0;
	for (int i = 0; i < input_img.rows; i++)
	{
		for (int j = 0; j < input_img.cols; j++)
		{
			avg_intensity += input_img.at<uchar>(i, j);
		}
	}
	avg_intensity = avg_intensity / (input_img.rows * input_img.cols);
	float intensity_c = 0.0;
	for (int i = 0; i < input_img.rows; i++)
	{
		for (int j = 0; j < input_img.cols; j++)
		{
			intensity_c += abs(input_img.at<uchar>(i, j) - avg_intensity);
		}
	}
	float fi = (float)intensity_c / region.getArea();
	if (fi > th_ins) {
		return 0;
	}
	else
		return 1;
}
int filterSize(Region region, float th_size)
{
	int size = region.getArea();
	if (size > th_size) {
		return 0;
	}
	else
		return 1;
}
void getCCForTextBox(Mat &src_img, Mat channel_img, char *imageFile, int channel, Mat img)
{
	MSER ms(5, 10, 15000);/**************************************/
	vector<vector<Point>> channel_regions;
	ms(channel_img, channel_regions, Mat());

	Mat copy_channle0 = channel_img.clone();
	Mat copy_channle1 = channel_img.clone();
	Mat copy_channle2 = channel_img.clone();
	Mat copy_channle3 = channel_img.clone();
	Mat copy_channle4 = channel_img.clone();
	Scalar color;
	// 计算regions之间的diff,然后merge
	vector<Region> rs;
	for (int i = 0; i < channel_regions.size(); i++)
	{
		//生成Region对象
		Region reg( channel_regions[i], i);
		float th_size = THR_FILTER_SIZE;
		float th_aspect = THR_FILTER_ASPECT;
		float th_ff = THR_FILTER_FILLFACTOR;
		reg.filterWH();
		if (!(filterSize(reg, th_size) && filterAspect(reg, th_aspect) && filterFillFactor(reg, th_ff)))
			reg.isHidden();
		rs.push_back(reg);

		//画出区域
		if (reg.isHidden() == 0)
			rectangle(copy_channle0, reg.getRect(), color, 2, 8, 0);
		color = Scalar(255, 0, 0);
	}
	//合并区域
	cout << "merge start!" << endl;


	mergeRegions(rs);
	//mergeTextRegions(rs, img);
	//mergeRegions(rs);
	for (int i = 0; i < rs.size(); i++)
	{
		if (rs[i].isHidden() == 0)
			rectangle(copy_channle4, rs[i].getRect(), color, 2, 8, 0);
	}
	//imshow("before remove", copy_channle4);
	removeSingleRegion(rs);
	for (int i = 0; i < rs.size(); i++)
	{
		if (rs[i].isHidden() == 0)
			rectangle(copy_channle3, rs[i].getRect(), color, 2, 8, 0);
	}
	//imshow("after remove", copy_channle3);
	cout << "merge end!" << endl;

	//draw merged region
	for (int i = 0; i < rs.size(); i++)
	{
		if (rs[i].isHidden() == 0)
			color = Scalar(255, 255, 255);
		else
			color = Scalar(0, 0, 0);
		rectangle(copy_channle1, rs[i].getRect(), color, 2, 8, 0);

		if (!rs[i].isHidden())
		{

			rectangle(copy_channle2, rs[i].getRect(), 255, 2, 8, 0);

			Region r = rs[i];
			Mat ot = Mat::zeros(r.getHeight() + 11, r.getWidth() + 11, CV_8UC1);	//用于找到区域后二值化
			cout << "start here!" << endl;
			vector<Point> rps = r.getPoints();
			for (int j = 0; j < rps.size(); j++)
			{
				Point p = rps[j];
				int x = p.x - r.getTL().x;
				int y = p.y - r.getTL().y;
				ot.at<uchar>(y + 5, x + 5) = 255;
			}
			cout << "end here" << endl;
			char fname[70];
			// save cc
			sprintf_s(fname, "TEST/%d_%d_%d_%d_%d_%d_%s", i, r.getTL().x, r.getTL().y, r.getBR().x, r.getBR().y, channel, ".jpg");
			imwrite(fname, ot);

			// draw cc to src img
			rectangle(src_img, rs[i].getRect(), 255, 2, 8, 0);
		}
	}
	//imshow("after filter", copy_channle0);
	//imshow("before merge", copy_channle1);
	//imshow("after merge", copy_channle2);
	waitKey(0);

}
int getTextBox(char* imageFile)
{
	cout << "Deviding dirty image into boxes" << endl;
	Mat src = imread(imageFile, 1);
	resize(src, src, cv::Size(SRC_WIDTH, SRC_HEIGHT));
	Mat input_img = src.clone();
	Mat bw = Mat::zeros(SRC_HEIGHT, SRC_WIDTH, CV_8UC1);	//用于找到区域后二值化

	if (!input_img.data) {
		cout << "open image file:" << imageFile << " failed!" << endl;
		return -1;
	}

	////bilateral filter
	cv::Mat bilateral_img01; // In cv::bilateralFilter(), output_array must be different frome input_array. So we prepared bilateral_img0101 for input_array.
	cv::Mat bilateral_img02; // We prepared bilateral_img0102 for output_array.

	bilateral_img01 = input_img.clone();

	for (int i = 0; i < num_bilateral_filtering01; ++i)
	{
		cv::bilateralFilter(bilateral_img01, bilateral_img02, 5, 40, 20, 4);
		bilateral_img01 = bilateral_img02.clone();
		bilateral_img02 = cv::Mat::zeros(input_img.cols, input_img.rows, CV_8UC3);
	}

	// 分离通道，H,L,S,GRAY
	vector<Mat> img_channels, v_hls;
	Mat hls, gray, rgb;
	cvtColor(bilateral_img01, gray, CV_BGR2GRAY);
	cvtColor(bilateral_img01, hls, CV_BGR2HLS);
	rgb = bilateral_img01.clone();
	cv::split(hls, v_hls);
	//img_channels.push_back(gray);
	img_channels.push_back(v_hls[0]);
	img_channels.push_back(v_hls[2]);
	img_channels.push_back(gray);
	// obtain regions for each channel
	Mat img = src.clone();
	for (int i = 0; i < img_channels.size(); i++)
	{
		getCCForTextBox(input_img, img_channels[i], imageFile, i, img);
	}
	//imshow("src_regions", input_img);
	//waitKey();
}



/*getDescriptorFunctions*/
void _se_getDescriptorFromImage(char* fileName, float *hist)
{
#ifdef _SESG_
#define MAX_DIM_DESCRIPTOR MAX_DIM_SE
#endif
	Mat input_img = imread(fileName);
	if (!input_img.data)
	{
		cout << "read image failed! exit!" << endl;
		exit(0);
	}
	int len;
	if (input_img.rows > input_img.cols)
		len = input_img.rows;
	else
		len = input_img.cols;
	len = MAX_DIM_DESCRIPTOR;
#if _SHOW_HIST_ == 1
#ifndef _SESG_
	imshow("input", input_img);
#endif
#endif
	//MSER取区域
	vector<vector<Point>> regions;
	MSER ms(1, 1);
	ms(input_img, regions, Mat());

	//二值化
	Mat bw;
	cvtColor(input_img, bw, CV_RGB2GRAY);
	int front_sum = 0;
	for (int i = 0; i < bw.rows; i++)
	{
		for (int j = 0; j < bw.cols; j++)
		{
			if (bw.at<uchar>(i, j) > 100)
			{
				front_sum++;
				bw.at<uchar>(i, j) = 255;
			}
			else
				bw.at<uchar>(i, j) = 0;
		}
	}

	//求边缘
	vector<SuperPoint> grad_vectors;
	vector<vector<SuperPoint>> grad_map;
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].size(); j++)
		{
			//int l = ;
			SuperPoint sp(regions[i][j].y, regions[i][j].x, bw);
			if (sp.getGradient() != 0)
				grad_vectors.push_back(sp);
		}
	}
	int grad_sum = 0;
	int pxl_sum = 0;

	for (int i = 0; i < bw.rows; i++)
	{
		vector<SuperPoint> row;
		for (int j = 0; j < bw.cols; j++)
		{
			pxl_sum++;
			SuperPoint sp(i, j, bw);
			row.push_back(sp);
			if (sp.getGradient() > 0)
				grad_sum++;
		}
		if (row.size())
			grad_map.push_back(row);
	}
	strokeEdge(hist, grad_map, bw, len);
	//imshow("block", input_img);
	waitKey(0);

#ifndef _SESG_
	Mat des(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < des.cols; i++)
	{
		des.at<float>(i) = hist[i];
	}
	/*画出descriptor直方图*/
	drawHist(des, _SHOW_HIST_, 15);
#endif
}

void _superG_getDescriptorFromImage(char* fileName, float *hist)
{
#ifdef _SESG_
#define MAX_DIM_DESCRIPTOR MAX_DIM_SG
#endif
	Mat input_img = imread(fileName);
	if (!input_img.data)
	{
		cout << "read image failed! exit!" << endl;
		exit(0);
	}
	int len;
	if (input_img.rows > input_img.cols)
		len = input_img.rows;
	else
		len = input_img.cols;
	len = MAX_DIM_DESCRIPTOR;
#if _SHOW_HIST_ == 1
#ifndef _SESG_
	imshow("input", input_img);
#endif
#endif
	//MSER取区域
	vector<vector<Point>> regions;
	MSER ms(1, 1);
	ms(input_img, regions, Mat());

	//二值化
	Mat bw;
	cvtColor(input_img, bw, CV_RGB2GRAY);
	int front_sum = 0;
	for (int i = 0; i < bw.rows; i++)
	{
		for (int j = 0; j < bw.cols; j++)
		{
			if (bw.at<uchar>(i, j) > 100)
			{
				front_sum++;
				bw.at<uchar>(i, j) = 255;
			}
			else
				bw.at<uchar>(i, j) = 0;
		}
	}
	//求边缘
	vector<SuperPoint> grad_vectors;
	vector<vector<SuperPoint>> grad_map;
	for (int i = 0; i < regions.size(); i++)
	{
		for (int j = 0; j < regions[i].size(); j++)
		{
			//int l = ;
			SuperPoint sp(regions[i][j].y, regions[i][j].x, bw);
			if (sp.getGradient() != 0)
				grad_vectors.push_back(sp);
		}
	}
	int grad_sum = 0;
	int pxl_sum = 0;

	for (int i = 0; i < bw.rows; i++)
	{
		vector<SuperPoint> row;
		for (int j = 0; j < bw.cols; j++)
		{
			pxl_sum++;
			SuperPoint sp(i, j, bw);
			row.push_back(sp);
			if (sp.getGradient() > 0)
				grad_sum++;
		}
		if (row.size())
			grad_map.push_back(row);
	}
	//遍历边缘统计梯度
	//可能有多条边缘，所以遍历整张图，找到没被遍历到的点，以这个点为起点往后遍历。这条边缘遍历结束之后重新循环。

	for (int i = 0; i < grad_map.size(); i++)
	{
		for (int j = 0; j < grad_map[i].size(); j++)
		{
			if (grad_map[i][j].getFlag() == 0 && grad_map[i][j].getGradient() != 0)
			{
				runEdge(grad_map[i][j], hist, grad_map, len);
				i = 0;
				j = 0;
			}
		}
	}

#ifndef _SESG_
	/*int top = 0;*/
	int sum = 0;
	for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
	{
		sum = sum + hist[i];
	}
	//normalizeHist(hist, MAX_DIM_DESCRIPTOR);

	//sortArry(hist, 9);
	cout << endl << "after sort:" << endl;

	//for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
	//{
	//	hist[i] = hist[i] / (float)grad_sum;
	//}

	cout << "sum: " << sum << endl;
	cout << "grad_sum: " << grad_sum << endl;
	cout << "front_sum: " << front_sum << endl;
	cout << "pxl_sum: " << pxl_sum << endl;
	if (grad_sum == 0 || pxl_sum == 0)
		hist[0] = 0;
	float percent = (float)sum / grad_sum * 100;
	float percent1 = (float)grad_sum / pxl_sum * 100;
	if (percent1 < FILTER_EDGE_PERCENT)
		for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
			hist[i] = 0;
	else//这里是过滤留下来的部分，要想保留hist原值，就写hist[i] = hist[i];想改成统一的值比如1，就写hist[i] = 1;
		for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
			hist[i] = percent1;
	//for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
	//	cout << i << ":" << hist[i] << endl;
	cout << "percent: " << percent << endl;
	cout << "percent1: " << percent1 << endl;

	Mat des(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < des.cols; i++)
		des.at<float>(i) = hist[i];
	/*画出descriptor直方图*/
	drawHist(des, _SHOW_HIST_, 15);
#endif

}

int _stroke_getDescriptorFromImage(char* imageFile, float* descriptor)
{
	cv::Mat input_img = cv::imread(imageFile, 1);
	if (!input_img.data) {
		cout << "open image file:" << imageFile << " failed!" << endl;
		return -1;
	}

	Mat stroke_bin;// (br.y - tl.y, br.x - tl.x, CV_8UC1);
	stroke_bin = input_img;
	/*取值结束*/
	Mat stroke = skel(stroke_bin);
	double stroke_max;
	minMaxLoc(stroke, NULL, &stroke_max, 0, 0);
	float stroke_w = (float)stroke_max / (MAX_DIM_DESCRIPTOR);
	cout << "stroke_max:" << stroke_max << endl;
	for (int i = 0; i < stroke.rows; i++)
	{
		for (int j = 0; j < stroke.cols; j++)
		{
			float v = stroke.at<float>(i, j);
			if (v == 0)
				continue;
			for (int i = 0; i < MAX_DIM_DESCRIPTOR; i++)
			{
				if (v >= i*stroke_w && v < (i + 1)*stroke_w)
				{
					descriptor[i]++;
					break;
				}
			}
		}
	}

	sortArry(descriptor, MAX_DIM_DESCRIPTOR);
#if _SHOW_HIST_==1
	imshow("input_img", input_img);
#endif
	Mat des(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < des.cols; i++)
		des.at<float>(i) = descriptor[i];
	/*画出descriptor直方图*/
	drawHist(des, _SHOW_HIST_);
}
void _sesg_getDescriptorFromImage(char* fileName, float* descriptor)
{
#ifdef _SESG_
#define MAX_DIM_DESCRIPTOR MAX_DIM_SE+MAX_DIM_SG
#endif

#ifndef _SESG_
#define MAX_DIM_SG 1
#define MAX_DIM_SE 1
#endif
	float se[MAX_DIM_SE] = { 0 };
	float sg[MAX_DIM_SG] = { 0 };

	_se_getDescriptorFromImage(fileName, se);
	_superG_getDescriptorFromImage(fileName, sg);
	int i = 0;
	for (i = 0; i < MAX_DIM_SE; i++)
	{
		descriptor[i] = se[i];
	}
	for (i = 0; i < MAX_DIM_SG; i++)
	{
		descriptor[i + MAX_DIM_SE] = sg[i];
	}

	Mat des(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < des.cols; i++)
	{
		des.at<float>(i) = descriptor[i];
	}
	/*画出descriptor直方图*/
#if _SHOW_HIST_ == 1
	Mat img = imread(fileName);
	imshow("input", img);
#endif
	drawHist(des, _SHOW_HIST_, 15);

}

/*read image*/
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
	// init Labels
	for (int i = 0; i < CLASS_NUM; i++){
		classLabels[i] = (float)i;
	}
	// Get Boxes
	//
#ifdef _DIV_IMG_
	getTextBox(DIRTY_IMG);
#endif
}

void initDetector(string dirtyImg){
	// init Labels

	for (int i = 0; i < CLASS_NUM; i++){
		classLabels[i] = (float)i;
	}
	// Get Boxes
	//
	string cmd = "cd ./TEST & del * /S /Q";
	system("cd ./TEST & del * /S /Q");
	char img[50];
	strcpy_s(img, dirtyImg.c_str());
	//	const char *img = dirtyImg.c_str();
	getTextBox(img);
}
int readTrainFiles(Cfname *trainFiles, int trainClassCount[], int trainFilesCount, float *labels)
{
	for (int i = 0, cur = 0; i < CLASS_NUM; cur += trainClassCount[i], i++)
	{
		readFiles(i, trainFiles + cur, labels + cur);
	}
	for (int i = 0; i < trainFilesCount; i++){
		//cout << labels[i] << ":" << trainFiles[i] << endl;
	}
	return trainFilesCount;
}
int readTestFiles(Cfname *testFiles, int testFilesCount)
{
	readFiles(TEST_LOC, testFiles, NULL);
	return testFilesCount;
}
void readRegionFromImageName(char *fname, vector<Region> &regions, int seq)
{
	Mat img = imread(fname);
	int pos[4];
	char fname_cp[50];
	strcpy_s(fname_cp, fname);
	char * split = "_";
	char *p;
	char *token;
	token = strtok_s(fname_cp, split, &p);
	for (int i = 0; i < 4; i++)
	{
		token = strtok_s(NULL, split, &p);
		pos[i] = atoi(token);
	}
	Point tl, br;
	vector<Point> points;
	tl.x = pos[0];
	tl.y = pos[1];
	br.x = pos[2];
	br.y = pos[3];
	points.push_back(tl);
	points.push_back(br);

	Region region(points, seq);
	regions.push_back(region);
}

/*在img中显示regions的矩形区域*/
void showResult(Mat img, vector<Region> regions, string winName = "result", Scalar color = cv::Scalar(255, 0, 0))
{


	for (int i = 0; i < regions.size(); i++)
	{
		if (regions[i].isHidden())
			continue;
		Point tl = regions[i].getTL();
		Point br = regions[i].getBR();

		cv::rectangle(img, tl, br, color, 2, 2, 0);
	}

	imshow(winName, img);
}

vector<vector<Region>> detector(string dirtyImg)
{
	initDetector(dirtyImg);
	int trainCount[CLASS_NUM];
	int classDirs[CLASS_NUM];
	int testFilesCount = getClassFilesNum(testDir);
	int trainFilesCount = getTrainFilesNum(classDirs);

	cout << "***********Read Files...***********" << endl;

	Cfname *imgFiles = new Cfname[trainFilesCount];
	Cfname *testFiles = new Cfname[testFilesCount];
	float *labels = new float[trainFilesCount];
	int rows_for_train = readTrainFiles(imgFiles, classDirs, trainFilesCount, labels);
	int rows_for_test = readTestFiles(testFiles, testFilesCount);
	//for (int i = 0; i < rows_for_train; i++)
	//{
	//	cout << imgFiles[i] << ":" << labels[i] << endl;
	//}
	vector<Region> cc_res;
	Dcount *descriptorsTest = new Dcount[rows_for_test];// ??用descriptor
	for (int i = 0; i < rows_for_test; i++)
	{
		for (int j = 0; j<MAX_DIM_DESCRIPTOR; j++)
		{
			descriptorsTest[i][j] = 0;
		}
	}

#ifdef _SVM_
	Dcount *descriptors = new Dcount[rows_for_train];///// ??用descriptor
	for (int i = 0; i < rows_for_train; i++)
	{
		for (int j = 0; j<MAX_DIM_DESCRIPTOR; j++)
		{
			descriptors[i][j] = 0;
		}
	}
	///// 提取??用?片descriptor
	cout << "***********Get descriptor for train...***********" << endl;
	for (int i = 0; i < rows_for_train; i++) {
		//cout << i << "/" << rows_for_train << endl;
		cout << endl << "file: " << imgFiles[i] << endl;
		getDescriptorFromImage(imgFiles[i], descriptors[i]);
	}
	cout << "des4train:" << endl;
	/*for (int i = 0; i < rows_for_test; i++)
	{
	for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++)
	cout << descriptorsTest[i][j] << " ";
	cout << endl;
	}*/
	Mat labelsMat(rows_for_train, 1, CV_32FC1, labels);
	Mat trainDataMat(rows_for_train, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < rows_for_train; i++) {
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++) {
			trainDataMat.at<float>(i, j) = descriptors[i][j];
		}
	}
	///// 提取完成
#endif
	///// 提取??用?片descriptor
	cout << "***********Get descriptor for test....***********" << endl;
#ifdef _HOG_

#else
	for (int i = 0; i < rows_for_test; i++)
	{
		//cout << i << "/" << rows_for_test << endl;
		char* fn = testFiles[i];
		cout << fn << endl;
		readRegionFromImageName(fn, cc_res, i);
		getDescriptorFromImage(testFiles[i], descriptorsTest[i]);
	}
	cout << "des4test:" << endl;
#endif
#ifdef _SVM_
	///// 初始化SVM参数
	cout << "Initial SVM..." << endl;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	CvSVM SVM;

	///// 使用?参数的SVM?行??
	for (int i = 0; i < rows_for_train; i++)
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++)
			if (descriptors[i][j] != 0)
				;// cout << "des_for_train[" << i << "][" << j << "]" << trainDataMat.at<float>(i, j) << endl;
	//system("pause");
	cout << endl << "Training..." << endl;
	SVM.train(trainDataMat, labelsMat, Mat(), Mat(), params);
	///// 使用默?参数的SVM?行??
	//SVM.train(trainDataMat, labelsMat);


	float result = 0; //SVM测试结果
	vector<Region> sresult[2];
	///// 根据测试图片的descriptor生成测试用数据，并进行测试
	cout << "Response of SVM:" << endl;
	Mat testDesMat(1, MAX_DIM_DESCRIPTOR, CV_32FC1);
	for (int i = 0; i < rows_for_test; i++)
	{
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++) {
			///// 生成测试数据
			testDesMat.at<float>(0, j) = descriptorsTest[i][j];
			if (0 != descriptorsTest[i][j])
				;// cout << "des_for_test[" << j << "] = " << testDesMat.at<float>(0, j) << ":" << descriptorsTest[i][j] << endl;
		}
		///// 进行测试
		result = SVM.predict(testDesMat);
		//cout << result << endl;
		cc_res[i].setChar(result);
		sresult[(int)result].push_back(cc_res[i]);
		//std::cout << " RESULT" << i << "---" << response << "---" << testFiles[i] << endl;
	}



	/*合并并输出结果*/
	Mat img = imread(dirtyImg);
	int src_width = img.cols;
	int src_height = img.rows;
	float x_resize = (float)src_width / SRC_WIDTH;
	float y_resize = (float)src_height / SRC_HEIGHT;
	//resize(img, img, cv::Size(800, 600));
	vector<Region> rr[2];
	for (int i = 0; i < 2; i++)
	{

		Mat res_img = img.clone();
		stringstream ss;
		ss << "svm-result-" << i;
		cout << "wn:" << ss.str() << "-" << i << endl;
		for (int j = 0; j < sresult[i].size(); j++)
		{
			sresult[i][j].setSeq(j);
			sresult[i][j].setParent(j);
		}
		mergeRegions(sresult[i]);
		mergeTextRegions(sresult[i], res_img);
		mergeRegions(sresult[i]);
		for (int j = 0; j < sresult[i].size(); j++)
		{
			int tlx = sresult[i][j].getTL().x * x_resize;
			int tly = sresult[i][j].getTL().y * y_resize;
			int brx = sresult[i][j].getBR().x * x_resize;
			int bry = sresult[i][j].getBR().y * y_resize;
			sresult[i][j].setTL(tlx, tly);
			sresult[i][j].setBR(brx, bry);
			if (!sresult[i][j].isHidden())
				rr[i].push_back(sresult[i][j]);
		}

		showResult(res_img, sresult[i], ss.str());
	}
	vector<vector<Region>> r;
	r.push_back(rr[0]);//dirty
	r.push_back(rr[1]);//text

#endif
#ifdef _KM_
	///// kmeans
	cout << "Response of K-Means:" << endl;
	Mat testKmeansData(rows_for_test, MAX_DIM_DESCRIPTOR, CV_32FC1);
	Mat kmeansLabels;
	for (int i = 0; i < rows_for_test; i++)
		for (int j = 0; j < MAX_DIM_DESCRIPTOR; j++)
		{
		testKmeansData.at<float>(i, j) = descriptorsTest[i][j];
		}
	cv::kmeans(testKmeansData, K_MEANS_K, kmeansLabels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 1.0), 10, KMEANS_PP_CENTERS);
	///// result of kmeans:
	vector<Region> kresult[K_MEANS_K];
	for (int i = 0; i < rows_for_test; i++)
	{
		int result = kmeansLabels.at<int>(i);
		if (result > K_MEANS_K || resize < 0)
		{
			cout << "Something wrong in result of k-means,exit! " << endl;
			exit(0);
		}
		cc_res[i].is_char = result;
		kresult[result].push_back(cc_res[i]);
	}
	/////kmeans over


	/*合并并输出结果*/
	Mat img = imread(DIRTY_IMG);
	int src_width = img.cols;
	int src_height = img.rows;

	//resize(img, img, cv::Size(800, 600));

	for (int i = 0; i < K_MEANS_K; i++)
	{
		Mat res_img = img.clone();
		stringstream ss;
		ss << "k-result-" << i;
		cout << "wn:" << ss.str() << "-" << i << endl;
		for (int j = 0; j < kresult[i].size(); j++)
			kresult[i][j].parent = kresult[i][j].seq = j;
		mergeRegions(kresult[i]);
		showResult(res_img, kresult[i], ss.str());
		}
#endif
	waitKey(0);
	///// ?出完?
	//system("pause");
	return r;
	}

vector<string> readImageFileNamesFromXML(string xml)
{
	vector<string> imgs;
	TiXmlDocument *doc = new TiXmlDocument(xml.c_str());
	doc->LoadFile();
	TiXmlElement *root = doc->RootElement();
	if (!(root->NoChildren()))
	{
		TiXmlElement * eImg = root->FirstChildElement();
		string im = eImg->FirstChildElement()->GetText();
		imgs.push_back(im);
		while (eImg->NextSiblingElement())
		{
			eImg = eImg->NextSiblingElement();
			string im = eImg->FirstChildElement()->GetText();
			imgs.push_back(im);
		}
	}
	for (int i = 0; i < imgs.size(); i++)
	{
		cout << imgs[i] << endl;
	}
	return imgs;
}

vector<ImageResult> readResultFromXML(string xml)
{
	vector<ImageResult> vir;
	TiXmlDocument * input = new TiXmlDocument(xml.c_str());
	input->LoadFile();
	TiXmlElement *root = input->RootElement();
	if (!(root->NoChildren()))
	{
		TiXmlElement * eImage = root->FirstChildElement();
		do{
			ImageResult ir;

			//file name
			TiXmlElement * eContent = eImage->FirstChildElement();
			ir.imgName = eContent->GetText();

			//resolution
			eContent = eContent->NextSiblingElement();
			ir.width = atoi(eContent->Attribute("x"));
			ir.height = atoi(eContent->Attribute("y"));

			//rects
			eContent = eContent->NextSiblingElement();
			if (!(eContent->NoChildren()))
			{
				eContent = eContent->FirstChildElement();
				do{
					int x = atoi(eContent->Attribute("x"));
					int y = atoi(eContent->Attribute("y"));
					int width = atoi(eContent->Attribute("width"));
					int height = atoi(eContent->Attribute("height"));
					Rect rect(x, y, width, height);
					ir.rects.push_back(rect);
					eContent = eContent->NextSiblingElement();
				} while (eContent);
			}
			vir.push_back(ir);
			eImage = eImage->NextSiblingElement();
		} while (eImage);
	}
	return vir;
}
int inRect(int x, int y, Rect r)
{
	if (x >= r.x && x < r.x + r.width && y >= r.y && y < r.y + r.height)
		return 1;
	else
		return 0;
}

int inRects(int x, int y, vector<Rect> rs)
{
	for (int i = 0; i < rs.size(); i++)
	{
		if (inRect(x, y, rs[i]))
			return 1;
	}
	return 0;
}
void drawHero(vector<Rect> rs, Mat &m)
{
	for (int i = 0; i < rs.size(); i++)
	{
		Rect r = rs[i];
		int cr = r.y - 1;
		for (int h = 0; h < r.height; h++)
		{
			int cc = r.x - 1;

			cr = cr++;
			for (int w = 0; w < r.width; w++)
			{
				cc = cc++;
				if (m.at<uchar>(cr, cc) == 0)
					m.at<uchar>(cr, cc) = 1;
			}
		}
	}
}
void drawBoss(vector<Rect> rs, Mat &m)
{
	for (int i = 0; i < rs.size(); i++)
	{
		int cr = rs[i].y - 1;
		for (int h = 0; h < rs[i].height; h++)
		{
			int cc = rs[i].x - 1;
			cr = cr++;
			for (int w = 0; w < rs[i].width; w++)
			{
				cc = cc++;
				if (m.at<uchar>(cr, cc) == 0)
					m.at<uchar>(cr, cc) = 2;
				else
					m.at<uchar>(cr, cc) = 3;
			}
		}
	}
}
void fight(ImageResult hero, ImageResult boss, string name)
{
	Mat im = imread(name);
	Mat m = Mat::zeros(im.rows, im.cols, CV_8UC3);

	//draw the hero
	int h = 0;
	int b = 0;
	int hb = 0;

	drawHero(hero.rects, m);
	drawBoss(boss.rects, m);


	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			int c = m.at<uchar>(i, j);
			if (c == 1)
				h++;
			else if (c == 2)
				b++;
			else if (c == 3)
			{
				h++;
				b++;
				hb++;
			}
		}
	}
	if (h == 0 || b == 0)
	{
		cout << "P: " << 0 << endl;
		cout << "R: " << 0 << endl;
		cout << "F: " << 0 << endl;
		return;
	}
	float P = (float)hb / h;
	float R = (float)hb / b;
	float F = 2 * (P*R) / (P + R);
	cout << "P: " << P << endl;
	cout << "R: " << R << endl;
	cout << "F: " << F << endl;
}

void battle()
{
	vector<ImageResult> hero = readResultFromXML("hero.xml");
	vector<ImageResult> boss = readResultFromXML("boss.xml");
	for (int h = 0; h < hero.size(); h++)
	{
		string iname = hero[h].imgName;
		for (int b = 0; b < boss.size(); b++)
		{
			if (boss[b].imgName == iname)
			{
				cout << "》》》----->Fight On " << iname << "<-----《《《 " << endl;
				fight(hero[h], boss[b], iname);
			}
		}
	}

}
void main()
{


	//delete patches in ./TEST, and create new patches for current picture.
	vector<vector<Region>> result;		//result of detection
	vector<ImageResult> imgs;	//image file name read from xml file
	//read image file name from xml file
	imgs = readResultFromXML("boss.xml");

#ifdef _DETECT_
	//create output xml
	TiXmlDocument *out = new TiXmlDocument("hero.xml");
	TiXmlDeclaration * eDec = new TiXmlDeclaration("1.0", "UTF-8", "");
	out->LinkEndChild(eDec);
	TiXmlElement * eTagset = new TiXmlElement("tagset");
	out->LinkEndChild(eTagset);

	for (int i = 0; i < imgs.size(); i++)
	{
		Mat imgtest = imread(imgs[i].imgName);
		if (!imgtest.data)
		{
			cout << "cant find " << imgs[i].imgName << "!!!!!!!!!!!!!!" << endl;
			continue;
		}
		const int TRUTH = 1;//here is the sequence number of the text regions, normally 0 or 1
		string curimg = imgs[i].imgName;
		result = detector(curimg.c_str());
		TiXmlElement * eImage = new TiXmlElement("image");
		eTagset->LinkEndChild(eImage);

		TiXmlElement * eImageName = new TiXmlElement("imageName");
		TiXmlText * imagename = new TiXmlText(curimg.c_str());
		eImageName->LinkEndChild(imagename);
		eImage->LinkEndChild(eImageName);

		TiXmlElement * eResolution = new TiXmlElement("resolution");
		eResolution->SetAttribute("x", imgs[i].width);
		eResolution->SetAttribute("y", imgs[i].height);
		eImage->LinkEndChild(eResolution);

		TiXmlElement * eTaggedRectangles = new TiXmlElement("taggedRectangles");
		for (int j = 0; j < result[TRUTH].size(); j++)
		{
			Region rect = result[TRUTH][j];
			TiXmlElement *eTaggedRectangle = new TiXmlElement("taggedRectangle");
			eTaggedRectangle->SetAttribute("x", rect.getTL().x);
			eTaggedRectangle->SetAttribute("y", rect.getTL().y);
			eTaggedRectangle->SetAttribute("width", rect.getWidth());
			eTaggedRectangle->SetAttribute("height", rect.getHeight());
			eTaggedRectangles->LinkEndChild(eTaggedRectangle);
		}
		eImage->LinkEndChild(eTaggedRectangles);
	}
	out->SaveFile();

#endif
#ifdef _FIGHT_
	cout << "read hero" << endl;
	vector<ImageResult> hero = readResultFromXML("hero.xml");

	cout << "read boss" << endl;
	vector<ImageResult> boss = readResultFromXML("boss.xml");
	for (int i = 0; i < imgs.size(); i++)
	{
		//fight
		cout << "fight" << endl;
		Mat img = imread(imgs[i].imgName);
		//boss
		cout << "boss" << endl;
		for (int j = 0; j < boss.size(); j++)
		{
			Scalar color(0, 0, 255);
			if (boss[j].imgName != imgs[i].imgName)
				continue;
			for (int k = 0; k < boss[j].rects.size(); k++)
			{
				cv::rectangle(img, boss[j].rects[k], color, 2);
			}
		}
		//hero
		for (int j = 0; j < boss.size(); j++)
		{
			Scalar color(255, 0, 0);
			if (hero[j].imgName != imgs[i].imgName)
				continue;
			for (int k = 0; k < hero[j].rects.size(); k++)
			{
				cv::rectangle(img, hero[j].rects[k], color, 2);
			}
		}
		imshow(imgs[i].imgName, img);
		waitKey(0);
	}
#endif
#ifdef _BATTLE_
	battle();
	system("pause");
#endif
}
