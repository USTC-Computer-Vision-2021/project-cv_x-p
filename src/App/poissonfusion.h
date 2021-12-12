#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseQR>

#include "imagegrad.h"
/*
Poisson图像融合
*/
class PoissonFusion
{
public:
	PoissonFusion();
	~PoissonFusion();


	/*
	将source的部分取mask的部分融合到target中；source (0, 0)的位置对齐到 target的 (x_pos, y_pos)位置。
	* source和target的channels数应当相等
	* mask应当只有1个channel
	*/
	cv::Mat GetFusion();


	/*设置source和mask。将检查格式是否合法，否则会抛出异常*/
	void SetSourceMask(const cv::Mat asource, const cv::Mat amask);

	/*设置target*/
	void SetTarget(const cv::Mat atarget) { target = atarget; }

	/*设置偏移距离。同时重新计算有效重叠部分*/
	void SetShiftPos(const cv::Point shift_pos) { pos = shift_pos; GenFeasibleRect(); }

	/*设置是否采用Mixed方法*/
	void SetMixedGrad(bool abool) { mixed_grad = abool; }

private:
	/*
	计算系数矩阵
	*/
	void GenCoefMatA();

	/*
	建立内点的索引
	*/
	void GenIndex();

	/*
	计算b
	*/
	void GenVecb(bool Mixed);


	/*
	预分解系数矩阵
	*/
	void PreFactorizeA();

	void GenFeasibleRect();

public:
	/*
	对矩阵中第irowy行,第jcolx列/坐标(jcolx, irowy)第ch个uchar地址的访问。不使用at<>函数
	*/
	static uchar* AddrMatAt(const cv::Mat& mat, const int irowy, const int jcolx, const int ch);

	/*
	对矩阵中第irowy行,第jcolx列/坐标(jcolx, irowy)元素首地址的访问。不使用at<>函数，无需事先知道矩阵类型
	*/
	static uchar* AddrMatAt(const cv::Mat& mat, const int irowy, const int jcolx);

	/*
	对矩阵中坐标 p 第ch个uchar地址的访问。不使用at<>函数，无需事先知道矩阵类型
	*/
	static uchar* AddrMatAt(const cv::Mat& mat, cv::Point p, const int ch);

	/*
	对矩阵中坐标 p 元素首地址的访问。不使用at<>函数，无需事先知道矩阵类型
	*/
	static uchar* AddrMatAt(const cv::Mat& mat, cv::Point p);

	static int CoorInRect(const int x, const int y, const cv::Rect& rect);
	static int CoorInRect(cv::Point p, const cv::Rect& rect);

	/*
	计算b的静态方法
	*/
	static void GenVecb(const Eigen::MatrixXd& laplace_map, const cv::Mat target, const cv::Point shift_pos, std::vector<cv::Point>& id_to_coor, const int* coor_to_id, const cv::Rect& frect, Eigen::MatrixXd& b);

	static void GenLaplaceMap(const cv::Mat& img, const std::vector<cv::Point>& id_to_coor, Eigen::MatrixXd& laplacemap);
	static void GenLaplaceMapMixed(const cv::Mat& sourceimg, const cv::Mat& targetimg, const cv::Point shift_pos, std::vector<cv::Point>& id_to_coor, Eigen::MatrixXd& laplacemap);


private:

	bool mixed_grad;

	bool coef_a_uptodate;

	cv::Mat mask;
	cv::Mat source;
	cv::Mat target;
	cv::Point pos;
	cv::Rect feasible_rect;

	int* coor_mapto_index;						//内点的坐标映射到编号
	std::vector<cv::Point> index_mapto_coor;	//内点的编号映射到坐标
	Eigen::SparseMatrix<double>		coef_mat_a;
	Eigen::MatrixXd						vec_b;
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;

};