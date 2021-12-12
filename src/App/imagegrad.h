#pragma once

#include <Eigen/dense>
#include <opencv2/core/core.hpp>

/*
提供了一些静态方法，对cv:Mat上差分求导
*/
class ImageGrad
{
public :
	//static Eigen::MatrixXd LaplaceMap(const cv::Mat& img);
	static Eigen::VectorXd DivGrad(const cv::Mat& img, cv::Point p);
	static double DivMixGrad(const cv::Mat& sourceimg, const cv::Mat& targetimg, const int channel, const cv::Point shift_pos, cv::Point p);
	static Eigen::VectorXd DivMixGrad(const cv::Mat& sourceimg, const cv::Mat& targetimg, const cv::Point shift_pos, cv::Point p);

	static uchar* AddrMatAt(const cv::Mat& mat, const int irowy, const int jcolx, const int ch)
	{
		return mat.data + mat.step[0] * irowy + mat.step[1] * jcolx + ch;
	}

	static uchar* AddrMatAt(const cv::Mat& mat, const int irowy, const int jcolx)
	{
		return mat.data + mat.step[0] * irowy + mat.step[1] * jcolx;
	}
	static uchar* AddrMatAt(const cv::Mat& mat, cv::Point p, const int ch)
	{
		return mat.data + mat.step[0] * p.y + mat.step[1] * p.x + ch;
	}
	static uchar* AddrMatAt(const cv::Mat& mat, cv::Point p)
	{
		return mat.data + mat.step[0] * p.y + mat.step[1] * p.x;
	}

	static int CoorInRect(const int x, const int y, const cv::Rect& rect)
	{
		return (y - rect.y)*rect.width + x - rect.x;
	}
};