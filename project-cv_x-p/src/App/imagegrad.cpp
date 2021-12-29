#include "ImageGrad.h"

double ImageGrad::DivMixGrad(const cv::Mat& sourceimg, const cv::Mat& targetimg, const int channel, const cv::Point shift_pos, cv::Point p)
{
	cv::Point direction[4] = {cv::Point(0,-1), cv::Point(0,1) ,cv::Point(-1,0) ,cv::Point(1, 0) };
	cv::Point ptemp;
	//0,1,2,3-上下左右
	int div = 0, v1, v2;
	for (int i = 0; i < 4; i++)
	{
		v1 = *AddrMatAt(sourceimg, p , channel) - *AddrMatAt(sourceimg, p + direction[i], channel);
		ptemp = p + shift_pos;
		v2 = *AddrMatAt(targetimg, ptemp, channel) - *AddrMatAt(targetimg, ptemp + direction[i], channel);
		div = div + (abs(v1) > abs(v2) ? v1 : v2);
	}
	return (double)div;
}

Eigen::VectorXd ImageGrad::DivMixGrad(const cv::Mat& sourceimg, const cv::Mat& targetimg, const cv::Point shift_pos, cv::Point p)
{
	//0,1,2,3-上下左右
	cv::Point direction[4] = { cv::Point(0,-1), cv::Point(0,1) ,cv::Point(-1,0) ,cv::Point(1, 0) };
	cv::Point ptemp;
	
	Eigen::VectorXd div(targetimg.channels());
	div.fill(0);
	int v1, v2;
	for (int i = 0; i < 4; i++)
	{
		for (int ch = 0; ch < targetimg.channels(); ch++)
		{
			v1 = *AddrMatAt(sourceimg, p, ch) - *AddrMatAt(sourceimg, p + direction[i], ch);
			ptemp = p + shift_pos;
			v2 = *AddrMatAt(targetimg, ptemp, ch) - *AddrMatAt(targetimg, ptemp + direction[i], ch);
			div(ch) = div(ch) + (double)(abs(v1) > abs(v2) ? v1 : v2);
		}
	}
	return div;
}

Eigen::VectorXd ImageGrad::DivGrad(const cv::Mat& img, cv::Point p)
{
	cv::Point direction[4] = { cv::Point(0,-1), cv::Point(0,1) ,cv::Point(-1,0) ,cv::Point(1, 0) };
	//0,1,2,3-上下左右
	Eigen::VectorXd div(img.channels());
	div.fill(0);
	int v1;
	for (int i = 0; i < 4; i++)
	{
		for (int ch = 0; ch < img.channels(); ch++)
		{
			v1 = *AddrMatAt(img, p, ch) - *AddrMatAt(img, p + direction[i], ch);
			div(ch) = div(ch) + (double)v1;
		}
	}
	return div;
}