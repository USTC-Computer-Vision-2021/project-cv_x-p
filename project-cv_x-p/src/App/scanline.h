#pragma once

#include <opencv2/core/core.hpp>
#include <vector>
#include <list>

//扫描线算法
class ScanLine
{
public:
	ScanLine(std::vector<cv::Point2f> corner_coordinate_);
	~ScanLine();
	//void SetPolygon(QPolygon apolygon);
	void FillPolygon(cv::Mat& mat);

private:
	void BuildEdgeTable();
	void UpdateActiveEdgeTable(int height);
	void CalcIntersect(int height);

private:
	struct Edge
	{
		int ymax;
		double dx;
		double x;

		inline bool operator<(const Edge& e2) const
		{
			return x < e2.x;
		}
	};

	int top_;
	int bottom_;
	int left_;
	int right_;
	int width_;
	int height_;

	std::vector<cv::Point2f> polygon_;

	std::vector<std::list<Edge>>	edge_table_;				//边表
	std::list<Edge>					active_edge_table_;		//活性边表
	std::vector<double>				intersect_list_;				//交点
//public:
//
//	static cv::Rect CalcRectRegion(const QPolygon& apolygon);
//	static cv::Rect CalcRectRegion(const QPolygon& apolygon, int awidth, int aheight);
//	static QPolygon ShiftPolygon(const QPolygon& apolygon, QPoint avec);
};