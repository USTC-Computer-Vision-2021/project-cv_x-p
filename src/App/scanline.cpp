#include "ScanLine.h"
#include <iostream>

ScanLine::ScanLine(std::vector<cv::Point2f> corner_coordinate_)
{
	polygon_ = corner_coordinate_;
	top_ = MIN(MIN(MIN(corner_coordinate_[0].y, corner_coordinate_[1].y), corner_coordinate_[2].y), corner_coordinate_[3].y);
	bottom_  = MAX(MAX(MAX(corner_coordinate_[0].y, corner_coordinate_[1].y), corner_coordinate_[2].y), corner_coordinate_[3].y);
	left_ = MIN(MIN(MIN(corner_coordinate_[0].x, corner_coordinate_[1].x), corner_coordinate_[2].x), corner_coordinate_[3].x);
	right_ = MAX(MAX(MAX(corner_coordinate_[0].x, corner_coordinate_[1].x), corner_coordinate_[2].x), corner_coordinate_[3].x);
}


ScanLine::~ScanLine()
{
}


//void ScanLine::SetPolygon(QPolygon apolygon)
//{
//	polygon = apolygon;
//
//	top = INT_MAX;
//	bottom = INT_MIN;
//	left = INT_MAX;
//	right = INT_MIN ;
//	for (QPolygon::iterator iter = polygon.begin(); iter != polygon.end(); iter++)
//	{
//		if (iter->x() < left)
//			left = iter->x();
//		if (iter->x() > right)
//			right = iter->x();
//		if (iter->y() < top)
//			top = iter->y();
//		if (iter->y() > bottom)
//			bottom = iter->y();
//	}
//}


void ScanLine::BuildEdgeTable()
{
	edge_table_.resize(bottom_ - top_ + 1);

	for (size_t i = 0, count = polygon_.size(); i < count; i++)
	{
		Edge e;
		cv::Point2f p, q;

		p = polygon_[i];
		q = polygon_[(i + 1) % count];
		
		if (p.y == q.y)
		{
			//水平的边;
			e.x = p.x;
			e.ymax = p.y;
			e.dx = 999999999;
			edge_table_[p.y - top_].insert(edge_table_[p.y - top_].end(), e);
			e.x = q.x;
			e.ymax = q.y;
			e.dx = 999999999;
			edge_table_[q.y - top_].insert(edge_table_[q.y - top_].end(), e);
		}
		else
		{
			//找边的y较小的顶点为p, 较大的为q
			if (q.y < p.y)
				std::swap(p, q);
			e.x = p.x;
			e.ymax = q.y;
			e.dx = (q.x - p.x) / (double)(q.y - p.y);
			edge_table_[p.y - top_].insert(edge_table_[p.y - top_].end(), e);
		}
	}
}

void ScanLine::UpdateActiveEdgeTable(int height_)
{
	std::list<Edge>::iterator iter = active_edge_table_.begin();

	while (iter != active_edge_table_.end())
	{
		if (iter->ymax < height_)
		{
			iter = active_edge_table_.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	iter = active_edge_table_.begin();
	while (iter != active_edge_table_.end())
	{
		iter->x += iter->dx;
		iter++;
	}

	active_edge_table_.insert(active_edge_table_.end(), edge_table_[height_ - top_].begin(), edge_table_[height_ - top_].end());
	active_edge_table_.sort();
}


void ScanLine::CalcIntersect(int height_)
{
	intersect_list_.clear();

	if (active_edge_table_.empty())
	{
		return;
	}

	std::list<Edge>::iterator iter = active_edge_table_.begin(), iter2;
	while (iter != active_edge_table_.end())
	{
		iter2 = iter;
		iter2++;
		if (active_edge_table_.end() == iter2)
		{
			intersect_list_.push_back(iter->x);
			iter++;
		}
		else if (fabs(iter->x - iter2->x) > 2.0)	
		{
			intersect_list_.push_back(iter->x);
			iter++;
		}
		else
		{
			if ((iter->ymax == height_) && (iter2->ymax == height_))//处理顶点的情况
			{
				intersect_list_.push_back(iter->x);
				intersect_list_.push_back(iter->x);
			}
			else if ((iter->ymax > height_) && (iter2->ymax > height_))
			{
			}
			else
			{
				intersect_list_.push_back(iter->x);
			}
			iter++;
			iter++;
		}
	}

	std::sort(intersect_list_.begin(), intersect_list_.end());
}


//cv::Rect CScanLine::CalcRectRegion(const QPolygon & apolygon)
//{
//	int atop = INT_MAX, abottom = INT_MIN, aleft = INT_MAX, aright = INT_MIN;
//	for (int i = 0; i< apolygon.size(); i++)
//	{
//		if (apolygon[i].x() < aleft)
//			aleft = apolygon[i].x();
//		if (apolygon[i].x() > aright)
//			aright = apolygon[i].x();
//		if (apolygon[i].y() < atop)
//			atop = apolygon[i].y();
//		if (apolygon[i].y() > abottom)
//			abottom = apolygon[i].y();
//	}
//	return cv::Rect(aleft, atop, aright - aleft, abottom - aleft);
//}

//cv::Rect CScanLine::CalcRectRegion(const QPolygon & apolygon, int awidth, int aheight)
//{
//	int atop = aheight, abottom = 0, aleft = awidth, aright = 0;
//	for (int i = 0; i < apolygon.size(); i++)
//	{
//		if (apolygon[i].x() < aleft)
//			aleft = apolygon[i].x();
//		if (apolygon[i].x() > aright)
//			aright = apolygon[i].x();
//		if (apolygon[i].y() < atop)
//			atop = apolygon[i].y();
//		if (apolygon[i].y() > abottom)
//			abottom = apolygon[i].y();
//	}
//	atop = std::max(0, atop - 1 );
//	aleft = std::max(0, aleft - 1);
//	aright = std::min(aright, awidth + 1);
//	abottom = std::min(abottom, aheight + 1);
//
//	return cv::Rect(aleft, atop, std::max(0, aright - aleft), std::max(0, abottom - atop));
//}
//
//QPolygon CScanLine::ShiftPolygon(const QPolygon & apolygon, QPoint avec)
//{
//	QPolygon  ret;
//	ret.reserve(apolygon.size());
//	for (int i = 0; i < apolygon.size(); i++)
//	{
//		ret.push_back(apolygon[i] + avec);
//	}
//	return ret;
//}


void ScanLine::FillPolygon(cv::Mat& mat)
{
	BuildEdgeTable();

	std::vector<int> intPts;
	width_ = mat.size().width;
	height_ = mat.size().height;

	for (int i = top_; i <= bottom_; i++)
	{
		UpdateActiveEdgeTable(i);
		CalcIntersect(i);

		bool status = false;				//初始在外
		intPts.clear();
		//相邻两个重合的交点相互抵消
		for (int j = 0; j < intersect_list_.size(); j++)
		{
			if (intPts.empty())
			{
				intPts.push_back((int)intersect_list_[j]);
			}
			else if ((int)intersect_list_[j] == intPts.back())
			{
				intPts.pop_back();
			}
			else
			{
				intPts.push_back((int)intersect_list_[j]);
			}
		}
		intPts.push_back(INT_MAX);
		//std::cout << "inPts size() = " << intPts.size() << std::endl;
		int index = 0;
		for (int j = left_ - 1; j <= right_; j++)
		{
			if (j == intPts[index])
			{
				status = !status;								//内外状态反转
				index++;
			}
			if (status)
			{
				if (j > 0 && j < width_ && i > 0 && i < height_)
				{
					mat.at<cv::uint8_t>(i, j) = 1;
				}
			}
		}
	}
}