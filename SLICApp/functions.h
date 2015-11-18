/*************************************************************************
/用来添加一些需要的函数
**************************************************************************/
#pragma once
#ifndef __FUNCTION__H
#define __FUNCTION__H
#include<vector>
using std::vector;
namespace yFunctions {
	void slicFeatureNeighborGray(const vector<vector <double> >& g, int imgrow, int imgcol, vector<vector<double> >& localfeature);
}


#endif