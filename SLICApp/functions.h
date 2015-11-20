/*************************************************************************
/用来添加一些需要的函数
**************************************************************************/
#pragma once
#ifndef __FUNCTION__H
#define __FUNCTION__H
#include <vector>
#include <string>
#include <fstream>
using namespace std;
namespace yFunctions {
	void slicFeatureNeighborGray(const vector<vector <double> >& g, int imgrow, int imgcol, vector<vector<double> >& localfeature);
	void saveSuperpixelLabels(
		const int*					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				addName,
		const string&				path); 
}

#endif