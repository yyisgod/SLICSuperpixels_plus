#pragma once
#ifndef __FUNCTION__H
#define __FUNCTION__H
#include<vector>
using std::vector;
namespace yFunctions {
	void slicfeature(const vector<vector <double> >& g, int imgrow, int imgcol, vector<vector<double> >& localfeature);
}


#endif