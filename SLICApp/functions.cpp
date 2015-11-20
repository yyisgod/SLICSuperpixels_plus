#include "stdafx.h"
#include "functions.h"

void yFunctions::slicFeatureNeighborGray(const vector<vector <double> >& g, int imgrow, int imgcol, vector<vector<double> >& localfeature)
{
	int N = 3;
	localfeature.assign(imgrow*imgcol, vector<double>(N*N, 0));

	for (int i = 0; i < imgrow; i++)
	{
		for (int j = 0; j < imgcol; j++)
		{
			//if (i == 0 && j == 0 || i == 0 && j == imgcol - 1 || i == imgrow - 1 && j == 0 || i == imgrow - 1 && j == imgcol - 1)
			if(i == 0 || j == 0 || i == imgrow - 1 || j == imgcol - 1)
			{
				for (int m = 0; m < N*N; m++)
				{
					localfeature[j + i*imgcol][m] = g[j + i*imgcol][0];
				}
			}
			else
			{
				localfeature[j + i*imgcol][0] = g[j - 1 + (i - 1)*imgcol][0];
				localfeature[j + i*imgcol][1] = g[j - 1 + i*imgcol][0];
				localfeature[j + i*imgcol][2] = g[j - 1 + (i + 1)*imgcol][0];

				localfeature[j + i*imgcol][3] = g[j + (i - 1)*imgcol][0];
				localfeature[j + i*imgcol][4] = g[j + i*imgcol][0];
				localfeature[j + i*imgcol][5] = g[j + (i + 1)*imgcol][0];

				localfeature[j + i*imgcol][6] = g[j + 1 + (i - 1)*imgcol][0];
				localfeature[j + i*imgcol][7] = g[j + 1 + i*imgcol][0];
				localfeature[j + i*imgcol][8] = g[j + 1 + (i + 1)*imgcol][0];
			}
		}
	}
}

//===========================================================================
///	saveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void yFunctions::saveSuperpixelLabels(
	const int*					labels,
	const int&					width,
	const int&					height,
	const string&				filename,
	const string&				addName,
	const string&				path) {

	char fname[256];
	_splitpath_s(filename.c_str(), NULL, 0, NULL, 0, fname, 256, NULL, 0);
	string temp = fname;
	string finalpath = path + temp + addName + string(".m");


	int sz = width*height;
	ofstream outfile;
	outfile.open(finalpath.c_str());
	outfile << " label = [";
	for (int i = 0; i < sz; i++) {
		outfile << labels[i] << ' ';
		if (i % width == width - 1)
			outfile << ';';
	}
	outfile << "];";
	outfile.close();
}
