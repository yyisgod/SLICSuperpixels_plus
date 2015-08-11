// SLIC.cpp: implementation of the SLIC class.
//
// Copyright (C) Radhakrishna Achanta 2012
// All rights reserved
// Email: firstname.lastname@epfl.ch
//////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <cfloat>
#include <cmath>
#include <iostream>
#include <fstream>
#include "SLIC.h"
#include <sstream>


//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC()
{
	m_depth = 3;
}

SLIC::~SLIC()
{

}

vector<vector<double>>& SLIC::getData()
{
	return m_data;
}

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
void SLIC::RGB2XYZ(
	const int&		sR,
	const int&		sG,
	const int&		sB,
	double&			X,
	double&			Y,
	double&			Z)
{
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)	r = R/12.92;
	else				r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = pow((B+0.055)/1.055,2.4);

	X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval, double& aval, double& bval)
{
	//------------------------
	// sRGB to XYZ conversion
	//------------------------
	double X, Y, Z;
	RGB2XYZ(sR, sG, sB, X, Y, Z);

	//------------------------
	// XYZ to LAB conversion
	//------------------------
	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = pow(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	lval = 116.0*fy-16.0;
	aval = 500.0*(fx-fy);
	bval = 200.0*(fy-fz);
}

//===========================================================================
///	doRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::doRGBtoLABConversion(const unsigned int*& ubuff)
{
	int sz = m_width*m_height;

	for( int j = 0; j < sz; j++ )
	{	
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >>  8) & 0xFF;
		int b = (ubuff[j]      ) & 0xFF;
		double l, a, bb;
		RGB2LAB( r, g, b, l, a, bb);
		m_data[j][0] = (l);
		m_data[j][1] = (a);
		m_data[j][2] = (bb);
	}
}


//=================================================================================
/// drawContoursAroundSegments
///
/// Internal contour drawing option exists. One only needs to comment the if
/// statement inside the loop that looks at neighbourhood.
//=================================================================================
void SLIC::drawContoursAroundSegments(
	unsigned int*&			ubuff,
	int*&					labels,
	const int&				width,
	const int&				height,
	const unsigned int&				color )
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

/*	int sz = width*height;

	vector<bool> istaken(sz, false);

	int mainindex(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					if( false == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )//change to 2 or 3 for thinner lines
			{
				ubuff[mainindex] = color;
				istaken[mainindex] = true;
			}
			mainindex++;
		}
	}*/


	int sz = width*height;
	vector<bool> istaken(sz, false);
	vector<int> contourx(sz);vector<int> contoury(sz);
	int mainindex(0);int cind(0);
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			int np(0);
			for( int i = 0; i < 8; i++ )
			{
				int x = k + dx8[i];
				int y = j + dy8[i];

				if( (x >= 0 && x < width) && (y >= 0 && y < height) )
				{
					int index = y*width + x;

					//if( false == istaken[index] )//comment this to obtain internal contours
					{
						if( labels[mainindex] != labels[index] ) np++;
					}
				}
			}
			if( np > 1 )
			{
				contourx[cind] = k;
				contoury[cind] = j;
				istaken[mainindex] = true;
				//img[mainindex] = color;
				cind++;
			}
			mainindex++;
		}
	}

	int numboundpix = cind;//int(contourx.size());
	for( int j = 0; j < numboundpix; j++ )
	{
		int ii = contoury[j]*width + contourx[j];
		ubuff[ii] = 0xffffff;

		for( int n = 0; n < 8; n++ )
		{
			int x = contourx[j] + dx8[n];
			int y = contoury[j] + dy8[n];
			if( (x >= 0 && x < width) && (y >= 0 && y < height) )
			{
				int ind = y*width + x;
				if(!istaken[ind]) ubuff[ind] = 0;
			}
		}
	}
}

void SLIC::loadImage(const unsigned int * ubuff, const int width, const int height)
{
	m_width = width;
	m_height = height;
	int sz = m_width*m_height;
	m_data.assign(sz, vector<double>(3));
	if (1)//LAB, the default option
	{
		doRGBtoLABConversion(ubuff);
	}
	else//RGB
	{
		for (int i = 0; i < sz; i++)
		{
			int l = ubuff[i] >> 16 & 0xff;
			int a = ubuff[i] >> 8 & 0xff;
			int b = ubuff[i] & 0xff;
			m_data[i][0] = (double(l));
			m_data[i][1] = (double(a));
			m_data[i][2] = (double(b));
		}
	}
}


//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::detectLabEdges(vector<double>& edges)
{
	int sz = m_width * m_height;

	edges.resize(sz,0);
	for( int j = 1; j < m_height-1; j++ )
	{
		for( int k = 1; k < m_width-1; k++ )
		{
			int i = j * m_width+k;

			double dx = (m_data[i - 1][0] - m_data[i + 1][0])*(m_data[i - 1][0] - m_data[i + 1][0]) +
						(m_data[i - 1][1] - m_data[i + 1][1])*(m_data[i - 1][1] - m_data[i + 1][1]) +
						(m_data[i - 1][2] - m_data[i + 1][2])*(m_data[i - 1][2] - m_data[i + 1][2]);

			double dy = (m_data[i-m_width][0]-m_data[i+m_width][0])*(m_data[i-m_width][0]-m_data[i+m_width][0]) +
						(m_data[i-m_width][1]-m_data[i+m_width][1])*(m_data[i-m_width][1]-m_data[i+m_width][1]) +
						(m_data[i-m_width][2]-m_data[i+m_width][2])*(m_data[i-m_width][2]-m_data[i+m_width][2]);

			//edges[i] = fabs(dx) + fabs(dy);
			edges[i] = dx*dx + dy*dy;
		}
	}
}

//===========================================================================
///	PerturbSeeds
//调整种子点到附近梯度最大的点上去
//===========================================================================
void SLIC::perturbSeeds(
	vector<vector<double> >&				kseeds,
	vector< vector<double> >&				kseedsxy,
        const vector<double>&                   edges)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};
	
	int numseeds = kseeds.size();

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = (int)kseedsxy[n][0];//original x
		int oy = (int)kseedsxy[n][1];//original y
		int oind = oy*m_width + ox;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = ny*m_width + nx;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if (storeind != oind)
		{
			kseedsxy[n][0] = storeind%m_width;
			kseedsxy[n][1] = storeind / m_width;
			for (int i = 0; i < m_depth; i++) {
				kseeds[n][i] = m_data[storeind][i];
			}
		}
	}
}


//===========================================================================
///	GetLABXYSeeds_ForGivenStepSize
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::getSeeds_ForGivenStepSize(
	vector<vector<double> >&				kseeds,
	vector<vector<double> >&				kseedsxy,
    const int&					STEP,
    const bool&					perturbseeds,
    const vector<double>&       edgemag)
{
    const bool hexgrid = false;
	int numseeds(0);
	int n(0);

	int xstrips = (int)(0.5+double(m_width)/double(STEP));
	int ystrips = (int)(0.5+double(m_height)/double(STEP));

    int xerr = m_width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = m_width - STEP*xstrips;}
    int yerr = m_height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = m_height- STEP*ystrips;}

	double xerrperstrip = double(xerr)/double(xstrips);
	double yerrperstrip = double(yerr)/double(ystrips);

	int xoff = STEP/2;
	int yoff = STEP/2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------
	kseeds.resize(numseeds);
	kseedsxy.resize(numseeds);

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = (int)(y*yerrperstrip);
		
		for( int x = 0; x < xstrips; x++ )
		{	
			kseeds[n] = vector<double>(m_depth);
			kseedsxy[n] = vector<double>(2);
			int xe = (int)(x*xerrperstrip);
            int seedx = (x*STEP+xoff+xe);
            if(hexgrid){ seedx = x*STEP+(xoff<<(y&0x1))+xe; seedx = min(m_width-1,seedx); }//for hex grid sampling
            int seedy = (y*STEP+yoff+ye);
            int i = seedy*m_width + seedx;
			
			for (int j = 0; j < m_depth; j++) {
				kseeds[n][j] = (m_data[i][j]);
			}
			kseedsxy[n][0] = (seedx); 
			kseedsxy[n][1] = (seedy);
			n++;
		}
	}

	
	if(perturbseeds)
	{
		perturbSeeds(kseeds, kseedsxy, edgemag);
	}
}


//===========================================================================
///	PerformSuperpixelSLIC
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
//===========================================================================
void SLIC::performSuperpixelSLIC(
	vector<vector<double> >&				kseeds,
	vector<vector<double> >&				kseedsxy,
        int*&					klabels,
        const int&				STEP,
        const vector<double>&                   edgemag,
	const double&				M)
{
	int sz = m_width*m_height;
	const int numk = kseeds.size();
	//----------------
	int offset = STEP;
        //if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------
	
	vector<double> clustersize(numk, 0);
	vector<double> inv(numk, 0);//to store 1/clustersize[k] values

	vector<vector<double> > sigma(numk, vector<double>(m_depth + 2,0));
	vector<double> distvec(sz, DBL_MAX);

	double invwt = 1.0/((STEP/M)*(STEP/M));

	int x1, y1, x2, y2;
	double dist;
	double distxy;
	for( int itr = 0; itr < 10; itr++ )
	{
		outSeeds(itr,kseeds,kseedsxy);
		distvec.assign(sz, DBL_MAX);
		for( int n = 0; n < numk; n++ )
		{
                        y1 = (int)max(0.0,			kseedsxy[n][1]-offset);
                        y2 = (int)min((double)m_height,	kseedsxy[n][1]+offset);
                        x1 = (int)max(0.0,			kseedsxy[n][0]-offset);
                        x2 = (int)min((double)m_width,	kseedsxy[n][0]+offset);


			for( int y = y1; y < y2; y++ )
			{
				for( int x = x1; x < x2; x++ )
				{
					int i = y*m_width + x;
					dist = 0;
					for (int j = 0; j < m_depth; j++) {
						dist += (m_data[i][j] - kseeds[n][j]) * (m_data[i][j] - kseeds[n][j]);
					}

					distxy =		(x - kseedsxy[n][0])*(x - kseedsxy[n][0]) +
									(y - kseedsxy[n][1])*(y - kseedsxy[n][1]);
					
					//------------------------------------------------------------------------
					dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
					//------------------------------------------------------------------------
					if( dist < distvec[i] )
					{
						distvec[i] = dist;
						klabels[i]  = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		//instead of reassigning memory on each iteration, just reset.
	
		sigma.assign(numk, vector<double>(m_depth + 2, 0));
		clustersize.assign(numk, 0);
		//------------------------------------
		//edgesum.assign(numk, 0);
		//------------------------------------

		{int ind(0);
		for( int r = 0; r < m_height; r++ )
		{
			for( int c = 0; c < m_width; c++ )
			{
				int i;
				for (i = 0; i < m_depth; i++) {
					sigma[klabels[ind]][i] += m_data[ind][i];
				}
				sigma[klabels[ind]][m_depth] += c;
				sigma[klabels[ind]][m_depth+1] += r;
				//------------------------------------
				//edgesum[klabels[ind]] += edgemag[ind];
				//------------------------------------
				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}}

		{for( int k = 0; k < numk; k++ )
		{
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/clustersize[k];//computing inverse now to multiply, than divide later
		}}
		
		{for( int k = 0; k < numk; k++ )
		{
			for (int i = 0; i < m_depth; i++) {
				kseeds[k][i] = sigma[k][i] * inv[k];
			}
			kseedsxy[k][0] = sigma[k][m_depth]*inv[k];  //x
			kseedsxy[k][1] = sigma[k][m_depth+1]*inv[k]; //y
			//------------------------------------
			//edgesum[k] *= inv[k];
			//------------------------------------
		}}
	}
}

//===========================================================================
///	PerformSuperpixelSLICy
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
//===========================================================================
//void SLIC::PerformSuperpixelSLICnew(
//	vector<double>&				kseedsl,
//	vector<double>&				kseedsa,
//	vector<double>&				kseedsb,
//	vector<double>&				kseedsx,
//	vector<double>&				kseedsy,
//        int*&					klabels,
//        const int&				STEP,
//        const vector<double>&                   edgemag,
//	const double&				M)
//{
//	int sz = m_width*m_height;
//	int numk = kseedsl.size();
//	//----------------
//	int offset = STEP;
//        //if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
//	//----------------
//	
//	
//	vector<double> distvec(sz, DBL_MAX);
//	vector<int> labelC(numk,-1);
//
//	double invwt = 1.0/((STEP/M)*(STEP/M));
//
//	//
//
//	int x1, y1, x2, y2;
//	double l, a, b;
//	double dist;
//	double distxy;
//	bool flag = 0;//标志是否需要加种子
//	for( int itr = 0; itr < 11; itr++ )
//	{	
//		distvec.assign(sz, DBL_MAX);
//		for( int n = 0; n < numk; n++ )
//		{
//			if(flag && (labelC[n] == -1))//加种子后筛选需要充分的区域
//				continue;
//            y1 = max(0.0,			kseedsy[n]-offset);
//            y2 = min((double)m_height,	kseedsy[n]+offset);
//            x1 = max(0.0,			kseedsx[n]-offset);
//            x2 = min((double)m_width,	kseedsx[n]+offset);
//
//
//			for( int y = y1; y < y2; y++ )
//			{
//				for( int x = x1; x < x2; x++ )
//				{
//					int i = y*m_width + x;
//					if(flag &&(labelC[n] != klabels[i])) continue;   //如果不是该区域的点，则抛弃
//
//					l = m_lvec[i];
//					a = m_avec[i];
//					b = m_bvec[i];
//
//					dist =			(l - kseedsl[n])*(l - kseedsl[n]) +
//									(a - kseedsa[n])*(a - kseedsa[n]) +
//									(b - kseedsb[n])*(b - kseedsb[n]);
//
//					distxy =		(x - kseedsx[n])*(x - kseedsx[n]) +
//									(y - kseedsy[n])*(y - kseedsy[n]);
//					
//					//------------------------------------------------------------------------
//					dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
//					//------------------------------------------------------------------------
//					if( dist < distvec[i] )
//					{
//						distvec[i] = dist;
//						klabels[i]  = n;
//					}
//					
//				}
//			}
//		}
//		//-----------------------------------------------------------------
//		// Recalculate the centroid and store in the seed values
//		//-----------------------------------------------------------------
//		//instead of reassigning memory on each iteration, just reset.
//	
//		if(itr == 9 )
//				flag = 1;
//		vector<double> clustersize(numk, 0);
//		
//
//		vector<double> sigmal(numk, 0);
//		vector<double> sigmaa(numk, 0);
//		vector<double> sigmab(numk, 0);
//		vector<double> sigmax(numk, 0);
//		vector<double> sigmay(numk, 0);
//		sigmal.assign(numk, 0);
//		sigmaa.assign(numk, 0);
//		sigmab.assign(numk, 0);
//		sigmax.assign(numk, 0);
//		sigmay.assign(numk, 0);
//		clustersize.assign(numk, 0);
//
//		//新东西
//		//something new
//		vector<double> meanl(numk,0);
//		vector<double> meana(numk,0);
//		vector<double> meanb(numk,0);
//		vector<double> stddist(numk,0);
//		vector<double> edgemax(numk, 0);
//		vector<double> edgemin(numk,0);
//		edgemax.assign(numk,0);//最大梯度
//		edgemin.assign(numk,DBL_MAX);//最小梯度
//		vector<int> emaxlabel(numk, 0);//对应像素
//		vector<int> eminlabel(numk,0);
//		emaxlabel.assign(numk,-1);
//		eminlabel.assign(numk,-1);
//		//------------------------------------
//		//edgesum.assign(numk, 0);
//		//------------------------------------
//
//		{int ind(0);
//		for( int r = 0; r < m_height; r++ )
//		{
//			for( int c = 0; c < m_width; c++ )
//			{
//				sigmal[klabels[ind]] += m_lvec[ind];
//				sigmaa[klabels[ind]] += m_avec[ind];
//				sigmab[klabels[ind]] += m_bvec[ind];
//				sigmax[klabels[ind]] += c;
//				sigmay[klabels[ind]] += r;
//
//				//------------------------------------
//				//edgesum[klabels[ind]] += edgemag[ind];
//				//------------------------------------
//				clustersize[klabels[ind]] += 1.0;
//				
//				if(edgemax[klabels[ind]] < edgemag[ind])
//				{
//					edgemax[klabels[ind]] = edgemag[ind];//该种子点类的最大梯度
//					emaxlabel[klabels[ind]] = ind;
//				}
//				if(edgemin[klabels[ind]] > edgemag[ind])
//				{
//					edgemin[klabels[ind]] = edgemag[ind];
//					eminlabel[klabels[ind]] = ind;
//				}
//				ind++;
//			}
//		}}
//		//出均值
//		vector<double> inv(numk, 0);//to store 1/clustersize[k] values
//		meanl.assign(numk,0);
//		meana.assign(numk,0);
//		meanb.assign(numk,0);
//		
//		{for( int k = 0; k < numk; k++ )
//		{
//			if( clustersize[k] <= 0 ) clustersize[k] = 1;
//			inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later 
//			meanl[k] = sigmal[k]*inv[k];
//			meana[k] = sigmaa[k]*inv[k];
//			meanb[k] = sigmab[k]*inv[k];
//
//		}}
//		
//		//出方差
//		stddist.assign(numk,0);
//		if(flag)
//		{int ind(0);
//		for( int r = 0; r < m_height; r++ )
//		{
//			for( int c = 0; c < m_width; c++ )
//			{
//				stddist[klabels[ind]] += (m_lvec[ind] - meanl[klabels[ind]])*(m_lvec[ind] - meanl[klabels[ind]])
//										+(m_avec[ind] - meana[klabels[ind]])*(m_avec[ind] - meana[klabels[ind]])
//										+(m_bvec[ind] - meanb[klabels[ind]])*(m_bvec[ind] - meanb[klabels[ind]]);
//				ind++;
//			}
//		}
//		for( int k = 0; k < numk; k++ )
//		{
//			stddist[k] = sqrt(stddist[k] * inv[k] /3);//因为点数为3倍
//
//		}
//		}
//
//	
//		int new_numk = numk;//新的种子数
//		{for( int k = 0; k < numk; k++ )
//		{
//			if((stddist[k] < 3)|| !flag)//不需要再分割
//			{
//				kseedsl[k] = meanl[k];
//				kseedsa[k] = meana[k];
//				kseedsb[k] = meanb[k];
//				kseedsx[k] = sigmax[k]*inv[k];
//				kseedsy[k] = sigmay[k]*inv[k];
//				//------------------------------------
//				//edgesum[k] *= inv[k];
//				//------------------------------------
//			}else
//			{
//				new_numk++;
//				kseedsl[k] = m_lvec[eminlabel[k]];
//				kseedsa[k] = m_avec[eminlabel[k]];
//				kseedsb[k] = m_bvec[eminlabel[k]];
//				kseedsx[k] = eminlabel[k] % m_width;
//				kseedsy[k] = eminlabel[k] / m_width;
//				kseedsl.push_back(m_lvec[emaxlabel[k]]);
//				kseedsa.push_back(m_avec[emaxlabel[k]]);
//				kseedsb.push_back(m_bvec[emaxlabel[k]]);
//				kseedsx.push_back(emaxlabel[k] % m_width);
//				kseedsy.push_back(emaxlabel[k] / m_width);
//				labelC[k] = k;
//				labelC.push_back(k);
//			}
//		}}
//		numk = new_numk;
//	}
//}



//===========================================================================
///	saveSuperpixelLabels
///
///	Save labels in raster scan order.
//===========================================================================
void SLIC::saveSuperpixelLabels(
	const int*&					labels,
	const int&					width,
	const int&					height,
	const string&				filename,
	const string&				path) 
{
#ifdef WINDOWS
        char fname[256];
        char extn[256];
        _splitpath(filename.c_str(), NULL, NULL, fname, extn);
        string temp = fname;
        string finalpath = path + temp + string(".dat");
#else
        string nameandextension = filename;
        size_t pos = filename.find_last_of("/");
        if(pos != string::npos)//if a slash is found, then take the filename with extension
        {
            nameandextension = filename.substr(pos+1);
        }
        string newname = nameandextension.replace(nameandextension.rfind(".")+1, 3, "dat");//find the position of the dot and replace the 3 characters following it.
        string finalpath = path+newname;
#endif

        int sz = width*height;
	ofstream outfile;
	outfile.open(finalpath.c_str(), ios::binary);
	for( int i = 0; i < sz; i++ )
	{
		outfile.write((const char*)&labels[i], sizeof(int));
	}
	outfile.close();
}




//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLIC::enforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int					width,
	const int					height,
	int*&						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	//nlabels.resize(sz, -1);
	for( int i = 0; i < sz; i++ ) nlabels[i] = -1;
	int label(0);
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	int oindex(0);
	int adjlabel(0);//adjacent label
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 4)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;
}







//===========================================================================
// setModel
// 2015/3/20
// set the slic method 
// 0:use the old method
// 1;use the new method
//===========================================================================
void SLIC::setModel(int model)
{
	m_model = model;
}
////===========================================================================
//// reCutBadRegion
//// 2015/3/20
//// deal with Bad Region
////===========================================================================
//void SLIC::reCutBadRegion(
//		vector<double>&				kseedsl,
//		vector<double>&				kseedsa,
//		vector<double>&				kseedsb,
//		vector<double>&				kseedsx,
//		vector<double>&				kseedsy,
//		int*&						klabels,
//		int							numlabels,
//		const int&					STEP,
//        const vector<double>&		edgemag,
//		const double&				M)
//{
//	int sz = m_width*m_height;
//	int numk = numlabels;
//	//----------------
//	int offset = STEP;
//        if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
//	//----------------
//	
//	
//	
//	vector<int> labelC(numk,-1);
//
//	
//	//-----------------------------------------------------------------
//	// Recalculate the centroid and store in the seed values
//	//-----------------------------------------------------------------
//	//instead of reassigning memory on each iteration, just reset.
//	
//	vector<double> clustersize(numk, 0);
//		
//
//	vector<double> sigmal(numk, 0);
//	vector<double> sigmaa(numk, 0);
//	vector<double> sigmab(numk, 0);
//	vector<double> sigmax(numk, 0);
//	vector<double> sigmay(numk, 0);
//	sigmal.assign(numk, 0);
//	sigmaa.assign(numk, 0);
//	sigmab.assign(numk, 0);
//	sigmax.assign(numk, 0);
//	sigmay.assign(numk, 0);
//	clustersize.assign(numk, 0);
//
//	//新东西
//	//something new
//	vector<double> meanl(numk,0);
//	vector<double> meana(numk,0);
//	vector<double> meanb(numk,0);
//	vector<double> stddist(numk,0);
//	//vector<double> edgemax(numk, 0);
//	//vector<double> edgemin(numk,0);
//	//edgemax.assign(numk,0);//最大梯度
//	//edgemin.assign(numk,DBL_MAX);//最小梯度
//	//vector<int> emaxlabel(numk, 0);//对应像素
//	//vector<int> eminlabel(numk,0);
//	/*emaxlabel.assign(numk,-1);
//	eminlabel.assign(numk,-1);*/
//	vector<double> normMax(numk,0);
//	vector<double> normMin(numk,DBL_MAX);
//	vector<int> nMaxLabel(numk,-1);
//	vector<int> nMinLabel(numk,-1);
//	//------------------------------------
//	//edgesum.assign(numk, 0);
//	//------------------------------------
//
//	{int ind(0);
//	for( int r = 0; r < m_height; r++ )
//	{
//		for( int c = 0; c < m_width; c++ )
//		{
//			sigmal[klabels[ind]] += m_lvec[ind];
//			sigmaa[klabels[ind]] += m_avec[ind];
//			sigmab[klabels[ind]] += m_bvec[ind];
//			sigmax[klabels[ind]] += c;
//			sigmay[klabels[ind]] += r;
//
//			//------------------------------------
//			//edgesum[klabels[ind]] += edgemag[ind];
//			//------------------------------------
//			clustersize[klabels[ind]] += 1.0;
//				
//			//if(edgemax[klabels[ind]] < edgemag[ind])
//			//{
//			//	edgemax[klabels[ind]] = edgemag[ind];//该种子点类的最大梯度
//			//	emaxlabel[klabels[ind]] = ind;
//			//}
//			//if(edgemin[klabels[ind]] > edgemag[ind])
//			//{
//			//	edgemin[klabels[ind]] = edgemag[ind];
//			//	eminlabel[klabels[ind]] = ind;
//			//}
//			
//			//---------------------------
//			// find min gray and max gray point
//			//-------------------------------------
//	/*		vector<double> average = calAverage(ind);
//			double norm = average[0]*average[0]+average[1]*average[1]+average[2]*average[2];
//			if(norm > normMax[klabels[ind]])
//			{
//				normMax[klabels[ind]] = norm;
//				nMaxLabel[klabels[ind]] = ind;
//			}
//			if(norm < normMin[klabels[ind]])
//			{
//				normMin[klabels[ind]] = norm;
//				nMinLabel[klabels[ind]] = ind;
//			}*/
//			ind++;
//		}
//	}}
//	//出均值
//	vector<double> inv(numk, 0);//to store 1/clustersize[k] values
//	meanl.assign(numk,0);
//	meana.assign(numk,0);
//	meanb.assign(numk,0);
//		
//	{for( int k = 0; k < numk; k++ )
//	{
//		if( clustersize[k] <= 0 ) clustersize[k] = 1;
//		inv[k] = 1.0/double(clustersize[k]);//computing inverse now to multiply, than divide later 
//		meanl[k] = sigmal[k]*inv[k];
//		meana[k] = sigmaa[k]*inv[k];
//		meanb[k] = sigmab[k]*inv[k];
//
//	}}
//		
//	//出方差
//	stddist.assign(numk,0);
//	{int ind(0);
//	for( int r = 0; r < m_height; r++ )
//	{
//		for( int c = 0; c < m_width; c++ )
//		{
//			stddist[klabels[ind]] += (m_lvec[ind] - meanl[klabels[ind]])*(m_lvec[ind] - meanl[klabels[ind]])
//									+(m_avec[ind] - meana[klabels[ind]])*(m_avec[ind] - meana[klabels[ind]])
//									+(m_bvec[ind] - meanb[klabels[ind]])*(m_bvec[ind] - meanb[klabels[ind]]);
//			ind++;
//		}
//	}
//	}
//	for( int k = 0; k < numk; k++ )
//	{
//		stddist[k] = sqrt(stddist[k] * inv[k] /3);//因为点数为3倍
//
//	}
//	
//
//	//用阈值分割来计算种子
//	vector<double> sigmal1(numk, 0);;//low average gray
//	vector<double> sigmal2(numk, 0);;//high 
//	vector<double> sigmaa1(numk, 0);;//low average gray
//	vector<double> sigmaa2(numk, 0);;//
//	vector<double> sigmab1(numk, 0);;//low average gray
//	vector<double> sigmab2(numk, 0);;//
//	vector<double> sigmax1(numk, 0);
//	vector<double> sigmax2(numk, 0);
//	vector<double> sigmay1(numk, 0);
//	vector<double> sigmay2(numk, 0);
//	vector<double> clustersize1(numk, 0);
//	vector<double> clustersize2(numk, 0);
//	{int ind(0);
//	for( int r = 0; r < m_height; r++ )
//	{
//		for( int c = 0; c < m_width; c++ )
//		{
//			if(stddist[klabels[ind]]  > D_VAR)
//			{
//				if(RGB2Gray(m_bvec[ind],m_avec[ind],m_lvec[ind]) 
//					< RGB2Gray(meanb[klabels[ind]],meana[klabels[ind]],meanl[klabels[ind]]))
//				{
//					sigmal1[klabels[ind]] += m_lvec[ind];
//					sigmaa1[klabels[ind]] += m_avec[ind];
//					sigmab1[klabels[ind]] += m_bvec[ind];
//					sigmax1[klabels[ind]] += c;
//					sigmay1[klabels[ind]] += r;
//					clustersize1[klabels[ind]]++;
//				}else
//				{
//					sigmal2[klabels[ind]] += m_lvec[ind];
//					sigmaa2[klabels[ind]] += m_avec[ind];
//					sigmab2[klabels[ind]] += m_bvec[ind];
//					sigmax2[klabels[ind]] += c;
//					sigmay2[klabels[ind]] += r;
//					clustersize2[klabels[ind]]++;
//				}
//				
//			}
//			ind++;
//			
//		}
//	}
//	
//	}
//	double inv1;
//	for( int k = 0; k < numk; k++ )
//	{
//		if(stddist[k] > D_VAR)
//		{
//			if( clustersize1[k] <= 0 ) clustersize1[k] = 1;
//			inv1 = 1.0/double(clustersize1[k]);//computing inverse now to multiply, than divide later 
//			sigmal1[k] = sigmal1[k]*inv1;
//			sigmaa1[k] = sigmaa1[k]*inv1;
//			sigmab1[k] = sigmab1[k]*inv1;
//			sigmax1[k] = sigmax1[k]*inv1;
//			sigmay1[k] = sigmay1[k]*inv1;
//
//			if( clustersize2[k] <= 0 ) clustersize2[k] = 1;
//			inv1 = 1.0/double(clustersize2[k]);//computing inverse now to multiply, than divide later 
//			sigmal2[k] = sigmal2[k]*inv1;
//			sigmaa2[k] = sigmaa2[k]*inv1;
//			sigmab2[k] = sigmab2[k]*inv1;
//			sigmax2[k] = sigmax2[k]*inv1;
//			sigmay2[k] = sigmay2[k]*inv1;
//		}
//	}
//	
//
//	//
//	int new_numk = numk;//新的种子数
//	kseedsl.assign(numk,0);
//	kseedsa.assign(numk,0);
//	kseedsb.assign(numk,0);
//	kseedsx.assign(numk,0);
//	kseedsy.assign(numk,0);
//	{for( int k = 0; k < numk; k++ )
//	{
//		if(stddist[k] > D_VAR)//再分割
//		{
//			new_numk++;
//			kseedsl[k] = sigmal1[k];
//			kseedsa[k] = sigmaa1[k];
//			kseedsb[k] = sigmab1[k];
//			kseedsx[k] = sigmax1[k];
//			kseedsy[k] = sigmay1[k];
//			kseedsl.push_back(sigmal2[k]);
//			kseedsa.push_back(sigmaa2[k]);
//			kseedsb.push_back(sigmab2[k]);
//			kseedsx.push_back(sigmax2[k]);
//			kseedsy.push_back(sigmay2[k]);
//			labelC[k] = k;
//			labelC.push_back(k);
//		}else
//		{
//			kseedsl[k] = meanl[k];
//			kseedsa[k] = meana[k];
//			kseedsb[k] = meanb[k];
//			kseedsx[k] = sigmax[k]*inv[k];
//			kseedsy[k] = sigmay[k]*inv[k];
//		}
//	}}
//	numk = new_numk;
//	outSeeds(10,kseedsl,kseedsx,kseedsy);
//	double invwt = 1.0/((STEP/M)*(STEP/M));
//	vector<double> distvec(sz, DBL_MAX);
//	
//
//	int x1, y1, x2, y2;
//	double l, a, b;
//	double dist;
//	double distxy;
//	
//	distvec.assign(sz, DBL_MAX);
//	for( int n = 0; n < numk; n++ )
//	{
//		if(labelC[n] == -1)//加种子后筛选需要充分的区域
//			continue;
//        y1 = max(0.0,			kseedsy[n]-offset);
//        y2 = min((double)m_height,	kseedsy[n]+offset);
//        x1 = max(0.0,			kseedsx[n]-offset);
//        x2 = min((double)m_width,	kseedsx[n]+offset);
//
//
//		for( int y = y1; y < y2; y++ )
//		{
//			for( int x = x1; x < x2; x++ )
//			{
//				int i = y*m_width + x;
//				if(labelC[n] != klabels[i]) continue;   //如果不是该区域的点，则抛弃
//
//				l = m_lvec[i];
//				a = m_avec[i];
//				b = m_bvec[i];
//
//				dist =			(l - kseedsl[n])*(l - kseedsl[n]) +
//								(a - kseedsa[n])*(a - kseedsa[n]) +
//								(b - kseedsb[n])*(b - kseedsb[n]);
//
//				distxy =		(x - kseedsx[n])*(x - kseedsx[n]) +
//								(y - kseedsy[n])*(y - kseedsy[n]);
//					
//				//------------------------------------------------------------------------
//				dist += distxy*invwt;//dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
//				//------------------------------------------------------------------------
//				if( dist < distvec[i] )
//				{
//					distvec[i] = dist;
//					klabels[i]  = n;
//				}
//					
//			}
//		}
//	}
//}
/****************************************************
/ use for cal the average in 3x3 field
/
/
/**************************************************/
//vector<double> SLIC::calAverage(int index)
//{
//	vector<double> AV;
//	int i = index / m_width;
//	int j = index % m_width;
//	int sz = m_width * m_height;
//	int dx[] = {-1,-1,-1,0,0,0,1,1,1};
//	int dy[] = {-1,0,1,-1,0,1,-1,0,1};
//	AV.assign(3,0);
//	int count = 0;
//	for (int k = 0; k < 9; k++)
//	{
//		int ik = i + dx[k];
//		int jk = j + dy[k];
//		int indexk = ik*m_width + jk;
//		if((indexk >= 0) && (indexk < sz))
//		{
//			AV[0] += m_lvec[indexk];
//			AV[1] += m_avec[indexk];
//			AV[2] += m_bvec[indexk];
//			count++;
//		}
//	}
//	for(int k = 0; k < 3; k++)
//		AV[k] /= count;
//	return AV;
//}
//

/***************************************************************************
*输出种子以供调试
***************************************************/
void SLIC::outSeeds(
		int							n,
		vector<vector<double> >&				kseeds,
		vector<vector<double> >&				kseedsxy)
{
	string str = "kseeds = [";
	string filename = "E:\\source\\Matlab\\SLIC\\Seed";
	std::stringstream ss;
	std::string str1;
	ss << n;
	ss >> str1;
	filename += str1;
	filename += ".m";
	ss.clear();
	str1.clear();

	int length = kseeds.size();


	for (int i = 0; i < length; i++)
	{
		str +=" struct('gray',";
		std::stringstream ss;
		std::string str1;
		ss << kseeds[i][0];
		ss >> str1;
		str += str1;
		str += ",'x',";
		str1.clear();
		ss.clear();
		ss << kseedsxy[i][1];
		ss >> str1;
		str += str1;
		str += ",'y',";
		ss.clear();
		str1.clear();
		ss << kseedsxy[i][0];
		ss >> str1;
		str += str1;
		str += ")";
	}
	
	str += "];";
	ofstream outFile(filename);
	outFile << str;
	outFile.close();
}

double SLIC::RGB2Gray(double red,double green,double blue){
	return red*0.299 + green * 0.587 + blue * 0.114;
}
//添加元素
void SLIC::push_vec(
	const vector<vector<double> >&	data
	) 
{
	if (data.size()) {
		int len = data[0].size();
		int oriLen = m_depth;
		m_depth += len;
		for (size_t i = 0; i < data.size(); i++) {
			m_data[i].resize(m_depth + len);
			for (int j = 0; j < len; j++) {
				m_data[i][j + oriLen] = data[i][j];
			}
		}
	}
}
/*=================================================================*/
//  main function for all SLIC-superpixel calculation
////////////////////////////////////////////////////////////////////////
void SLIC::doSLIC(
	const unsigned int*			ubuff,
	const int					width,
	const int					height,
	int*&						klabels,
	int&						numlabels,
	const int&					K,
	const double&               compactness,
	const vector<vector<double> >& exData)
{
	//calculate intital superpixels size through number of superpixels K
	const int superpixelsize = (int)(0.5 + double(width*height) / double(K));
	const int STEP = int(sqrt(double(superpixelsize)) + 0.5);

	if (!m_data.size()) {
		loadImage(ubuff, width, height);
	}

	int sz = m_width*m_height;
	for (int s = 0; s < sz; s++) klabels[s] = -1;
	
	//calculatee initial seeds
	bool perturbSeeds(true);//perturb seeds is not absolutely necessary, one can set this flag to false,可以优化初始种子点
	vector<double> edgemag(0);
	if (perturbSeeds) detectLabEdges(edgemag);

	vector<vector<double> > kseeds(0);
	vector<vector<double> > kseedsxy(0);
	m_depth = m_data[0].size(); //vector length();

	//if there are more data want to add to m_data,
	if (exData.size())
		push_vec(exData);
	getSeeds_ForGivenStepSize(kseeds, kseedsxy, STEP, perturbSeeds, edgemag);

	if (0 && m_model)
		//PerformSuperpixelSLICnew(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP, edgemag, compactness);
		return;
	else
		performSuperpixelSLIC(kseeds, kseedsxy, klabels, STEP, edgemag, compactness);
	numlabels = kseeds.size();

	int* nlabels = new int[sz];
	enforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, int(double(sz) / double(STEP*STEP)));
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	/*if (m_model)
	{
	reCutBadRegion(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, numlabels, STEP, edgemag, compactness);
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, double(sz) / double(STEP*STEP));
	{for (int i = 0; i < sz; i++) klabels[i] = nlabels[i]; }
	}*/
	if (nlabels) delete[] nlabels;
}
