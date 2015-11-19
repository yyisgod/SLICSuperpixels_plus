// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the superpixel method described in:
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
// "SLIC Superpixels",
// EPFL Technical Report no. 149300, June 2010.
//
// Yy added a improvement and reconfiguration the code at 2015.
//===========================================================================
//	Copyright (c) 2012 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
//////////////////////////////////////////////////////////////////////

#ifndef _SLIC_H
#define _SLIC_H

#include <vector>
#include <string>
#include <algorithm>

#define __DEBUG__ 0
using namespace std;

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();
	
	//============================================================================
	// return m_data
	//============================================================================
	vector<vector<double> >& getData();
	
	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void saveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				path);

	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void drawContoursAroundSegments(
		unsigned int*&				segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );
	//============================================================================
	//Load image (unsigned int*&) -> m_data(vector)
	//============================================================================
	void loadImage(
		const unsigned int*	ubuff,
		const int					width,
		const int					height);

private:
	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void performSuperpixelSLIC(
		vector< vector<double> >&	kseeds,
		vector<vector<double> >&	kseedsxy,
		int*&						klabels,
		const int&					STEP,
        const vector<double>&		edgemag,
		const double&				m = 10.0);
	//================================================================================
	// ¥¶¿ÌµÙªµ«¯”Ú
	//================================================================================
	void reCutBadRegion(
		vector<vector<double> >&	kseeds,
		vector<vector<double> >&	kseedsxy,
		int*						klabels,
		int							numlabels,
		const int&					STEP,
		const vector<double>&		edgemag,
		const double&				m = 10.0);
	
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void getSeeds_ForGivenStepSize(
		vector<vector<double> >&				kseeds,
		vector<vector<double> >&				kseedsxy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edgemag);
	
	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void perturbSeeds(
		vector<vector<double> >&				kseeds,
		vector< vector<double> >&				kseedsxy,
		const vector<double>&		edges);

	void RGB2XYZ(const int & sR, const int & sG, const int & sB, double & X, double & Y, double & Z);

	//============================================================================
	// sRGB to CIELAB conversion (uses RGB2XYZ function)
	//============================================================================
	void RGB2LAB(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						lval,
		double&						aval,
		double&						bval);
	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void doRGBtoLABConversion(const unsigned int*&		ubuff);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void detectEdges( vector<double>&edges);

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void enforceLabelConnectivity(
		const int*					labels,
		const int					width,
		const int					height,
		int*						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

	void enforceLabelConnectivityNew(const int * labels, const int width, const int height, int * nlabels, int & numlabels, const int & K);

	//============================================================================
	// push more vector back the m_data;
	//============================================================================
	void push_vec(
		const vector<vector<double> >&	data
		);

	//============================================================================
	// through RGB calculate Gray Level;
	//============================================================================
	double RGB2Gray(double red, double green, double blue);

	//============================================================================
	// out seeds data for debug
	//============================================================================
	void outSeeds(
		int							n,
		vector<vector<double> >&	kseedsl,
		vector<vector<double> >&	kseedsxy);
public:
	//============================================================================
	// set m_model for select old/new method.
	//============================================================================
	void setModel(int model, double std = 0);

	//============================================================================
	// the entry for SLIC alogrithm
	//============================================================================
	void SLIC::doSLIC(
		const unsigned int*			ubuff,
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
		const int&					K,
		const double&               compactness,
		const vector<vector<double> >& exData = vector<vector<double> >());

private:
	int										m_width;
	int										m_height;
	int										m_depth;

	vector<vector<double> >					m_data;		//perserve image data, one line for one pixel's feature vector

	//use for control the model(such as new method or old method 
	//3bit: ABC, A = 0/1:old/new method. BC = 00(Lab),01(RGB),10(input 24bit,want Gray),11(Gray)
	int										m_model;

	double									D_VAR;	//use for judge bad region
													
};
#endif 
