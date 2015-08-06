// SLIC.h: interface for the SLIC class.
//===========================================================================
// This code implements the superpixel method described in:
//
// Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk,
// "SLIC Superpixels",
// EPFL Technical Report no. 149300, June 2010.
//===========================================================================
//	Copyright (c) 2012 Radhakrishna Achanta [EPFL]. All rights reserved.
//===========================================================================
//////////////////////////////////////////////////////////////////////

#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_


#include <vector>
#include <string>
#include <algorithm>
using namespace std;
#define D_VAR 15.7

class SLIC  
{
public:
	SLIC();
	virtual ~SLIC();
	//============================================================================
	// Superpixel segmentation for a given step size (superpixel size ~= step*step)
	//============================================================================
        void DoSuperpixelSegmentation_ForGivenSuperpixelSize(
        const unsigned int*                            ubuff,//Each 32 bit unsigned int contains ARGB pixel values.
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
                const int&					superpixelsize,
                const double&                                   compactness);
	//============================================================================
	// Superpixel segmentation for a given number of superpixels
	//============================================================================
        void DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(
        const unsigned int*                             ubuff,
		const int					width,
		const int					height,
		int*&						klabels,
		int&						numlabels,
                const int&					K,//required number of superpixels
                const double&                                   compactness);//10-20 is a good value for CIELAB space
	//============================================================================
	// Supervoxel segmentation for a given step size (supervoxel size ~= step*step*step)
	//============================================================================
	void DoSupervoxelSegmentation(
		unsigned int**&		ubuffvec,
		const int&					width,
		const int&					height,
		const int&					depth,
		int**&						klabels,
		int&						numlabels,
                const int&					supervoxelsize,
                const double&                                   compactness);
	//============================================================================
	// Save superpixel labels in a text file in raster scan order
	//============================================================================
	void SaveSuperpixelLabels(
		const int*&					labels,
		const int&					width,
		const int&					height,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Save supervoxel labels in a text file in raster scan, depth order
	//============================================================================
	void SaveSupervoxelLabels(
		const int**&				labels,
		const int&					width,
		const int&					height,
		const int&					depth,
		const string&				filename,
		const string&				path);
	//============================================================================
	// Function to draw boundaries around superpixels of a given 'color'.
	// Can also be used to draw boundaries around supervoxels, i.e layer by layer.
	//============================================================================
	void DrawContoursAroundSegments(
		unsigned int*&				segmentedImage,
		int*&						labels,
		const int&					width,
		const int&					height,
		const unsigned int&			color );

private:
	//============================================================================
	// The main SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLIC(
		vector< vector<double> >&				kseeds,
		vector<vector<double> >&				kseedsxy,
		int*&						klabels,
		const int&					STEP,
                const vector<double>&		edgemag,
		const double&				m = 10.0);
	//============================================================================
	// The new SLIC algorithm for generating superpixels
	//============================================================================
	void PerformSuperpixelSLICnew(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*&						klabels,
		const int&					STEP,
                const vector<double>&		edgemag,
		const double&				m = 10.0);
	
	//============================================================================
	// Pick seeds for superpixels when step size of superpixels is given.
	//============================================================================
	void GetSeeds_ForGivenStepSize(
		vector<vector<double> >&				kseeds,
		vector<vector<double> >&				kseedsxy,
		const int&					STEP,
		const bool&					perturbseeds,
		const vector<double>&		edgemag);
	
	//============================================================================
	// Move the superpixel seeds to low gradient positions to avoid putting seeds
	// at region boundaries.
	//============================================================================
	void PerturbSeeds(
		vector<vector<double> >&				kseeds,
		vector< vector<double> >&				kseedsxy,
		const vector<double>&		edges);
	//============================================================================
	// Detect color edges, to help PerturbSeeds()
	//============================================================================
	void DetectLabEdges( vector<double>&				edges);
	//============================================================================
	// sRGB to XYZ conversion; helper for RGB2LAB()
	//============================================================================
	void RGB2XYZ(
		const int&					sR,
		const int&					sG,
		const int&					sB,
		double&						X,
		double&						Y,
		double&						Z);
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
	void DoRGBtoLABConversion(const unsigned int*&		ubuff);
	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int*					labels,
		const int					width,
		const int					height,
		int*&						nlabels,//input labels that need to be corrected to remove stray labels
		int&						numlabels,//the number of labels changes in the end if segments are removed
		const int&					K); //the number of superpixels desired by the user

	//============================================================================
	// push more vector back the m_data;
	//============================================================================
	void push_vec(
		const vector<vector<double> >&	data
		);
	//================================================================================
	// This field is used by yy self
	//================================================================================
public:
	void setModel(int model);
	void reCutBadRegion(
		vector<double>&				kseedsl,
		vector<double>&				kseedsa,
		vector<double>&				kseedsb,
		vector<double>&				kseedsx,
		vector<double>&				kseedsy,
		int*&						klabels,
		int							numlabels,
		const int&					STEP,
        const vector<double>&		edgemag,
		const double&				m = 10.0);
	
	void outSeeds(
		int							n,
		vector<vector<double> >&				kseedsl,
		vector<vector<double> >&				kseedsxy);

private:
	//===================
	//calculate the average of vector l a b
	//return vector [l a b]
	//===========================================
	vector<double> calAverage(int index);
	double RGB2Gray(double red,double green,double blue);

public:
	void SLIC::DoSLIC(
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

	vector<vector<double> >					m_data;

	int										m_model;   ///use for control the model(such as new method or old method , 0 = old, 1 = new
};

#endif // !defined(_SLIC_H_INCLUDED_)
