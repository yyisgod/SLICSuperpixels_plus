
// SLICAppDlg.cpp : 实现文件
//

#include "stdafx.h"
#include "SLICApp.h"
#include "SLICAppDlg.h"
#include "afxdialogex.h"
#include "PictureHandler.h"
#include "SLIC.h"
#include "functions.h"
using namespace yFunctions;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
	enum { IDD = IDD_ABOUTBOX };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CSLICAppDlg 对话框



CSLICAppDlg::CSLICAppDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(CSLICAppDlg::IDD, pParent)
	, m_IsNew(FALSE)
	, m_nums(100)
	, m_m(10)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CSLICAppDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_CHECK_NEW, m_IsNew);
	DDX_Text(pDX, IDC_EDIT_NUM, m_nums);
	DDX_Text(pDX, IDC_EDIT_M, m_m);
}

BEGIN_MESSAGE_MAP(CSLICAppDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON_OPEN, &CSLICAppDlg::OnBnClickedButtonOpen)
END_MESSAGE_MAP()


// CSLICAppDlg 消息处理程序

BOOL CSLICAppDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO:  在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CSLICAppDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CSLICAppDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CSLICAppDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CSLICAppDlg::OnBnClickedButtonOpen()
{
	_CrtSetBreakAlloc(489);
	PictureHandler picHand;
	vector<string> picvec(0);
	picvec.resize(0);
	GetPictures(picvec);//user chooses one or more pictures
	string saveLocation = "C:\\rktemp\\";
	BrowseForFolder(saveLocation);

	int numPics(picvec.size());
	int model = m_IsNew ? 1 : 0;
	for (int k = 0; k < numPics; k++)
	{
		UINT* img = NULL;
		int width(0);
		int height(0);

		picHand.GetPictureBuffer(picvec[k], img, width, height);
		int sz = width*height;
		//---------------------------------------------------------
		if (m_nums < 20 || m_nums > sz / 4) m_nums = sz / 200;//i.e the default size of the superpixel is 200 pixels
		if (m_m < 1.0 || m_m > 80.0) m_m = 20.0;
		//---------------------------------------------------------
		int* labels = new int[sz];
		int numlabels(0);
		SLIC slic;
		slic.setModel(model);
		slic.loadImage(img, width, height);
		vector<vector<double> > exData;
		slicfeature(slic.getData(), height, width, exData);
		slic.doSLIC(img, width, height, labels, numlabels,m_nums, m_m,exData);
		slic.drawContoursAroundSegments(img, labels, width, height, 0);
		delete[] labels;

		string addPath = "_C";//append outfilename
		std::stringstream ss;
		std::string str1;
		ss << m_nums;
		ss >> str1;
		addPath += str1;
		addPath += "_M";
		str1.clear();
		ss.clear();
		//std::stringstream ss1;
		//std::string str2;
		ss << m_m;
		ss >> str1;
		addPath += str1;
		str1.clear();
		ss.clear();
		ss << model;
		ss >> str1;
		addPath += "_m";
		addPath += str1;

		picHand.SavePicture(img, width, height, picvec[k], saveLocation, 1, addPath);// 0 is for BMP and 1 for JPEG)
		if (img) delete[] img;
	}
	AfxMessageBox(L"Done!", 0, 0);
}


//=================================================================================
///	GetPictures
///
///	This function collects all the pictures the user chooses into a vector.
//=================================================================================
void CSLICAppDlg::GetPictures(vector<string>& picvec)
{
	CFileDialog cfd(TRUE, NULL, NULL, OFN_OVERWRITEPROMPT, L"*.*|*.*|", NULL);
	cfd.m_ofn.Flags |= OFN_ALLOWMULTISELECT;

	//cfd.PostMessage(WM_COMMAND, 40964, NULL);

	CString strFileNames;
	cfd.m_ofn.lpstrFile = strFileNames.GetBuffer(2048);
	cfd.m_ofn.nMaxFile = 2048;

	BOOL bResult = cfd.DoModal() == IDOK ? TRUE : FALSE;
	strFileNames.ReleaseBuffer();

	//if(cfd.DoModal() == IDOK)
	if (bResult)
	{
		POSITION pos = cfd.GetStartPosition();
		while (pos)
		{
			CString imgFile = cfd.GetNextPathName(pos);
			PictureHandler ph;
			string name = ph.Wide2Narrow(imgFile.GetString());
			picvec.push_back(name);
		}
	}
	else return;
}

//===========================================================================
///	BrowseForFolder
///
///	The main function
//===========================================================================
bool CSLICAppDlg::BrowseForFolder(string& folderpath)
{
	IMalloc* pMalloc = 0;
	if (::SHGetMalloc(&pMalloc) != NOERROR)
		return false;

	BROWSEINFO bi;
	memset(&bi, 0, sizeof(bi));

	bi.hwndOwner = m_hWnd;
	bi.lpszTitle = L"Please select a folder and press 'OK'.";

	LPITEMIDLIST pIDL = ::SHBrowseForFolder(&bi);
	if (pIDL == NULL)
		return false;

	TCHAR buffer[_MAX_PATH];
	if (::SHGetPathFromIDList(pIDL, buffer) == 0)
		return false;
	PictureHandler pichand;
	folderpath = pichand.Wide2Narrow(buffer);
	folderpath.append("\\");
	return true;
}
