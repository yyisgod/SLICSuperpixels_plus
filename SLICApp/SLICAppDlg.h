
// SLICAppDlg.h : 头文件
//
#include <vector>
#include <string>
using namespace std;
#pragma once


// CSLICAppDlg 对话框
class CSLICAppDlg : public CDialogEx
{
// 构造
public:
	CSLICAppDlg(CWnd* pParent = NULL);	// 标准构造函数

// 对话框数据
	enum { IDD = IDD_SLICAPP_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonOpen();

private:

	bool BrowseForFolder(string& folderpath);
	void GetPictures(vector<string>& picvec);
public:
	BOOL m_IsNew;
	// 分类数目
	int m_nums;
	// M参数
	double m_m;
};
