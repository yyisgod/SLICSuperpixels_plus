
// SLICAppDlg.h : ͷ�ļ�
//
#include <vector>
#include <string>
using namespace std;
#pragma once


// CSLICAppDlg �Ի���
class CSLICAppDlg : public CDialogEx
{
// ����
public:
	CSLICAppDlg(CWnd* pParent = NULL);	// ��׼���캯��

// �Ի�������
	enum { IDD = IDD_SLICAPP_DIALOG };

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV ֧��


// ʵ��
protected:
	HICON m_hIcon;

	// ���ɵ���Ϣӳ�亯��
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
	// ������Ŀ
	int m_nums;
	// M����
	double m_m;
};
