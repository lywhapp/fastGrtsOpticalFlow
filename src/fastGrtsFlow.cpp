/*
 * @Author: Yuanwei Li, lywhbbj@126.com, TortoiseShell 
 * @Date: 2020-11-02 15:45:56 
 * @Last Modified by: Yuanwei Li, lywhbbj@126.com, TortoiseShell
 * @Last Modified time: 2020-11-02 18:30:32
 */


//#include "stdafx.h"
#include "omp.h"
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <numeric>
#include <string>
#include <time.h> 

#include <opencv2/opencv.hpp>
// #include <opencv2/video/tracking.hpp>
//#include "opencv2/video/tracking_c.h"
// #include <opencv2/highgui/highgui.hpp>
//#include "opencv2/highgui/highgui_c.h"
// #include <opencv2\imgproc\imgproc.hpp>
//#include <xmmintrin.h>

#include "ANN/PatchMatch.h"

//#define DebugCode
#define WaitTime 10000000
int PointCount[4]={0};

#define cubesize_global 3
#define PyramidLevelNum  3
#define SmoothSize   19

//===========================================
#define  NoExternalCorrection
#define myType float  
#define TypeCache int
#define WindowSizeOf3DGrdts 5 
#define PydFactor 2  

#define KernelWeight_Edge 2
#define KernelWeight_Center 8
#define KernelCenterAdd (KernelWeight_Center-KernelWeight_Edge*KernelWeight_Edge)
#define KernelNormFactor (KernelWeight_Center+4*KernelWeight_Edge+4)/(4+2*KernelWeight_Edge)

#define LevelsGradient3D  1
#define Use2dTo3d
#define  Scale3d
const float KernelFor3DGradient[5]=//{0.0,-1.0,0.0,1.0,0.0};
{0.2222,   -1.7778,         0.0,    1.7778   -0.2222};



struct CacheTermXYZ
{
	TypeCache LineData[8];
	TypeCache LocalRecord[WindowSizeOf3DGrdts*8];
	int LocalEndIndex;
	int MatchIndex;
	const int* grtXYZ2;
};

struct ImagePyramidLevel
{
	int w;
	int h;
	myType* Gradient3DMat;
	uchar* SrcImg[10];
	const uchar* SrcImgReordered[10];

	int* GradientY[10];
	int* GradientYReordered[10];
	int* GradientZ[10];
	int* GradientZReordered[10];

	int* ResultOrientationX;
	int* ResultOrientationY;
	int* Weight;
	int* Magnitude;

	
	myType* OpticalFlow_Ori;
	myType* TermsXYZ;
	myType* OpticalFlow_Warped;
	myType* OpticalInterFlow;
	float* MeanRowFlow;
	int* GradientXYZ[10];
	int* GradientXYZReordered[10];

	
	BITMAP* rgbImg[10];
	BITMAP* ann[10];
	BITMAP* rgbImgReordered[10];
};
ImagePyramidLevel PyramidLevel[PyramidLevelNum];

struct OpticalFlowResult
{
	float* OrientationX;
	float* OrientationY;
	int* Weight;
	int* InterOrientationX;
	int* InterOrientationY;
	int* InterWeight;
	int* InterMagnitude;
};
OpticalFlowResult myFinalOpticalFlowResult;

inline int FastArcTan(float x)
{
	if(x>1.0)
	{
		x=1/x;
		return 90-(int)(45.0*x - x*(fabs(x) - 1)*(14.020277+ 3.79871*fabs(x)));
	}
	else
		if(x>=-1.0)
		{
			return (int)(45.0*x - x*(fabs(x) - 1)*(14.020277+ 3.79871*fabs(x)));
		}
		else
		{
			x=1/x;
			return -90-(int)(45.0*x - x*(fabs(x) - 1)*(14.020277+ 3.79871*fabs(x)));
		}
}

static inline float arccos_radians(float x) {
	union { float f; unsigned int b; } z;
	z.f=x;
	bool x_is_negative = (z.b&0x80000000);
	z.b&=0x7fffffff;
	float s=sqrtf(z.f);
	float u=1.57079632679489661923f-1.05199081698724154807f*z.f;
	u=(z.b<0x3f400000)?u:2.f*sqrtf(1.f-s);
	return x_is_negative?3.14159265358979323846f-u:u;
}

inline int arccos_degrees(float x) {
	union { float f; unsigned int b; } z;
	z.f=x;
	bool x_is_negative = (z.b&0x80000000);
	z.b&=0x7fffffff;
	float s=sqrtf(z.f);
	float u=1.57079632679489661923f-1.05199081698724154807f*z.f;
	u=(z.b<0x3f400000)?u:2.f*sqrtf(1.f-s);
	u=x_is_negative?3.14159265358979323846f-u:u;
	return (int)(57.29577951308232087684f*u);
}


BITMAP *create_2channels_bitmap(int w, int h) {
  BITMAP *ans = new BITMAP(w, h);
  ans->data = new unsigned char[2*4*w*h]; // always 32bit
  ans->line = new unsigned char*[h];
  ans->c=2;
  for (int y = 0; y < h; y++) 
    ans->line[y] = &ans->data[y*4*w*2];
  return ans;
}

void InitPyramidLevel(cv::Mat srcImg,ImagePyramidLevel LevelData[5],const int LevelNum,OpticalFlowResult* myFinalOpticalFlowResult=NULL)
{
	cv::Mat img;
	srcImg.copyTo(img);
	for(int i=0;i<LevelNum;i++)
	{
		LevelData->w=img.cols;
		LevelData->h=img.rows;
		LevelData->Gradient3DMat=new myType[LevelsGradient3D*3*LevelData->w*LevelData->h];
		memset(LevelData->Gradient3DMat,0,LevelsGradient3D*3*LevelData->w*LevelData->h*sizeof(myType));

		LevelData->OpticalFlow_Ori=new myType[LevelData->w*LevelData->h*2];
		memset(LevelData->OpticalFlow_Ori,0,2*LevelData->w*LevelData->h*sizeof(myType));
		LevelData->TermsXYZ=new myType[(LevelData->w+1)*(LevelData->h+1)*8];
		memset(LevelData->TermsXYZ,0,(LevelData->w+1)*(LevelData->h+1)*8*sizeof(myType));

		LevelData->OpticalFlow_Warped=new myType[LevelData->w*LevelData->h*2];
		memset(LevelData->OpticalFlow_Warped,0,2*LevelData->w*LevelData->h*sizeof(myType));

		LevelData->OpticalInterFlow=new myType[(LevelData->w+1)*(LevelData->h+1)*2];
		memset(LevelData->OpticalInterFlow,0,2*(LevelData->w+1)*(LevelData->h+1)*sizeof(myType));

		LevelData->MeanRowFlow=new float[LevelData->w*2];

		
		
		for(int j=0;j<cubesize_global;j++)
		{
			LevelData->GradientXYZ[j]=new int[3*LevelData->w*LevelData->h];
			memset(LevelData->GradientXYZ[j],0,3*LevelData->w*LevelData->h*sizeof(int));
			LevelData->SrcImg[j]=new uchar[LevelData->w*LevelData->h];
#ifdef IsProRefineWithGradients
			if(i==0||i==PyramidLevelNum-1)
				LevelData->rgbImg[j]=create_2channels_bitmap(LevelData->w,LevelData->h);
#else
			if(i==0)
				LevelData->rgbImg[j]=create_bitmap(LevelData->w,LevelData->h);
			else if(i==PyramidLevelNum-1)
				LevelData->rgbImg[j]=create_2channels_bitmap(LevelData->w,LevelData->h);
#endif
			
		}
		cv::pyrDown(img,img);
		LevelData++;
	}
}


void Fast2DGradient(cv::Mat imgSrc,int* GradientXYZ)
{
	
	const uchar* ptrImg=imgSrc.data;
	const int w=imgSrc.cols;
	const int h=imgSrc.rows;
	if(cubesize_global==3)
	{
		int dx_speed1=0,dx_speed2=0,dx_speed3=0;
		int dy_speed1=0,dy_speed2=0,dy_speed3=0;
		ptrImg+=((cubesize_global/2)*2);

		GradientXYZ+=(cubesize_global/2)*w*3;
		//GradientY+=(cubesize_global/2)*w;
		//GradientZ+=(cubesize_global/2)*w;

		for(int y=cubesize_global/2;y<h-cubesize_global/2;y++)
		{
			const uchar* ptrImgMidRright=ptrImg+y*w;
			const uchar* tempMidLeft=ptrImgMidRright-(cubesize_global/2)*2;

			dx_speed1=(*(tempMidLeft))*KernelWeight_Edge+
				*(tempMidLeft-w)+
				*(tempMidLeft+w),
				dx_speed2=(*(tempMidLeft+1))*KernelWeight_Edge+
				*(tempMidLeft+1-w)+
				*(tempMidLeft+1+w);

			dy_speed1=-*(tempMidLeft-w)+
				*(tempMidLeft+w),
				dy_speed2=-*(tempMidLeft+1-w)+
				*(tempMidLeft+1+w);

			GradientXYZ+=cubesize_global/2*3;
			for(int x=cubesize_global/2;x<w-cubesize_global/2;x++)
			{
				dx_speed3=(*(ptrImgMidRright))*KernelWeight_Edge+
					*(ptrImgMidRright-w)+
					*(ptrImgMidRright+w);
				dy_speed3=-*(ptrImgMidRright-w)+
					*(ptrImgMidRright+w);

				(*GradientXYZ++)=dx_speed3-dx_speed1;
				(*GradientXYZ++)=dy_speed1+KernelWeight_Edge*dy_speed2+dy_speed3;
				(*GradientXYZ++)=dx_speed3+KernelWeight_Edge*dx_speed2+KernelCenterAdd*(*(ptrImgMidRright-1))+dx_speed1;

				dy_speed1=dy_speed2;
				dy_speed2=dy_speed3;

				dx_speed1=dx_speed2;
				dx_speed2=dx_speed3;
				ptrImgMidRright++;
			}
			GradientXYZ+=cubesize_global/2*3;
		}
	}
	else
#if 1
	{
		int dx_speed1=0,dx_speed2=0,dx_speed3=0,dx_speed4=0,dx_speed5=0;
		int dy_speed1=0,dy_speed2=0,dy_speed3=0,dy_speed4=0,dy_speed5=0;
		ptrImg+=((cubesize_global/2)*2);

		GradientXYZ+=(cubesize_global/2)*w*3;
		//GradientY+=(cubesize_global/2)*w;
		//GradientZ+=(cubesize_global/2)*w;

		for(int y=cubesize_global/2;y<h-cubesize_global/2;y++)
		{
			const uchar* ptrImgMidRright=ptrImg+y*w;
			const uchar* tempMidLeft=ptrImgMidRright-(cubesize_global/2)*2;

			dx_speed1=*(tempMidLeft)+
				*(tempMidLeft-w)+
				*(tempMidLeft+w)+
				*(tempMidLeft-w*2)+
				*(tempMidLeft+w*2),
				dx_speed2=*(tempMidLeft+1)+
				*(tempMidLeft+1-w)+
				*(tempMidLeft+1+w)+
				*(tempMidLeft+1-2*w)+
				*(tempMidLeft+1+2*w),
				dx_speed3=*(tempMidLeft+2)+
				*(tempMidLeft+2-w)+
				*(tempMidLeft+2+w)+
				*(tempMidLeft+2-2*w)+
				*(tempMidLeft+2+2*w),
				dx_speed4=*(tempMidLeft+3)+
				*(tempMidLeft+3-w)+
				*(tempMidLeft+3+w)+
				*(tempMidLeft+3-2*w)+
				*(tempMidLeft+3+2*w);

			dy_speed1=*(tempMidLeft-w)*KernelFor3DGradient[1]+
				*(tempMidLeft+w)*KernelFor3DGradient[3]+
				*(tempMidLeft-2*w)*KernelFor3DGradient[0]+
				*(tempMidLeft+2*w)*KernelFor3DGradient[4],
				dy_speed2=*(tempMidLeft+1-w)*KernelFor3DGradient[1]+
				*(tempMidLeft+1+w)*KernelFor3DGradient[3]+
				*(tempMidLeft+1-2*w)*KernelFor3DGradient[0]+
				*(tempMidLeft+1+2*w)*KernelFor3DGradient[4];
			dy_speed3=*(tempMidLeft+2-w)*KernelFor3DGradient[1]+
				*(tempMidLeft+2+w)*KernelFor3DGradient[3]+
				*(tempMidLeft+2-2*w)*KernelFor3DGradient[0]+
				*(tempMidLeft+2+2*w)*KernelFor3DGradient[4],
				dy_speed4=*(tempMidLeft+3-w)*KernelFor3DGradient[1]+
				*(tempMidLeft+3+w)*KernelFor3DGradient[3]+
				*(tempMidLeft+3-2*w)*KernelFor3DGradient[0]+
				*(tempMidLeft+3+2*w)*KernelFor3DGradient[4];


			GradientXYZ+=cubesize_global/2*3;

			int dyInter_speed=dy_speed1+dy_speed2+dy_speed3+dy_speed4;
			int dzInter_speed=dx_speed4+dx_speed3+dx_speed2+dx_speed1;
			for(int x=cubesize_global/2;x<w-cubesize_global/2;x++)
			{
				
				dx_speed5=*(ptrImgMidRright)+
					*(ptrImgMidRright-w)+
					*(ptrImgMidRright+w)+
					*(ptrImgMidRright-2*w)+
					*(ptrImgMidRright+2*w);
				dy_speed5=-*(ptrImgMidRright-w)
					+*(ptrImgMidRright+w)
					-*(ptrImgMidRright-2*w)
					+*(ptrImgMidRright+2*w);

				const int gx=dx_speed5+dx_speed4
					-dx_speed2-dx_speed1;
				const int gy=dyInter_speed+dy_speed5;
				const int gz=dx_speed5+dzInter_speed;
				(*GradientXYZ++)=gx;
				(*GradientXYZ++)=gy;
				(*GradientXYZ++)=gz;

				dzInter_speed=gz-dx_speed1;
				dyInter_speed=gy-dy_speed1;

				dy_speed1=dy_speed2;
				dy_speed2=dy_speed3;
				dy_speed3=dy_speed4;
				dy_speed4=dy_speed5;

				dx_speed1=dx_speed2;
				dx_speed2=dx_speed3;
				dx_speed3=dx_speed4;
				dx_speed4=dx_speed5;
				ptrImgMidRright++;
			}
			GradientXYZ+=cubesize_global/2*3;
		}
	}
#endif
}


void V2_FastSmoothOpticalFlow(myType* mOptFlow,myType* mInterOptFlow,const int w,const int h,const int WindowSizeOfSmooth)
{
	const int w2=w+1;
	const int RowLineSize=w2*2;
	memset(mInterOptFlow,0,sizeof(myType)*RowLineSize);
	for(int y=0;y<h;y++)
	{
		const int startIndex=y*w*2;
		const int startIndex2=(1+(y+1)*w2)*2;
		const myType* srcXYZ=mOptFlow+startIndex;
		myType* dstXYZ=mInterOptFlow+startIndex2;
		for(int i=-2;i<0;i++)
			*(dstXYZ+i)=0;

		myType rowX=0.0,rowY=0.0;
		for(int x=0;x<w;x++)
		{
			const myType dx=*(srcXYZ++),dy=*(srcXYZ++);
			rowX+=dx;
			rowY+=dy;

			*dstXYZ=*(dstXYZ-RowLineSize)+rowX;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowY;dstXYZ++;
		}
	}
	
	const int RowWindowSize=WindowSizeOfSmooth*2;
	const int PointsNum=WindowSizeOfSmooth*WindowSizeOfSmooth;
	const int wBorder=w-WindowSizeOfSmooth;
	for (int y=WindowSizeOfSmooth/2;y<h-WindowSizeOfSmooth/2;y++)
	{
		myType* curRowFlow=mOptFlow+(y*w+WindowSizeOfSmooth/2)*2;
		const myType* curRowTermsUp=mInterOptFlow+(y-WindowSizeOfSmooth/2)*RowLineSize;
		const myType* curRowTermsDown=curRowTermsUp+WindowSizeOfSmooth*RowLineSize;

		for (int x=WindowSizeOfSmooth/2;x<w-WindowSizeOfSmooth/2;x++)
		{
			const myType TermX=(*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize))
				/(WindowSizeOfSmooth*WindowSizeOfSmooth);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermY=(*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize))
				/(WindowSizeOfSmooth*WindowSizeOfSmooth);;
			curRowTermsUp++;curRowTermsDown++;
			
			if(x<WindowSizeOfSmooth)
			{
				 myType* FloBorder=curRowFlow-WindowSizeOfSmooth/2*2;
				*(FloBorder++)=TermX;
				*(FloBorder)=TermY;
			}
			if(x>wBorder)
			{
				myType* FloBorder=curRowFlow+WindowSizeOfSmooth/2*2;
				*(FloBorder++)=TermX;
				*(FloBorder)=TermY;
			}
			*(curRowFlow++)=TermX;
			*(curRowFlow++)=TermY;
		}
	}
	memcpy(mOptFlow,mOptFlow+w*2*(WindowSizeOfSmooth/2),sizeof(float)*2*w*(WindowSizeOfSmooth/2));
	memcpy(mOptFlow+w*2*(h-WindowSizeOfSmooth/2),mOptFlow+w*2*(h-WindowSizeOfSmooth/2*2),sizeof(float)*2*w*(WindowSizeOfSmooth/2));
}

void V3_GrtsRefine(ImagePyramidLevel* levelData2,IplImage* MaskOcc=NULL)
{
	static CacheTermXYZ mCacheTermXYZ; 
	

	const int w=levelData2->w;
	const int h=levelData2->h;
	
	const int borderH=h-WindowSizeOf3DGrdts/2;
	const int borderW=w-WindowSizeOf3DGrdts/2;
	const int PointsNum=WindowSizeOf3DGrdts*WindowSizeOf3DGrdts;

	const int* grtXYZ1=levelData2->GradientXYZReordered[cubesize_global/2];
	//const int* grtY1=levelData2->GradientYReordered[cubesize_global/2];
	//const int* grtZ1=levelData2->GradientZReordered[cubesize_global/2];

	const int* grtXYZ2=levelData2->GradientXYZReordered[cubesize_global/2+1];
	//const int* grtY2=levelData2->GradientYReordered[cubesize_global/2+1];
	//const int* grtZ2=levelData2->GradientZReordered[cubesize_global/2+1];

	//const uchar* img1=levelData2->SrcImgReordered[cubesize_global/2];
	//const uchar* img2=levelData2->SrcImgReordered[cubesize_global/2+1];

	const int wGxyz=w*3;
	const int wFloxy=w*2;
	int Dxyz[WindowSizeOf3DGrdts*WindowSizeOf3DGrdts*3];
	uchar* ptrMaskOcc=(uchar*)MaskOcc->imageData;
	for(int y=WindowSizeOf3DGrdts/2;y<borderH;y++)
	{
		//const int startIndex=y*w;
		myType* FlowWarped=levelData2->OpticalFlow_Warped+y*wFloxy+WindowSizeOf3DGrdts/2*2;
		//myType* FlowCurLevel=levelData2->OpticalFlow_Ori+y*wFloxy+WindowSizeOf3DGrdts/2*2;
		uchar* rowMaskOcc=ptrMaskOcc+MaskOcc->widthStep*y;
		//=======================================================
		
		mCacheTermXYZ.MatchIndex=0;

		for(int x=WindowSizeOf3DGrdts/2;x<borderW;x++)
		{
			
			const int x2=cvRound(x+*(FlowWarped++));
			const int y2=cvRound(y+*(FlowWarped++));
			if ((!rowMaskOcc[x])||
				x2<WindowSizeOf3DGrdts/2||y2<WindowSizeOf3DGrdts/2||
				x2>=borderW||y2>=borderH)
			{
				
				mCacheTermXYZ.MatchIndex=0;

				FlowWarped+=2;
				
				continue;
			}

			const int dxInt=x2-x;
			const int dyInt=y2-y;		
			if((!dxInt)&&(!dyInt))
			{
				
				mCacheTermXYZ.MatchIndex=0;

				
				FlowWarped[-2]=0;
				FlowWarped[-1]=0;
				continue;
			}
			
			const int cacheIndexNeed=dxInt+dyInt*1000+200000;
			//1======================================================================
			
			if (mCacheTermXYZ.MatchIndex!=cacheIndexNeed)
			{
				
				mCacheTermXYZ.MatchIndex=cacheIndexNeed;
				mCacheTermXYZ.grtXYZ2=grtXYZ2+(dxInt)*3+(dyInt)*wGxyz;
				mCacheTermXYZ.LocalEndIndex=0;
				
				TypeCache* TermXYZ=mCacheTermXYZ.LineData;
				TypeCache* RecordXYZ=mCacheTermXYZ.LocalRecord;
				memset(RecordXYZ,0,sizeof(TypeCache)*WindowSizeOf3DGrdts*8);
				memset(TermXYZ,0,sizeof(TypeCache)*8);
				for (int yy=-WindowSizeOf3DGrdts/2;yy<=WindowSizeOf3DGrdts/2;yy++)
				{
					RecordXYZ=mCacheTermXYZ.LocalRecord;
					const int indexTrans=(x-WindowSizeOf3DGrdts/2)*3+(y+yy)*wGxyz;
					const int* disp1=grtXYZ1+indexTrans;
					const int* disp2=mCacheTermXYZ.grtXYZ2+indexTrans;
					for (int xx=0;xx<WindowSizeOf3DGrdts;xx++)
					{				
						const int gtX=*(disp1++)+*(disp2++);
						const int gtY=*(disp1++)+*(disp2++);
						const int gtZ=*(disp2++)-*(disp1++);
						(*RecordXYZ++)+=gtX;
						(*RecordXYZ++)+=gtY;
						(*RecordXYZ++)+=gtZ;
						(*RecordXYZ++)+=gtX*gtX;
						(*RecordXYZ++)+=gtY*gtY;
						(*RecordXYZ++)+=gtX*gtY;
						(*RecordXYZ++)+=gtX*gtZ;
						(*RecordXYZ++)+=gtY*gtZ;
					}
				}
				RecordXYZ=mCacheTermXYZ.LocalRecord;
				for (int xx=0;xx<WindowSizeOf3DGrdts;xx++)
				{
					TermXYZ[0]+=(*RecordXYZ++);
					TermXYZ[1]+=(*RecordXYZ++);
					TermXYZ[2]+=(*RecordXYZ++);
					TermXYZ[3]+=(*RecordXYZ++);
					TermXYZ[4]+=(*RecordXYZ++);
					TermXYZ[5]+=(*RecordXYZ++);
					TermXYZ[6]+=(*RecordXYZ++);
					TermXYZ[7]+=(*RecordXYZ++);
				}
			}
			else
			//3=======================================================
			{
				{
					TypeCache* TermXYZ=mCacheTermXYZ.LineData;
					TypeCache* RecordXYZ=mCacheTermXYZ.LocalRecord;
					const int indexTrans=(x+WindowSizeOf3DGrdts/2)*3+(y-WindowSizeOf3DGrdts/2)*wGxyz;
					const int* disp1=grtXYZ1+indexTrans;
					const int* disp2=mCacheTermXYZ.grtXYZ2+indexTrans;
					TypeCache* curLocalRecord=mCacheTermXYZ.LocalRecord+mCacheTermXYZ.LocalEndIndex;
					mCacheTermXYZ.LocalEndIndex=mCacheTermXYZ.LocalEndIndex==((WindowSizeOf3DGrdts-1)*8)?
						0:mCacheTermXYZ.LocalEndIndex+8;
					
					TermXYZ[0]-=curLocalRecord[0];
					TermXYZ[1]-=curLocalRecord[1];
					TermXYZ[2]-=curLocalRecord[2];
					TermXYZ[3]-=curLocalRecord[3];
					TermXYZ[4]-=curLocalRecord[4];
					TermXYZ[5]-=curLocalRecord[5];
					TermXYZ[6]-=curLocalRecord[6];
					TermXYZ[7]-=curLocalRecord[7];
					memset(curLocalRecord,0,8*sizeof(TypeCache));
					for(int yy=0;yy<WindowSizeOf3DGrdts;yy++)
					{
						const int gtX=disp1[0]+disp2[0];
						const int gtY=disp1[1]+disp2[1];
						const int gtZ=disp2[2]-disp1[2];
						curLocalRecord[0]+=gtX;
						curLocalRecord[1]+=gtY;
						curLocalRecord[2]+=gtZ;
						curLocalRecord[3]+=gtX*gtX;
						curLocalRecord[4]+=gtY*gtY;
						curLocalRecord[5]+=gtX*gtY;
						curLocalRecord[6]+=gtX*gtZ;
						curLocalRecord[7]+=gtY*gtZ;
						disp1+=wGxyz;
						disp2+=wGxyz;
					}
					TermXYZ[0]+=curLocalRecord[0];
					TermXYZ[1]+=curLocalRecord[1];
					TermXYZ[2]+=curLocalRecord[2];
					TermXYZ[3]+=curLocalRecord[3];
					TermXYZ[4]+=curLocalRecord[4];
					TermXYZ[5]+=curLocalRecord[5];
					TermXYZ[6]+=curLocalRecord[6];
					TermXYZ[7]+=curLocalRecord[7];
				}
			}
			//4=======================================================
			TypeCache* TermXYZ=mCacheTermXYZ.LineData;
			
			const myType af=PointsNum*TermXYZ[3]-TermXYZ[0]*TermXYZ[0];
			const myType bt=PointsNum*TermXYZ[5]-TermXYZ[0]*TermXYZ[1];
			const myType gm=PointsNum*TermXYZ[4]-TermXYZ[1]*TermXYZ[1];
			const myType dz=(bt*bt-af*gm)/2*KernelNormFactor+0.0001;
			const myType z1=PointsNum*TermXYZ[6]-TermXYZ[0]*TermXYZ[2];
			const myType z2=PointsNum*TermXYZ[7]-TermXYZ[1]*TermXYZ[2];

			myType dx=(gm*z1-bt*z2)/dz;
			myType dy=(af*z2-bt*z1)/dz;

			
			const bool isBad=(abs((int)dx)>>3)||(abs((int)dy)>>3);
			FlowWarped[-2]=((isBad?0.0:dx)+dxInt);
			FlowWarped[-1]=((isBad?0.0:dy)+dyInt);
			
		}
	}
}

void V2_Fast_warpFlowToNextLevel(ImagePyramidLevel* levelData1,ImagePyramidLevel* levelData2,IplImage* MaskOcc=NULL)
{
	//static CacheTermXYZ mCacheTermXYZ; 

	const int w=levelData2->w;
	const int h=levelData2->h;

	cv::Mat flo1(levelData1->h,levelData1->w,CV_32FC2,(void*)levelData1->OpticalFlow_Warped);
	cv::Mat flo2(h,w,CV_32FC2,(void*)levelData2->OpticalFlow_Warped);

	myType* preFlow=levelData1->OpticalFlow_Warped;

	for (int i=0;i<levelData1->h*levelData1->w;i++)
	{
		(*preFlow++)*=PydFactor;
		(*preFlow++)*=PydFactor;
	}
	cv::pyrUp(flo1,flo2,flo2.size());
	//return;
	
	const int borderH=h-cubesize_global/2;
	const int borderW=w-cubesize_global/2;

	const int* grtXYZ1=levelData2->GradientXYZReordered[cubesize_global/2];
	const int* grtXYZ2=levelData2->GradientXYZReordered[cubesize_global/2+1];
	const int wGxyz=w*3;
	const int wFloxy=w*2;
	
	for (int y=cubesize_global/2;y<h-cubesize_global/2;y++)
	{
		const int indexTrans=cubesize_global/2*3+y*wGxyz;
		myType* G3DMat=levelData2->Gradient3DMat+indexTrans;
		const int* grtXYZ1=levelData2->GradientXYZReordered[1]+indexTrans;
		const int* grtXYZ2=levelData2->GradientXYZReordered[2]+indexTrans;

		myType* FlowWarped=levelData2->OpticalFlow_Warped+y*wFloxy+cubesize_global/2*2;
		for (int x=cubesize_global/2;x<w-cubesize_global/2;x++)
		{
			const int x2=cvRound(x+*(FlowWarped++));
			const int y2=cvRound(y+*(FlowWarped++));
			
			if (x2<cubesize_global/2||y2<cubesize_global/2||
				x2>=borderW||y2>=borderH)
			{
				G3DMat+=3;
				grtXYZ1+=3;
				grtXYZ2+=3;
				continue;
			}
			const int dxInt=x2-x;
			const int dyInt=y2-y;		
			
			if((!dxInt)&&(!dyInt))
			{
				G3DMat+=3;
				grtXYZ1+=3;
				grtXYZ2+=3;
				FlowWarped[-2]=0.0;
				FlowWarped[-1]=0.0;
				continue;
			}
			
			FlowWarped[-2]=dxInt;
			FlowWarped[-1]=dyInt;
			const int* disp1=grtXYZ1;
			const int* disp2=grtXYZ2+(dxInt)*3+(dyInt)*wGxyz;	
			*(G3DMat++)=*(disp1++)+*(disp2++);
			*(G3DMat++)=*(disp1++)+*(disp2++);
			*(G3DMat++)=*(disp2++)-*(disp1++);
			grtXYZ1+=3;
			grtXYZ2+=3;
		}
	}
	
	const int w2=w+1;
	const int RowLineSize=w2*8;
	//memset(TermsXYZ,0,sizeof(myType)*RowLineSize);
	for(int y=0;y<h;y++)
	{
		const int startIndex=y*w*3;
		const int startIndex2=(1+(y+1)*w2)*8;
		const myType* srcXYZ=levelData2->Gradient3DMat+startIndex;
		myType* dstXYZ=levelData2->TermsXYZ+startIndex2;

		myType rowX=0,rowY=0,rowZ=0,rowX2=0,rowY2=0,rowXY=0,rowXZ=0,rowYZ=0;
		for(int x=0;x<w;x++)
		{
			const myType dx=*(srcXYZ++),dy=*(srcXYZ++),dz=*(srcXYZ++);
			rowX+=dx;
			rowY+=dy;
			rowZ+=dz;
			rowX2+=dx*dx;
			rowY2+=dy*dy;
			rowXY+=dx*dy;
			rowXZ+=dx*dz;
			rowYZ+=dy*dz;

			*dstXYZ=*(dstXYZ-RowLineSize)+rowX;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowY;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowZ;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowX2;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowY2;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowXY;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowXZ;dstXYZ++;
			*dstXYZ=*(dstXYZ-RowLineSize)+rowYZ;dstXYZ++;
		}
	}
	uchar* ptrMaskOcc=MaskOcc?(uchar*)MaskOcc->imageData:NULL;
	
	const int RowWindowSize=WindowSizeOf3DGrdts*8;
	const int PointsNum=WindowSizeOf3DGrdts*WindowSizeOf3DGrdts;
	myType* curOpticalFlow=levelData2->OpticalFlow_Warped;
	for (int y=WindowSizeOf3DGrdts/2;y<h-WindowSizeOf3DGrdts/2;y++)
	{
		uchar* rowMaskOcc=ptrMaskOcc?(ptrMaskOcc+MaskOcc->widthStep*y):NULL;

		myType* curRowFlow=curOpticalFlow+y*wFloxy+WindowSizeOf3DGrdts/2*2;
		const myType* curRowTermsUp=levelData2->TermsXYZ+(y-WindowSizeOf3DGrdts/2)*RowLineSize;
		const myType* curRowTermsDown=curRowTermsUp+WindowSizeOf3DGrdts*RowLineSize;
		for (int x=WindowSizeOf3DGrdts/2;x<w-WindowSizeOf3DGrdts/2;x++)
		{
			
			const myType TermX=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermY=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermZ=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermX2=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermY2=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermXY=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermXZ=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			const myType TermYZ=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;

			const myType af=PointsNum*TermX2-TermX*TermX;
			const myType bt=PointsNum*TermXY-TermX*TermY;
			const myType gm=PointsNum*TermY2-TermY*TermY;
			const myType dz=(bt*bt-af*gm)/2*KernelNormFactor+0.0001;

			const myType z1=PointsNum*TermXZ-TermX*TermZ;
			const myType z2=PointsNum*TermYZ-TermY*TermZ;

			

			const myType dx=(gm*z1-bt*z2)/dz;
			const myType dy=(af*z2-bt*z1)/dz;
			const bool isBad=(abs((int)dx)>>2)||(abs((int)dy)>>2);


			if(rowMaskOcc&&rowMaskOcc[x]==100)
			{
				rowMaskOcc[x]=(isBad?0:255);
			}
			{
				*(curRowFlow++)+=(isBad?0.0:dx);
				*(curRowFlow++)+=(isBad?0.0:dy);
			}

		}
	}
}

void V2_ProcessOneLevel(ImagePyramidLevel* levelData,const int levelNum,cv::Mat ImgForGrts,cv::Mat ImgForAnn,int CurrentStartIndex,int cur3DGradientStartIndex=0)
{
	//levelData+=levelNum;
	const int ImgW=levelData->w;
	const int ImgH=levelData->h;
	int CurrentEndIndex=CurrentStartIndex-1;
	if(CurrentStartIndex==0)
		CurrentEndIndex=cubesize_global-1;

	int cur3DGradientEndIndex=cur3DGradientStartIndex-1;
	if(cur3DGradientStartIndex==0)
		cur3DGradientEndIndex=LevelsGradient3D-1;
	
	memcpy(levelData->SrcImg[CurrentEndIndex],ImgForGrts.data,ImgW*ImgH);
	
	Fast2DGradient(ImgForGrts,levelData->GradientXYZ[CurrentEndIndex]);

	if(levelNum==(PyramidLevelNum-1))
		convert_bitmap_withGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgForGrts,ImgW,ImgH,ImgForAnn.channels());
	else if(levelNum==0)
	{
#ifdef IsProRefineWithGradients
		convert_bitmap_withGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgForGrts,ImgW,ImgH,ImgForAnn.channels());
#else
		convert_bitmap_withoutGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgW,ImgH,ImgForAnn.channels());
#endif
	}	

	int currentMiddleFrame=0;
	for(int i=0,tempIndex=CurrentStartIndex;i<cubesize_global;i++)
	{
		levelData->rgbImgReordered[i]=levelData->rgbImg[tempIndex];

		levelData->SrcImgReordered[i]=levelData->SrcImg[tempIndex];
		levelData->GradientXYZReordered[i]=levelData->GradientXYZ[tempIndex];
		levelData->GradientYReordered[i]=levelData->GradientY[tempIndex];
		levelData->GradientZReordered[i]=levelData->GradientZ[tempIndex];
		if(i==cubesize_global/2)
			currentMiddleFrame=tempIndex;
		tempIndex++;
		if(tempIndex>=cubesize_global)
			tempIndex=0;
	}
	
	if(cubesize_global==3)
	{
		
		for(int y=cubesize_global/2;y<ImgH-cubesize_global/2;y++)
		{
			const int indexTrans=(cubesize_global/2+y*ImgW)*3;
			myType* G3DMat=levelData->Gradient3DMat+indexTrans;
			const int* grtXYZ1=levelData->GradientXYZReordered[1]+indexTrans;
			const int* grtXYZ2=levelData->GradientXYZReordered[2]+indexTrans;
			for(int x=cubesize_global/2;x<ImgW-cubesize_global/2;x++)
			{
					const int Dx=/*levelData->GradientXReordered[0][count]+*/(*grtXYZ1++)+(*grtXYZ2++);
					const int Dy=/*levelData->GradientYReordered[0][count]+*/(*grtXYZ1++)+(*grtXYZ2++);
					const int Dz=(*grtXYZ2++)-(*grtXYZ1++);
					
					*(G3DMat++)=Dx;
					*(G3DMat++)=Dy;
					*(G3DMat++)=Dz;
			}
		}
	}//if(cubesize_global==3)
	else
#ifdef UseCube3For3dGt
	
#else
	{
		for(int y=cubesize_global/2, count=ImgW*(cubesize_global/2);y<ImgH-cubesize_global/2;y++)
		{
			count+=cubesize_global/2;
			for(int x=cubesize_global/2;x<ImgW-cubesize_global/2;x++)
			{
				myType* G3DMat=levelData->Gradient3DMat+3*LevelsGradient3D*count+cur3DGradientEndIndex*3;
				*(G3DMat)=levelData->GradientXYZReordered[2][3*count]+levelData->GradientXYZReordered[3][3*count];
				*(G3DMat+1)=levelData->GradientXYZReordered[2][3*count+1]+levelData->GradientXYZReordered[3][3*count+1];
				*(G3DMat+2)=-levelData->GradientXYZReordered[2][3*count+2]+
					levelData->GradientXYZReordered[3][3*count+2];
				count++;
			}
			count+=cubesize_global/2;
		}
	}
#endif

}

void InitProcessOneLevel(ImagePyramidLevel* levelData,const int levelNum,cv::Mat ImgForGrts,cv::Mat ImgForAnn,int CurrentStartIndex)
{
	
	const int ImgW=levelData->w;
	const int ImgH=levelData->h;
	int CurrentEndIndex=CurrentStartIndex-1;
	if(CurrentStartIndex==0)
		CurrentEndIndex=cubesize_global-1;
	memcpy(levelData->SrcImg[CurrentEndIndex],ImgForGrts.data,ImgW*ImgH);
	Fast2DGradient(ImgForGrts,levelData->GradientXYZ[CurrentEndIndex]);
	

	if(levelNum==(PyramidLevelNum-1))
		convert_bitmap_withGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgForGrts,ImgW,ImgH,ImgForAnn.channels());
	else if(levelNum==0)
	{
#ifdef IsProRefineWithGradients
		convert_bitmap_withGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgForGrts,ImgW,ImgH,ImgForAnn.channels());
#else
		convert_bitmap_withoutGradients(levelData->rgbImg[CurrentEndIndex],ImgForAnn.data,ImgW,ImgH,ImgForAnn.channels());
#endif
	}
	int currentMiddleFrame=0;
	for(int i=0,tempIndex=CurrentStartIndex;i<cubesize_global;i++)
	{
		levelData->rgbImgReordered[i]=levelData->rgbImg[tempIndex];

		levelData->SrcImgReordered[i]=levelData->SrcImg[tempIndex];
		levelData->GradientXYZReordered[i]=levelData->GradientXYZ[tempIndex];
		levelData->GradientYReordered[i]=levelData->GradientY[tempIndex];
		levelData->GradientZReordered[i]=levelData->GradientZ[tempIndex];
		if(i==cubesize_global/2)
			currentMiddleFrame=tempIndex;
		tempIndex++;
		if(tempIndex>=cubesize_global)
			tempIndex=0;
	}
}

float maxOfArray(float* ptr,int len)
{
	float maxval=0.0;
	for(int i=0;i<len;i++)
	{
		if(fabs(ptr[i])>maxval)
			maxval=fabs(ptr[i]);
	}
	return maxval;
}
void OpticalComputeMainLoop_Fast(char* videoName)
{
	int currentStartIndex = 1;
	int cur3DGradientStartIndex = 1;
	cv::VideoCapture mVideoCapture(videoName);
	int rateSkip = cvRound(mVideoCapture.get(CV_CAP_PROP_FPS) / 10);
	
	cv::Mat srcImg, grayImg, rgbImg, preRgbImg;
	
	for (int i = 0; i<1; i++)
	{
		
		mVideoCapture.read(srcImg);
		cv::GaussianBlur(srcImg, srcImg, cv::Size(7, 7), 0.6, 0.6);
		srcImg.copyTo(rgbImg);
		
		if (i == 0)
		{
			preRgbImg = srcImg.clone();
			InitPyramidLevel(rgbImg, PyramidLevel, PyramidLevelNum, &myFinalOpticalFlowResult);
		}
		for (int levelCount = 0; levelCount<PyramidLevelNum; levelCount++)
		{
			if (levelCount)
				cv::pyrDown(rgbImg, rgbImg);
			cv::cvtColor(rgbImg, grayImg, CV_RGB2GRAY);
			//cv::cvtColor(rgbImg, rgbImg,CV_RGB2HSV);
			InitProcessOneLevel(PyramidLevel + levelCount, levelCount, grayImg, rgbImg, currentStartIndex);
		}
		currentStartIndex++;
		if (currentStartIndex >= cubesize_global)
			currentStartIndex = 0;
	}

	// double TCostPart1 = 0.0, TCostPart2 = 0.0;
	//==========================================
	IplImage* MaskOcclused = cvCreateImage(cvSize(PyramidLevel[PyramidLevelNum - 1].w, PyramidLevel[PyramidLevelNum - 1].h), 8, 1);
	IplImage* curMaskOcc = cvCreateImage(cvSize(PyramidLevel[0].w, PyramidLevel[0].h), 8, 1);
	BITMAP* OccReplaceMap = create_bitmap(PyramidLevel[PyramidLevelNum - 1].w, PyramidLevel[PyramidLevelNum - 1].h);
	BITMAP *ann = create_bitmap(PyramidLevel[0].w, PyramidLevel[0].h);
	int countFrames = cubesize_global / 2 + 1;
	
	while (1)
	{
		
		for (int j = 0; j < 6; j++) mVideoCapture.read(srcImg);
		
		if (srcImg.empty())
			break;
		cv::GaussianBlur(srcImg, srcImg, cv::Size(7, 7), 0.6, 0.6);
		srcImg.copyTo(rgbImg);
		
		
		for (int levelCount = 0; levelCount<PyramidLevelNum; levelCount++)
		{
			if (levelCount)
				cv::pyrDown(rgbImg, rgbImg);
			cv::cvtColor(rgbImg, grayImg, CV_RGB2GRAY);
			
			V2_ProcessOneLevel(PyramidLevel + levelCount, levelCount, grayImg, rgbImg, currentStartIndex, cur3DGradientStartIndex);

#ifdef NoExternalCorrection
			if (levelCount == PyramidLevelNum - 1)
			{

				InitFlowFieldByANN(PyramidLevel[levelCount].rgbImgReordered[cubesize_global - 2], PyramidLevel[levelCount].rgbImgReordered[cubesize_global - 1], PyramidLevel[levelCount].OpticalFlow_Warped, MaskOcclused, OccReplaceMap);

				FillBorder(PyramidLevel[levelCount].OpticalFlow_Warped, PyramidLevel[levelCount].rgbImgReordered[cubesize_global - 1], 7);
				
				V2_FastSmoothOpticalFlow((PyramidLevel + levelCount)->OpticalFlow_Warped, (PyramidLevel + levelCount)->OpticalInterFlow, (PyramidLevel + levelCount)->w, (PyramidLevel + levelCount)->h, 7/*SmoothSizeOfPerLevel[levelCount]*/);
				V3_GrtsRefine(/*ImagePyramidLevel* levelData1,*/PyramidLevel + levelCount, MaskOcclused);
				V2_FastSmoothOpticalFlow((PyramidLevel + levelCount)->OpticalFlow_Warped, (PyramidLevel + levelCount)->OpticalInterFlow, (PyramidLevel + levelCount)->w, (PyramidLevel + levelCount)->h, 7/*SmoothSizeOfPerLevel[levelCount]*/);
				V3_GrtsRefine(/*ImagePyramidLevel* levelData1,*/PyramidLevel + levelCount, MaskOcclused);
				V2_FastSmoothOpticalFlow((PyramidLevel + levelCount)->OpticalFlow_Warped, (PyramidLevel + levelCount)->OpticalInterFlow, (PyramidLevel + levelCount)->w, (PyramidLevel + levelCount)->h, 7/*SmoothSizeOfPerLevel[levelCount]*/);
				V3_GrtsRefine(/*ImagePyramidLevel* levelData1,*/PyramidLevel + levelCount, MaskOcclused);

				V2_FastSmoothOpticalFlow((PyramidLevel + levelCount)->OpticalFlow_Warped, (PyramidLevel + levelCount)->OpticalInterFlow, (PyramidLevel + levelCount)->w, (PyramidLevel + levelCount)->h, 7/*SmoothSizeOfPerLevel[levelCount]*/);
			}
#endif		
		}
		
#ifndef NoExternalCorrection
		memcpy(PyramidLevel[PyramidLevelNum - 1].OpticalFlow_Warped, PyramidLevel[PyramidLevelNum - 1].OpticalFlow_Ori, 2 * sizeof(float)*PyramidLevel[PyramidLevelNum - 1].w*PyramidLevel[PyramidLevelNum - 1].h);
#endif
		for (int i_level = PyramidLevelNum - 2; i_level >= 0; i_level--)
		{
			if (i_level == 0)
			{
				InitNextOccMask(MaskOcclused, curMaskOcc);
				V2_Fast_warpFlowToNextLevel(PyramidLevel + 1 + i_level, PyramidLevel + i_level, curMaskOcc);
			}
			else
			{
				V2_Fast_warpFlowToNextLevel(PyramidLevel + 1 + i_level, PyramidLevel + i_level, NULL);
			}
			V2_FastSmoothOpticalFlow((PyramidLevel + i_level)->OpticalFlow_Warped, (PyramidLevel + i_level)->OpticalInterFlow, (PyramidLevel + i_level)->w, (PyramidLevel + i_level)->h, 9);
			if (i_level == 0)
			{
				
				V2_FastSmoothOpticalFlow((PyramidLevel + i_level)->OpticalFlow_Warped, (PyramidLevel + i_level)->OpticalInterFlow, (PyramidLevel + i_level)->w, (PyramidLevel + i_level)->h, 15);
				PropagateANNOnly(ann, PyramidLevel[i_level].rgbImgReordered[cubesize_global - 2], PyramidLevel[i_level].rgbImgReordered[cubesize_global - 1], PyramidLevel[i_level].OpticalFlow_Warped, curMaskOcc);
				
				ReplaceOccPixels(PyramidLevel[i_level].OpticalFlow_Warped, curMaskOcc, OccReplaceMap, 7);

				FillBorder(PyramidLevel[i_level].OpticalFlow_Warped, PyramidLevel[i_level].rgbImgReordered[cubesize_global - 1], 50, curMaskOcc);
				
			}
			
		}
		
		V2_FastSmoothOpticalFlow(PyramidLevel->OpticalFlow_Warped, PyramidLevel->OpticalInterFlow, PyramidLevel->w, PyramidLevel->h, 5);
		//=====================================================
		
		currentStartIndex++;
		if (currentStartIndex >= cubesize_global)
			currentStartIndex = 0;

		cur3DGradientStartIndex++;
		if (cur3DGradientStartIndex >= LevelsGradient3D)
			cur3DGradientStartIndex = 0;
		//=====================================================
		{
			const float* ptrFlow = PyramidLevel->OpticalFlow_Warped;

			
			for (int y = 0; y<PyramidLevel->h; y++)
			{
				for (int x = 0; x<PyramidLevel->w; x++)
				{
					if (x % 30 == 0 && y % 30 == 0)
					{
						cv::line(preRgbImg, cv::Point(x, y), cv::Point(x + ptrFlow[0] * 1.0, y + ptrFlow[1] * 1.0), cv::Scalar(0, 0, 100));
						cv::circle(preRgbImg, cv::Point(x, y), 3, cv::Scalar(100, 0, 0));
					}
					ptrFlow += 2;
				}
			}
			
		}
		//=====================================================
		
		memset(PointCount, 0, 4 * 4);

		imshow("Flow Vector(Draw on Img1-> Img2)", preRgbImg);
		int key = cvWaitKey(10);

		if (key == 'q' || key == 'Q')
			return;
		
		countFrames++;
		//=====================================================
		preRgbImg = srcImg;
	}
}

int main(int argc,char *argv[])
{
	OpticalComputeMainLoop_Fast(argv[1]);
	return 0;
}