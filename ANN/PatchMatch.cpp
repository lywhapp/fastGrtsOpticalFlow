// PatchMatch.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include "PatchMatch.h"
#include <opencv2/imgproc/imgproc_c.h>
#define TStart  double exec_time_146144821;\
					  {exec_time_146144821= cvGetTickCount();}
#define TEnd  {exec_time_146144821 = (cvGetTickCount() - exec_time_146144821)/1000.0/cvGetTickFrequency();printf("%.2lfms,%.1lf fps\n",exec_time_146144821,1000/exec_time_146144821);}

//static char AdobePatentID_P876E1[] = "AdobePatentID=\"P876E1\""; // AdobePatentID="P876E1"
//static char AdobePatentID_P962[] = "AdobePatentID=\"P962\""; // AdobePatentID="P962"

void init_params(Params *p);

#define MODE_IMAGE  0
#define MODE_VECB   1
#define MODE_VECF   2

typedef unsigned int uint;

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

int is_float(const mxArray *A) {
	return mxIsDouble(A) || mxIsSingle(A);
}

extern int xform_scale_table[NUM_SCALES];
extern void CodeFlowAsColor(FlowType* Flow,int w,int h);
/*void logf(const char *s) {
FILE *f = fopen("log.txt", "a");
fprintf(f, s);
fclose(f);
}*/

void CenterAnnField(BITMAP* ann, const int patch_w)
{
	const int step=(patch_w/2)*ann->w*4;
	const int rowSize=(ann->w-patch_w/2);
	const int dxy=(patch_w/2)|((patch_w/2)<<12);
	for(int y=ann->h-patch_w-1;y>=patch_w/2;y--)
	{
		int* des=(int*)(ann->line[y]+4*(patch_w/2));
		int* src=(int*)(ann->line[y]-step);
		//memcpy(des,src,rowSize);
		for(int i=-(patch_w/2);i<0;i++)
			des[i]=0;
		for(int i=0;i<rowSize;i++)
			des[i]=src[i]+dxy;
	}
	memset(ann->data,0,ann->w*(patch_w/2)*4);
}



void PropagateFill(BITMAP* ann,BITMAP* Img,IplImage* MaskOcclused/*,const float gamma_c2,const float gamma_d2*//*ƽ������*/)
{
//#define SqrtL2(x,y) ((x)*(x)+(y)*(y))
#define NewX(ans,x,step,rowAnn)		{(ans)=((rowAnn)[(x)-(step)]+(step));}
#define NewY(ans,x,step,rowAnnPre)		{(ans)=((rowAnnPre)[(x)]+(step)*(0x1000));}
//#define NewX(ans,x,step,rowAnn)		{(ans)=0;}
//#define NewY(ans,x,step,rowAnnPre)		{(ans)=0;}

	static int* scoreMat=new int[ann->w*ann->h];
	for(int i=0;i<ann->w*ann->h;i++)	scoreMat[i]=INT_MAX;
	uchar* MaskOcc=(uchar*)MaskOcclused->imageData;
	for(int i=0;i<4;i++)//ѭ����ɢN��
	{
		const int step=i%2?1:-1;//stepΪ1ʱ���£�Ϊ-1����
		const int xstart = (step==1)?1:(ann->w-2), xfinal = (step==1)?ann->w:-1; 
		const int ystart = (step==1)?1:(ann->h-2), yfinal = (step==1)?ann->h:-1;
		//int xlastGood=ystart,ylastGood=xstart;
		for(int y=ystart;y!=yfinal;y+=step)
		{
			uchar* ptrOcc=MaskOcc+MaskOcclused->widthStep*y;
			int* ptrScore=scoreMat+ann->w*y;
			int* rowAnn=(int*)ann->line[y];
			int* rowRgb=(int*)Img->line[y];
			for(int x=xstart;x!=xfinal;x+=step)
			{
				if(ptrOcc[x]<255)
				{
					uint rgbCenter=rowRgb[x];
					int r = (rgbCenter&255);
					int g = ((rgbCenter>>8)&255);
					int b = ((rgbCenter>>16)&255);
					
					if(ptrOcc[x-step]) 
					{
						uint rgbComp=rowRgb[x-step];
						int dr12 = (r)-(rgbComp&255);
						int dg12 = (g)-((rgbComp>>8)&255);
						int db12 = (b)-((rgbComp>>16)&255);
						int score=dr12*dr12+dg12*dg12+db12*db12;
						if(score<ptrScore[x])
						{
							NewX(rowAnn[x],x,step,rowAnn)
							ptrScore[x]=score;
							ptrOcc[x]=100;
						}
					}
					  
					if(ptrOcc[x-step*MaskOcclused->widthStep]) 
					{
						const int preOffest=step*ann->w;
						uint rgbComp=rowRgb[x-preOffest];
						int dr12 = (r)-(rgbComp&255);
						int dg12 = (g)-((rgbComp>>8)&255);
						int db12 = (b)-((rgbComp>>16)&255);
						int score=dr12*dr12+dg12*dg12+db12*db12;
						if(score<ptrScore[x])
						{
							NewY(rowAnn[x],x,step,rowAnn-preOffest)
							ptrScore[x]=score;
							ptrOcc[x]=100;
						}
					}
				}
			}
		}
	}
}

#define maxForFill(x,y)	((x)>(y)?(x):(y))
#define minForFill(x,y)	((x)<(y)?(x):(y))

inline void FillOneRow(FlowType* rowFlow,int* rowImg,int* scoreRow,const int dy,const int widthImg,const int halfBd,uchar* rowOccMask,const int wStepOccMask)
{
	FlowType* cmpFlow=rowFlow+dy*widthImg*2;
	int* cmpImg=rowImg+dy*widthImg;
	uchar* cmpOccMask=rowOccMask+dy*wStepOccMask;
	//const int xmax=widthImg-halfBd-1;
	memset(scoreRow,127,widthImg*sizeof(int));
	for(int x=0;x<widthImg;x++)
	{
		if(rowOccMask[x]==255)
			continue;
		uint rgb=rowImg[x];
		int r = (rgb&255);
		int g = ((rgb>>8)&255);
		int b =((rgb>>16)&255);
		for(int dx=x-1;dx<=x+1;dx++)
		{
			if(dx<0||dx>=widthImg||(!cmpOccMask[dx]))
				continue;
			uint rgbCmp=cmpImg[dx];
			int rCmp = (rgbCmp&255)-r;
			int gCmp = ((rgbCmp>>8)&255)-g;
			int bCmp = ((rgbCmp>>16)&255)-b;
			int score=rCmp*rCmp+gCmp*gCmp+bCmp*bCmp;
			if(score<scoreRow[x])
			{
				scoreRow[x]=score;
				rowFlow[2*x]=cmpFlow[2*dx];
				rowFlow[2*x+1]=cmpFlow[2*dx+1];
				rowOccMask[x]=100;
			}
		}
	}
}

inline void FillOneCol(FlowType* colFlow,int* colImg,int* scoreCol,const int dx,const int widthImg,const int heightImg,const int halfBd,uchar* colOccImg,const int wStepOccMask)
{
	//const int ymax=heightImg-halfBd-1;
	const int wStepFlow=widthImg*2;
	FlowType* cmpFlow=colFlow+dx*2;
	int* cmpImg=colImg+dx;
	uchar* cmpOccMask=colOccImg+dx;
	memset(scoreCol,127,heightImg*sizeof(int));
	for(int y=0;y<heightImg;y++)
	{
		if(colOccImg[y*wStepOccMask]==255)
			continue;
		uint rgb=colImg[y*widthImg];
		int r = (rgb&255);
		int g = ((rgb>>8)&255);
		int b = ((rgb>>16)&255);
		for(int dy=y-1;dy<=y+1;dy++)
		{
			if(dy<0||dy>=heightImg||(!cmpOccMask[dy*wStepOccMask]))
				continue;
			uint rgbCmp=cmpImg[dy*widthImg];
			int rCmp = (rgbCmp&255)-r;
			int gCmp = ((rgbCmp>>8)&255)-g;
			int bCmp = ((rgbCmp>>16)&255)-b;
			int score=rCmp*rCmp+gCmp*gCmp+bCmp*bCmp;
			if(score<scoreCol[y])
			{
				scoreCol[y]=score;
				colFlow[y*wStepFlow]=cmpFlow[wStepFlow*dy];
				colFlow[y*wStepFlow+1]=cmpFlow[wStepFlow*dy+1];
				colOccImg[y*wStepOccMask]=100;
			}
		}
	}
}

inline void FillOneRow(FlowType* rowFlow,int* rowImg,int* scoreRow,const int dy,const int widthImg,const int halfBd)
{
	FlowType* cmpFlow=rowFlow+dy*widthImg*2;
	int* cmpImg=rowImg+dy*widthImg;
	const int xmax=widthImg-halfBd-1;
	memset(scoreRow,127,widthImg*sizeof(int));
	for(int x=0;x<widthImg;x++)
	{
		uint rgb=rowImg[x];
		int r = (rgb&255);
		int g = ((rgb>>8)&255);
		int b = ((rgb>>16)&255);
		for(int dx=x-1;dx<=x+1;dx++)
		{
			if(dx<halfBd||dx>xmax)
				continue;
			uint rgbCmp=cmpImg[dx];
			int rCmp = (rgbCmp&255)-r;
			int gCmp = ((rgbCmp>>8)&255)-g;
			int bCmp =  ((rgbCmp>>16)&255)-b;
			int score=rCmp*rCmp+gCmp*gCmp+bCmp*bCmp;
			if(score<scoreRow[x])
			{
				scoreRow[x]=score;
				rowFlow[2*x]=cmpFlow[2*dx];
				rowFlow[2*x+1]=cmpFlow[2*dx+1];
			}
		}
	}
}

inline void FillOneCol(FlowType* colFlow,int* colImg,int* scoreCol,const int dx,const int widthImg,const int heightImg,const int halfBd)
{
	const int ymax=heightImg-halfBd-1;
	const int wStepFlow=widthImg*2;
	FlowType* cmpFlow=colFlow+dx*2;
	int* cmpImg=colImg+dx;
	memset(scoreCol,127,heightImg*sizeof(int));
	for(int y=0;y<heightImg;y++)
	{
		uint rgb=colImg[y*widthImg];
		int r = (rgb&255);
		int g = ((rgb>>8)&255);
		int b = ((rgb>>16)&255);
		for(int dy=y-1;dy<=y+1;dy++)
		{
			if(dy<halfBd||dy>ymax)
				continue;
			uint rgbCmp=cmpImg[dy*widthImg];
			int rCmp = (rgbCmp&255)-r;
			int gCmp = ((rgbCmp>>8)&255)-g;
			int bCmp =  ((rgbCmp>>16)&255)-b;
			int score=rCmp*rCmp+gCmp*gCmp+bCmp*bCmp;
			if(score<scoreCol[y])
			{
				scoreCol[y]=score;
				colFlow[y*wStepFlow]=cmpFlow[wStepFlow*dy];
				colFlow[y*wStepFlow+1]=cmpFlow[wStepFlow*dy+1];
				//if(fabs(colFlow[y*wStepFlow])+fabs(colFlow[y*wStepFlow+1])>1000.0)
				//	printf("dad");
			}
		}
	}
}

void FillBorder(FlowType* Flow,BITMAP* Img,const int bdWidth)
{
	const int halfBd=bdWidth/2,wStepFlow=Img->w*2,wImg=Img->w,hImg=Img->h;
	int* rowScore=new int[wImg];
	{
		for(int y=halfBd-1;y>=0;y--)
			FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,1,wImg,halfBd);
		for(int y=hImg-halfBd;y<hImg;y++)
			FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,-1,wImg,halfBd);
	}
	delete rowScore;

	int* colScore=new int[hImg];
	{
		for(int x=halfBd-1;x>=0;x--)
			FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,1,wImg,hImg,halfBd);
		for(int x=wImg-halfBd;x<wImg;x++)
			FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,-1,wImg,hImg,halfBd);
	}
	delete colScore;
}
void FillBorder(FlowType* Flow,BITMAP* Img,const int bdWidth,IplImage* OccMask)
{
	const int halfBd=bdWidth/2,wStepFlow=Img->w*2,wImg=Img->w,hImg=Img->h;
	int* rowScore=new int[wImg];
	//if(OccMask)
	{
		const int wStep=OccMask->widthStep;
		uchar* ptrOccMask=(uchar*)OccMask->imageData;
		for(int y=halfBd-1;y>=0;y--)
			FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,1,wImg,halfBd,ptrOccMask+wStep*y,wStep);
		for(int y=hImg-halfBd;y<hImg;y++)
			FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,-1,wImg,halfBd,ptrOccMask+wStep*y,wStep);
	}
	//else
	//{
	//	for(int y=halfBd-1;y>=0;y--)
	//		FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,1,wImg,halfBd,0);
	//	for(int y=hImg-halfBd;y<hImg;y++)
	//		FillOneRow(Flow+y*wStepFlow,(int*)Img->line[y],rowScore,-1,wImg,halfBd,0);
	//}
	delete rowScore;

	int* colScore=new int[hImg];
	//if(OccMask)
	{
		const int wStep=OccMask->widthStep;
		uchar* ptrOccMask=(uchar*)OccMask->imageData;
		for(int x=halfBd-1;x>=0;x--)
			FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,1,wImg,hImg,halfBd,ptrOccMask+x,wStep);
		for(int x=wImg-halfBd;x<wImg;x++)
			FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,-1,wImg,hImg,halfBd,ptrOccMask+x,wStep);
	}
	//else
	//{
	//	for(int x=halfBd-1;x>=0;x--)
	//		FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,1,wImg,hImg,halfBd,0,0);
	//	for(int x=wImg-halfBd;x<wImg;x++)
	//		FillOneCol(Flow+x*2,(int*)Img->data+x,colScore,-1,wImg,hImg,halfBd,0,0);
	//}
	delete colScore;
}
void ConvertAnnAndOcclusionFill(FlowType* Flow,BITMAP* ann,BITMAP* annInv,BITMAP* Img,IplImage* MaskOcclused,const int patch_w,BITMAP* OccReplaceMap)
{
#define MaxIters 30
#define TempLabel 100
	const int W=MaskOcclused->width,H=MaskOcclused->height,halfW=patch_w/2,widestep=MaskOcclused->widthStep;
	//===================================================
	uchar* MaskOcc=(uchar*)MaskOcclused->imageData+halfW*widestep+halfW;
	int* ptrAnn=(int*)ann->data;
	int* ptrAnnInv=(int*)annInv->data;
	int nNeedFill=0/*halfW*(H+W)*2-4*halfW*halfW*/;

	//memset(MaskOcclused->imageData,TempLabel,widestep*halfW);
	//memset(MaskOcclused->imageData+widestep*(H-halfW),TempLabel,widestep*halfW);
	//for(int y=halfW;y<H-halfW;y++)
	//{
	//	memset(MaskOcclused->imageData+y*widestep,TempLabel,halfW);
	//	memset(MaskOcclused->imageData+y*widestep+W-halfW,TempLabel,halfW);
	//}
	
	for (int y = 0; y <=H-patch_w; y++) 
	{
		FlowType *flo_row = Flow+(y+halfW)*W*2+halfW*2;
		float xFloLast=flo_row[-W*2],yFloLast=flo_row[-W*2+1];
		//uchar* ptrOcc=MaskOcc+(y+halfW)*widestep+halfW;
		int *replace_row = ((int*)OccReplaceMap->line[y+halfW])+halfW;
		for (int x = 0; x <= W-patch_w; x++) 
		{
			int xGood=x,yGood=y,pp;
			//int xLast=x,yLast=y;
			int iter=0;
			while(MaskOcc[xGood+yGood*widestep]!=255)
			{
				if(iter>MaxIters) 
				{
					break;
					//��ε�����δ����������滻Ϊһ���ڽ���
					//while(1)
					//{
					//	xGood=xLast+(rand()%3)-1;
					//	if(xGood>=0&&xGood<=W-patch_w)
					//		break;
					//}
					//while (1)
					//{
					//	yGood=yLast+(rand()%3)-1;
					//	if(yGood>=0&&yGood<=H-patch_w)
					//		break;
					//}
					//
					//xLast=xGood;
					//yLast=yGood;
					//iter=0;
					//continue;
				}
				iter++;
				pp = ptrAnn[xGood+yGood*W];
				xGood=INT_TO_X(pp);
				yGood=INT_TO_Y(pp);//�ҳ��ڶ�֡�ж�Ӧ�ĵ�
				pp=ptrAnnInv[xGood+yGood*W];
				xGood=INT_TO_X(pp);
				yGood=INT_TO_Y(pp);//�����ҳ���һ֡�еĶ�Ӧ�㣬����õ����Ƶĵ�
				
			}
			if(iter>MaxIters)
			{
				flo_row[0] = xFloLast;
				flo_row[1] = yFloLast;
				//MaskOcc[x+y*widestep]=TempLabel;
				nNeedFill++;
				replace_row[x]=UnknownANN;
				//printf("Ѱ���滻����ʱδ����\n");
				//system("pause");
			}
			else
			{
				
				pp = ptrAnn[xGood+yGood*W];
				xFloLast=INT_TO_X(pp)-xGood;
				yFloLast=INT_TO_Y(pp)-yGood;
				flo_row[0] = xFloLast;
				flo_row[1] = yFloLast;
				//if(!iter)
				replace_row[x]=XY_TO_INT(xGood+halfW,yGood+halfW);
				//MaskOcc[x+y*widestep]=Flag_OccPixel;//�ɹ�׷�ٵ���Ч���ƿ���Ϊ��������
			}
			//if(fabs(flo_row[0])+fabs(flo_row[1])>100.0)
			//	printf("dsad");
			flo_row+=2;
		}
	}
	//MaskOcc=(uchar*)MaskOcclused->imageData;
	//for(int i=0;i<H*widestep;i++)
	//{
	//	if(MaskOcc[i]==TempLabel)
	//		MaskOcc[i]=255;
	//}
	//���߽�����
	//FillBorder(Flow,Img,patch_w);
	// printf("NeedFill:%d\n",nNeedFill);
	
}

void PropagateFill(FlowType* Flow,BITMAP* Img,IplImage* MaskOcclused/*,const float gamma_c2,const float gamma_d2*//*ƽ������*/)
{
//#define SqrtL2(x,y) ((x)*(x)+(y)*(y))
#define NewX(ans,x,step,rowAnn)		{(ans)=((rowAnn)[(x)-(step)]+(step));}
#define NewY(ans,x,step,rowAnnPre)		{(ans)=((rowAnnPre)[(x)]+(step)*(0x1000));}
//#define NewX(ans,x,step,rowAnn)		{(ans)=0;}
//#define NewY(ans,x,step,rowAnnPre)		{(ans)=0;}

	const int W=MaskOcclused->width,H=MaskOcclused->height;
	
	static int* scoreMat=new int[W*H];
	for(int i=0;i<W*H;i++)	scoreMat[i]=INT_MAX;
	uchar* MaskOcc=(uchar*)MaskOcclused->imageData;
	for(int i=0;i<4;i++)//ѭ����ɢN��
	{
		const int step=i%2?1:-1;//stepΪ1ʱ���£�Ϊ-1����
		const int xstart = (step==1)?1:(W-2), xfinal = (step==1)?W:-1; 
		const int ystart = (step==1)?1:(H-2), yfinal = (step==1)?H:-1;
		//int xlastGood=ystart,ylastGood=xstart;
		for(int y=ystart;y!=yfinal;y+=step)
		{
			uchar* ptrOcc=MaskOcc+MaskOcclused->widthStep*y;
			int* ptrScore=scoreMat+W*y;
			FlowType* rowFlow=Flow+y*2*W;
			int* rowRgb=(int*)Img->line[y];
			for(int x=xstart;x!=xfinal;x+=step)
			{
				if(ptrOcc[x]<255)
				{
					uint rgbCenter=rowRgb[x];
					int r = (rgbCenter&255);
					int g = ((rgbCenter>>8)&255);
					int b = ((rgbCenter>>16)&255);
					//float score=(-(x-xlastGood)*(x-xlastGood)/gamma_d2-(x-xlastGood,y-ylastGood)/gamma_c2);
					//
					if(ptrOcc[x-step]) 
					{
						uint rgbComp=rowRgb[x-step];
						int dr12 = (r)-(rgbComp&255);
						int dg12 = (g)-((rgbComp>>8)&255);
						int db12 = (b)-((rgbComp>>16)&255);
						int score=dr12*dr12+dg12*dg12+db12*db12;
						if(score<ptrScore[x])
						{
							//NewX(rowAnn[x],x,step,rowAnn)
							FlowType* tmp=rowFlow+2*x;
							memcpy(tmp,tmp-step*2,2*sizeof(FlowType));
							ptrScore[x]=score;
							ptrOcc[x]=100;
						}
					}
					  
					if(ptrOcc[x-step*MaskOcclused->widthStep]) 
					{
						const int preOffest=step*W;
						uint rgbComp=rowRgb[x-preOffest];
						int dr12 = (r)-(rgbComp&255);
						int dg12 = (g)-((rgbComp>>8)&255);
						int db12 = (b)-((rgbComp>>16)&255);
						int score=dr12*dr12+dg12*dg12+db12*db12;
						if(score<ptrScore[x])
						{
							//NewY(rowAnn[x],x,step,rowAnn-preOffest)
							FlowType* tmp=rowFlow+2*x;
							memcpy(tmp,tmp-preOffest*2,2*sizeof(FlowType));
							ptrScore[x]=score;
							ptrOcc[x]=100;
						}
					}
				}
			}
		}
	}
}
void cvtFlowtoANN(BITMAP *ann,FlowType* flo,const int patchSize,IplImage* MaskOcc)
{
	const int halfW=patchSize/2;
	const int w=ann->w-patchSize,h=ann->h-patchSize;
	if(MaskOcc)
	{
		uchar* ptrMaskOcc=(uchar*)MaskOcc->imageData;
		int widthStep=MaskOcc->widthStep;
		for (int y = 0; y <=h; y++) 
		{
			uchar* rowMaskOcc=ptrMaskOcc+(y+halfW)*widthStep;
			FlowType *flo_row = flo+(y+halfW)*ann->w*2+halfW*2;
			int *ann_row = (int *) ann->line[y];
			for (int x = 0; x <= w; x++) 
			{
				int xf=cvRound(flo_row[0]+x);
				int yf=cvRound(flo_row[1]+y);
				flo_row+=2;
				if(xf<0||xf>w||yf<0||yf>h)
				{
					ann_row[x]=UnknownANN/*XY_TO_INT(x,y)*/;/**/
				}
				else if(!rowMaskOcc[(x+halfW)])
					ann_row[x]=UnknownANN/*XY_TO_INT(x,y)*/;
				else
					ann_row[x]=XY_TO_INT(xf,yf);
			}
			memset(ann_row+ann->w-patchSize+1,0,4*(patchSize-1));
		}
	}	else
	{
		for (int y = 0; y <=h; y++) 
		{
			FlowType *flo_row = flo+(y+halfW)*ann->w*2+halfW*2;
			int *ann_row = (int *) ann->line[y];
			for (int x = 0; x <= w; x++) 
			{
				int xf=cvRound(flo_row[0]+x);
				int yf=cvRound(flo_row[1]+y);
				flo_row+=2;
				if(xf<0||xf>w||yf<0||yf>h)
					ann_row[x]=UnknownANN/*XY_TO_INT(x,y)*/;/**/
				else
					ann_row[x]=XY_TO_INT(xf,yf);
			}
			memset(ann_row+ann->w-patchSize+1,0,4*(patchSize-1));
		}
	}
}
void cvtNNtoFlow(BITMAP *ann,FlowType* flo,const int patchSize)
{
	const int halfW=patchSize/2;
	for (int y = 0; y <=ann->h-patchSize; y++) 
	{
		FlowType *flo_row = flo+(y+halfW)*ann->w*2+halfW*2;
		int *ann_row = (int *) ann->line[y];
		for (int x = 0; x <= ann->w-patchSize; x++) 
		{
			int pp = ann_row[x];
			if(pp!=UnknownANN)
			{
				flo_row[0] = INT_TO_X(pp)-x;
				flo_row[1] = INT_TO_Y(pp)-y;
			}
			//else
			//{
			//	int dx= INT_TO_X(pp)-x;
			//	int dy = INT_TO_Y(pp)-y;
			//}
			flo_row+=2;
		}
	}
}
void CheckOccupyConflict(BITMAP* annA,IplImage* MaskConflict,IplImage* MaskOcclused,const int patch_w)
{
	memset(MaskConflict->imageData,0,MaskConflict->widthStep*MaskConflict->height);
	const int halfW=(patch_w/2);
	
	const int w=annA->w-patch_w,h=annA->h-patch_w,wStep=MaskConflict->widthStep;
	uchar* ConMask=(uchar*)MaskConflict->imageData+wStep*halfW+halfW;
	uchar* OccMask=(uchar*)MaskOcclused->imageData+wStep*halfW+halfW;
	for (int y = 0; y <=h; y++) 
	{
		int *ann_row = ((int*) annA->line[y]);
		uchar* rowOccMask=OccMask+y*wStep;
		for (int x = 0; x <= w; x++) 
		{
			//if(x==205&&y==9)
			//	printf("dd");
			if(rowOccMask[x]==0)
				continue;
			const int pp = ann_row[x];
			const int x1 = INT_TO_X(pp);
			const int y1 = INT_TO_Y(pp);
			ConMask[x1+y1*wStep]+=1;
		}
	}
	cvThreshold(MaskConflict,MaskConflict,2,255,CV_THRESH_BINARY);
}
// stack RGB to a single 32bit integer
void convert_bitmap_withGradients(BITMAP* rgbImg,uchar* ptrRgbImg,cv::Mat GrayImg,const int w,const int h,const int channels) 
{
	cv::Mat gx,gy;
	//cv::cvtColor(tmp, tmp,CV_RGB2HSV);
	cv::Sobel(GrayImg,gx,CV_16S,1,0,5);
	//convertScaleAbs( gx, gx );
	cv::Sobel(GrayImg,gy,CV_16S,0,1,5);
	//convertScaleAbs( gy, gy );
	//float maxval=0;
	short* ptrDx=(short*)gx.data;
	short* ptrDy=(short*)gy.data;

	const int widestep=w*channels;
	//unsigned char *data =ptrImg;
	
	BITMAP *ans = rgbImg;
	for (int y = 0; y < h; y++) {
		uint *row = (uint *) ans->line[y];
		uchar *rowSrc = ptrRgbImg+ y * widestep;
		short* rowDx=ptrDx+y*w;
		short* rowDy=ptrDy+y*w;
		for (int x = 0; x < w; x++) 
		{
			int r = rowSrc[0];
			int g = rowSrc[1];
			int b = rowSrc[2];
			
			short dx=rowDx[x];
			short dy=rowDy[x];
			float len=sqrt(dx*dx+dy*dy);
			int grt=len>250.0?125:(len*0.5);
			
			dx=dx/len*60;
			dy=dy/len*60;

			
			rowSrc+=3;
			
			row[0]=r|(g<<8)|(b<<16)|(grt<<24);
			row[1]=((ushort)dx)|(dy<<16);
			
			row+=2;
		}
	}
}
void convert_bitmap_withoutGradients(BITMAP* rgbImg,uchar* ptrRgbImg,const int w,const int h,const int channels) 
{
	const int widestep=w*channels;
	//unsigned char *data =ptrImg;
	BITMAP *ans = rgbImg;
	for (int y = 0; y < h; y++) {
		uint *row = (uint *) ans->line[y];
		uchar *rowSrc = ptrRgbImg+ y * widestep;
		for (int x = 0; x < w; x++) 
		{
			int r = rowSrc[0];
			int g = rowSrc[1];
			int b = rowSrc[2];		
			rowSrc+=3;
			
			*(row)=r|(g<<8)|(b<<16);
			row++;
		}
	}
}
void CheckOcclusion(BITMAP* annA,BITMAP* annB,IplImage* MaskOcclused,const int patch_w)
{
	const int halfW=(patch_w/2);
	const int xMin=halfW;
	const int xMax=annA->w-halfW-1;
	const int yMin=halfW;
	const int yMax=annA->h-halfW-1;
	const int dxy=(halfW<<12)|halfW;
	int* ptrBToA=(int*)annB->data;
	for (int y = 0; y < annA->h; y++) 
	{
		uchar* ptrMask=(uchar*)MaskOcclused->imageData+MaskOcclused->widthStep*y;
		if(y<yMin||y>yMax)
		{
			memset(ptrMask,0,annA->w);
			continue;
		}
		int *ann_row = ((int*) annA->line[y-halfW])-halfW;
		for (int x = 0; x < annA->w; x++) 
		{
			if(x<xMin||x>xMax)
			{
				ptrMask[x]=0;
				continue;
			}
			//if(x==205&&y==9)
			//	printf("dd");
			int pp = ann_row[x];
			const int x1 = INT_TO_X(pp);
			const int y1 = INT_TO_Y(pp);
			const int pp2=ptrBToA[y1*annB->w+x1]+dxy;
			const int x2 = INT_TO_X(pp2)/*+halfW*/;
			const int y2 = INT_TO_Y(pp2)/*+halfW*/;
			ptrMask[x]=(abs(x-x2)+abs(y-y2)<=5)?255:0;
		}
	}
}
void PropagateANNOnly(IplImage* A,IplImage* B,FlowType* Flow,IplImage* MaskOcclused)
{
	int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;//����ͼ��Ĵ�С
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	//BITMAP *ann_sim_final = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	static Params *p = new Params();
	
	if (mode == MODE_IMAGE) 
	{
		a = convert_bitmap(A);
		b = convert_bitmap(B);
		//borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	int enrich_mode = 0;

	p->algo = ALGO_CPU;
	p->patch_w = 7;
	p->nn_iters =5;
	p->rs_max =15;
	p->rs_min = 1;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;
	p->do_randomSearch=0;
	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;

	if (enrich_mode) 
	{
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);
	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	BITMAP *ann = NULL; // NN field
	//BITMAP *annInv = NULL; // NN field
	BITMAP *annd=NULL;
	//BITMAP *anndInv=NULL;
	//BITMAP *annd_final = NULL; // NN patch distance field

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	TStart
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			//annd_final = annd;
			//ann_sim_final = ann_sim;
		} 
		else 
		{
			//ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			ann=create_bitmap(A->width,A->height);
			cvtFlowtoANN(ann,Flow,p->patch_w,0);
			//annInv=CopyBitmap(ann);
			//=======================================
			annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
			
			cvtNNtoFlow(ann,Flow,p->patch_w);
			
			if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
			//annd_final = annd;
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = create_bitmap(annd->w, annd->h);
		//clear(annd_final);
		delete annd;	
	}
	TEnd
	destroy_region_masks(amaskm);
	
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	destroy_bitmap(a);
	destroy_bitmap(b);
	destroy_bitmap(ann);
	//destroy_bitmap(annInv);
	destroy_bitmap(annd);
	//destroy_bitmap(anndInv);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	//destroy_bitmap(ann);
	//destroy_bitmap(annd_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}

void PropagateANNOnly(BITMAP* ann,BITMAP* A,BITMAP* B,FlowType* Flow,IplImage* MaskOcclused)
{
	int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;//����ͼ��Ĵ�С
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	//BITMAP *ann_sim_final = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	static Params *p = new Params();
	//RecomposeParams *rp = new RecomposeParams();
	//BITMAP *borig = NULL;
	if (mode == MODE_IMAGE) 
	{
		a = (A);
		b = (B);
		//borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	int enrich_mode = 0;

	p->algo = ALGO_CPU;
	p->patch_w = 5;
	p->nn_iters =3;
	p->rs_max =15;
	p->rs_min = 1;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;
	p->do_randomSearch=0;
	p->SubPixelsFlow=Flow;
	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;

	if (enrich_mode) 
	{
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);
	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	//BITMAP *ann = NULL; // NN field
	//BITMAP *annInv = NULL; // NN field
	BITMAP *annd=NULL;
	//BITMAP *anndInv=NULL;
	//BITMAP *annd_final = NULL; // NN patch distance field

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	//TStart
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			//annd_final = annd;
			//ann_sim_final = ann_sim;
		} 
		else 
		{
			//ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			//ann=create_bitmap(A->w,A->h);
			cvtFlowtoANN(ann,Flow,p->patch_w,MaskOcclused);
			//annInv=CopyBitmap(ann);
			//=======================================
			//TStart			
#ifdef  IsProRefineWithGradients
			annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
#else
			annd = init_dist_v2(p, a, b, ann, bmask, NULL, amaskm);
			nn_v2(p, a, b, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
#endif
			

			if(p->SubPixelsFlow==NULL)
			cvtNNtoFlow(ann,Flow,p->patch_w);

			
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = create_bitmap(annd->w, annd->h);
		//clear(annd_final);
		delete annd;	
	}
	//TEnd
	destroy_region_masks(amaskm);
	
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	//destroy_bitmap(a);
	//destroy_bitmap(b);
	//destroy_bitmap(ann);
	//destroy_bitmap(annInv);
	destroy_bitmap(annd);
	//destroy_bitmap(anndInv);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	//destroy_bitmap(ann);
	//destroy_bitmap(annd_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}
void ReplaceOccPixels(FlowType* Flow,IplImage* MaskOcclused,BITMAP* OccReplaceMap,const int patch_w)
{
#define mSwap(x,y,tmp)	{(tmp)=(x);(x)=(y);(y)=(tmp);}
	const int xScale=cvRound(MaskOcclused->width/(float)OccReplaceMap->w);
	const int yScale=cvRound(MaskOcclused->height/(float)OccReplaceMap->h);
	uchar* MaskOcc=(uchar*)MaskOcclused->imageData;
	const int xBorder=(patch_w/2)*xScale,yBorder=(patch_w/2)*yScale;
	const int wMap=OccReplaceMap->w,W=MaskOcclused->width;
	const int numPad=xScale*yScale;
	

	for(int y=yBorder;y<MaskOcclused->height-yBorder;y++)
	{
		uchar* rowMaskOcc=MaskOcc+y*MaskOcclused->widthStep;
		int yd=y/yScale;
		int* rowReplaceMap=(int*)OccReplaceMap->line[yd];
		FlowType* dstFlow=Flow+(y*W)*2;
		for(int x=xBorder;x<MaskOcclused->width-xBorder;x++)
		{
			if(rowMaskOcc[x]==0)
			{
				int xd=x/xScale;			
				int pp=rowReplaceMap[xd];
				if(pp!=UnknownANN)
				{
					int xNew= INT_TO_X(pp)*xScale;
					int yNew= INT_TO_Y(pp)*yScale;
					FlowType* rowFlow=Flow+(yNew*W+xNew)*2;
					//int count=0;
					FlowType meanX=0.0,meanY=0.0;
					for(int yy=0;yy<yScale;yy++)
					{
						for(int xx=0;xx<xScale;xx++)
						{
							//flowPad[count][0]=rowFlow[xx*2];
							//flowPad[count][1]=rowFlow[xx*2+1];
							meanX+=rowFlow[xx*2];
							meanY+=rowFlow[xx*2+1];
							//count++;
						}
						rowFlow+=(2*W);
					}
					//int tmp=0;
					meanX/=numPad;meanY/=numPad;
					
					{
					dstFlow[x*2]=meanX/*flowPad[count/2-1][0]*/;
					dstFlow[x*2+1]=meanY/*flowPad[count/2-1][1]*/;
					}
					rowMaskOcc[x]=255;
				}
			}
		}
	}
}
void RemoveOutlines(IplImage* MaskOcclused,const int WindowSizeOfSmooth)
{
	const int w=MaskOcclused->width,h=MaskOcclused->height,wstep=MaskOcclused->widthStep;
	const int RowLineSize=w+1;
	static uint* mInterOptFlow=new uint[(w+1)*(h+1)];
	uchar* mOptFlow=(uchar*)MaskOcclused->imageData;
	memset(mInterOptFlow,0,sizeof(uint)*RowLineSize);
	for(int y=0;y<h;y++)
	{
		const int startIndex=y*wstep;
		const int startIndex2=(1+(y+1)*RowLineSize);
		const uchar* srcXYZ=mOptFlow+startIndex;
		uint* dstXYZ=mInterOptFlow+startIndex2;
		*(dstXYZ-1)=0;
		uint rowX=0;
		for(int x=0;x<w;x++)
		{
			rowX+=((*(srcXYZ++))>>7);
			*dstXYZ=*(dstXYZ-RowLineSize)+rowX;
			dstXYZ++;
		}
	}
	const int RowWindowSize=WindowSizeOfSmooth;
	const uint PointsNum=WindowSizeOfSmooth*WindowSizeOfSmooth*0.6;
	for (int y=WindowSizeOfSmooth/2;y<h-WindowSizeOfSmooth/2;y++)
	{
		uchar* curRowFlow=mOptFlow+(y*wstep+WindowSizeOfSmooth/2);
		const uint* curRowTermsUp=mInterOptFlow+(y-WindowSizeOfSmooth/2)*RowLineSize;
		const uint* curRowTermsDown=curRowTermsUp+WindowSizeOfSmooth*RowLineSize;
		for (int x=WindowSizeOfSmooth/2;x<w-WindowSizeOfSmooth/2;x++)
		{
			const uint TermX=*curRowTermsUp-*(curRowTermsUp+RowWindowSize)-*curRowTermsDown+*(curRowTermsDown+RowWindowSize);
			curRowTermsUp++;curRowTermsDown++;
			if((*curRowFlow)&&TermX<PointsNum)
			{
				*(curRowFlow)=0;
			}
			curRowFlow++;
		}
	}
}
void InitFlowFieldByANN(BITMAP* A,BITMAP* B,FlowType* Flow,IplImage* MaskOcclused,BITMAP* OccReplaceMap)
{
	int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	//BITMAP *ann_sim_final = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	static Params *p = new Params();
	//RecomposeParams *rp = new RecomposeParams();
	//BITMAP *borig = NULL;
	if (mode == MODE_IMAGE) 
	{
		a = A;
		b = B;
		//borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	int enrich_mode = 0;

	p->algo = ALGO_CPU;
	p->patch_w = 5;
	p->nn_iters =5;
	p->rs_max =15;
	p->rs_min = 2;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;
	//p->do_randomSearch=0;
	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;

	if (enrich_mode) 
	{
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);//��ʼ��OpenMP��صĲ���
	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	BITMAP *ann = NULL; // NN field
	BITMAP *annInv = NULL; // NN field
	BITMAP *annd=NULL;
	BITMAP *anndInv=NULL;
	//BITMAP *annd_final = NULL; // NN patch distance field

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	//TStart
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			//annd_final = annd;
			//ann_sim_final = ann_sim;
		} 
		else //
		{
			ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			annInv=CopyBitmap(ann);
			//=======================================
			annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
			//=======================================	
			anndInv = init_dist(p, b, a, annInv, 0, NULL, 0);
			nn(p, b, a, annInv, anndInv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0); 
			
			CheckOcclusion(ann,annInv,MaskOcclused,p->patch_w);
			
			RemoveOutlines(MaskOcclused,5);
			
			ConvertAnnAndOcclusionFill(Flow,ann,annInv,a,MaskOcclused,p->patch_w,OccReplaceMap);
			
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = create_bitmap(annd->w, annd->h);
		//clear(annd_final);
		delete annd;	
	}
	//TEnd
	destroy_region_masks(amaskm);
	
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	//destroy_bitmap(a);
	//destroy_bitmap(b);
	destroy_bitmap(ann);
	destroy_bitmap(annInv);
	destroy_bitmap(annd);
	destroy_bitmap(anndInv);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	//destroy_bitmap(ann);
	//destroy_bitmap(annd_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}

void InitFlowFieldByANN(IplImage* A,IplImage* B,FlowType* Flow,IplImage* MaskOcclused)
{
	int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;//����ͼ��Ĵ�С
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	//BITMAP *ann_sim_final = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	static Params *p = new Params();
	//RecomposeParams *rp = new RecomposeParams();
	//BITMAP *borig = NULL;
	if (mode == MODE_IMAGE) 
	{
		a = convert_bitmap(A);
		b = convert_bitmap(B);
		//borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	int enrich_mode = 0;

	p->algo = ALGO_CPU;
	p->patch_w = 7;
	p->nn_iters =5;
	p->rs_max =15;
	p->rs_min = 1;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;
	//p->do_randomSearch=0;
	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;

	if (enrich_mode) 
	{
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);//
	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	BITMAP *ann = NULL; // NN field
	BITMAP *annInv = NULL; // NN field
	BITMAP *annd=NULL;
	BITMAP *anndInv=NULL;
	//BITMAP *annd_final = NULL; // NN patch distance field

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	TStart
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			//annd_final = annd;
			//ann_sim_final = ann_sim;
		} 
		else 
		{
			ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			annInv=CopyBitmap(ann);
			//=======================================
			annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
			//=======================================	
			anndInv = init_dist(p, b, a, annInv, 0, NULL, 0);
			nn(p, b, a, annInv, anndInv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0); 
			//CenterAnnField(ann, p->patch_w);
			//CenterAnnField(annInv, p->patch_w);
			//=======================================
			//IplImage* MaskOcclused=cvCreateImage(cvGetSize(A),8,1);
			CheckOcclusion(ann,annInv,MaskOcclused,p->patch_w);
			//cvErode(MaskOcclused,MaskOcclused);
			cvShowImage("Occlosion",MaskOcclused);

			//cvtNNtoFlow(ann,Flow,p->patch_w);
			//PropagateFill(Flow,a,MaskOcclused);

			ConvertAnnAndOcclusionFill(Flow,ann,annInv,a,MaskOcclused,p->patch_w,0);

			//PropagateFill(ann,a,MaskOcclused);
			//if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
			//annd_final = annd;
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, 0, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, 0, NULL, amaskm, p->cores);  
		//annd_final = create_bitmap(annd->w, annd->h);
		//clear(annd_final);
		delete annd;	
	}
	TEnd
	destroy_region_masks(amaskm);
	
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	destroy_bitmap(a);
	destroy_bitmap(b);
	destroy_bitmap(ann);
	destroy_bitmap(annInv);
	destroy_bitmap(annd);
	destroy_bitmap(anndInv);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	//destroy_bitmap(ann);
	//destroy_bitmap(annd_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}
void ComputeBothSideANN(IplImage* A,IplImage* B,FlowType* Flow) 
{
	
	int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;//����ͼ��Ĵ�С
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	Params *p = new Params();
	RecomposeParams *rp = new RecomposeParams();
	BITMAP *borig = NULL;
	if (mode == MODE_IMAGE) 
	{
		a = convert_bitmap(A);
		b = convert_bitmap(B);
		borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	p->algo = ALGO_CPU;
	int enrich_mode = 0;

	p->patch_w = 7;
	p->nn_iters =5 ;
	p->rs_max =100/4.0;
	p->rs_min = 1;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;

	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;
	if (ann_window&&!awinsize&&!win_size) {
		mexErrMsgTxt("\nUsing ann_window - either awinsize or win_size should be defined.\n");
	}

	if (enrich_mode) {
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);//
	if (sim_mode) 
	{
		//init_xform_tables(scalemin, scalemax, 1);
	}

	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	BITMAP *ann = NULL; // NN field
	BITMAP *annInv = NULL; // NN field
	BITMAP *annd_final = NULL; // NN patch distance field
	BITMAP *ann_sim_final = NULL;

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	TStart
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			annd_final = annd;
			ann_sim_final = ann_sim;
		} 
		else //
		{
			ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			annInv=CopyBitmap(ann);
			//=======================================
			BITMAP *annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
			//=======================================	
			BITMAP *anndInv = init_dist(p, b, a, annInv, 0, NULL, 0);
			nn(p, b, a, annInv, anndInv, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0); 
			//CenterAnnField(ann, p->patch_w);
			//CenterAnnField(annInv, p->patch_w);
			//=======================================
			IplImage* MaskOcclused=cvCreateImage(cvGetSize(A),8,1);
			CheckOcclusion(ann,annInv,MaskOcclused,p->patch_w);
			cvtNNtoFlow(ann,Flow,p->patch_w);
			PropagateFill(Flow,a,MaskOcclused);
			//PropagateFill(ann,a,MaskOcclused);
			cvShowImage("Occlosion",MaskOcclused);
			if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
			annd_final = annd;
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
		annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
		annd_final = create_bitmap(annd->w, annd->h);
		clear(annd_final);
		delete annd;	
	}
	TEnd
		destroy_region_masks(amaskm);
	//TEnd
	//=============================================================
	//if(Flow)
	//	cvtNNtoFlow(ann,Flow,p->patch_w);
	IplImage* Results=cvCreateImage(cvGetSize(B),8,3);
	//VoteNN(Results, B,ann) ;
	cvShowImage("Src",A);
	cvShowImage("Des",B);
	//cvShowImage("ReBuild",Results);
	cvWaitKey(100000000);
	cvReleaseImage(&Results);
	//=============================================================
	// clean up
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	delete p;
	delete rp;
	destroy_bitmap(a);
	destroy_bitmap(borig);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	destroy_bitmap(ann);
	destroy_bitmap(annd_final);
	destroy_bitmap(ann_sim_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}

void ComputeANN(IplImage* A,IplImage* B,FlowType* Flow) 
{
	TStart
		int mode = MODE_IMAGE;
	int aw = -1, ah = -1, bw = -1, bh = -1;//
	BITMAP *a = NULL, *b = NULL, *ann_prev = NULL, *ann_window = NULL, *awinsize = NULL;
	VECBITMAP<unsigned char> *ab = NULL, *bb = NULL;
	VECBITMAP<float> *af = NULL, *bf = NULL;

	Params *p = new Params();
	RecomposeParams *rp = new RecomposeParams();
	BITMAP *borig = NULL;
	if (mode == MODE_IMAGE) 
	{
		a = convert_bitmap(A);
		b = convert_bitmap(B);
		borig = b;
		aw = a->w; ah = a->h;
		bw = b->w; bh = b->h;
	} 
	else if (mode == MODE_VECB) 
	{
		//ab = convert_vecbitmap<unsigned char>(A);
		//bb = convert_vecbitmap<unsigned char>(B);
		if (ab->n != bb->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = ab->w; ah = ab->h;
		bw = bb->w; bh = bb->h;
		p->vec_len = ab->n;
	} 
	else if (mode == MODE_VECF) 
	{
		//af = convert_vecbitmap<float>(A);
		//bf = convert_vecbitmap<float>(B);
		if (af->n != bf->n) { mexErrMsgTxt("3rd dimension differs"); }
		aw = af->w; ah = af->h;
		bw = bf->w; bh = bf->h;
		p->vec_len = af->n;
	}

	double *win_size = NULL;
	BITMAP *amask = NULL, *bmask = NULL;

	//double scalemin = 0.5, scalemax = 2.0;  // The product of these must be one.
	/* parse parameters */
	int i = 2;
	int sim_mode = 0;
	int knn_chosen = -1;
	p->algo = ALGO_CPU;
	int enrich_mode = 0;

	p->patch_w = 7;
	p->nn_iters =5 ;
	p->rs_max =100000;
	p->rs_min = 1;
	p->rs_ratio = 0.5;
	p->rs_iters =1.0;
	p->cores = 1;
	bmask = NULL; // XC+
	p->window_h = p->window_w =INT_MAX;

	// [ann_prev=NULL], [ann_window=NULL], [awinsize=NULL], 
	if(0)
	{
		//ann_prev = convert_field(p, NULL, bw, bh, clip_count);       // Bug fixed by Connelly
		//ann_window = convert_field(p, NULL, bw, bh, clip_count);      
		//awinsize = convert_winsize_field(p, NULL, aw, ah);  
	}
	knn_chosen = 0;
	if (ann_window&&!awinsize&&!win_size) {
		mexErrMsgTxt("\nUsing ann_window - either awinsize or win_size should be defined.\n");
	}

	if (enrich_mode) {
		int nn_iters = p->nn_iters;
		p->enrich_iters = nn_iters/2;
		p->nn_iters = 2;
		if (A != B) { mexErrMsgTxt("\nOur implementation of enrichment requires that image A = image B.\n"); }
		if (mode == MODE_IMAGE) {
			b = a;
		} else {
			mexErrMsgTxt("\nEnrichment only implemented for 3 channel uint8 inputs.\n");
		}
	}
	init_params(p);//
	if (sim_mode) 
	{
		//init_xform_tables(scalemin, scalemax, 1);
	}

	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;

	BITMAP *ann = NULL; // NN field
	BITMAP *annd_final = NULL; // NN patch distance field
	BITMAP *ann_sim_final = NULL;

#ifdef WithKnn
	VBMP *vann_sim = NULL;
	VBMP *vann = NULL;
	VBMP *vannd = NULL;
#endif
	if (mode == MODE_IMAGE) 
	{
		// input as RGB image
		if (!a || !b) { mexErrMsgTxt("internal error: no a or b image"); }
		if (knn_chosen > 1) 
		{
#ifdef WithKnn
			p->knn = knn_chosen;
			if (sim_mode) { mexErrMsgTxt("rotating+scaling patches not implemented with knn (actually it is implemented it is not exposed by the wrapper)"); }
			PRINCIPAL_ANGLE *pa = NULL;
			vann_sim = NULL;
			vann = knn_init_nn(p, a, b, vann_sim, pa);
			vannd = knn_init_dist(p, a, b, vann, vann_sim);
			knn(p, a, b, vann, vann_sim, vannd, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pa);
#endif
			//      sort_knn(p, vann, vann_sim, vannd);
		} else if (sim_mode) 
		{
			BITMAP *ann_sim = NULL;
			ann = sim_init_nn(p, a, b, ann_sim);
			BITMAP *annd = sim_init_dist(p, a, b, ann, ann_sim);
			sim_nn(p, a, b, ann, ann_sim, annd);
			if (ann_prev) { mexErrMsgTxt("when searching over rotations+scales, previous guess is not supported"); }
			annd_final = annd;
			ann_sim_final = ann_sim;
		} 
		else //����������ͨ��ͼ���ANN
		{
			ann = init_nn(p, a, b, bmask, NULL, amaskm, 1, ann_window, awinsize);
			BITMAP *annd = init_dist(p, a, b, ann, bmask, NULL, amaskm);
			nn(p, a, b, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores, ann_window, awinsize); 
			if (ann_prev) minnn(p, a, b, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
			annd_final = annd;
		}
	} 
	else if(mode == MODE_VECB) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		//    mexPrintf("mode vecb_xc %dx%dx%d, %dx%dx%d\n", ab->w, ab->h, ab->n, bb->w, bb->h, bb->n);
		// input as uint8 discriptors per pixel
		if (!ab || !bb) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<unsigned char>(p, ab, bb, bmask, NULL, amaskm);
		VECBITMAP<int> *annd = XCvec_init_dist<unsigned char, int>(p, ab, bb, ann, bmask, NULL, amaskm);
		XCvec_nn<unsigned char, int>(p, ab, bb, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<unsigned char, int>(p, ab, bb, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
		annd_final = vecbitmap_to_bitmap(annd);
		delete annd;
	} else if(mode == MODE_VECF) 
	{
		if (sim_mode) { mexErrMsgTxt("internal error: rotation+scales not implemented with descriptor mode"); }
		if (knn_chosen > 1) { mexErrMsgTxt("internal error: kNN not implemented with descriptor mode"); }
		// input as float/double discriptors per pixel
		if (!af || !bf) { mexErrMsgTxt("internal error: no a or b image"); }
		ann = XCvec_init_nn<float>(p, af, bf, bmask, NULL, amaskm);
		VECBITMAP<float> *annd = XCvec_init_dist<float, float>(p, af, bf, ann, bmask, NULL, amaskm);
		XCvec_nn<float, float>(p, af, bf, ann, annd, amaskm, bmask, 0, 0, rp, 0, 0, 0, NULL, p->cores); 
		if (ann_prev) XCvec_minnn<float, float>(p, af, bf, ann, annd, ann_prev, bmask, 0, 0, rp, NULL, amaskm, p->cores);  
		annd_final = create_bitmap(annd->w, annd->h);
		clear(annd_final);
		delete annd;	
	}
	TEnd
		destroy_region_masks(amaskm);
	//TEnd
	//=============================================================
	if(Flow)
		cvtNNtoFlow(ann,Flow,p->patch_w);
	IplImage* Results=cvCreateImage(cvGetSize(B),8,3);
	VoteNN(Results, B,ann) ;
	cvShowImage("Src",A);
	cvShowImage("Des",B);
	cvShowImage("ReBuild",Results);
	cvWaitKey(100000000);
	cvReleaseImage(&Results);
	//=============================================================
	// clean up
#ifdef WithKnn
	delete vann;
	delete vann_sim;
	delete vannd;
#endif
	delete p;
	delete rp;
	destroy_bitmap(a);
	destroy_bitmap(borig);
	delete ab;
	delete bb;
	delete af;
	delete bf;
	destroy_bitmap(ann);
	destroy_bitmap(annd_final);
	destroy_bitmap(ann_sim_final);
	if (ann_prev) destroy_bitmap(ann_prev);
	if (ann_window) destroy_bitmap(ann_window);
	if (awinsize) destroy_bitmap(awinsize);
}

void InitNextOccMask(IplImage* preMask,IplImage* curMask)
{
	//
	//
#define SetBigMaskVal(val,w,h)		{uchar* ptr=rowCur+x*xScale;\
														for(int i=0;i<(h);i++)\
														{\
															memset(ptr,(val),(w));\
															ptr+=curWS;\
														}}

	const int xScale=cvRound(curMask->width/(float)preMask->width);
	const int yScale=cvRound(curMask->height/(float)preMask->height);
	const int preWS=preMask->widthStep,
				  curWS=curMask->widthStep;
	uchar* ptrPre=(uchar*)preMask->imageData;
	uchar* ptrCur=(uchar*)curMask->imageData;

	memset(ptrCur+curWS,255,curWS*(curMask->height-2));
	memset(ptrCur,0,curWS*yScale);
	memset(ptrCur+curWS*(curMask->height-yScale),0,curWS*yScale);
	for(int y=0;y<curMask->height;y++)
	{
		memset(ptrCur+y*curWS,0,xScale);
		memset(ptrCur+y*curWS+curMask->width-xScale,0,xScale);
		//ptrCur[y*curWS+0]=0;
		//ptrCur[y*curWS+curMask->width-1]=0;
	}
	for(int y=1;y<preMask->height-1;y++)
	{
		uchar* rowPre=ptrPre+y*preWS;
		uchar* rowPreUp=rowPre-preWS;
		uchar* rowPreDown=rowPre+preWS;

		uchar* rowCur=ptrCur+y*yScale*curWS;

		for(int x=1;x<preMask->width-1;x++)
		{
			if(!rowPre[x])
			{
				//int ww=xScale,hh=yScale;
				if(rowPre[x-1]||rowPre[x+1]||rowPreUp[x]||rowPreDown[x])
				{//
					SetBigMaskVal(100,xScale,yScale)
				}
				else
				{
					SetBigMaskVal(0,xScale,yScale)
				}
			}
		}
	}
}