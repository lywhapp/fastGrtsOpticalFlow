#include "allegro_emu.h"
//#include "mex.h"
#include "nn.h"
//#include "matrix.h"
#include "simnn.h"
#include "mexutil.h"

#include "knn.h"


#define Flag_GoodPixel 255	
#define Flag_PossibleGoodPixel 100		
#define Flag_OccPixel 0		
void ComputeANN(IplImage* A,IplImage* B,FlowType* Flow=NULL) ;
void ComputeBothSideANN(IplImage* A,IplImage* B,FlowType* Flow=NULL) ;
void InitFlowFieldByANN(IplImage* A,IplImage* B,FlowType* Flow,IplImage* MaskOcclused);
void InitFlowFieldByANN(BITMAP* A,BITMAP* B,FlowType* Flow,IplImage* MaskOcclused,BITMAP* OccReplaceMap);
void PropagateANNOnly(IplImage* A,IplImage* B,FlowType* Flow,IplImage* MaskOcclused);
void PropagateANNOnly(BITMAP* ann,BITMAP* A,BITMAP* B,FlowType* Flow,IplImage* MaskOcclused);
void convert_bitmap_withoutGradients(BITMAP* rgbImg,uchar* ptrImg,const int w,const int h,const int channels);
void convert_bitmap_withGradients(BITMAP* rgbImg,uchar* ptrRgbImg,cv::Mat GrayImg,const int w,const int h,const int channels) ;

void InitNextOccMask(IplImage* preMask,IplImage* curMask);
void ReplaceOccPixels(FlowType* Flow,IplImage* MaskOcclused,BITMAP* OccReplaceMap,const int patch_w);
void FillBorder(FlowType* Flow,BITMAP* Img,const int bdWidth,IplImage* OccMask);
void FillBorder(FlowType* Flow,BITMAP* Img,const int bdWidth);