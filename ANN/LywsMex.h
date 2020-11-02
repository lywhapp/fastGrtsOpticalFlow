#ifndef LywsMex
#define LywsMex
#include <stdio.h>
#include <stdlib.h>
#define mexUint8 0
#define mexSingle 1
#define mexDouble 2
#define mexInt32 3
#define mexChar 4
struct mxArray
{
	unsigned char* data;
	int type;
	int w;
	int h;
	int channel;
};
void mexErrMsgTxt(char* msg);
int mxGetNumberOfDimensions(const mxArray *A);
int* mxGetDimensions(const mxArray *A);
bool mxIsUint8(const mxArray *A);
bool mxIsSingle(const mxArray *A);
bool mxIsDouble(const mxArray *A);
bool mxIsInt32(const mxArray *A);
bool mxIsChar(const mxArray *A);
unsigned char* mxGetData(const mxArray *A);
#endif