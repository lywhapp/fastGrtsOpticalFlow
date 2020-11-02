#include "LywsMex.h"
void mexErrMsgTxt(char* msg)
{
	printf("error: %s\n",msg);
	exit(-1);
}
int mxGetNumberOfDimensions(const mxArray *A)
{
	return (A->w>0)+(A->h>0)+(A->channel>0);
}
int* mxGetDimensions(const mxArray *A)
{
	static int Dim[3];
	Dim[0]=A->h;
	Dim[1]=A->w;
	Dim[2]=A->channel;
	return Dim;
}
bool mxIsUint8(const mxArray *A)
{
	return A->type==mexUint8;
}
bool mxIsSingle(const mxArray *A)
{
	return A->type==mexSingle;
}
bool mxIsDouble(const mxArray *A)
{
	return A->type==mexDouble;
}
bool mxIsInt32(const mxArray *A)
{
	return A->type==mexInt32;
}
bool mxIsChar(const mxArray *A)
{
	return A->type==mexChar;
}
unsigned char* mxGetData(const mxArray *A)
{
	return A->data;
}