
// MATLAB interface, for vote().

#include "allegro_emu.h"
//#include "mex.h"
#include "nn.h"
//#include "matrix.h"
#include "mexutil.h"

void cvtBITMAPToCv(IplImage* Des,BITMAP *Src)
{
	uchar* data=(uchar*)Des->imageData;
	int widestep=Des->widthStep;
	for(int y=0;y<Src->h;y++)
	{
		int *row = (int *) Src->line[y];
		for(int x=0;x<Src->w;x++)
		{
			unsigned char *base = data + x*3 + y * widestep;
			int val=row[x];
			base[0]=val&255;
			base[1]=(val>>8)&255;
			base[2]=(val>>16)&255;
		}
	}
}
void VoteNN(IplImage* Results, IplImage* Source,BITMAP *ann, BITMAP *bnn) 
{
	BITMAP *b = convert_bitmap(Source);

	Params *p = new Params();
	// [bnn=[]], [algo='cpu'], [patch_w=7], [bmask=[]], [bweight=[]], [coherence_weight=1], [complete_weight=1], [amask=[]], [aweight=[]]
	BITMAP *bmask = NULL, *bweight = NULL, *amask = NULL, *aweight = NULL, *ainit = NULL;
	double coherence_weight = 1, complete_weight = 1;
	int i = 2;
	p->algo = ALGO_CPU;
	p->patch_w = 7;
	bmask = NULL;
	bweight = NULL;
	coherence_weight = 1;
	complete_weight = 1;
	amask = NULL;
	aweight =NULL;
	ainit = NULL;

	//int aclip = 0, bclip = 0;
	//int nclip = aclip + bclip;
	//if (nclip) printf("Warning: clipped %d votes (%d a -> b, %d b -> a)\n", nclip, aclip, bclip);

	RegionMasks *amaskm = amask ? new RegionMasks(p, amask): NULL;
	//BITMAP *a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit);
	// David Jacobs -- Added mask_self_only as true for multiple.
	BITMAP *a = vote(p, b, ann, bnn, bmask, bweight, coherence_weight, complete_weight, amaskm, aweight, ainit, NULL, NULL, 1);
	cvtBITMAPToCv(Results,a);
	destroy_region_masks(amaskm);

	delete p;
	destroy_bitmap(a);
	destroy_bitmap(ainit);
	destroy_bitmap(b);
	//destroy_bitmap(ann);
	destroy_bitmap(bnn);
	destroy_bitmap(bmask);
	destroy_bitmap(bweight);
	//  destroy_bitmap(amask);
	destroy_bitmap(aweight);
}
