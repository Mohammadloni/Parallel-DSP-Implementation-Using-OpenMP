/**********************************************************
*   Copyright (C) SEI, 2015.
*   All rights reserved.
*
*   ===== TransMat.c =====
*   def file for CPU 
*
*   Version 1.00
*
*   Created:	 Mohamamd Loni
*	
***********************************************************/

#include "stdafx.h"
#include "Filter.h"
#define CHUNK_SIZE 133
#define NEW_OMP
#define VRE(Arr,I,J,R) Arr->data[J * R + I].re // Value RE
#define VIM(Arr,I,J,R) Arr->data[J * R + I].im // Value IM

extern void Filter3Coef( Array_real_T *b, Array_real_T *a, const Array_creal_T *x, int32_T dim, Array_creal_T *y )
{
	int32_T Start_Index;
	int32_T idx2;
	int32_T R_C;
	int32_T idx1;
	creal_T *x_Ptr;
	creal_T *y_Ptr;
	BOOLEAN Omp_En = TRUE;

	ASSERT(b->size[0] == 3);
	ASSERT(a->size[0] == 3);
	ASSERT(dim == 2 || dim == 1);
	ASSERT(x->size[1] >= 3);

#ifndef NEW_OMP
	// TODO : This Implementation reduce precision
	if ( dim == 2 ) {
		R_C = x->size[0]; // Row Count
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) private(idx1, idx2) firstprivate(R_C) if(Omp_En) 
		for (idx1 = 0; idx1 < R_C; idx1++)
		{
			if ( x->size[1] >= 2 )
			{
				VRE(y, idx1, 0, R_C) = b->data[0] * VRE(x, idx1, 0, R_C);
				VIM(y, idx1, 0, R_C) = b->data[0] * VIM(x, idx1, 0, R_C);

				VRE(y, idx1, 1, R_C) = b->data[0] * VRE(x, idx1, 1, R_C) + b->data[1] * VRE(x, idx1, 0, R_C) - a->data[1] * VRE(y, idx1, 0, R_C);
				VIM(y, idx1, 1, R_C) = b->data[0] * VIM(x, idx1, 1, R_C) + b->data[1] * VIM(x, idx1, 0, R_C) - a->data[1] * VIM(y, idx1, 0, R_C);

			} else {
				VRE(y, idx1, 0, R_C) = b->data[0] * VRE(x, idx1, 0, R_C);
				VIM(y, idx1, 0, R_C) = b->data[0] * VIM(x, idx1, 0, R_C);
			}

			for (idx2 = 2; idx2 < x->size[1]; idx2++) 
			{
				VRE(y, idx1, idx2, R_C) = b->data[0] * VRE(x, idx1, idx2, R_C) + b->data[1] * VRE(x, idx1, (idx2-1), R_C) 
					+ b->data[2] * VRE(x, idx1, (idx2-2), R_C) - a->data[1] * VRE(y,idx1, (idx2-1), R_C) - a->data[2] * VRE(y, idx1, (idx2-2), R_C);

				VIM(y, idx1, idx2, R_C) = b->data[0] * VIM(x, idx1, idx2, R_C) + b->data[1] * VIM(x, idx1, (idx2-1), R_C) 
					+ b->data[2] * VIM(x, idx1, (idx2-2), R_C) - a->data[1] * VIM(y,idx1, (idx2-1), R_C) - a->data[2] * VIM(y, idx1, (idx2-2), R_C);
			}
		}
	} else {
		R_C = x->size[0]; // Row Count
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) private(idx1, idx2, Start_Index, x_Ptr, y_Ptr) firstprivate(R_C) if(Omp_En) 
		for (idx1 = 0; idx1 < x->size[1]; idx1++)
		{
			Start_Index = idx1 * R_C;
			x_Ptr = &x->data[Start_Index];
			y_Ptr = &y->data[Start_Index];

			if ( x->size[0] >= 2 )
			{
				y_Ptr[0].re = b->data[0] * x_Ptr[0].re;
				y_Ptr[0].im = b->data[0] * x_Ptr[0].im;

				y_Ptr[1].re = b->data[0] * x_Ptr[1].re + b->data[1] * x_Ptr[0].re - a->data[1] * y_Ptr[0].re;
				y_Ptr[1].im = b->data[0] * x_Ptr[1].im + b->data[1] * x_Ptr[0].im - a->data[1] * y_Ptr[0].im;

			} else {
				y_Ptr[0].re = b->data[0] * x_Ptr[0].re;
				y_Ptr[0].im = b->data[0] * x_Ptr[0].im;
			}

			for (idx2 = 2; idx2 < x->size[0]; idx2++) 
			{
				y_Ptr[idx2].re = b->data[0] * x_Ptr[idx2].re + b->data[1] * x_Ptr[idx2-1].re 
					+ b->data[2] * x_Ptr[idx2-2].re - a->data[1] * y_Ptr[idx2-1].re - a->data[2] * y_Ptr[idx2-2].re;
				y_Ptr[idx2].im = b->data[0] * x_Ptr[idx2].im + b->data[1] * x_Ptr[idx2-1].im 
					+ b->data[2] * x_Ptr[idx2-2].im - a->data[1] * y_Ptr[idx2-1].im - a->data[2] * y_Ptr[idx2-2].im;
			}
		}
	}
#else
	if ( dim == 2 ) {
		R_C = x->size[0]; // Row Count
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) private(idx1, idx2) firstprivate(R_C) if(Omp_En)
		for (idx1 = 0; idx1 < R_C; idx1++)
		{
			VRE(y, idx1, 0, R_C) = b->data[0] * VRE(x, idx1, 0, R_C);
			VIM(y, idx1, 0, R_C) = b->data[0] * VIM(x, idx1, 0, R_C);

			VRE(y, idx1, 1, R_C) = b->data[0] * VRE(x, idx1, 1, R_C) + b->data[1] * VRE(x, idx1, 0, R_C) - a->data[1] * VRE(y, idx1, 0, R_C);
			VIM(y, idx1, 1, R_C) = b->data[0] * VIM(x, idx1, 1, R_C) + b->data[1] * VIM(x, idx1, 0, R_C) - a->data[1] * VIM(y, idx1, 0, R_C);

			for (idx2 = 2; idx2 < x->size[1]; idx2++) 
			{
				VRE(y, idx1, idx2, R_C) = b->data[0] * VRE(x, idx1, idx2, R_C) + b->data[1] * VRE(x, idx1, (idx2-1), R_C) 
					+ b->data[2] * VRE(x, idx1, (idx2-2), R_C) - a->data[1] * VRE(y,idx1, (idx2-1), R_C) - a->data[2] * VRE(y, idx1, (idx2-2), R_C);

				VIM(y, idx1, idx2, R_C) = b->data[0] * VIM(x, idx1, idx2, R_C) + b->data[1] * VIM(x, idx1, (idx2-1), R_C) 
					+ b->data[2] * VIM(x, idx1, (idx2-2), R_C) - a->data[1] * VIM(y,idx1, (idx2-1), R_C) - a->data[2] * VIM(y, idx1, (idx2-2), R_C);
			}
		}
	} else {
		R_C = x->size[0]; // Row Count
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) private(idx1, idx2, Start_Index, x_Ptr, y_Ptr) firstprivate(R_C) if(Omp_En) 
		for (idx1 = 0; idx1 < x->size[1]; idx1++)
		{
			Start_Index = idx1 * R_C;
			x_Ptr = &x->data[Start_Index];
			y_Ptr = &y->data[Start_Index];

			y_Ptr[0].re = b->data[0] * x_Ptr[0].re;
			y_Ptr[0].im = b->data[0] * x_Ptr[0].im;

			y_Ptr[1].re = b->data[0] * x_Ptr[1].re + b->data[1] * x_Ptr[0].re - a->data[1] * y_Ptr[0].re;
			y_Ptr[1].im = b->data[0] * x_Ptr[1].im + b->data[1] * x_Ptr[0].im - a->data[1] * y_Ptr[0].im;

			for (idx2 = 2; idx2 < R_C; idx2++) 
			{
				y_Ptr[idx2].re = b->data[0] * x_Ptr[idx2].re + b->data[1] * x_Ptr[idx2-1].re 
					+ b->data[2] * x_Ptr[idx2-2].re - a->data[1] * y_Ptr[idx2-1].re - a->data[2] * y_Ptr[idx2-2].re;
				y_Ptr[idx2].im = b->data[0] * x_Ptr[idx2].im + b->data[1] * x_Ptr[idx2-1].im 
					+ b->data[2] * x_Ptr[idx2-2].im - a->data[1] * y_Ptr[idx2-1].im - a->data[2] * y_Ptr[idx2-2].im;
			}
		}
//		R_C = x->size[0]; // Row Count
//#pragma omp parallel for schedule(dynamic, CHUNK_SIZE) private(idx1, idx2) firstprivate(R_C) if(Omp_En)
//		for (idx1 = 0; idx1 < x->size[1]; idx1++)
//		{
//			VRE(y, 0, idx1, R_C) = b->data[0] * VRE(x, 0, idx1, R_C);
//			VIM(y, 0, idx1, R_C) = b->data[0] * VIM(x, 0, idx1, R_C);
//
//			VRE(y, 1, idx1, R_C) = b->data[0] * VRE(x, 1, idx1, R_C) + b->data[1] * VRE(x, 0, idx1, R_C) - a->data[1] * VRE(y, 0, idx1, R_C);
//			VIM(y, 1, idx1, R_C) = b->data[0] * VIM(x, 1, idx1, R_C) + b->data[1] * VIM(x, 0, idx1, R_C) - a->data[1] * VIM(y, 0, idx1, R_C);
//
//			for (idx2 = 2; idx2 < x->size[0]; idx2++) 
//			{
//				VRE(y, idx2, idx1, R_C) = b->data[0] * VRE(x, idx2, idx1, R_C) + b->data[1] * VRE(x, (idx2-1), idx1, R_C) 
//					+ b->data[2] * VRE(x, (idx2-2), idx1, R_C) - a->data[1] * VRE(y, (idx2-1), idx1, R_C) - a->data[2] * VRE(y, (idx2-2), idx1, R_C);
//
//				VIM(y, idx2, idx1, R_C) = b->data[0] * VIM(x, idx2, idx1, R_C) + b->data[1] * VIM(x, (idx2-1), idx1, R_C) 
//					+ b->data[2] * VIM(x, (idx2-2), idx1, R_C) - a->data[1] * VIM(y, (idx2-1), idx1, R_C) - a->data[2] * VIM(y, (idx2-2), idx1, R_C);
//			}
//		}
	}
#endif NEW_OMP

	y->size[0] = x->size[0];
	y->size[1] = x->size[1];
}
