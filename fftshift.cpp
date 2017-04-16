/***********************************************************
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
#include "fftshift.h"
#include <iostream>

#define OMP_ENABLE TRUE

void fftshift2(Array_creal_T *x)
{

  ASSERT(x->numDimensions == 2);

  creal_T xtmp;
  creal_T lastVal;
  int32_T vlend2;
  int32_T b;
  int32_T midoffset;
  int32_T j;
  int32_T ia;
  int32_T ib;
  int32_T k;
  real_T xtmp_re;
  real_T xtmp_im;
  int32_T ic;
  if (x->size[1] <= 1) {
	  midoffset = x->size[0] / 2;
	  if ( midoffset << 1 != x->size[0] ) {
		  lastVal = x->data[midoffset];
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
		  for (j = 0; j < midoffset; j++) {
			  xtmp = x->data[j];
			  x->data[j] = x->data[midoffset + j + 1];
			  x->data[midoffset + j] = xtmp;
		  }
		  x->data[x->size[0] - 1] = lastVal;
	  } else {
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
		  for (j = 0; j < midoffset; j++) {
			  xtmp = x->data[j];
			  x->data[j] = x->data[midoffset + j];
			  x->data[midoffset + j] = xtmp;
		  }
	  }
  } else {
    vlend2 = x->size[1];
    vlend2 /= 2;
    b = x->size[0];
    midoffset = vlend2 * x->size[0];
    if (vlend2 << 1 == x->size[1]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp_re, xtmp_im) firstprivate(b, midoffset, vlend2) if(OMP_ENABLE)
      for (j = 0; j < b; j++) {
        ia = j;
        ib = j + midoffset;
        for (k = 1; k <= vlend2; k++) {
          xtmp_re = x->data[ia].re;
          xtmp_im = x->data[ia].im;
          x->data[ia] = x->data[ib];
          x->data[ib].re = xtmp_re;
          x->data[ib].im = xtmp_im;
          ia += b;
          ib += b;
        }
      }
    } else {
		
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp_re, xtmp_im) firstprivate(b, midoffset, vlend2) if(OMP_ENABLE)
      for (j = 0; j < b; j++) {
        ia = j;
        ib = j + midoffset;
        xtmp_re = x->data[ib].re;
        xtmp_im = x->data[ib].im;
        for (k = 1; k <= vlend2; k++) {
          ic = ib + b;
          x->data[ib] = x->data[ia];
          x->data[ia] = x->data[ic];
          ia += b;
          ib = ic;
        }
        x->data[ic].re = xtmp_re;
        x->data[ic].im = xtmp_im;
      }
    }
  }
}

void fftshift2(Array_creal64_T *x)
{
	ASSERT(x->numDimensions == 2);

	creal64_T xtmp;
	creal64_T lastVal;
	int32_T vlend2;
	int32_T b;
	int32_T midoffset;
	int32_T j;
	int32_T ia;
	int32_T ib;
	int32_T k;
	real64_T xtmp_re;
	real64_T xtmp_im;
	int32_T ic;
	if (x->size[1] <= 1) {
		midoffset = x->size[0] / 2;
		if ( midoffset << 1 != x->size[0] ) {
			lastVal = x->data[midoffset];
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j + 1];
				x->data[midoffset + j] = xtmp;
			}
			x->data[x->size[0] - 1] = lastVal;
		} else {
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j];
				x->data[midoffset + j] = xtmp;
			}
		}
	} else {
		vlend2 = x->size[1];
		vlend2 /= 2;
		b = x->size[0];
		midoffset = vlend2 * x->size[0];
		if (vlend2 << 1 == x->size[1]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp_re, xtmp_im) firstprivate(b, midoffset, vlend2) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j;
				ib = j + midoffset;
				for (k = 1; k <= vlend2; k++) {
					xtmp_re = x->data[ia].re;
					xtmp_im = x->data[ia].im;
					x->data[ia] = x->data[ib];
					x->data[ib].re = xtmp_re;
					x->data[ib].im = xtmp_im;
					ia += b;
					ib += b;
				}
			}
		} else {
			
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp_re, xtmp_im) firstprivate(b, midoffset, vlend2) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j;
				ib = j + midoffset;
				xtmp_re = x->data[ib].re;
				xtmp_im = x->data[ib].im;
				for (k = 1; k <= vlend2; k++) {
					ic = ib + b;
					x->data[ib] = x->data[ia];
					x->data[ia] = x->data[ic];
					ia += b;
					ib = ic;
				}
				x->data[ic].re = xtmp_re;
				x->data[ic].im = xtmp_im;
			}
		}
	}
}

void fftshift1(Array_creal_T *x)
{
	ASSERT(x->numDimensions == 2);

	int32_T vlend2;
	int32_T b;
	int32_T midoffset;
	int32_T j;
	int32_T ia;
	int32_T ib;
	int32_T k;
	real_T xtmp_re;
	real_T xtmp_im;
	int32_T ic;
	if (x->size[1] <= 1) {
	} else {
		vlend2 = x->size[0];
		vlend2 /= 2;
		b = x->size[1];
		midoffset = vlend2;
		if (vlend2 << 1 == x->size[0]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp_re, xtmp_im) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				for (k = 1; k <= vlend2; k++) {
					xtmp_re = x->data[ia].re;
					xtmp_im = x->data[ia].im;
					x->data[ia] = x->data[ib];
					x->data[ib].re = xtmp_re;
					x->data[ib].im = xtmp_im;
					ia++;
					ib++;
				}
			}
		} else {
			
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp_re, xtmp_im) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				xtmp_re = x->data[ib].re;
				xtmp_im = x->data[ib].im;
				for (k = 1; k <= vlend2; k++) {
					ic = ib + 1;
					x->data[ib] = x->data[ia];
					x->data[ia] = x->data[ic];
					ia++;
					ib = ic;
				}
				x->data[ib].re = xtmp_re;
				x->data[ib].im = xtmp_im;
			}
		}
	}
}

void fftshift1(Array_creal64_T *x)
{
	ASSERT(x->numDimensions == 2);

	int32_T vlend2;
	int32_T b;
	int32_T midoffset;
	int32_T j;
	int32_T ia;
	int32_T ib;
	int32_T k;
	real64_T xtmp_re;
	real64_T xtmp_im;
	int32_T ic;
	if (x->size[1] <= 1) {
	} else {
		vlend2 = x->size[0];
		vlend2 /= 2;
		b = x->size[1];
		midoffset = vlend2;
		if (vlend2 << 1 == x->size[0]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp_re, xtmp_im) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				for (k = 1; k <= vlend2; k++) {
					xtmp_re = x->data[ia].re;
					xtmp_im = x->data[ia].im;
					x->data[ia] = x->data[ib];
					x->data[ib].re = xtmp_re;
					x->data[ib].im = xtmp_im;
					ia++;
					ib++;
				}
			}
		} else {
		
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp_re, xtmp_im) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				xtmp_re = x->data[ib].re;
				xtmp_im = x->data[ib].im;
				for (k = 1; k <= vlend2; k++) {
					ic = ib + 1;
					x->data[ib] = x->data[ia];
					x->data[ia] = x->data[ic];
					ia++;
					ib = ic;
				}
				x->data[ib].re = xtmp_re;
				x->data[ib].im = xtmp_im;
			}
		}
	}
}

void fftshift1(Array_real64_T *x)
{
	ASSERT(x->numDimensions == 2);

	int32_T vlend2;
	int32_T b;
	int32_T midoffset;
	real64_T lastVal;
	int32_T j;
	int32_T ia;
	int32_T ib;
	int32_T k;
	real64_T xtmp;
	int32_T ic;
	if (x->size[1] <= 1) {
		midoffset = x->size[0] / 2;
		if ( midoffset << 1 != x->size[0] ) {
			lastVal = x->data[midoffset];
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j + 1];
				x->data[midoffset + j] = xtmp;
			}
			x->data[x->size[0] - 1] = lastVal;
		} else {
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j];
				x->data[midoffset + j] = xtmp;
			}
		}
	} else {
		vlend2 = x->size[0];
		vlend2 /= 2;
		b = x->size[1];
		midoffset = vlend2;
		if (vlend2 << 1 == x->size[0]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				for (k = 1; k <= vlend2; k++) {
					xtmp = x->data[ia];
					x->data[ia] = x->data[ib];
					x->data[ib] = xtmp;
					ia++;
					ib++;
				}
			}
		} else {
			
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j * x->size[0];
				ib = j * x->size[0] + midoffset;
				xtmp = x->data[ib];
				for (k = 1; k <= vlend2; k++) {
					ic = ib + 1;
					x->data[ib] = x->data[ia];
					x->data[ia] = x->data[ic];
					ia++;
					ib = ic;
				}
				x->data[ib] = xtmp;
			}
		}
	}
}

void fftshift2(Array_real64_T *x)
{
	ASSERT(x->numDimensions == 2);

	int32_T vlend2;
	int32_T b;
	int32_T midoffset;
	int32_T j;
	int32_T ia;
	int32_T ib;
	int32_T k;
	real64_T xtmp;
	real64_T lastVal;
	int32_T ic;
	if (x->size[1] <= 1) {
		midoffset = x->size[0] / 2;
		if ( midoffset << 1 != x->size[0] ) {
			lastVal = x->data[midoffset];
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j + 1];
				x->data[midoffset + j] = xtmp;
			}
			x->data[x->size[0] - 1] = lastVal;
		} else {
#pragma omp parallel for private(j, xtmp) firstprivate(midoffset) if(OMP_ENABLE)
			for (j = 0; j < midoffset; j++) {
				xtmp = x->data[j];
				x->data[j] = x->data[midoffset + j];
				x->data[midoffset + j] = xtmp;
			}
		}
	} else {
		vlend2 = x->size[1];
		vlend2 /= 2;
		b = x->size[0];
		midoffset = vlend2 * x->size[0];
		if (vlend2 << 1 == x->size[1]) {
#pragma omp parallel for private(k, j, ia, ib, xtmp) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j;
				ib = j + midoffset;
				for (k = 1; k <= vlend2; k++) {
					xtmp = x->data[ia];
					x->data[ia] = x->data[ib];
					x->data[ib] = xtmp;
					ia += b;
					ib += b;
				}
			}
		} else {
			// TODO : This is not tested, I am not sure correct.
#pragma omp parallel for private(j, k, ia, ib, ic, xtmp) firstprivate(b, midoffset) if(OMP_ENABLE)
			for (j = 0; j < b; j++) {
				ia = j;
				ib = j + midoffset;
				xtmp = x->data[ib];
				for (k = 1; k <= vlend2; k++) {
					ic = ib + b;
					x->data[ib] = x->data[ia];
					x->data[ia] = x->data[ic];
					ia += b;
					ib = ic;
				}
				x->data[ib] = xtmp;
			}
		}
	}
}
