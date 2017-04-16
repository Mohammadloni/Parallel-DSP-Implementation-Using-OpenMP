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
#include "ifft.h"
#include "fft.h"
#include "fftw3.h"
#include <iostream>
#include <omp.h>

void ifft(const Array_creal32_T *x,int32_T dim, Array_creal32_T *y);
void ifft(const Array_creal64_T *x,int32_T dim, Array_creal64_T *y);

void scale(creal32_T *x, real32_T scale)
{
	x->re=x->re*scale;
	x->im=x->im*scale;
}

void scale(creal64_T *x, real64_T scale)
{
	x->re=x->re*scale;
	x->im=x->im*scale;
}

void scale(real_T x, real_T scale)
{
	x=x*scale;
}

//void scale(real32_T x, real32_T scale)
//{
//	x=x*scale;
//}

/**********************************************************************************************//**
* @fn	void ifft(const Array_creal_T *x, int32_T dim, Array_creal_T *y)
*
* @brief	Matlab y=ifft(x,[],dim)
*					apply one dimensional ifft transform on rows (dim==2) or columns (dim==1) of a matrix
*					the matrix format is column-major
* @author	AMIN
*
* @param [in] x		input matrix.
* @param	dim		dimension of tranform.
* @param [out] y	output matrix.
**************************************************************************************************/
void ifft(const Array_creal_T *x,int32_T dim, Array_creal_T *y)
{
	if ( sizeof(*x->data) == sizeof(creal64_T) && sizeof(*y->data) == sizeof(creal64_T) )
		ifft((Array_creal64_T*)x, dim, (Array_creal64_T*)y);
	else
		ifft((Array_creal32_T*)x, dim, (Array_creal32_T*)y);
}

void ifft(const Array_creal32_T *x,int32_T dim, Array_creal32_T *y)
{

	if(dim ==2)
	{
		fftwf_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;									/*We are computing 1d transforms*/
		int32_T n[]={x->size[1]};						/*1d transforms on row*/
		int32_T howmany=x->size[0];						/*Equal to the number of rows*/
		int32_T idist=1,odist=1;						/*the distance between the first two elements of two successive row*/
		int32_T istride=x->size[0],ostride=x->size[0];	/*the distance between the two successive elements in a row*/
		int32_T *inembed=n,*onembed=n;

		p1=fftwf_plan_many_dft(rank,n,howmany,reinterpret_cast<fftwf_complex*>(x->data),inembed,istride,idist,
			reinterpret_cast<fftwf_complex*>(y->data),onembed,ostride,odist,FFTW_BACKWARD,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftwf_execute(p1);

		if(x->size[1]>0)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[1];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(&y->data[i],(1.0/s));
		}

		fftwf_destroy_plan(p1);
	}
	else if(dim ==1)
	{
		fftwf_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;								/*We are computing 1d transforms*/
		int32_T n[]={x->size[0]};					/*1d transforms on column*/
		int32_T howmany=x->size[1];					/*Equal to the number of columns*/
		int32_T idist=x->size[0],odist=x->size[0];	/*the distance between the first two elements of two successive column*/
		int32_T istride=1,ostride=1;				/*the distance between the two successive elements in a column*/
		int32_T *inembed=n,*onembed=n;

		p1=fftwf_plan_many_dft(rank,n,howmany,reinterpret_cast<fftwf_complex*>(x->data),inembed,istride,idist,
			reinterpret_cast<fftwf_complex*>(y->data),onembed,ostride,odist,FFTW_BACKWARD,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftwf_execute(p1);

		if(x->size[0]>1)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[0];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(&y->data[i],(1.0/s));
		}

		fftwf_destroy_plan(p1);
	}
}

void ifft(const Array_creal64_T *x,int32_T dim, Array_creal64_T *y)
{

	if(dim ==2)
	{
		fftw_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;									/*We are computing 1d transforms*/
		int32_T n[]={x->size[1]};						/*1d transforms on row*/
		int32_T howmany=x->size[0];						/*Equal to the number of rows*/
		int32_T idist=1,odist=1;						/*the distance between the first two elements of two successive row*/
		int32_T istride=x->size[0],ostride=x->size[0];	/*the distance between the two successive elements in a row*/
		int32_T *inembed=n,*onembed=n;

		p1=fftw_plan_many_dft(rank,n,howmany,reinterpret_cast<fftw_complex*>(x->data),inembed,istride,idist,
			reinterpret_cast<fftw_complex*>(y->data),onembed,ostride,odist,FFTW_BACKWARD,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftw_execute(p1);

		if(x->size[1]>0)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[1];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(&y->data[i],(1.0/s));
		}

		fftw_destroy_plan(p1);
	}
	else if(dim ==1)
	{
		fftw_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;								/*We are computing 1d transforms*/
		int32_T n[]={x->size[0]};					/*1d transforms on column*/
		int32_T howmany=x->size[1];					/*Equal to the number of columns*/
		int32_T idist=x->size[0],odist=x->size[0];	/*the distance between the first two elements of two successive column*/
		int32_T istride=1,ostride=1;				/*the distance between the two successive elements in a column*/
		int32_T *inembed=n,*onembed=n;

		p1=fftw_plan_many_dft(rank,n,howmany,reinterpret_cast<fftw_complex*>(x->data),inembed,istride,idist,
			reinterpret_cast<fftw_complex*>(y->data),onembed,ostride,odist,FFTW_BACKWARD,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftw_execute(p1);

		if(x->size[0]>1)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[0];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(&y->data[i],(1.0/s));
		}

		fftw_destroy_plan(p1);
	}
}

#if 0
void ifft(const Array_real_T *x,int32_T dim, Array_real_T *y)
{
	if(dim ==2)
	{
		fftw_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;									/*We are computing 1d transforms*/
		int32_T n[]={x->size[1]};						/*1d transforms on row*/
		int32_T howmany=x->size[0];						/*Equal to the number of rows*/
		int32_T idist=1,odist=1;						/*the distance between the first two elements of two successive row*/
		int32_T istride=x->size[0],ostride=x->size[0];	/*the distance between the two successive elements in a row*/
		int32_T *inembed=n,*onembed=n;

		fftw_r2r_kind kinds[] = {FFTW_REDFT00};

		p1=fftw_plan_many_r2r(rank,n,howmany,reinterpret_cast<double*>(x->data),inembed,istride,idist,
			reinterpret_cast<double*>(y->data),onembed,ostride,odist, kinds,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftw_execute(p1);

		if(x->size[1]>0)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[1];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(y->data[i],(1.0/s));
		}

		fftw_destroy_plan(p1);
		fftw_cleanup();
	}
	else if(dim ==1)
	{
		fftw_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;								/*We are computing 1d transforms*/
		int32_T n[]={x->size[0]};					/*1d transforms on column*/
		int32_T howmany=x->size[1];					/*Equal to the number of columns*/
		int32_T idist=x->size[0],odist=x->size[0];	/*the distance between the first two elements of two successive column*/
		int32_T istride=1,ostride=1;				/*the distance between the two successive elements in a column*/
		int32_T *inembed=n,*onembed=n;

		fftw_r2r_kind kinds[] = {FFTW_REDFT00};

		p1=fftw_plan_many_r2r(rank,n,howmany,reinterpret_cast<double*>(x->data),inembed,istride,idist,
			reinterpret_cast<double*>(y->data),onembed,ostride,odist, kinds,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftw_execute(p1);

		if(x->size[0]>1)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[0];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale(y->data[i],(1.0/s));
		}

		fftw_destroy_plan(p1);
		fftw_cleanup();
	}
}

void ifft(const Array_real32_T *x,int32_T dim, Array_real32_T *y)
{
	if(dim ==2)
	{
		fftwf_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;									/*We are computing 1d transforms*/
		int32_T n[]={x->size[1]};						/*1d transforms on row*/
		int32_T howmany=x->size[0];						/*Equal to the number of rows*/
		int32_T idist=1,odist=1;						/*the distance between the first two elements of two successive row*/
		int32_T istride=x->size[0],ostride=x->size[0];	/*the distance between the two successive elements in a row*/
		int32_T *inembed=n,*onembed=n;

		fftwf_r2r_kind kinds[] = {FFTW_REDFT00};

		p1=fftwf_plan_many_r2r(rank,n,howmany,reinterpret_cast<float*>(x->data),inembed,istride,idist,
			reinterpret_cast<float*>(y->data),onembed,ostride,odist, kinds,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftwf_execute(p1);

		if(x->size[1]>0)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[1];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale((real32_T)y->data[i],(real32_T)(1.0/s));
		}

		fftwf_destroy_plan(p1);
		fftwf_cleanup();
	}
	else if(dim ==1)
	{
		fftwf_plan p1;
		int32_T i, loop_ub;
		real_T s;
		int32_T rank=1;								/*We are computing 1d transforms*/
		int32_T n[]={x->size[0]};					/*1d transforms on column*/
		int32_T howmany=x->size[1];					/*Equal to the number of columns*/
		int32_T idist=x->size[0],odist=x->size[0];	/*the distance between the first two elements of two successive column*/
		int32_T istride=1,ostride=1;				/*the distance between the two successive elements in a column*/
		int32_T *inembed=n,*onembed=n;

		fftwf_r2r_kind kinds[] = {FFTW_REDFT00};

		p1=fftwf_plan_many_r2r(rank,n,howmany,reinterpret_cast<float*>(x->data),inembed,istride,idist,
			reinterpret_cast<float*>(y->data),onembed,ostride,odist, kinds,FFTW_ESTIMATE);

		if(p1==NULL) {
			std::cout << "Configure FFTW plan problem " << std::endl;
		}

		fftwf_execute(p1);

		if(x->size[0]>1)
		{
			loop_ub = y->size[0]*y->size[1];
			s=x->size[0];
#pragma omp parallel for
			for(i=0;i<loop_ub;i++)
				scale((real32_T)y->data[i],(real32_T)(1.0/s));
		}

		fftwf_destroy_plan(p1);
		fftwf_cleanup();
	}
}
#endif