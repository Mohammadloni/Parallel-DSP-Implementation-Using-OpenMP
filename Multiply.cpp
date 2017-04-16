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

#include <string.h>
#include <stdlib.h>
#include "Data_Type.h"
#include "Passive _util.h"
#include "def.h"
#include "TransMat.h"
#include <math.h>

// ========= defines =========
	//*** #define VAL(Arr,I,J,R) Arr[J * R + I]
	//*** #define V(Arr,I,J) VAL(Arr,I,J,3)

	//#define V(Arr,I,J,P) VAL(Arr,I,J,P,3,64)   // 3D matris calculation
	//#define VAL(Arr,I,J,P,R,s) Arr[J * R + I+(P-1)*s]
	#define ABS(A)	  A.re * A.re +A.im * A.im
	#define NORM(A)	  sqrtf(ABS(A))

	#define V(Arr,I,J) VAL(Arr,I,J,8)
	#define VAL(Arr,I,J,R) Arr[J * R + I]
	
	#define MUL_C_RE_conj(A,B)  A.re * B.re + A.im * B.im
	#define MUL_C_IM_conj(A,B)  A.im * B.re - A.re * B.im 

	#define MUL_C_RE(A,B)  A.re * B.re - A.im * B.im
	#define MUL_C_IM(A,B)  A.re * B.im + A.im * B.re

	#define DIV_C_RE(A,B)  (A.re * B.re + A.im * B.im)/(B.re*B.re+B.im*B.im)
	#define DIV_C_IM(A,B)  (A.im * B.re - A.re * B.im)/(B.re*B.re+B.im*B.im)
//============================
void AbsNative(Array_creal_T *BufferA,Array_real_T *ABSout,int32_T length);
//==============================================================
//					Mul8x8
//==============================================================
void inline Mul8x8(creal_T *A, creal_T *B, creal_T *C1)
{

	creal_T A1;
	creal_T B1;
	
	int i,t;
	for(t=0;t<8;t++){
		for(i=0;i<8;i++){
			A1.re = A[t].re;
			A1.im = A[t].im;
			B1.re = B[i].re;
			B1.im = B[i].im;

			C1[8*t+i].re = MUL_C_RE_conj(A1,B1);
			C1[8*t+i].im = MUL_C_IM_conj(A1,B1);

			//C[8*t+i].re = MUL_C_RE_conj(A[i],B[t]); // for test
			//C[8*t+i].im = MUL_C_IM_conj(A[i],B[t]);

		}
	}
	
}
//==============================================================
//					MulV8x8_Cub
//==============================================================
void MulV8x8_Cub(Array_creal_T *BufferA, Array_creal_T *BufferB, Array_creal_T *BufferC, int32_T BatchDim)
{
	int32_T indx;
	int32_T loop_up;
	int32_T Dim2;
	int32_T Dim1;

	Dim1 = BufferA->size[0];
	Dim2 = BufferA->size[0] * BufferB->size[0];
	loop_up = BufferA->size[1];
#pragma omp parallel for private(indx) firstprivate(loop_up, Dim2)
	for (indx = 0; indx < loop_up; indx++)
		Mul8x8(&BufferA->data[indx * Dim1], &BufferB->data[indx * Dim1], &BufferC->data[indx * Dim2]);
}
//==============================================================
//					Mul_Undrtest
//==============================================================
void Mul_Undrtest(Array_creal_T *BufferA, Array_creal_T *BufferB, Array_creal_T *BufferC)
{
	int32_T s,i,row = BufferA->size[0];
	int32_T loop = BufferA->size[0]*BufferA->size[1];
	creal_T sum,dump;
	for(i=0;i<row;i++){
			sum.re = 0;
			sum.im = 0;
		for(s=0;s<row;s++){
			dump.re = (MUL_C_RE(BufferA->data[i+s*8] , BufferB->data[s]));
			dump.im = (MUL_C_IM(BufferA->data[i+s*8] , BufferB->data[s]));
			sum.re += (MUL_C_RE(BufferA->data[i+s*8] , BufferB->data[s]));
			sum.im += (MUL_C_IM(BufferA->data[i+s*8] , BufferB->data[s]));
		}
		BufferC->data[i].re = sum.re;
		BufferC->data[i].im = sum.im;
	}
}
//==============================================================
//					Mul_UndrtestVec
//==============================================================
void Mul_UndrtestVec(Array_creal_T *BufferA, Array_creal_T *BufferB, creal_T *BufferC)
{
	int32_T s,row = BufferA->size[0];
	creal_T sum;

		sum.re = 0;
		sum.im = 0;
		for(s=0;s<row;s++){
			sum.re += MUL_C_RE_conj(BufferA->data[s] , BufferB->data[s]);
			sum.im += MUL_C_IM_conj(BufferA->data[s] , BufferB->data[s]);
		}
		BufferC->re = sum.re;
		BufferC->im = sum.im;
}
//==============================================================
//					MulVecMat
//==============================================================
void MulVecMat(Array_creal_T *InVec, Array_creal_T *InMat, Array_creal_T *OutVec)
{
	int32_T s,i,loop,col = InVec->size[0];
	creal_T sum;
	loop = InMat->size[0]*InMat->size[1];

		for(i=0;i<loop;i+=col){
			sum.re = 0;
			sum.im = 0;
			for(s=0;s<col;s++){
				sum.re += MUL_C_RE(InVec->data[s] , InMat->data[i+s]);
				sum.im += MUL_C_IM(InVec->data[s] , InMat->data[i+s]);
			}
			OutVec->data[i].re = sum.re;
			OutVec->data[i].im = sum.im;
		}
}
//==============================================================
//					MulVecMat
//==============================================================
void MulMatVecConj(Array_creal_T *InMat, Array_creal_T *InVec, Array_creal_T *OutVec)
{
	int32_T s,i,loop,col = InVec->size[0],counter = 0;
	creal_T sum;
	loop = InMat->size[0]*InMat->size[1];

		for(i=0;i<loop;i+=col){ 
			sum.re = 0;
			sum.im = 0;
			for(s=0;s<col;s++){
				sum.re += MUL_C_RE_conj(InMat->data[i+s] , InVec->data[s]);
				sum.im += MUL_C_IM_conj(InMat->data[i+s] , InVec->data[s]);
			}
			OutVec->data[counter].re = sum.re;
			OutVec->data[counter].im = sum.im;
			counter++;
		}
}

//==============================================================
//					MulConjNative
//==============================================================
void MulConjNative(creal_T *A, creal_T *B, Array_creal_T *C1,int32_T length)
{

	creal_T A1;
	creal_T B1;
	
	int32_T i;
	
		for(i=0;i<length;i++){
			A1.re = A[i].re;
			A1.im = A[i].im;
			B1.re = B[i].re;
			B1.im = B[i].im;

			C1->data[i].re = MUL_C_RE_conj(A1,B1);
			C1->data[i].im = MUL_C_IM_conj(A1,B1);

			//C[8*t+i].re = MUL_C_RE_conj(A[i],B[t]);	// for Optimization
			//C[8*t+i].im = MUL_C_IM_conj(A[i],B[t]);	// for Optimization

		}
	
}
//==============================================================
//					MulConjNative
//==============================================================
void MulCVecCSConjNative(creal_T *A, creal_T *B, creal_T *C1,int32_T length)/* Opt: Acumulator*/
{

	creal_T A1;
	creal_T B1;
	
	int32_T i;
	
		C1->re = 0;
		C1->im = 0;

		for(i=0;i<length;i++){
			A1.re = A[i].re;
			A1.im = A[i].im;
			B1.re = B[i].re;
			B1.im = B[i].im;

			C1->re += MUL_C_RE_conj(A1,B1);
			C1->im += MUL_C_IM_conj(A1,B1);

			//C[8*t+i].re = MUL_C_RE_conj(A[i],B[t]);	// for Optimization
			//C[8*t+i].im = MUL_C_IM_conj(A[i],B[t]);	// for Optimization

		}
	
}

//==============================================================
//					Dev_UndrtestVec
//==============================================================
void Dev_UndrtestC(Array_creal_T *BufferA, creal_T BufferB, Array_creal_T *BufferC)
{
	int32_T s,row;
	creal_T f;
	row = BufferA->size[0];

	for(s=0;s<row;s++){
		BufferC->data[s].re = DIV_C_RE(BufferA->data[s] , BufferB);
		BufferC->data[s].im = DIV_C_IM(BufferA->data[s] , BufferB);
		}
}
//==============================================================
//					Dev_UndrtestVec
//==============================================================
void DevNative(Array_creal_T *BufferA, creal_T BufferB, Array_creal_T *BufferC)
{
	int32_T s,loop;
	creal_T f;
	loop = BufferA->size[0]*BufferA->size[1];

	for(s=0;s<loop;s++){
		f.re = DIV_C_RE(BufferA->data[s] , BufferB);
		f.im = DIV_C_IM(BufferA->data[s] , BufferB);
		BufferC->data[s].re = f.re;
		BufferC->data[s].im = f.im;
	}
}
//==============================================================
//					Dev_UndrtestVec
//==============================================================
void DevNative(creal_T *BufferA,int32_T length, creal_T BufferB, creal_T *BufferC)
{
	int32_T s;
	creal_T f;

	for(s=0;s<length;s++){
		f.re = DIV_C_RE(BufferA[s] , BufferB);
		f.im = DIV_C_IM(BufferA[s] , BufferB);
		BufferC[s].re = f.re;
		BufferC[s].im = f.im;
	}
}
//==============================================================
//					SumNative 1
//==============================================================
void SumNative(creal_T *sumR,Array_creal_T *B){

	int32_T i,t;

	sumR->re=0;
	sumR->im=0;

	//if(B->numDimensions == 1){
		for(i=0;i<B->size[0];i++){

			sumR->re +=B->data[i].re;
			sumR->im +=B->data[i].im;
		}
	//}

}
//==============================================================
//					SumNative 1
//==============================================================
void SumNative(creal_T *sumR,Array_creal_T *B,int32_T length){

	int32_T i,t;

	sumR->re=0;
	sumR->im=0;

		for(i=0 ; i<length ; i++){

			sumR->re +=(B->data[i].re);
			sumR->im +=(B->data[i].im);
		}
	//}

}
//==============================================================
//					SumNative 1
//==============================================================
void SumNative(creal_T *sumR,Array_creal_T *B,int32_T length,int32_T p){

	int32_T i;
	//creal_T DmpSum;
	sumR->re=0;
	sumR->im=0;

		for(i=0 ; i<length ; i++){
			sumR->re +=(B->data[i].re);
			sumR->im +=(B->data[i].im);
		}
	//}

}

//==============================================================
//					SumNative
//==============================================================
void SumNative(Array_creal_T *sumR,Array_creal_T *B){


		int32_T i,t;
		creal_T sumD;
		for(i=0;i<B->size[1];i++){
		sumD.re = 0;
		sumD.im = 0;  
			for(t=0;t<B->size[0];t++){

				sumD.re +=B->data[t*B->size[1]+i].re;
				sumD.im +=B->data[t*B->size[1]+i].im;
			}

			sumR->data[i].re = sumD.re;
			sumR->data[i].im = sumD.im;
		}

}
//==============================================================
//					SumNative
//==============================================================
void SumNative(Array_creal_T *sumR,Array_creal_T *B,int32_T length){


		int32_T i,t;
		creal_T sumD;

		for(i=0;i<length;i++){
			sumD.re = 0;
			sumD.im = 0;
			for(t=0;t<B->size[0];t++){

				sumD.re +=B->data[t*length+i].re;
				sumD.im +=B->data[t*length+i].im;
			}

			sumR->data[i].re = sumD.re;
			sumR->data[i].im = sumD.im;
		}

}
//==============================================================
//					NormNative
//==============================================================
void NormNative(Array_creal_T *BufferA,real_T *SumDump){ 

	int32_T i,loop;

	loop = BufferA->numDimensions == 2 ? (BufferA->size[0]*BufferA->size[1]):(BufferA->size[0]);
	*SumDump = 0;


	for(i=0;i<loop;i++){

		*SumDump+=ABS(BufferA->data[i]);
	}

	*SumDump = sqrtf(*SumDump);

}
//==============================================================
//					NormNative
//==============================================================
void NormNative(Array_creal_T *BufferA,real_T *SumDump,int32_T row,int32_T col){

	int32_T i,loop;

	loop =  row * col;
	*SumDump = 0;


	for(i=0;i<loop;i++){

		*SumDump+=ABS(BufferA->data[i]);
	}

	*SumDump = sqrtf(*SumDump);
}

//==============================================================
//					NormNative
//==============================================================
void NormNative(creal_T *BufferA,real_T *SumDump,int32_T length){

	int32_T i,loop;

	*SumDump = 0;
	
	for(i=0;i<length;i++){

		*SumDump+=ABS(BufferA[i]);
	}

	*SumDump = sqrtf(*SumDump);
}
//==============================================================
//					AbsNative
//==============================================================
void AbsNative(Array_creal_T *BufferA,Array_real_T *ABSout){

	int32_T i;

	creal_T A1;
	creal_T B1;


	for(i=0;i<BufferA->size[0];i++){

		A1.re = BufferA->data[i].re;
		A1.im = BufferA->data[i].im;
		ABSout->data[i] = NORM(A1);
		//ABSout->data[i] = ABS(BufferA->data[i]); // for Optimization

	}

}
//==============================================================
//					AbsNative
//==============================================================
void AbsNative(Array_creal_T *BufferA,Array_real_T *ABSout,int32_T length){

	int32_T i;

	creal_T A1;
	creal_T B1;


	for(i=0;i<length;i++){ // Opt: A1 ghesmate miyani hazf shavad

		A1.re = BufferA->data[i].re;
		A1.im = BufferA->data[i].im;
		ABSout->data[i] = NORM(A1);
		// ABSout->data[i] = ABS(BufferA->data[i]); // for Optimization
	}
}
//==============================================================
//					FindMaxNative
//==============================================================
void FindMaxNative(Array_real_T *BufferA,real_T *maxOut,int32_T length){

	int32_T i;

	*maxOut = 0;

	for(i=0;i<length;i++){
		if(BufferA->data[i]>*maxOut)
			*maxOut = BufferA->data[i];
		else
			continue;
	}
}
//==============================================================
//					FindMaxNative
//==============================================================
void FindMaxNative(Array_real_T *BufferA,real_T *maxOut,int32_T *MaxIndx,int32_T length){

	int32_T i;
	*maxOut = 0;	
	
	for(i=0;i<length;i++){
		if(BufferA->data[i]>*maxOut){
			*maxOut = BufferA->data[i];
			*MaxIndx = i;
		}
		else
			continue;
	}
}
//==============================================================
//					FindMaxNative
//==============================================================
void FindMaxNative(Array_creal_T *BufferA,creal_T *maxOut,int32_T *MaxIndx,int32_T length){

	int32_T i;
	real_T MaxOutDump;
	Array_real_T *dump;
	PASSIVEInit_real_T(&dump,1);
	dump->size[0]= length;
	PASSIVEAlloc_real_T(&dump);

	MaxOutDump = 0;
	maxOut->re = 0;
	maxOut->im = 0;

	AbsNative(BufferA,dump,length);

	for(i=0;i<length;i++){
		if(dump->data[i] > MaxOutDump){
			MaxOutDump = dump->data[i];
			maxOut->re = BufferA->data[i].re;
			maxOut->im = BufferA->data[i].im;
			*MaxIndx = i;
		}
		else
			continue;
	}
	PASSIVEFree_real_T(&dump);

}
//==============================================================
//					MemCopy
//==============================================================
void MemCopy(creal_T *BuffB,int32_T length,int32_T colmn,Array_creal_T *BuffA){

	int32_T i,t;
	int32_T row = BuffA->size[0];
	//int32_T colmn = BuffB->size[1];
	//int32_T	ofset = (row - length)*2;

	for(t=0;t<row;t++){
		for(i=0;i<length;i++){
			BuffA->data[t*length+i].re = BuffB[t*colmn+i].re;
			BuffA->data[t*length+i].im = BuffB[t*colmn+i].im;
		}
	}

}
//==============================================================
//					MeanNative
//==============================================================
void MeanNative(Array_creal_T *BuffA,creal_T *BuffB){

	int32_T i,loop;
	creal_T sumR;
	loop = BuffA->numDimensions == 1 ? BuffA->size[0] : BuffA->size[0]*BuffA->size[1];
	sumR.re = 0;//BuffA->data[i].re;
	sumR.im = 0;//BuffA->data[i].im;
	for(i=0;i<loop;i++){
		
		sumR.re += BuffA->data[i].re;
		sumR.im += BuffA->data[i].im;
	}

	BuffB->re = sumR.re/loop; 
	BuffB->im = sumR.im/loop; 

}
//==============================================================
//					CircShiftCol
//==============================================================
void CircShiftCol(Array_creal_T *BuffA/*2048x10*/,int32_T shift,Array_creal_T *BuffB){

	int32_T i,NumChunk,Chunk,s,d,t,l;	
	creal_T sumR;
	NumChunk = BuffA->size[0];
	Chunk = BuffA->size[1];
	//Transpose()
	Array_creal_T *dump;
	PASSIVEInit_creal_T(&dump,1);
	dump->size[0]= BuffA->size[1];	//CP
	PASSIVEAlloc_creal_T(&dump);


	/*if(shift == 0)
		memcpy((void*)BuffB->data,(void*)BuffA->data,BuffA->size[0]*BuffA->size[1]*sizeof(creal_T));

	for(i=0;i<shift;i++){
													  //  10  
		memcpy((void*)dump->data,(void*)BuffA->data,BuffA->size[1]*sizeof(creal_T));
		for(s=1;s<loop-1;s++){
			memcpy((void*)&BuffB->data[(s-1)*BuffA->size[1]],(void*)&BuffA->data[(s)*BuffA->size[1]],BuffA->size[1]*sizeof(creal_T));
		}
		memcpy((void*)&BuffB->data[BuffA->size[1]*(BuffA->size[0]-1)],(void*)dump->data,BuffA->size[1]*sizeof(creal_T));
	}*/

	/*	memcpy((void*)BuffB->data,(void*)BuffA->data,NumChunk*Chunk*sizeof(creal_T));

	for(i=0;i<shift;i++){
													  //  10  
		memcpy((void*)dump->data,(void*)BuffB->data,Chunk*sizeof(creal_T));
		for(s=1;s<NumChunk-1;s++){
			memcpy((void*)&BuffB->data[(s-1)*Chunk],(void*)&BuffB->data[(s)*Chunk],Chunk*sizeof(creal_T));
		}
		memcpy((void*)&BuffB->data[Chunk*(NumChunk-1)],(void*)dump->data,Chunk*sizeof(creal_T));
	}*/

	memcpy((void*)BuffB->data,(void*)BuffA->data,NumChunk*Chunk*sizeof(creal_T));

	for(i=0;i<shift;i++){
													  //  10  
		//memcpy((void*)dump->data,(void*)BuffB->data,Chunk*sizeof(creal_T));
		for(d=0;d<10;d++){
			dump->data[d].re = BuffB->data[d].re;
			dump->data[d].im = BuffB->data[d].im;
		
		}
		for(s=1;s<NumChunk;s++){
			//memcpy((void*)&BuffB->data[(s-1)*Chunk],(void*)&BuffB->data[(s)*Chunk],Chunk*sizeof(creal_T));
			for(t=0;t<10;t++){
				BuffB->data[(s-1)*Chunk+t].re = BuffB->data[(s)*Chunk+t].re;
				BuffB->data[(s-1)*Chunk+t].im = BuffB->data[(s)*Chunk+t].im;
		
			}
		}
		//memcpy((void*)&BuffB->data[Chunk*(NumChunk-1)],(void*)dump->data,Chunk*sizeof(creal_T));
		for(l=0;l<10;l++){
			BuffB->data[Chunk*(NumChunk-1)+l].re = dump->data[l].re;
			BuffB->data[Chunk*(NumChunk-1)+l].im = dump->data[l].im;
		
		}
	}
	
	//TransMatrixNative(BuffB);
	//TransMatrixNoDimChangNative(BuffB);
	PASSIVEFree_creal_T(&dump);
}
//==============================================================
//					Sig_Gen
//==============================================================
void Sig_Gen(Array_creal_T *SigAnt,real_T deltaf0,Array_creal_T *Sig){

	
	int32_T i,s,row = SigAnt->size[0],col = SigAnt->size[1];
	//*** creal_T dump1;

	Array_creal_T *dump;
	PASSIVEInit_creal_T(&dump,2);
	dump->size[0]= SigAnt->size[0];	//CP
	dump->size[1]= SigAnt->size[1];	//CP
	PASSIVEAlloc_creal_T(&dump);

	Array_real_T *n4;
	PASSIVEInit_real_T(&n4,2);
	n4->size[0]= SigAnt->size[0];	//CP
	n4->size[1]= SigAnt->size[1];	//CP
	PASSIVEAlloc_real_T(&n4);


	//data_stream4n=sig_ant.*PassivFrqOfstCoef_f;	//data_stream4n=sig_ant.*(exp(-(1*1i*2*pi*deltaf0*n4))); // Write SIgGen Function
	for(s=0;s<row;s++)
		for(i=0;i<col;i++){

			n4->data[s*col+i] = i;
		}

	for(s=0;s<row;s++)
		for(i=0;i<col;i++){
			dump->data[s*col+i].re = cos(2*M_PI*deltaf0*n4->data[s*col+i]);
			dump->data[s*col+i].im = -sin(2*M_PI*deltaf0*n4->data[s*col+i]);
		}

	for(s=0;s<row;s++)
		for(i=0;i<col;i++){
			Sig->data[s*col+i].re = MUL_C_RE(SigAnt->data[s*col+i],dump->data[s*col+i]) ;
			Sig->data[s*col+i].im = MUL_C_IM(SigAnt->data[s*col+i],dump->data[s*col+i]) ;
		}
//================================================		// Opt: for balaeii hazf shode
	/*for(i=0;i<col;i++){

		Sig[i] = MUL_C_RE(SigAnt* cos(2*pi*deltaf0*i));
		Sig[i] = MUL_C_IM(SigAnt* sin(2*pi*deltaf0*i));
		Sig[i+col] = SigAnt[i+col]* cos(2*pi*deltaf0*i);
		Sig[i+col] = SigAnt[i+col]* sin(2*pi*deltaf0*i);
		Sig[i+2*col] = SigAnt[i+2*col]* cos(2*pi*deltaf0*i);
		Sig[i+2*col] = SigAnt[i+2*col]* sin(2*pi*deltaf0*i);
		Sig[i+3*col] = SigAnt[i+3*col]* cos(2*pi*deltaf0*i);
		Sig[i+3*col] = SigAnt[i+3*col]* sin(2*pi*deltaf0*i);
		Sig[i+4*col] = SigAnt[i+4*col]* cos(2*pi*deltaf0*i);
		Sig[i+4*col] = SigAnt[i+4*col]* sin(2*pi*deltaf0*i);
		Sig[i+5*col] = SigAnt[i+5*col]* cos(2*pi*deltaf0*i);
		Sig[i+5*col] = SigAnt[i+5*col]* sin(2*pi*deltaf0*i);
		Sig[i+6*col] = SigAnt[i+6*col]* cos(2*pi*deltaf0*i);
		Sig[i+6*col] = SigAnt[i+6*col]* sin(2*pi*deltaf0*i);
		Sig[i+7*col] = SigAnt[i+7*col]* cos(2*pi*deltaf0*i);
		Sig[i+7*col] = SigAnt[i+7*col]* sin(2*pi*deltaf0*i);
	
	}*/

	PASSIVEFree_creal_T(&dump);
	PASSIVEFree_real_T(&n4);

}

//==============================================================
//					ModifySign
//==============================================================
void ModifySign(Array_creal_T *data_sendfft_shift,Array_int32_T *cn_plt,Array_creal_T *ModSig){

	int32_T i;
	for(i=0;i<cn_plt->size[0];i++){

		ModSig->data[i].re = data_sendfft_shift->data[(cn_plt->data[i]-1)].re;
		ModSig->data[i].im = data_sendfft_shift->data[(cn_plt->data[i]-1)].im;
	}


}
//==============================================================
//					ModifySign
//==============================================================
void ModifySign(creal_T *data_sendfft_shift,Array_int32_T *cn_plt,Array_creal_T *ModSig){

	int32_T i;
	for(i=0;i<cn_plt->size[0];i++){

		ModSig->data[i].re = data_sendfft_shift[(cn_plt->data[i]-1)].re;
		ModSig->data[i].im = data_sendfft_shift[(cn_plt->data[i]-1)].im;
	}


}

//==============================================================
//					MinusNative
//==============================================================
void MinusNative(creal_T *InBuff1,creal_T *InBuff2,int32_T length,int32_T jump,creal_T *OutBuff){


	int32_T t,count;

		for(count=0,t=0;t<=length;t+=jump ,count++){

			OutBuff[count].re = InBuff1[t].re - InBuff2[t].re;
			OutBuff[count].im = InBuff1[t].im - InBuff2[t].im;
			
		}


}

//==============================================================
//					FindMaxNative
//==============================================================
void FindMinNative(real_T *BufferA,real_T *MinOut,int32_T *MinIndx,int32_T length){

	int32_T i;

	*MinOut = BufferA[0];	

	for(i=0;i<length;i++){
		if(BufferA[i] < *MinOut){
			*MinOut = BufferA[i];
			*MinIndx = i;
		}
		else
			continue;
	}
}





