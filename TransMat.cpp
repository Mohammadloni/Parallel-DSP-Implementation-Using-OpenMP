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

//=================================================================
//						TransMatrixNative
//=================================================================
void TransMatrixNative( Array_real64_T *BufferMatrix )
{
	real64_T *data;
	int32_T indx;
	uint32_T loop_up;
	uint32_T indx_RowT;
	uint32_T indx_ColT;
	uint32_T Row_SizeT;
	uint32_T Col_SizeT;
	uint32_T Row_Size;
	uint32_T Col_Size;

	data = (real64_T *) PASSIVE_Alloc(BufferMatrix->allocatedSize);
	
	Row_Size = BufferMatrix->size[0];
	Col_Size = BufferMatrix->size[1];
	loop_up = Row_Size * Col_Size;
	Col_SizeT = Row_Size;
	Row_SizeT = Col_Size;

#pragma omp parallel for private(indx, indx_RowT, indx_ColT) firstprivate(loop_up, Row_SizeT, Row_Size)
	for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Row_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Row_Size; // index For Data of Transpose Matrix
		data[indx_ColT * Row_SizeT + indx_RowT] = BufferMatrix->data[indx];
	}

	PASSIVE_Free(BufferMatrix->data);
	BufferMatrix->data = data;

	BufferMatrix->size[0] = Row_SizeT;
	BufferMatrix->size[1] = Col_SizeT;
}
//=================================================================
//						TransMatrixRowMajorNative
//=================================================================
void TransMatrixNative( Array_creal_T *BufferMatrix )
{
	creal_T *data;
	int32_T indx;
	uint32_T loop_up;
	uint32_T indx_RowT;
	uint32_T indx_ColT;
	uint32_T Row_SizeT;
	uint32_T Col_SizeT;
	uint32_T Row_Size;
	uint32_T Col_Size;

	data = (creal_T *) PASSIVE_Alloc(BufferMatrix->allocatedSize);

	Row_Size = BufferMatrix->size[0];
	Col_Size = BufferMatrix->size[1];
	loop_up = Row_Size * Col_Size;
	Col_SizeT = Row_Size;
	Row_SizeT = Col_Size;

//#pragma omp parallel for private(indx, indx_RowT, indx_ColT) firstprivate(loop_up, Row_SizeT, Row_Size)
	/*for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Row_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Row_Size; // index For Data of Transpose Matrix
		data[indx_ColT * Row_SizeT + indx_RowT] = BufferMatrix->data[indx];
	}*/

	for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Col_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Col_Size; // index For Data of Transpose Matrix
		//if((indx_ColT * Row_Size + indx_RowT) == 10)
		data[indx_ColT * Row_Size + indx_RowT] = BufferMatrix->data[indx];
	}
	
	PASSIVE_Free(BufferMatrix->data);
	BufferMatrix->data = data;

	BufferMatrix->size[0] = Row_SizeT;
	BufferMatrix->size[1] = Col_SizeT;
}


//=================================================================
//						TransMatrixNative
//=================================================================
void TransMatrixNative( const Array_creal_T *BufferMatrixIn , Array_creal_T *BufferMatrixOut )
{
	int32_T indx;
	uint32_T loop_up;
	uint32_T indx_RowT;
	uint32_T indx_ColT;
	uint32_T Row_SizeT;
	uint32_T Col_SizeT;
	uint32_T Row_Size;
	uint32_T Col_Size;

	Row_Size = BufferMatrixIn->size[0];
	Col_Size = BufferMatrixIn->size[1];
	loop_up = Row_Size * Col_Size;
	Col_SizeT = Row_Size;
	Row_SizeT = Col_Size;

	// Opt:
//#pragma omp parallel for private(indx, indx_RowT, indx_ColT) firstprivate(loop_up, Row_SizeT, Row_Size)
	for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Row_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Row_Size; // index For Data of Transpose Matrix
		BufferMatrixOut->data[indx_ColT * Row_SizeT + indx_RowT].re = BufferMatrixIn->data[indx].re;
		BufferMatrixOut->data[indx_ColT * Row_SizeT + indx_RowT].im = BufferMatrixIn->data[indx].im;
	}

	BufferMatrixOut->size[0] = Row_SizeT;
	BufferMatrixOut->size[1] = Col_SizeT;
}
//=================================================================
//						TransMatrixReshapeNative
//=================================================================
void TransMatrixReshapeNative( const Array_creal_T *BufferMatrixIn ,int32_T row,int32_T colm, Array_creal_T *BufferMatrixOut )
{
	int32_T indx,s;
	uint32_T loop_up;
	uint32_T indx_RowT;
	uint32_T indx_ColT;
	uint32_T Row_SizeT;
	uint32_T Col_SizeT;
	uint32_T Row_Size;
	uint32_T Col_Size;

	Row_Size = row;
	Col_Size = colm;
	loop_up = Row_Size * Col_Size;
	Col_SizeT = Row_Size;
	Row_SizeT = Col_Size;

//Opt : #pragma omp parallel for private(indx, indx_RowT, indx_ColT) firstprivate(loop_up, Row_SizeT, Row_Size)
	/*for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Row_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Row_Size; // index For Data of Transpose Matrix
		BufferMatrixOut->data[indx_ColT * Row_SizeT + indx_RowT].re = BufferMatrixIn->data[indx].re;
		BufferMatrixOut->data[indx_ColT * Row_SizeT + indx_RowT].im = BufferMatrixIn->data[indx].im;
	}*/
	for(s=0;s<colm;s++)
	for (indx = 0; indx < row; indx++)
	{
		BufferMatrixOut->data[s*row+indx].re = BufferMatrixIn->data[s*row+indx].re;
		BufferMatrixOut->data[s*row+indx].im = BufferMatrixIn->data[s*row+indx].im;
	}

	BufferMatrixOut->size[0] = row;
	BufferMatrixOut->size[1] = colm;
}
//=================================================================
//						TransMatrixNoDimChangNative
//=================================================================
void TransMatrixNoDimChangNative( Array_creal_T *BufferMatrix )
{
	creal_T *data;
	int32_T indx;
	uint32_T loop_up;
	uint32_T indx_RowT;
	uint32_T indx_ColT;
	uint32_T Row_SizeT;
	uint32_T Col_SizeT;
	uint32_T Row_Size;
	uint32_T Col_Size;

	data = (creal_T *) PASSIVE_Alloc(BufferMatrix->allocatedSize);

	Row_Size = BufferMatrix->size[0];
	Col_Size = BufferMatrix->size[1];
	loop_up = Row_Size * Col_Size;
	Col_SizeT = Row_Size;
	Row_SizeT = Col_Size;

#pragma omp parallel for private(indx, indx_RowT, indx_ColT) firstprivate(loop_up, Row_SizeT, Row_Size)
	for (indx = 0; indx < loop_up; indx++)
	{
		indx_ColT = indx % Row_Size; // index For Data of Transpose Matrix
		indx_RowT = indx / Row_Size; // index For Data of Transpose Matrix
		data[indx_ColT * Row_SizeT + indx_RowT] = BufferMatrix->data[indx];
	}

	PASSIVE_Free(BufferMatrix->data);
	BufferMatrix->data = data;

}
