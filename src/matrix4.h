/**
 * Project Name: vector4
 * Module Name: 
 * Filename: matrix4.h
 * Creator: Yaokai Liu
 * Create Date: 2025-05-03
 * Copyright (c) 2025 Yaokai Liu. All rights reserved.
 **/

#ifndef VECTOR4_MATRIX4_H
#define VECTOR4_MATRIX4_H

#include "type.h"

/// a[i,j] = a[i,j] + b[i,j]
void intMatAdd(IMat4 a, const IMat4 b);
/// a[i,j] = a[i,j] + b[i,j]
void intMatSub(IMat4 a, const IMat4 b);
/// a[i,j] = a[i,j] * b
void intMatScale(IMat4 a, int32_t b);
/// a[i,j] = a[i,j] * b[i]
void intMatScaleVec(IMat4 a, const IVec4 b);
/// a[i,j] = a[i,j] * b[i,j]
void intMatScaleMat(IMat4 a, const IMat4 b);
/// b = transpose(a * b)
void intMatMulVec(const IMat4 a, IVec4 b);
/// b = a * b
void intMatLMulMat(const IMat4 a, IMat4 b);
/// a = a * b
void intMatRMulMat(IMat4 a, const IMat4 b);
/// a = transpose(a)
void intMatTrCopy(IMat4 des, const IMat4 a);

/// a[i,j] = a[i,j] + b[i,j]
void floatMatAdd(FMat4 a, const FMat4 b);
/// a[i,j] = a[i,j] + b[i,j]
void floatMatSub(FMat4 a, const FMat4 b);
/// a[i,j] = a[i,j] * b
void floatMatScale(FMat4 a, float b);
/// a[i,j] = a[i,j] * b[i]
void floatMatScaleVec(FMat4 a, const FVec4 b);
/// a[i,j] = a[i,j] * b[i,j]
void floatMatScaleMat(FMat4 a, const FMat4 b);
/// b = transpose(a * b)
void floatMatMulVec(const FMat4 a, FVec4 b);
/// b = a * b
void floatMatLMulMat(const FMat4 a, FMat4 b);
/// a = a * b
void floatMatRMulMat(FMat4 a, const FMat4 b);
/// a = transpose(a)
void floatMatTrCopy(FMat4 des, const FMat4 a);
#endif //VECTOR4_MATRIX4_H
