/**
 * Project Name: vector4
 * Module Name:
 * Filename: vector4.h
 * Creator: Yaokai Liu
 * Create Date: 2024-07-21
 * Copyright (c) 2024 Yaokai Liu. All rights reserved.
 **/

#ifndef VECTOR4_VECTOR4_H
#define VECTOR4_VECTOR4_H
#include "type.h"

#define vecShuffle(src, des, o) do {             \
  __m128 a, b;                                   \
  int __r = _MM_SHUFFLE(o[0], o[1], o[2], o[3]); \
  a = _mm_load_ps((float *) src);                \
  b = _mm_shuffle_ps(a, a, __r);                 \
  __mm_store_ps((float *) des, b);               \
} while (false)

#define vecSwap(vec1, vec2) do {    \
  __m128 a, b;                      \
  a = _mm_load_ps((float *) vec1);  \
  b = _mm_load_ps((float *) vec2);  \
  __mm_store_ps((float *) vec1, b); \
  __mm_store_ps((float *) vec2, a); \
} while (false)

/// Addend: a[i] = a[i] + b[i]
void intVecAdd(IVec4 summand, const IVec4 addend);
/// Subtract:  a[i] = a[i] - b[i]
void intVecSub(IVec4 minuend, const IVec4 subtrahend);
/// Scale:  a[i] = a[i] * b
void intVecScale(IVec4 vector, int32_t rate);
/// Cover Multiple: a[i] = a[i] * b[i]
void intVecCoverMul(IVec4 multiplier, const IVec4 multiplicand);
/// Rover Multiple: a[i] = a[i] * b[n - i]
void intVecRoverMul(IVec4 multiplier, const IVec4 multiplicand);
/// Inner Product: sum(a[i] * b[i])
int32_t intVecInnerProduct(const IVec4 multiplier, const IVec4 multiplicand);
/// Convolution Product: sum(a[i] * b[n - i])
int32_t intVecConvProduct(const IVec4 multiplier, const IVec4 multiplicand);


/// a = transpose(a * b)
void intVecMulMat(IVec4 a, const IMat4 b);

/// Addend: a[i] = a[i] + b[i]
void floatVecAdd(FVec4 summand, const FVec4 addend);
/// Subtract:  a[i] = a[i] - b[i]
void floatVecSub(FVec4 minuend, const FVec4 subtrahend);
/// Scale:  a[i] = a[i] * b
void floatVecScale(FVec4 vector, float rate);
/// Cover Multiple: a[i] = a[i] * b[i]
void floatVecCoverMul(FVec4 multiplier, const FVec4 multiplicand);
/// Rover Multiple: a[i] = a[i] * b[n - i]
void floatVecRoverMul(FVec4 multiplier, const FVec4 multiplicand);
/// Inner Product: sum(a[i] * b[i])
float floatVecInnerProduct(const FVec4 multiplier, const FVec4 multiplicand);
/// Convolution Product: sum(a[i] * b[n - i])
float floatVecConvProduct(const FVec4 multiplier, const FVec4 multiplicand);
/// Cross Multiple: a[i] = a[i] × b[i]
void floatVec3CrossMul(FVec4 multiplier, const FVec4 multiplicand);

/// a = transpose(a * b)
void floatVecMulMat(FVec4 a, const FMat4 b);

#endif //VECTOR4_VECTOR4_H

