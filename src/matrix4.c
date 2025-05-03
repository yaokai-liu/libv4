/**
 * Project Name: vector4
 * Module Name: 
 * Filename: matrix4.c
 * Creator: Yaokai Liu
 * Create Date: 2025-05-03
 * Copyright (c) 2025 Yaokai Liu. All rights reserved.
 **/

#include "matrix4.h"
#include "vector4.h"
#include <nmmintrin.h>

inline void intMatAdd(IMat4 a, const IMat4 b) {
  intVecAdd(a[0], b[0]);
  intVecAdd(a[1], b[1]);
  intVecAdd(a[2], b[2]);
  intVecAdd(a[3], b[3]);
}

inline void intMatSub(IMat4 a, const IMat4 b) {
  intVecSub(a[0], b[0]);
  intVecSub(a[1], b[1]);
  intVecSub(a[2], b[2]);
  intVecSub(a[3], b[3]);
}


inline void intMatScale(IMat4 a, const int32_t b) {
  intVecScale(a[0], b);
  intVecScale(a[1], b);
  intVecScale(a[2], b);
  intVecScale(a[3], b);
}

inline void intMatScaleVec(IMat4 a, const IVec4 b) {
  intVecScale(a[0], b[0]);
  intVecScale(a[1], b[1]);
  intVecScale(a[2], b[2]);
  intVecScale(a[3], b[3]);
}

inline void intMatScaleMat(IMat4 a, const IMat4 b) {
  intVecCoverMul(a[0], b[0]);
  intVecCoverMul(a[1], b[1]);
  intVecCoverMul(a[2], b[2]);
  intVecCoverMul(a[3], b[3]);
}

inline void intMatMulVec(const IMat4 a, IVec4 b) {
  __m128i m1;
  IVec4 c = {
    [0] = intVecInnerProduct(a[0], b),
    [1] = intVecInnerProduct(a[1], b),
    [2] = intVecInnerProduct(a[2], b),
    [3] = intVecInnerProduct(a[3], b),
  };
  m1 = _mm_load_si128((const __m128i *) c);
  _mm_store_si128((__m128i *) b, m1);
}

inline void intMatTrCopy(IMat4 des, const IMat4 src) {
  __m128 b0 = _mm_load_ps( (float *) src[0]);
  __m128 b1 = _mm_load_ps( (float *) src[1]);
  __m128 b2 = _mm_load_ps( (float *) src[2]);
  __m128 b3 = _mm_load_ps( (float *) src[3]);
  __m128 t0 = _mm_unpacklo_ps(b0, b1);
  __m128 t1 = _mm_unpackhi_ps(b0, b1);
  __m128 t2 = _mm_unpacklo_ps(b2, b3);
  __m128 t3 = _mm_unpackhi_ps(b2, b3);
  __m128 w = _mm_movelh_ps(t0, t2);
  __m128 x = _mm_movehl_ps(t2, t0);
  __m128 y = _mm_movelh_ps(t1, t3);
  __m128 z = _mm_movehl_ps(t3, t1);
  _mm_store_ps((float *) des[0], w);
  _mm_store_ps((float *) des[1], x);
  _mm_store_ps((float *) des[2], y);
  _mm_store_ps((float *) des[3], z);
}

inline void intMatLMulMat(const IMat4 a, IMat4 b) {
  intMatMulVec(a, b[0]);
  intMatMulVec(a, b[1]);
  intMatMulVec(a, b[2]);
  intMatMulVec(a, b[3]);
}

inline void intMatRMulMat(IMat4 a, const IMat4 b) {
  intVecMulMat(a[0], b);
  intVecMulMat(a[1], b);
  intVecMulMat(a[2], b);
  intVecMulMat(a[3], b);
}

inline void floatMatAdd(FMat4 a, const FMat4 b) {
  floatVecAdd(a[0], b[0]);
  floatVecAdd(a[1], b[1]);
  floatVecAdd(a[2], b[2]);
  floatVecAdd(a[3], b[3]);
}

inline void floatMatSub(FMat4 a, const FMat4 b) {
  floatVecSub(a[0], b[0]);
  floatVecSub(a[1], b[1]);
  floatVecSub(a[2], b[2]);
  floatVecSub(a[3], b[3]);
}


inline void floatMatScale(FMat4 a, const float b) {
  floatVecScale(a[0], b);
  floatVecScale(a[1], b);
  floatVecScale(a[2], b);
  floatVecScale(a[3], b);
}

inline void floatMatScaleVec(FMat4 a, const FVec4 b) {
  floatVecScale(a[0], b[0]);
  floatVecScale(a[1], b[1]);
  floatVecScale(a[2], b[2]);
  floatVecScale(a[3], b[3]);
}

inline void floatMatScaleMat(FMat4 a, const FMat4 b) {
  floatVecCoverMul(a[0], b[0]);
  floatVecCoverMul(a[1], b[1]);
  floatVecCoverMul(a[2], b[2]);
  floatVecCoverMul(a[3], b[3]);
}

inline void floatMatMulVec(const FMat4 a, FVec4 b) {
  __m128 m1;
  FVec4 c = {
    [0] = floatVecInnerProduct(a[0], b),
    [1] = floatVecInnerProduct(a[1], b),
    [2] = floatVecInnerProduct(a[2], b),
    [3] = floatVecInnerProduct(a[3], b),
  };
  m1 = _mm_load_ps(c);
  _mm_store_ps(b, m1);
}

inline void floatMatTrCopy(FMat4 des, const FMat4 src) {
  __m128 b0 = _mm_load_ps(src[0]);
  __m128 b1 = _mm_load_ps(src[1]);
  __m128 b2 = _mm_load_ps(src[2]);
  __m128 b3 = _mm_load_ps(src[3]);
  __m128 t0 = _mm_unpacklo_ps(b0, b1);
  __m128 t1 = _mm_unpackhi_ps(b0, b1);
  __m128 t2 = _mm_unpacklo_ps(b2, b3);
  __m128 t3 = _mm_unpackhi_ps(b2, b3);
  __m128 w = _mm_movelh_ps(t0, t2);
  __m128 x = _mm_movehl_ps(t2, t0);
  __m128 y = _mm_movelh_ps(t1, t3);
  __m128 z = _mm_movehl_ps(t3, t1);
  _mm_store_ps(des[0], w);
  _mm_store_ps(des[1], x);
  _mm_store_ps(des[2], y);
  _mm_store_ps(des[3], z);
}

inline void floatMatLMulMat(const FMat4 a, FMat4 b) {
  floatMatTrCopy(b, b);
  floatMatMulVec(a, b[0]);
  floatMatMulVec(a, b[1]);
  floatMatMulVec(a, b[2]);
  floatMatMulVec(a, b[3]);
  floatMatTrCopy(b, b);
}

inline void floatMatRMulMat(FMat4 a, const FMat4 b) {
  FMat4 c = {};
  floatMatTrCopy(c, b);
  floatVecMulMat(a[0], c);
  floatVecMulMat(a[1], c);
  floatVecMulMat(a[2], c);
  floatVecMulMat(a[3], c);
}
