/**
 * Project Name: vector4
 * Module Name:
 * Filename: vector4.c
 * Creator: Yaokai Liu
 * Create Date: 2024-07-21
 * Copyright (c) 2024 Yaokai Liu. All rights reserved.
 **/

#include "vector4.h"
#include <nmmintrin.h>

inline void intVecAdd(IVec4 summand, const IVec4 addend) {
  __m128i a, b, c;
  a = _mm_load_si128((const __m128i *) summand);
  b = _mm_load_si128((const __m128i *) addend);
  c = _mm_add_epi32(a, b);
  _mm_store_si128((__m128i *) summand, c);
}

inline void intVecSub(IVec4 minuend, const IVec4 subtrahend) {
  __m128i a, b, c;
  a = _mm_load_si128((const __m128i *) minuend);
  b = _mm_load_si128((const __m128i *) subtrahend);
  c = _mm_sub_epi32(a, b);
  _mm_store_si128((__m128i *) minuend, c);
}

inline void intVecScale(IVec4 vector, int32_t rate) {
  __m128i a, b, c;
  a = _mm_load_si128((const __m128i *) vector);
  b = _mm_set1_epi32(rate);
  c = _mm_mullo_epi32(a, b);
  _mm_store_si128((__m128i *) vector, c);
}

inline void intVecCoverMul(IVec4 multiplier, const IVec4 multiplicand) {
  __m128i a, b, c;
  a = _mm_load_si128((const __m128i *) multiplier);
  b = _mm_load_si128((const __m128i *) multiplicand);
  c = _mm_mullo_epi32(a, b);
  _mm_store_si128((__m128i *) multiplier, c);
}

inline void intVecRoverMul(IVec4 multiplier, const IVec4 multiplicand) {
  __m128i a, b, c;
  a = (__m128i) _mm_load_ps((const float *) multiplier);
  b = (__m128i) _mm_loadr_ps((const float *) multiplicand);
  c = _mm_mullo_epi32(a, b);
  _mm_store_si128((__m128i *) multiplier, c);
}

inline int32_t intVecInnerProduct(const IVec4 multiplier, const IVec4 multiplicand) {
  __m128i a, b, c, d;
  a = (__m128i) _mm_load_ps((const float *) multiplier);
  b = (__m128i) _mm_load_ps((const float *) multiplicand);
  c = _mm_mullo_epi32(a, b);
  d = _mm_setzero_si128();
  c = _mm_hadd_epi32(c, d);
  c = _mm_hadd_epi32(c, d);
  int32_t register res;
  __asm__ __volatile__ ("movd %[c], %[res]": [res] "=r"(res): [c] "x"(c));
  return res;
}

inline int32_t intVecConvProduct(const IVec4 multiplier, const IVec4 multiplicand) {
  __m128i a, b, c, d;
  a = (__m128i) _mm_load_ps((const float *) multiplier);
  b = (__m128i) _mm_loadr_ps((const float *) multiplicand);
  c = _mm_mullo_epi32(a, b);
  d = _mm_setzero_si128();
  c = _mm_hadd_epi32(c, d);
  c = _mm_hadd_epi32(c, d);
  int32_t register res;
  __asm__ __volatile__ ("movd %[c], %[res]": [res] "=r"(res): [c] "x"(c));
  return res;
}

inline void intVecMulMat(IVec4 a, const IMat4 b) {
  __m128i m1;
  IVec4 c = {
    [0] = intVecInnerProduct(a, b[0]),
    [1] = intVecInnerProduct(a, b[1]),
    [2] = intVecInnerProduct(a, b[2]),
    [3] = intVecInnerProduct(a, b[3]),
  };
  m1 = _mm_load_si128((const __m128i *) c);
  _mm_store_si128((__m128i *) a, m1);
}


inline void floatVecAdd(FVec4 summand, const FVec4 addend) {
  __m128 a, b, c;
  a = _mm_load_ps(summand);
  b = _mm_load_ps( addend);
  c = _mm_add_ps(a, b);
  _mm_store_ps(summand, c);
}

inline void floatVecSub(FVec4 minuend, const FVec4 subtrahend) {
  __m128 a, b, c;
  a = _mm_load_ps(minuend);
  b = _mm_load_ps(subtrahend);
  c = _mm_sub_ps(a, b);
  _mm_store_ps(minuend, c);
}

inline void floatVecScale(FVec4 vector, float rate) {
  __m128 a, b, c;
  a = _mm_load_ps(vector);
  b = _mm_set1_ps(rate);
  c = _mm_mul_ps(a, b);
  _mm_store_ps(vector, c);
}

inline void floatVecCoverMul(FVec4 multiplier, const FVec4 multiplicand) {
  __m128 a, b, c;
  a = _mm_load_ps(multiplier);
  b = _mm_load_ps(multiplicand);
  c = _mm_mul_ps(a, b);
  _mm_store_ps(multiplier, c);
}

inline void floatVecRoverMul(FVec4 multiplier, const FVec4 multiplicand) {
  __m128 a, b, c;
  a = _mm_load_ps(multiplier);
  b = _mm_loadr_ps(multiplicand);
  c = _mm_mul_ps(a, b);
  _mm_store_ps(multiplier, c);
}

inline float floatVecInnerProduct(const FVec4 multiplier, const FVec4 multiplicand) {
  __m128 a, b, c, d;
  a = _mm_load_ps(multiplier);
  b = _mm_load_ps(multiplicand);
  c = _mm_mul_ps(a, b);
  d = _mm_setzero_ps();
  c = _mm_hadd_ps(c, d);
  c = _mm_hadd_ps(c, d);
  return _mm_cvtss_f32(c);
}

inline float floatVecConvProduct(const FVec4 multiplier, const FVec4 multiplicand) {
  __m128 a, b, c, d;
  a = _mm_load_ps(multiplier);
  b = _mm_loadr_ps(multiplicand);
  c = _mm_mul_ps(a, b);
  d = _mm_setzero_ps();
  c = _mm_hadd_ps(c, d);
  c = _mm_hadd_ps(c, d);
  return _mm_cvtss_f32(c);
}

inline void floatVec3CrossMul(FVec4 multiplier, const FVec4 multiplicand) {
  __m128 a, b, c;
  a = _mm_load_ps(multiplier);
  b = _mm_load_ps(multiplicand);
  constexpr int r1 = _MM_SHUFFLE(3, 0, 2, 1);
  constexpr int r2 = _MM_SHUFFLE(3, 1, 2, 0);
  __m128 tmp0 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,0,2,1));
  __m128 tmp1 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,1,0,2));
  __m128 tmp2 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3,1,0,2));
  __m128 tmp3 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3,0,2,1));
  c = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
  _mm_store_ps(multiplier, c);
}

inline void floatVecMulMat(FVec4 a, const FMat4 b) {
  __m128 m1;
  FVec4 c = {
    [0] = floatVecInnerProduct(a, b[0]),
    [1] = floatVecInnerProduct(a, b[1]),
    [2] = floatVecInnerProduct(a, b[2]),
    [3] = floatVecInnerProduct(a, b[3]),
  };
  m1 = _mm_load_ps(c);
  _mm_store_ps(a, m1);
}
