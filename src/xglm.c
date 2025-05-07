/**
 * Project Name: xglm
 * Module Name: 
 * Filename: xglm.c
 * Creator: Yaokai Liu
 * Create Date: 2025-05-03
 * Copyright (c) 2025 Yaokai Liu. All rights reserved.
 **/

#include "xglm.h"
#include <nmmintrin.h>
#include <math.h>

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

inline void intVecMulMat(IVec4 a, const IMat4 B) {
  __m128i v = _mm_load_si128((__m128i *) a);
  __m128i B_0 = _mm_load_si128((__m128i *) B[0]);
  __m128i B_1 = _mm_load_si128((__m128i *) B[1]);
  __m128i B_2 = _mm_load_si128((__m128i *) B[2]);
  __m128i B_3 = _mm_load_si128((__m128i *) B[3]);
  B_0 = _mm_mul_epi32(v, B_0);
  B_1 = _mm_mul_epi32(v, B_1);
  B_2 = _mm_mul_epi32(v, B_2);
  B_3 = _mm_mul_epi32(v, B_3);
  __m128i s = _mm_hadd_epi32(B_0, B_1);
  __m128i t = _mm_hadd_epi32(B_2, B_3);
  __m128i w = _mm_hadd_epi32(s, t);
  _mm_store_si128((__m128i *) a, w);
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
  constexpr int r2 = _MM_SHUFFLE(3, 1, 0, 2);
  __m128 tmp0 = _mm_shuffle_ps(a, a, r1);
  __m128 tmp1 = _mm_shuffle_ps(b, b, r2);
  __m128 tmp2 = _mm_shuffle_ps(a, a, r2);
  __m128 tmp3 = _mm_shuffle_ps(b, b, r1);
  c = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
  _mm_store_ps(multiplier, c);
}

inline void floatVecMulMat(FVec4 a, const FMat4 B) {
  __m128 v = _mm_load_ps(a);
  __m128 B_0 = _mm_load_ps(B[0]);
  __m128 B_1 = _mm_load_ps(B[1]);
  __m128 B_2 = _mm_load_ps(B[2]);
  __m128 B_3 = _mm_load_ps(B[3]);
  B_0 = _mm_mul_ps(v, B_0);
  B_1 = _mm_mul_ps(v, B_1);
  B_2 = _mm_mul_ps(v, B_2);
  B_3 = _mm_mul_ps(v, B_3);
  __m128 s = _mm_hadd_ps(B_0, B_1);
  __m128 t = _mm_hadd_ps(B_2, B_3);
  __m128 w = _mm_hadd_ps(s, t);
  _mm_store_ps(a, w);
}

inline void floatAffineNormalize(FAffPoint4 P) {
  _mm_store_ps(P, _mm_mul_ps(_mm_load_ps(P), _mm_set1_ps(1.0f / P[3])));
}

inline void floatAffineSub(const FAffPoint4 A, const FAffPoint4 B, FVec4 d) {
  __m128 a128 = _mm_mul_ps(_mm_load_ps(A), _mm_set1_ps(1.0f / A[3]));
  __m128 b128 = _mm_mul_ps(_mm_load_ps(B), _mm_set1_ps(1.0f / B[3]));
  _mm_store_ps(d, _mm_sub_ps(a128, b128));
}

inline void floatAffineReflect(FAffPoint4 P, const FAffPoint4 O, const FVec4 d) {
  // Affine Point P
  const __m128 a128 = _mm_mul_ps(_mm_load_ps(P), _mm_set1_ps(1.0f / P[3]));
  // Affine Point O
  const __m128 b128 = _mm_mul_ps(_mm_load_ps(O), _mm_set1_ps(1.0f / O[3]));
  // Vector3 OP
  const __m128 c128 = _mm_sub_ps(a128, b128);
  // Vector3 d
  const __m128 d128 = _mm_sub_ps(_mm_load_ps(d), _mm_set_ps(0, 0, 0, d[3]));
  // calculate OP·d and sqrt(d·d)
  const __m128 z128 = _mm_setzero_ps();
  __m128 t128 = _mm_mul_ps(c128, d128);
  __m128 s128 = _mm_mul_ps(d128, d128);
  t128 = _mm_hadd_ps(_mm_hadd_ps(t128, z128), z128);
  s128 = _mm_hadd_ps(_mm_hadd_ps(s128, z128), z128);
  t128 = _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(0, 0, 0, 0));
  s128 = _mm_shuffle_ps(s128, s128, _MM_SHUFFLE(0, 0, 0, 0));
  s128 = _mm_sqrt_ps(s128);
  // calculate 2(OP·d)/sqrt(d·d)
  const __m128 r128 = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(2.0f), t128), s128);
  // P - 2(OP·d)/sqrt(d·d) * d
  const __m128 result = _mm_sub_ps(_mm_mul_ps(r128, d128), a128);
  // return
  _mm_store_ps(P, result);
}

void floatAffineFlip(FAffPoint4 P, const FAffPoint4 O, const FVec4 d) {
  // Affine Point P
  __m128 a128 = _mm_mul_ps(_mm_load_ps(P), _mm_set1_ps(1.0f / P[3]));
  // Affine Point O
  __m128 b128 = _mm_mul_ps(_mm_load_ps(O), _mm_set1_ps(1.0f / O[3]));
  // Vector3 OP
  __m128 c128 = _mm_sub_ps(a128, b128);
  // Vector3 d
  __m128 d128 = _mm_sub_ps(_mm_load_ps(d), _mm_set_ps(d[3], 0, 0, 0));
  // calculate OP·d and sqrt(d·d)
  __m128 t128 = _mm_mul_ps(c128, d128);
  __m128 s128 = _mm_mul_ps(d128, d128);
  __m128 z128 = _mm_setzero_ps();
  t128 = _mm_hadd_ps(_mm_hadd_ps(t128, z128), z128);
  s128 = _mm_hadd_ps(_mm_hadd_ps(s128, z128), z128);
  t128 = _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(0, 0, 0, 0));
  s128 = _mm_shuffle_ps(s128, s128, _MM_SHUFFLE(0, 0, 0, 0));
  s128 = _mm_sqrt_ps(s128);
  // calculate 2(OP·d)/sqrt(d·d)
  __m128 r128 = _mm_div_ps(_mm_mul_ps(_mm_set1_ps(2.0f), t128), s128);
  // O + 2(OP·d)/sqrt(d·d) * d - OP
  __m128 result = _mm_sub_ps(_mm_add_ps(b128, _mm_mul_ps(r128, d128)), c128);
  // return
  _mm_store_ps(P, result);
}

inline void intMatDiag(IMat4 A, const IVec4 b) {
  __m128i A_0 = _mm_set_epi32(0, 0, 0, b[0]);
  __m128i A_1 = _mm_set_epi32(0, 0, b[1], 0);
  __m128i A_2 = _mm_set_epi32(0, b[2], 0, 0);
  __m128i A_3 = _mm_set_epi32(b[3], 0, 0, 0);
  _mm_store_si128((__m128i *) A[0], A_0);
  _mm_store_si128((__m128i *) A[1], A_1);
  _mm_store_si128((__m128i *) A[2], A_2);
  _mm_store_si128((__m128i *) A[3], A_3);
}

inline void intMatAdd(IMat4 A, const IMat4 B) {
  intVecAdd(A[0], B[0]);
  intVecAdd(A[1], B[1]);
  intVecAdd(A[2], B[2]);
  intVecAdd(A[3], B[3]);
}

inline void intMatSub(IMat4 A, const IMat4 B) {
  intVecSub(A[0], B[0]);
  intVecSub(A[1], B[1]);
  intVecSub(A[2], B[2]);
  intVecSub(A[3], B[3]);
}


inline void intMatScale(IMat4 A, const int32_t b) {
  intVecScale(A[0], b);
  intVecScale(A[1], b);
  intVecScale(A[2], b);
  intVecScale(A[3], b);
}

inline void intMatScaleVec(IMat4 A, const IVec4 b) {
  intVecScale(A[0], b[0]);
  intVecScale(A[1], b[1]);
  intVecScale(A[2], b[2]);
  intVecScale(A[3], b[3]);
}

inline void intMatScaleMat(IMat4 A, const IMat4 B) {
  intVecCoverMul(A[0], B[0]);
  intVecCoverMul(A[1], B[1]);
  intVecCoverMul(A[2], B[2]);
  intVecCoverMul(A[3], B[3]);
}

inline void intMatMulVec(const IMat4 A, IVec4 b) {
  __m128i A_0 = _mm_load_si128((__m128i *) A[0]);
  __m128i A_1 = _mm_load_si128((__m128i *) A[1]);
  __m128i A_2 = _mm_load_si128((__m128i *) A[2]);
  __m128i A_3 = _mm_load_si128((__m128i *) A[3]);
  __m128i v = _mm_load_si128((__m128i *) b);
  A_0 = _mm_mul_epi32(v, A_0);
  A_1 = _mm_mul_epi32(v, A_1);
  A_2 = _mm_mul_epi32(v, A_2);
  A_3 = _mm_mul_epi32(v, A_3);
  __m128i s = _mm_hadd_epi32(A_0, A_1);
  __m128i t = _mm_hadd_epi32(A_2, A_3);
  __m128i w = _mm_hadd_epi32(s, t);
  _mm_store_si128((__m128i *) b, w);
}

inline void intMatTrCopy(IMat4 M_T, const IMat4 A) {
  __m128 b0 = _mm_load_ps( (float *) A[0]);
  __m128 b1 = _mm_load_ps( (float *) A[1]);
  __m128 b2 = _mm_load_ps( (float *) A[2]);
  __m128 b3 = _mm_load_ps( (float *) A[3]);
  __m128 t0 = _mm_unpacklo_ps(b0, b1);
  __m128 t1 = _mm_unpackhi_ps(b0, b1);
  __m128 t2 = _mm_unpacklo_ps(b2, b3);
  __m128 t3 = _mm_unpackhi_ps(b2, b3);
  __m128 w = _mm_movelh_ps(t0, t2);
  __m128 x = _mm_movehl_ps(t2, t0);
  __m128 y = _mm_movelh_ps(t1, t3);
  __m128 z = _mm_movehl_ps(t3, t1);
  _mm_store_ps((float *) M_T[0], w);
  _mm_store_ps((float *) M_T[1], x);
  _mm_store_ps((float *) M_T[2], y);
  _mm_store_ps((float *) M_T[3], z);
}

inline void intMatLMulMat(const IMat4 A, IMat4 B) {
  intMatMulVec(A, B[0]);
  intMatMulVec(A, B[1]);
  intMatMulVec(A, B[2]);
  intMatMulVec(A, B[3]);
}

inline void intMatRMulMat(IMat4 A, const IMat4 B) {
  intVecMulMat(A[0], B);
  intVecMulMat(A[1], B);
  intVecMulMat(A[2], B);
  intVecMulMat(A[3], B);
}

inline void floatMatDiag(FMat4 A, const FVec4 b) {
  __m128 A_0 = _mm_setr_ps(b[0], 0.0f, 0.0f, 0.0f);
  __m128 A_1 = _mm_setr_ps(0.0f, b[1], 0.0f, 0.0f);
  __m128 A_2 = _mm_setr_ps(0.0f, 0.0f, b[2], 0.0f);
  __m128 A_3 = _mm_setr_ps(0.0f, 0.0f, 0.0f, b[3]);
  _mm_store_ps(A[0], A_0);
  _mm_store_ps(A[1], A_1);
  _mm_store_ps(A[2], A_2);
  _mm_store_ps(A[3], A_3);
}

inline void floatMatAdd(FMat4 A, const FMat4 B) {
  floatVecAdd(A[0], B[0]);
  floatVecAdd(A[1], B[1]);
  floatVecAdd(A[2], B[2]);
  floatVecAdd(A[3], B[3]);
}

inline void floatMatSub(FMat4 A, const FMat4 B) {
  floatVecSub(A[0], B[0]);
  floatVecSub(A[1], B[1]);
  floatVecSub(A[2], B[2]);
  floatVecSub(A[3], B[3]);
}


inline void floatMatScale(FMat4 A, const float b) {
  floatVecScale(A[0], b);
  floatVecScale(A[1], b);
  floatVecScale(A[2], b);
  floatVecScale(A[3], b);
}

inline void floatMatScaleVec(FMat4 A, const FVec4 b) {
  floatVecScale(A[0], b[0]);
  floatVecScale(A[1], b[1]);
  floatVecScale(A[2], b[2]);
  floatVecScale(A[3], b[3]);
}

inline void floatMatScaleMat(FMat4 A, const FMat4 B) {
  floatVecCoverMul(A[0], B[0]);
  floatVecCoverMul(A[1], B[1]);
  floatVecCoverMul(A[2], B[2]);
  floatVecCoverMul(A[3], B[3]);
}

inline void floatMatMulVec(const FMat4 A, FVec4 b) {
  __m128 v = _mm_load_ps(b);
  __m128 A_0 = _mm_load_ps(A[0]);
  __m128 A_1 = _mm_load_ps(A[1]);
  __m128 A_2 = _mm_load_ps(A[2]);
  __m128 A_3 = _mm_load_ps(A[3]);
  A_0 = _mm_mul_ps(v, A_0);
  A_1 = _mm_mul_ps(v, A_1);
  A_2 = _mm_mul_ps(v, A_2);
  A_3 = _mm_mul_ps(v, A_3);
  __m128 s = _mm_hadd_ps(A_0, A_1);
  __m128 t = _mm_hadd_ps(A_2, A_3);
  __m128 w = _mm_hadd_ps(s, t);
  _mm_store_ps(b, w);
}

inline void floatMatTrCopy(FMat4 M_T, const FMat4 M) {
  __m128 b0 = _mm_load_ps(M[0]);
  __m128 b1 = _mm_load_ps(M[1]);
  __m128 b2 = _mm_load_ps(M[2]);
  __m128 b3 = _mm_load_ps(M[3]);
  __m128 t0 = _mm_unpacklo_ps(b0, b1);
  __m128 t1 = _mm_unpackhi_ps(b0, b1);
  __m128 t2 = _mm_unpacklo_ps(b2, b3);
  __m128 t3 = _mm_unpackhi_ps(b2, b3);
  __m128 w = _mm_movelh_ps(t0, t2);
  __m128 x = _mm_movehl_ps(t2, t0);
  __m128 y = _mm_movelh_ps(t1, t3);
  __m128 z = _mm_movehl_ps(t3, t1);
  _mm_store_ps(M_T[0], w);
  _mm_store_ps(M_T[1], x);
  _mm_store_ps(M_T[2], y);
  _mm_store_ps(M_T[3], z);
}

inline void floatMatLMulMat(const FMat4 A, FMat4 B) {
  floatMatTrCopy(B, B);
  floatMatMulVec(A, B[0]);
  floatMatMulVec(A, B[1]);
  floatMatMulVec(A, B[2]);
  floatMatMulVec(A, B[3]);
  floatMatTrCopy(B, B);
}

inline void floatMatRMulMat(FMat4 A, const FMat4 B) {
  FMat4 C = {};
  floatMatTrCopy(C, B);
  floatVecMulMat(A[0], C);
  floatVecMulMat(A[1], C);
  floatVecMulMat(A[2], C);
  floatVecMulMat(A[3], C);
}


inline void matFromAffineScale(FMat4 M, const FVec4 /* treat as FVec3 */ rate) {
  __m128 T_0 = _mm_setr_ps(rate[0], 0, 0, 0);
  __m128 T_1 = _mm_setr_ps(0, rate[1], 0, 0);
  __m128 T_2 = _mm_setr_ps(0, 0, rate[2], 0);
  __m128 T_3 = _mm_setr_ps(0, 0, 0, 1);
  _mm_store_ps(M[0], T_0);
  _mm_store_ps(M[1], T_1);
  _mm_store_ps(M[2], T_2);
  _mm_store_ps(M[3], T_3);
}

inline void matFromAffineShear(FMat4 M, const FVec4 /* treat as FVec3 */ she) {
  __m128 T_0 = _mm_setr_ps(1, she[1], she[2], 0);
  __m128 T_1 = _mm_setr_ps(she[0], 1, she[2], 0);
  __m128 T_2 = _mm_setr_ps(she[0], she[1], 1, 0);
  __m128 T_3 = _mm_setr_ps(0, 0, 0, 1);
  _mm_store_ps(M[0], T_0);
  _mm_store_ps(M[1], T_1);
  _mm_store_ps(M[2], T_2);
  _mm_store_ps(M[3], T_3);
}

inline void matFromAffineShift(FMat4 M, const FVec4 /* treat as FVec3 */ dis) {
  __m128 T_0 = _mm_setr_ps(1, 0, 0, dis[0]);
  __m128 T_1 = _mm_setr_ps(0, 1, 0, dis[1]);
  __m128 T_2 = _mm_setr_ps(0, 0, 1, dis[2]);
  __m128 T_3 = _mm_setr_ps(0, 0, 0, 1);
  _mm_store_ps(M[0], T_0);
  _mm_store_ps(M[1], T_1);
  _mm_store_ps(M[2], T_2);
  _mm_store_ps(M[3], T_3);
}

inline void matFromAffineRotate(FMat4 M, const FVec4 /* treat as FVec3 */ axis, float angle) {
  const __m128 d128 = _mm_sub_ps(_mm_load_ps(axis), _mm_set_ps(axis[3], 0, 0, 0));
  const __m128 d128_0 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(0, 0, 0, 0));
  const __m128 d128_1 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(1, 1, 1, 1));
  const __m128 d128_2 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(2, 2, 2, 2));
  const __m128 i128_0 = _mm_setr_ps(cosf(angle), 0.0f,0.0f, 0.0f);
  const __m128 i128_1 = _mm_setr_ps(0.0f, cosf(angle),0.0f, 0.0f);
  const __m128 i128_2 = _mm_setr_ps(0.0f, 0.0f, cosf(angle),0.0f);
  const __m128 j128_0 = _mm_mul_ps(_mm_set1_ps(sinf(angle)), _mm_setr_ps(    0.0f, -axis[2],  axis[1],0.0f));
  const __m128 j128_1 = _mm_mul_ps(_mm_set1_ps(sinf(angle)), _mm_setr_ps( axis[2],     0.0f, -axis[0],0.0f));
  const __m128 j128_2 = _mm_mul_ps(_mm_set1_ps(sinf(angle)), _mm_setr_ps(-axis[1],  axis[0],     0.0f,0.0f));
  const __m128 k128_0 = _mm_mul_ps(_mm_set1_ps(1.0f - cosf(angle)), _mm_mul_ps(d128_0, d128));
  const __m128 k128_1 = _mm_mul_ps(_mm_set1_ps(1.0f - cosf(angle)), _mm_mul_ps(d128_1, d128));
  const __m128 k128_2 = _mm_mul_ps(_mm_set1_ps(1.0f - cosf(angle)), _mm_mul_ps(d128_2, d128));
  const __m128 result_0 = _mm_add_ps(_mm_add_ps(i128_0, j128_0), k128_0);
  const __m128 result_1 = _mm_add_ps(_mm_add_ps(i128_1, j128_1), k128_1);
  const __m128 result_2 = _mm_add_ps(_mm_add_ps(i128_2, j128_2), k128_2);
  const __m128 result_3 = _mm_setr_ps(0, 0, 0, 1);
  _mm_store_ps(M[0], result_0);
  _mm_store_ps(M[1], result_1);
  _mm_store_ps(M[2], result_2);
  _mm_store_ps(M[3], result_3);
}

inline void matFromAffineReflect(FMat4 M, const FVec4 /* treat as FVec3 */ axis) {
  __m128 d128 = _mm_sub_ps(_mm_load_ps(axis), _mm_set_ps(axis[3], 0, 0, 0));
  __m128 s128 = _mm_mul_ps(d128, d128);
  s128 = _mm_hadd_ps(_mm_hadd_ps(s128, _mm_setzero_ps()), _mm_setzero_ps());
  s128 = _mm_shuffle_ps(s128, s128, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 d128_0 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 d128_1 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 d128_2 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(2, 2, 2, 2));
  d128_0 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_0, d128));
  d128_1 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_1, d128));
  d128_2 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_2, d128));
  d128_0 = _mm_sub_ps(_mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f), _mm_div_ps(d128_0, s128));
  d128_1 = _mm_sub_ps(_mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f), _mm_div_ps(d128_1, s128));
  d128_2 = _mm_sub_ps(_mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f), _mm_div_ps(d128_2, s128));
  _mm_store_ps(M[0], d128_0);
  _mm_store_ps(M[1], d128_1);
  _mm_store_ps(M[2], d128_2);
  _mm_store_ps(M[3], _mm_setr_ps(0, 0, 0, 1));
}

inline void matFromAffineFlip(FMat4 M, const FVec4 /* treat as FVec3 */ axis) {
  __m128 d128 = _mm_sub_ps(_mm_load_ps(axis), _mm_set_ps(axis[3], 0, 0, 0));
  __m128 s128 = _mm_mul_ps(d128, d128);
  s128 = _mm_hadd_ps(_mm_hadd_ps(s128, _mm_setzero_ps()), _mm_setzero_ps());
  s128 = _mm_shuffle_ps(s128, s128, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 d128_0 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(0, 0, 0, 0));
  __m128 d128_1 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(1, 1, 1, 1));
  __m128 d128_2 = _mm_shuffle_ps(d128, d128, _MM_SHUFFLE(2, 2, 2, 2));
  d128_0 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_0, d128));
  d128_1 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_1, d128));
  d128_2 = _mm_mul_ps(_mm_set1_ps(2.0f), _mm_mul_ps(d128_2, d128));
  d128_0 = _mm_sub_ps(_mm_div_ps(d128_0, s128), _mm_setr_ps(1.0f, 0.0f, 0.0f, 0.0f));
  d128_1 = _mm_sub_ps(_mm_div_ps(d128_1, s128), _mm_setr_ps(0.0f, 1.0f, 0.0f, 0.0f));
  d128_2 = _mm_sub_ps(_mm_div_ps(d128_2, s128), _mm_setr_ps(0.0f, 0.0f, 1.0f, 0.0f));
  _mm_store_ps(M[0], d128_0);
  _mm_store_ps(M[1], d128_1);
  _mm_store_ps(M[2], d128_2);
  _mm_store_ps(M[3], _mm_setr_ps(0, 0, 0, 1));
}

inline void matFromLookAt(FMat4 M, const FAffPoint4 eye, const FVec4 look, const FVec4 up) {
  constexpr int r1 = _MM_SHUFFLE(3, 0, 2, 1);
  constexpr int r2 = _MM_SHUFFLE(3, 1, 0, 2);
  __m128 e128 = _mm_div_ps(_mm_load_ps(eye), _mm_set1_ps(eye[3]));
  __m128 l128 = _mm_sub_ps(_mm_load_ps(look), _mm_set_ps(look[3], 0, 0, 0));
  __m128 u128 = _mm_sub_ps(_mm_load_ps(up), _mm_set_ps(up[3], 0, 0, 0));
  __m128 tmp0 = _mm_shuffle_ps(u128, u128, r1);
  __m128 tmp1 = _mm_shuffle_ps(l128, l128, r2);
  __m128 tmp2 = _mm_shuffle_ps(u128, u128, r2);
  __m128 tmp3 = _mm_shuffle_ps(l128, l128, r1);
  __m128 r128 = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
  tmp0 = _mm_shuffle_ps(l128, l128, r1);
  tmp1 = _mm_shuffle_ps(r128, r128, r2);
  tmp2 = _mm_shuffle_ps(l128, l128, r2);
  tmp3 = _mm_shuffle_ps(r128, r128, r1);
  u128 = _mm_sub_ps(_mm_mul_ps(tmp0, tmp1), _mm_mul_ps(tmp2, tmp3));
  __m128 s128 = _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(l128, l128), _mm_setzero_ps()), _mm_setzero_ps());
  __m128 t128 = _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(u128, u128), _mm_setzero_ps()), _mm_setzero_ps());
  __m128 w128 = _mm_hadd_ps(_mm_hadd_ps(_mm_mul_ps(r128, r128), _mm_setzero_ps()), _mm_setzero_ps());
  l128 = _mm_div_ps(l128, _mm_sqrt_ps(_mm_shuffle_ps(s128, s128, _MM_SHUFFLE(0, 0, 0, 0))));
  u128 = _mm_div_ps(u128, _mm_sqrt_ps(_mm_shuffle_ps(t128, t128, _MM_SHUFFLE(0, 0, 0, 0))));
  r128 = _mm_div_ps(r128, _mm_sqrt_ps(_mm_shuffle_ps(w128, w128, _MM_SHUFFLE(0, 0, 0, 0))));
  e128 = _mm_mul_ps(e128, _mm_setr_ps(-1.0f, -1.0f, -1.0f, 1));
  tmp0 = _mm_mul_ps(e128, r128);
  tmp1 = _mm_mul_ps(e128, u128);
  tmp2 = _mm_mul_ps(e128, l128);
  tmp3 = _mm_setr_ps(0, 0, 0, 1);
  e128 = _mm_hadd_ps(_mm_hadd_ps(tmp0, tmp1), _mm_hadd_ps(tmp2, tmp3));
  r128 = _mm_insert_ps(r128, e128, ((0 << 6) | (3 << 4) | 0));
  u128 = _mm_insert_ps(u128, e128, ((1 << 6) | (3 << 4) | 0));
  l128 = _mm_insert_ps(l128, e128, ((2 << 6) | (3 << 4) | 0));
  _mm_store_ps(M[0], r128);
  _mm_store_ps(M[1], u128);
  _mm_store_ps(M[2], l128);
  _mm_store_ps(M[3], _mm_setr_ps(0, 0, 0, 1));
}


inline void matFromOrthoProjection(FMat4 M, const float a[2], const float b[2], const float c[2]) {
  float s1 = a[0] + a[1], d1 = a[1] - a[0];
  float s2 = b[0] + b[1], d2 = b[1] - b[0];
  float s3 = c[0] + c[1], d3 = c[1] - c[0];
  __m128 T_0 = _mm_setr_ps(2 / d1, 0, 0, -s1 / d1);
  __m128 T_1 = _mm_setr_ps(0, 2 / d2, 0, -s2 / d2);
  __m128 T_2 = _mm_setr_ps(0, 0, 2 / d3, -s3 / d3);
  __m128 T_3 = _mm_setr_ps(0, 0, 0, 1);
  _mm_store_ps(M[0], T_0);
  _mm_store_ps(M[1], T_1);
  _mm_store_ps(M[2], T_2);
  _mm_store_ps(M[3], T_3);
}

inline void matFromPersProjection(FMat4 M, const float a[2], const float b[2], const float c[2]) {
  float s1 = a[0] + a[1], d1 = a[1] - a[0];
  float s2 = b[0] + b[1], d2 = b[1] - b[0];
  float s3 = c[0] + c[1], d3 = c[1] - c[0];
  float r  =c[0] * c[1];
  __m128 T_0 = _mm_setr_ps(2 * c[0] / d1, 0, -s1 / d1, 0);
  __m128 T_1 = _mm_setr_ps(0, 2 * c[0] / d2, -s2 / d2, 0);
  __m128 T_2 = _mm_setr_ps(0, 0,  - s3 / d3,  2 * r / d3);
  __m128 T_3 = _mm_setr_ps(0, 0, 1, 0);
  _mm_store_ps(M[0], T_0);
  _mm_store_ps(M[1], T_1);
  _mm_store_ps(M[2], T_2);
  _mm_store_ps(M[3], T_3);
}


inline void matAffineScale(FMat4 M, const FVec4 /* treat as FVec3 */ rate) {
  const FVec4 _rate = {rate[0], rate[1], rate[2], rate[3] ? rate[3] : 1.0f};
  floatMatScaleVec(M, _rate);
}

inline void matAffineShear(FMat4 M, const FVec4 /* treat as FVec3 */ she) {
  FMat4 T = {};
  matFromAffineShear(T, she);
  floatMatLMulMat(T, M);
}

inline void matAffineShift(FMat4 M, const FVec4 /* treat as FVec3 */ dis) {
  FMat4 T = {};
  matFromAffineShift(T, dis);
  floatMatLMulMat(T, M);
}

inline void matAffineRotate(FMat4 M, const FVec4 /* treat as FVec3 */ axis, const float angle) {
  FMat4 T = {};
  matFromAffineRotate(T, axis, angle);
  floatMatLMulMat(T, M);
}

inline void matAffineReflect(FMat4 M, const FVec4 /* treat as FVec3 */ axis) {
  FMat4 T = {};
  matFromAffineReflect(T, axis);
  floatMatLMulMat(T, M);
}

inline void matAffineFlip(FMat4 M, const FVec4 /* treat as FVec3 */ axis) {
  FMat4 T = {};
  matFromAffineFlip(T, axis);
  floatMatLMulMat(T, M);
}
