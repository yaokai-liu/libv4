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

void inline intVecAdd(IVector summand, const IVector addend) {
    __m128i a, b, c;
    a = _mm_load_si128((const __m128i *) summand);
    b = _mm_load_si128((const __m128i *) addend);
    c = _mm_add_epi32(a, b);
    _mm_store_si128((__m128i *) summand, c);
}

void inline intVecSub(IVector minuend, const IVector subtrahend) {
    __m128i a, b, c;
    a = _mm_load_si128((const __m128i *) minuend);
    b = _mm_load_si128((const __m128i *) subtrahend);
    c = _mm_sub_epi32(a, b);
    _mm_store_si128((__m128i *) minuend, c);
}

void inline intVecScale(IVector vector, int32_t rate) {
    __m128i a, b, c;
    a = _mm_load_si128((const __m128i *) vector);
    b = _mm_set1_epi32(rate);
    c = _mm_mullo_epi32(a, b);
    _mm_store_si128((__m128i *) vector, c);
}

void inline intVecCoverMul(IVector multiplier, const IVector multiplicand) {
    __m128i a, b, c;
    a = _mm_load_si128((const __m128i *) multiplier);
    b = _mm_load_si128((const __m128i *) multiplicand);
    c = _mm_mullo_epi32(a, b);
    _mm_store_si128((__m128i *) multiplier, c);
}

void inline intVecRoverMul(IVector multiplier, const IVector multiplicand) {
    __m128i a, b, c;
    a = (__m128i) _mm_load_ps((const float *) multiplier);
    b = (__m128i) _mm_loadr_ps((const float *) multiplicand);
    c = _mm_mullo_epi32(a, b);
    _mm_store_si128((__m128i *) multiplier, c);
}

int32_t intVecInnerProduct(const IVector multiplier, const IVector multiplicand) {
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

int32_t intVecConvProduct(const IVector multiplier, const IVector multiplicand) {
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


void inline floatVecAdd(FVector summand, const FVector addend) {
    __m128 a, b, c;
    a = _mm_load_ps(summand);
    b = _mm_load_ps( addend);
    c = _mm_add_ps(a, b);
    _mm_store_ps(summand, c);
}

void inline floatVecSub(FVector minuend, const FVector subtrahend) {
    __m128 a, b, c;
    a = _mm_load_ps(minuend);
    b = _mm_load_ps(subtrahend);
    c = _mm_sub_ps(a, b);
    _mm_store_ps(minuend, c);
}

void inline floatVecScale(FVector vector, float rate) {
    __m128 a, b, c;
    a = _mm_load_ps(vector);
    b = _mm_set1_ps(rate);
    c = _mm_mul_ps(a, b);
    _mm_store_ps(vector, c);
}

void inline floatVecCoverMul(FVector multiplier, const FVector multiplicand) {
    __m128 a, b, c;
    a = _mm_load_ps(multiplier);
    b = _mm_load_ps(multiplicand);
    c = _mm_mul_ps(a, b);
    _mm_store_ps(multiplier, c);
}

void inline floatVecRoverMul(FVector multiplier, const FVector multiplicand) {
    __m128 a, b, c;
    a = _mm_load_ps(multiplier);
    b = _mm_loadr_ps(multiplicand);
    c = _mm_mul_ps(a, b);
    _mm_store_ps(multiplier, c);
}

float floatVecInnerProduct(const FVector multiplier, const FVector multiplicand) {
    __m128 a, b, c, d;
    a = _mm_load_ps(multiplier);
    b = _mm_load_ps(multiplicand);
    c = _mm_mul_ps(a, b);
    d = _mm_setzero_ps();
    c = _mm_hadd_ps(c, d);
    c = _mm_hadd_ps(c, d);
    return _mm_cvtss_f32(c);
}

float floatVecConvProduct(const FVector multiplier, const FVector multiplicand) {
    __m128 a, b, c, d;
    a = _mm_load_ps(multiplier);
    b = _mm_loadr_ps(multiplicand);
    c = _mm_mul_ps(a, b);
    d = _mm_setzero_ps();
    c = _mm_hadd_ps(c, d);
    c = _mm_hadd_ps(c, d);
    return _mm_cvtss_f32(c);
}

void floatVec3CrossMul(FVector multiplier, const FVector multiplicand) {
    __m128 a, b, c, c1, c2, c3, c4;
    int r1 = _MM_SHUFFLE(2, 3, 1, 0);
    int r2 = _MM_SHUFFLE(3, 1, 2, 0);
    a = _mm_load_ps(multiplier);
    b = _mm_load_ps(multiplicand);
    c1 = _mm_shuffle_ps(a, a, r1);
    c2 = _mm_shuffle_ps(b, b, r2);
    c3 = _mm_shuffle_ps(a, a, r2);
    c4 = _mm_shuffle_ps(b, b, r1);
    c1 = _mm_mul_ps(c1, c2);
    c3 = _mm_mul_ps(c3, c4);
    c = _mm_sub_ps(c1, c3);
    _mm_store_ps(multiplier, c);
}
