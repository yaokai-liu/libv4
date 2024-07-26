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

#include <stdint.h>

typedef int32_t IVector[4];
typedef float FVector[4];

#define VEC_ALIGN  [[gnu::aligned(16)]]

void intVecAdd(IVector summand, const IVector addend);
void intVecSub(IVector minuend, const IVector subtrahend);
void intVecScale(IVector vector, int32_t rate);
void intVecCoverMul(IVector multiplier, const IVector multiplicand);
void intVecRoverMul(IVector multiplier, const IVector multiplicand);
int32_t intVecInnerProduct(const IVector multiplier, const IVector multiplicand);
int32_t intVecConvProduct(const IVector multiplier, const IVector multiplicand);


void floatVecAdd(FVector summand, const FVector addend);
void floatVecSub(FVector minuend, const FVector subtrahend);
void floatVecScale(FVector vector, float rate);
void floatVecCoverMul(FVector multiplier, const FVector multiplicand);
void floatVecRoverMul(FVector multiplier, const FVector multiplicand);
float floatVecInnerProduct(const FVector multiplier, const FVector multiplicand);
float floatVecConvProduct(const FVector multiplier, const FVector multiplicand);
void floatVec3CrossMul(FVector multiplier, const FVector multiplicand);

#endif //VECTOR4_VECTOR4_H

