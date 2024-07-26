/**
 * Project Name: vector4
 * Module Name: test
 * Filename: main.c
 * Creator: Yaokai Liu
 * Create Date: 2024-07-21
 * Copyright (c) 2024 Yaokai Liu. All rights reserved.
 **/

#include <stdio.h>
#include "vector4.h"

int main() {
    IVector vecA VEC_ALIGN = {1, 2, 3, 4};
    IVector vecB VEC_ALIGN = {1, 2, 3, 4};
//    intVecAdd(vecA, vecB);
//    intVecSub(vecA, vecB);
//    intVecScale(vecA, 2);
//    intVecCoverMul(vecA, vecB);
//    intVecRoverMul(vecA, vecB);
//    int p = intVecInnerProduct(vecA, vecB);
    int p = intVecConvProduct(vecA, vecB);
//    printf("{%u, %u, %u, %u}\n", vecA[0], vecA[1], vecA[2], vecA[3]);
    printf("%d\n", p);
    FVector vecC VEC_ALIGN = {1.4f, 2.3f, 3.2f, 4.1f};
    FVector vecD VEC_ALIGN = {1.4f, 2.3f, 3.2f, 4.1f};
//    floatVecAdd(vecC, vecD);
//    floatVecSub(vecC, vecD);
//    floatVecScale(vecC, 3.0f);
//    floatVecCoverMul(vecC, vecD);
//    floatVecRoverMul(vecC, vecD);
//    float q = floatVecInnerProduct(vecC, vecD);
    float q = floatVecConvProduct(vecC, vecD);
//    printf("{%g, %g, %g, %g}\n", vecC[0], vecC[1], vecC[2], vecC[3]);
    printf("%g\n", q);
    return 0;
}
