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
#include "matrix4.h"

int main() {
  IVec4 vecA = {1, 2, 3, 4};
  IVec4 vecB = {1, 2, 3, 4};
//  intVecAdd(vecA, vecB);
//  intVecSub(vecA, vecB);
//  intVecScale(vecA, 2);
//  intVecCoverMul(vecA, vecB);
//  intVecRoverMul(vecA, vecB);
//  int p = intVecInnerProduct(vecA, vecB);
  int p = intVecConvProduct(vecA, vecB);
//  printf("{%u, %u, %u, %u}\n", vecA[0], vecA[1], vecA[2], vecA[3]);
  printf("%d\n", p);
  FVec4 vecC = {1.0f, 2.0f, 3.0f, 0.0f};
  FVec4 vecD = {3.0f, 7.0f, 1.0f, 0.0f};
//  floatVecAdd(vecC, vecD);
//  floatVecSub(vecC, vecD);
//  floatVecScale(vecC, 3.0f);
//  floatVecCoverMul(vecC, vecD);
//  floatVecRoverMul(vecC, vecD);
  floatVec3CrossMul(vecC, vecD);
//  float q = floatVecInnerProduct(vecC, vecD);
  float q = floatVecConvProduct(vecC, vecD);
  printf("{%g, %g, %g, %g}\n", vecC[0], vecC[1], vecC[2], vecC[3]);
  printf("%g\n", q);
//  FMat4 mat1 = {{ 1.0f, 2.0f, 3.0f, 4.0f },
//          { 1.0f, 2.0f, 3.0f, 4.0f },
//          { 1.0f, 2.0f, 3.0f, 4.0f },
//          { 1.0f, 2.0f, 3.0f, 4.0f }};
  FMat4 mat2 = {{ 5.0f, 6.0f, 7.0f, 8.0f },
                { 5.0f, 6.0f, 7.0f, 8.0f },
                { 5.0f, 6.0f, 7.0f, 8.0f },
                { 5.0f, 6.0f, 7.0f, 8.0f }};
  IMat4 mat3 = {{ 11, 12, 13, 14 },
                { 21, 22, 23, 24 },
                { 31, 32, 33, 34 },
                { 41, 42, 43, 44 }};
//  floatMatRMulMat(mat1, mat2);
//  for (uint32_t i = 0; i < 4; i ++) {
//    printf("%g, %g, %g, %g\n", mat1[i][0], mat1[i][1], mat1[i][2], mat1[i][3]);
//  }
  intMatTrCopy(mat3, mat3);
  for (uint32_t i = 0; i < 4; i ++) {
    printf("%d, %d, %d, %d\n", mat3[i][0], mat3[i][1], mat3[i][2], mat3[i][3]);
  }
  floatMatTrCopy(mat2, mat2);
  for (uint32_t i = 0; i < 4; i ++) {
  printf("%g, %g, %g, %g\n", mat2[i][0], mat2[i][1], mat2[i][2], mat2[i][3]);
  }
  return 0;
}
