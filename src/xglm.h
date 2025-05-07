/**
 * Project Name: xglm
 * Module Name: 
 * Filename: xglm.h
 * Creator: Yaokai Liu
 * Create Date: 2025-05-03
 * Copyright (c) 2025 Yaokai Liu. All rights reserved.
 **/

#ifndef XGL_MATH_XGLM_H
#define XGL_MATH_XGLM_H

#include <stdint.h>
/// Vector that has 2 integer elements
typedef int32_t IVec2[2];
/// Vector that has 2 float number elements
typedef float FVec2[2];
/// 2x2 Matrix that elements are integer
typedef int32_t IMat2[2];
/// 2x2 Matrix that elements are float point number
typedef float FMat2[2];
/// Vector that has 4 integer elements
typedef int32_t IVec4[4];
/// Vector that has 4 float number elements
typedef float FVec4[4];
/// 4x4 Matrix that elements are integer
typedef int32_t IMat4[4][4];
/// 4x4 Matrix that elements are float point number
typedef float FMat4[4][4];
/// Affine Coordinate that all components are integer
typedef int32_t IAffPoint4[4];
/// Affine Coordinate that all components are float point number
typedef float   FAffPoint4[4];


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


/// a = transpose(a * B)
void intVecMulMat(IVec4 a, const IMat4 B);

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

/// a = transpose(a * B)
void floatVecMulMat(FVec4 a, const FMat4 B);


/// A = diag(b[0], b[1], b[2], b[3])
void intMatDiag(IMat4 A, const IVec4 b);
/// A[i,j] = A[i,j] + B[i,j]
void intMatAdd(IMat4 A, const IMat4 B);
/// A[i,j] = A[i,j] - B[i,j]
void intMatSub(IMat4 A, const IMat4 B);
/// A[i,j] = A[i,j] * b
void intMatScale(IMat4 A, int32_t b);
/// A[i,j] = A[i,j] * b[i]
void intMatScaleVec(IMat4 A, const IVec4 b);
/// A[i,j] = A[i,j] * B[i,j]
void intMatScaleMat(IMat4 A, const IMat4 B);
/// b = transpose(A * b)
void intMatMulVec(const IMat4 A, IVec4 b);
/// B = A * B
void intMatLMulMat(const IMat4 A, IMat4 B);
/// A = A * B
void intMatRMulMat(IMat4 A, const IMat4 B);
/// A = transpose(A)
void intMatTrCopy(IMat4 M_T, const IMat4 A);

/// A = diag(b[0], b[1], b[2], b[3])
void floatMatDiag(FMat4 A, const FVec4 b);
/// A[i,j] = A[i,j] + B[i,j]
void floatMatAdd(FMat4 A, const FMat4 B);
/// A[i,j] = A[i,j] - B[i,j]
void floatMatSub(FMat4 A, const FMat4 B);
/// A[i,j] = A[i,j] * b
void floatMatScale(FMat4 A, float b);
/// A[i,j] = A[i,j] * b[i]
void floatMatScaleVec(FMat4 A, const FVec4 b);
/// A[i,j] = A[i,j] * B[i,j]
void floatMatScaleMat(FMat4 A, const FMat4 B);
/// b = transpose(A * b)
void floatMatMulVec(const FMat4 A, FVec4 b);
/// B = A * B
void floatMatLMulMat(const FMat4 A, FMat4 B);
/// A = A * B
void floatMatRMulMat(FMat4 A, const FMat4 B);
/// M = transpose(M)
void floatMatTrCopy(FMat4 M_T, const FMat4 M);

/// Unitize:
/// @formula P = (P[0]/P[3], P[1]/P[3], P[2]/P[3], 1)
void floatAffineNormalize(FAffPoint4 P);
/// @formula d = A - B = BA
void floatAffineSub(const FAffPoint4 A, const FAffPoint4 B, FVec4 d);
/**
 * Reflect:
 * @description The mirror point of P determined by the plane passing through O and orthogonal to d
 * @formula P = P - 2(OP·d)/sqrt(d·d) * d
 */
void floatAffineReflect(FAffPoint4 P, const FAffPoint4 O, const FVec4 d);
/**
 * Reflect:
 * @description The flip point of P with respect to O about d
 * @formula P = O + 2(OP·d)/sqrt(d·d) * d - OP
 */
void floatAffineFlip(FAffPoint4 P, const FAffPoint4 O, const FVec4  /* treat as FVec3 */ d);


/// make M be the scale effect
void matFromAffineScale(FMat4 M, const FVec4 /* treat as FVec3 */ rate);
/// make M be the shear effect
void matFromAffineShear(FMat4 M, const FVec4 /* treat as FVec3 */ she);
/// make M be the shift effect
void matFromAffineShift(FMat4 M, const FVec4 /* treat as FVec3 */ dis);
/// make M be the rotate effect
void matFromAffineRotate(FMat4 M, const FVec4 /* treat as FVec3 */ axis, float angle);
/// make M be the reflect effect
void matFromAffineReflect(FMat4 M, const FVec4 /* treat as FVec3 */ axis);
/// make M be the flip effect
void matFromAffineFlip(FMat4 M, const FVec4 /* treat as FVec3 */ axis);


/**
 * make M be a view transformation
 * @param M where the output store
 * @param eye the position the camera at
 * @param look target direction the camera look at
 * @param up up direction of the camera, which must be orthogonal with the param `look`
 */
void matFromLookAt(FMat4 M, const FAffPoint4 eye, const FVec4 look, const FVec4 up);
/**
 * make M be a orthographic projection matrix
 * @param M where the output store
 * @param a box [a1, a2] of axis x, requires a1 < a2
 * @param b box [b1, b2] of axis y, requires b1 < b2
 * @param c box [c1, c2] of axis z, requires c1 < c2
 */
void matFromOrthoProjection(FMat4 M, const float a[2], const float b[2], const float c[2]);
/**
 * make M be a perspective projection matrix
 * @param M where the output store
 * @param a box [a1, a2] of axis x
 * @param b box [b1, b2] of axis y
 * @param c box [c1, c2] of axis z
 */
void matFromPersProjection(FMat4 M, const float a[2], const float b[2], const float c[2]);


/// left multiply a scale effect to the M
void matAffineScale(FMat4 M, const FVec4 /* treat as FVec3 */ rate);
/// left multiply a shear effect to the M
void matAffineShear(FMat4 M, const FVec4 /* treat as FVec3 */ she);
/// left multiply a rotate effect to the M
void matAffineRotate(FMat4 M, const FVec4 /* treat as FVec3 */ axis, float angle);
/// left multiply a shift effect to the M
void matAffineShift(FMat4 M, const FVec4 /* treat as FVec3 */ dis);
/// left multiply a reflect effect to the M
void matAffineReflect(FMat4 M, const FVec4 /* treat as FVec3 */ axis);
/// left multiply a flip effect to the M
void matAffineFlip(FMat4 M, const FVec4 /* treat as FVec3 */ axis);


#endif //XGL_MATH_XGLM_H
