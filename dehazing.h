/*
 * dehazing.h
 *
 *  Created on: Apr 8, 2015
 *      Author: river
 */

#ifndef DEHAZING_H_
#define DEHAZING_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CEIL(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))

//dark channel prior
void dark_channel(
		float *image,
		float *img_grey,
		float *dark_channel,
		int height,
		int width,
		dim3 blocks,
		dim3 grids
		);

//air light (RGB of maximum dark prior channle pixel)
void air_light(
		float *image,
		float *dark,
		int height,
		int width,
		dim3 blocks,
		dim3 grids
		);

void dehaze(
	float *image,
	float *dark,
	float *t,
	int height,
	int width,
	dim3 blocks,
	dim3 grids
	);

void transmission(
	float *image,
	float *t,
	int height,
	int width,
	dim3 blocks,
	dim3 grids
	);

void gfilter(
	float *filter,
	float *img_gray,
	float *trans,
	int height,
	int width,
	dim3 blocks,
	dim3 grids
	);//filter: guided imaging filter result

#endif /* DEHAZING_H_ */


