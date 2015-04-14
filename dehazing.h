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


void dark_channel(
		float *image,
		float *dark_channel,
		int height,
		int width,
		dim3 blocks,
		dim3 grids
		);

void air_light(
		float *image,
		float *dark,
		int size,
		dim3 blocks,
		dim3 grids
		);

#endif /* DEHAZING_H_ */


