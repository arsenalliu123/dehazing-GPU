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

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace cv;


//Type of Min and Max value
typedef struct _MinMax
{
	double min;
	double max;
}MinMax;

typedef struct _Image
{
	float blue;
	float red;
	float green;
}Image;

typedef struct _TransImage
{
	float grey;
}TransImage;

void gpu_func(
		gpu::DevMem2Df mat,
		gpu::DevMem2Df trans_mat,
		Vec<float, 3> airlight,
		gpu::DevMem2Df dest,
		int _PriorSize,
		int height,
		int width,
		int t0);


#endif /* DEHAZING_H_ */


