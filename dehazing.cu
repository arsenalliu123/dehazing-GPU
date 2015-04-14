#include "dehazing.h"
#include "stdio.h"




//convenient macros
#define IN_GRAPH(x,y,h,w) ((x>=0)&&(x<h)&&(y>=0)&&(y<w))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)

/*
 * dark_channel host wrapper and kernel
 */
__global__
void dark_channel_kernel(float3 *image, float *dark, int height, int width){
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) * width + blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ unsigned int min_value[1];
	min_value[0] = 255;
	if(i < height * width){
		unsigned int val = 255 * min(image[i].x, min(image[i].y, image[i].z));
		atomicMin(min_value, val);
	}
	__syncthreads();
	if(i < height * width){
		dark[i] = (*min_value)/255.f;
	}

}


void dark_channel(float *image,float *dark_channel,int height, int width, dim3 blocks,dim3 grids){
	dark_channel_kernel<<<grids, blocks>>> ((float3 *)image, dark_channel, height, width);
}

/*
 * air_light host wrapper and kernel
 */


__global__
void dehazing_img_kernel1(float3 *image, float *dark, int height, int width, float3 *int_image, float *int_dark){
	const int b_r = (width-1)/15+1;
	const int b_n = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = b_n / b_r * 15 * width + b_n % b_r * 15;
	//printf("%d %d %d %d\n", b_n, i, threadIdx.x , width*height);
	extern __shared__ float3 tmp_image[];
	float *tmp_dark = (float *)(tmp_image + blockDim.x);
	if(i < width * height){
		tmp_image[threadIdx.x] = image[i];
		tmp_dark[threadIdx.x] = dark[i];
		__syncthreads();
		for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
			if(threadIdx.x < stride){
				if(tmp_dark[threadIdx.x + stride] > tmp_dark[threadIdx.x]){
					tmp_dark[threadIdx.x] = tmp_dark[threadIdx.x + stride];
					tmp_image[threadIdx.x] = tmp_image[threadIdx.x + stride];
				}
			}
			__syncthreads();
		}
		if(threadIdx.x == 0){
			int_image[blockIdx.x] = tmp_image[threadIdx.x];
			int_dark[blockIdx.x] = tmp_dark[threadIdx.x];
		}
	}
}

__global__
void dehazing_img_kernel2(float3 *image, int size, float3 *int_image, float *int_dark){
	
	extern __shared__ float3 tmp_image[];
	float *tmp_dark = (float *)(tmp_image + blockDim.x);
	tmp_image[threadIdx.x] = int_image[threadIdx.x];
	tmp_dark[threadIdx.x] = int_dark[threadIdx.x];
	__syncthreads();
	for(unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){
			if(tmp_dark[threadIdx.x + stride] > tmp_dark[threadIdx.x]){
				tmp_dark[threadIdx.x] = tmp_dark[threadIdx.x + stride];
				tmp_image[threadIdx.x] = tmp_image[threadIdx.x + stride];
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0){
		image[size] = tmp_image[threadIdx.x];
	}
}

void air_light(float *image, float *dark, int height, int width, dim3 blocks, dim3 grids){
	float3 *int_image = NULL;
	float *int_dark = NULL;
	//printf("%d\n", grids.x);
	cudaMalloc((void **)(&int_image), sizeof(float3)*grids.x);
	cudaMalloc((void **)(&int_dark), sizeof(float)*grids.x);
	float *xx = (float *)malloc(sizeof(float)*height*width);
	CUDA_CHECK_RETURN(cudaMemcpy(xx, dark, height * width * sizeof(float), cudaMemcpyDeviceToHost));
	//for(int i=0;i<height*width;i++){printf("%.2f ", xx[i]);}
	int shared_size_1 = blocks.x*(sizeof(float3)+sizeof(float));
	int shared_size_2 = grids.x*(sizeof(float3)+sizeof(float));
	dehazing_img_kernel1<<<grids, blocks, shared_size_1>>> ((float3 *)image, dark, height, width, int_image, int_dark);
	dehazing_img_kernel2<<<1, grids, shared_size_2>>> ((float3 *)image, height*width, int_image, int_dark);

}

//Calculate Dark Channel
//J^{dark}(x)=min( min( J^c(y) ) )
/*
__global__
void DarkChannel(Image *image, int height, int width, int erosion_width)
{

	GpuMat dark=GpuMat::zeros(img.rows,img.cols,CV_32FC1);
	GpuMat dark_out=GpuMat::zeros(img.rows,img.cols,CV_32FC1);
	for(int i=0;i<img.rows;i++)
	{
		for(int j=0;j<img.cols;j++)
		{
			dark.at<float>(i,j)=min(min(img.at<Vec<float,3>>(i,j)[0],img.at<Vec<float,3>>(i,j)[1]),min(img.at<Vec<float,3>>(i,j)[0],img.at<Vec<float,3>>(i,j)[2]));
		}
	}
	erode(dark,dark_out,Mat::ones(_PriorSize,_PriorSize,CV_32FC1));
	return dark_out;


	__shared__ float buffer[];

	int startx = blockIdx.x * blockDim.x;
	int starty = blockIdx.y * blockDim.y;

	int x = startx + threadIdx.x;
	int y = starty + threadIdx.y;


	int tid = x*height+y;
	int above_tid = tid-7*width;
	if(threadIdx.x < 7){
		if(IN_GRAPH(x-7,y,height,width)){
			buffer[threadIdx.x*(blockDim.y+14)+(threadIdx.y+7)] =
					min(min(image[above_tid].green, image[above_tid].red),image[above_id].blue);
		}
		else{
			buffer[threadIdx.x*(blockDim.y+14)+(threadIdx.y+7)] = 0;
		}
	}
	if(threadIdx.x > blockDim.x - 8){

	}

	if(IN_GRAPH(x,y,height,width)){
		buffer[(threadIdx.x+7)*(blockDim.y+14)+(threadIdx.y+7)] =
				min(min(image[tid].green, image[tid].red),image[tid].blue);
	}


}
*/

