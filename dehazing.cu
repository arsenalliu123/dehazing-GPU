#include "dehazing.h"
#include "stdio.h"




//convenient macros
#define IN_GRAPH(x,y,h,w) ((x>=0)&&(x<h)&&(y>=0)&&(y<w))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)

/*
 * dark_channel host wrapper and kernel
 */
//first kernel calculate min of RGB
__global__
void dark_channel_kernel(float3 *image, float *dark, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		dark[i] = min(image[i].x, min(image[i].y, image[i].z));
	}
}

//second kernel calculate min of 15 X 15 ceil
__global__
void prior_kernel(float *dark, int height, int width, int window){
	extern __shared__ float buffer[];
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		const int si = (threadIdx.x + window) * (blockDim.y + window * 2) + threadIdx.y + window;
		buffer[si] = dark[i];
		if(threadIdx.x < window && IN_GRAPH(x-window, y, height, width) ){
			buffer[si - (blockDim.y + window * 2) * window] = dark[i - window * width];
			if(threadIdx.y < window &&
				IN_GRAPH(x-window, y-window, height, width) ){
				buffer[si - (blockDim.y + window * 2) * window - window]
			       = dark[i - window * width - window];
			}
			if(threadIdx.y >= blockDim.y - window &&
				IN_GRAPH(x-window, y+window, height, width) ){
				buffer[si - (blockDim.y + window * 2) * window + window]
			       = dark[i - window * width + window];
			}
		}
		if(threadIdx.x >= blockDim.x - window && IN_GRAPH(x+window, y, height, width) ){
			buffer[si + (blockDim.y + window * 2) * window] = dark[i + window * width];
			if(threadIdx.y >= blockDim.y - window &&
				IN_GRAPH(x+window, y+window, height, width) ){
					buffer[si + (blockDim.y + window * 2) * window + window]
					       = dark[i + window * width + window];
			}
			if(threadIdx.y < window &&
				IN_GRAPH(x+window, y-window, height, width) ){
					buffer[si + (blockDim.y + window * 2) * window - window]
					       = dark[i + window * width - window];
			}

		}
		if(threadIdx.y >= blockDim.y - window && IN_GRAPH(x, y+window, height, width) ){
			buffer[si + window] = dark[i + window];
		}
		if(threadIdx.y < window && IN_GRAPH(x, y-window, height, width) ){
			buffer[si - window] = dark[i - window];
		}

		__syncthreads();
		
		float minval = 1.0;
		for(int startx = 0; startx < window * 2 + 1; startx++){
			for(int starty = 0; starty < window * 2 + 1; starty++){
				if(IN_GRAPH(x-window+startx, y-window+starty, height, width)){
				minval = min(
						buffer[
						       (threadIdx.x+startx)*
						       (blockDim.y + window * 2) +
						       threadIdx.y + starty], minval);
				}
			}
		}
		dark[i] = minval;

	}
}

void dark_channel(float *image,float *dark_channel,int height, int width, dim3 blocks,dim3 grids){
	dark_channel_kernel<<<grids, blocks>>> ((float3 *)image, dark_channel, height, width);
	int window = 7;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	prior_kernel<<<grids, blocks, shared_size>>>(dark_channel, height, width, window);
}

/*
 * air_light host wrapper and kernel
 */

//first kernel reduce to < 1024 values for next kernel
__global__
void dehazing_img_kernel1(
		float3 *image, float *dark,
		int height, int width,
		float3 *int_image, float *int_dark){
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
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

//calculate air light
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
	//float *xx = (float *)malloc(sizeof(float)*height*width);
	//CUDA_CHECK_RETURN(cudaMemcpy(xx, dark, height * width * sizeof(float), cudaMemcpyDeviceToHost));
	//for(int i=0;i<height*width;i++){printf("%.2f ", xx[i]);}
	int shared_size_1 = blocks.x*(sizeof(float3)+sizeof(float));
	int shared_size_2 = grids.x*(sizeof(float3)+sizeof(float));
	dehazing_img_kernel1<<<grids, blocks, shared_size_1>>> ((float3 *)image, dark, height, width, int_image, int_dark);
	dehazing_img_kernel2<<<1, grids, shared_size_2>>> ((float3 *)image, height*width, int_image, int_dark);

}

__global__
void transmission_kernel(float3 *image, float transmission, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		tx = image[i].x/image[height*width].x;
		ty = image[i].y/image[height*width].y;
		tz = image[i].z/image[height*width].z;
		transmission[i] = 1 - 0.75*min(tx, min(ty, tz));
	}
}

void transmission(float3 *image, float3 *t, int height, int width, dim3 blocks,dim3 grids){
	transmission_kernel<<<grids, blocks>>> ((float3 *)image, transmission, height, width);
	int window = 7;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	prior_kernel<<<grids, blocks, shared_size>>>(transmission, height, width, window);
}

__global__
void dehaze_kernel(float3 *image, float *dark, float t, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		image[i].x = (image[i].x - image[height*width].x*(1-t[i])) / t[i];
		image[i].y = (image[i].y - image[height*width].y*(1-t[i])) / t[i];
		image[i].z = (image[i].z - image[height*width].z*(1-t[i])) / t[i];
	}
}

void dehaze(float3 *image,float *dark, float *t, int height, int width, dim3 blocks,dim3 grids){
	dehaze_kernel<<<grids, blocks>>> (image, dark, t, height, width);
}
