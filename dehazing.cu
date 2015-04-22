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
void prior_kernel(float *dark, float *new_dark, int height, int width, int window){
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
		
		float minval = 255.0;
		for(int startx = 0; startx < window * 2 + 1; startx++){
			for(int starty = 0; starty < window * 2 + 1; starty++){
				if(IN_GRAPH(x-window+startx, y-window+starty, height, width)){
				minval = min(
						buffer[(threadIdx.x+startx)*(blockDim.y + window * 2) + threadIdx.y + starty],
						 minval
					);
				}
			}
		}
		new_dark[i] = minval;
		/*
		float minval = 255.0;
		for(int startx = 0; startx < window * 2 + 1; startx++){
			for(int starty = 0; starty < window * 2 + 1; starty++){
				if(IN_GRAPH(x-window+startx, y-window+starty, height, width)){
					minval = min(dark[i+(startx-window)*width+starty-window], minval);
					//if(minval-(int)minval>0){printf("%d %d %.2f\n", x-window+startx, y-window+starty, minval);}
				}
			}
		}	
		//if(minval-(int)minval>0){printf("%.2f\n", minval);}

		buffer[threadIdx.x*blockDim.y + threadIdx.y] = minval;
		__syncthreads();
		new_dark[i] = buffer[threadIdx.x*blockDim.y + threadIdx.y];*/
	}
}

void dark_channel(float *image,float *dark_channel,int height, int width, dim3 blocks,dim3 grids){
	
	float *tmp_dark;
	cudaMalloc((void **)(&tmp_dark), sizeof(float)*height*width);
	dark_channel_kernel<<<grids, blocks>>> ((float3 *)image, tmp_dark, height, width);
	
	int window = 7;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	prior_kernel<<<grids, blocks, shared_size>>>(tmp_dark, dark_channel, height, width, window);
	cudaFree(tmp_dark);
}

/*
 * air_light host wrapper and kernel
 */

//first kernel reduce to < 1024 values for next kernel
__global__
void airlight_kernel1(
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
void airlight_kernel2(float3 *image, int size, float3 *int_image, float *int_dark){

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
			//printf("%.2f %.2f %.2f %.2f\n", tmp_image[0].x,tmp_image[0].y,tmp_image[0].z, tmp_dark[0]);
			}
		}
		__syncthreads();
	}
	if(threadIdx.x == 0){
		//float factor = 1.0;
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
	airlight_kernel1<<<grids, blocks, shared_size_1>>> ((float3 *)image, dark, height, width, int_image, int_dark);
	airlight_kernel2<<<1, grids, shared_size_2>>> ((float3 *)image, height*width, int_image, int_dark);

}

__global__
void transmission1_kernel(float3 *image, float *t, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	float tx, ty, tz;
	if(x < height && y < width){
		tx = image[i].x/image[height*width].x;
		ty = image[i].y/image[height*width].y;
		tz = image[i].z/image[height*width].z;
		t[i] = min(tx, min(ty, tz));
	}
}

__global__
void transmission2_kernel(float *dark, float *new_dark, int height, int width, int window){
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
						       threadIdx.y + starty
						], minval);
				}
			}
		}
		new_dark[i] = 1-0.95*minval;
	}
}



void transmission(float *image, float *t, int height, int width, dim3 blocks,dim3 grids){
	float *tmp_trans;
	cudaMalloc((void **)&tmp_trans, sizeof(float)*height*width);
	transmission1_kernel<<<grids, blocks>>> ((float3 *)image, tmp_trans, height, width);
	int window = 7;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	transmission2_kernel<<<grids, blocks, shared_size>>>(tmp_trans, t, height, width, window);
	cudaFree(tmp_trans);
}

__global__
void dehaze_kernel(float3 *image, float *dark, float *t, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		image[i].x = (image[i].x - image[height*width].x)/max(0.1, t[i]) + image[height*width].x;
		image[i].y = (image[i].y - image[height*width].y)/max(0.1, t[i]) + image[height*width].y;
		image[i].z = (image[i].z - image[height*width].z)/max(0.1, t[i]) + image[height*width].z;

	}
}

void dehaze(float *image,float *dark, float *t, int height, int width, dim3 blocks,dim3 grids){
	dehaze_kernel<<<grids, blocks>>> ((float3 *)image, dark, t, height, width);
}

__global__
void boxfilter_kernel(float *img_in, float *img_res, float *patch, int r, int height, int width){//r: local window radius
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		img_res[i] = img_in[i];
	}
}

__global__
void matmul_kernel(float *a, float *b, float *res, int height, int width){
//b=a.*b
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		res[i] = a[i]*b[i];
	}
}

__global__
void var_kernel(float *a, float *b, float *c, float *d, int height, int width){
//d = a-b.*c
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		d[i] = a[i]-b[i]*c[i];
	}
}

__global__
void compab_kernel(float *a, float *b, float *cov_IP, float *var_I, float *mean_P, float *mean_I, int height, int width){
//a=cov_IP./(var_I.+eps);
	//eps = 10^-6
//b=mean_P-a.*mean_I;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		a[i] = cov_IP[i]/var_I[i] + 0.000001;
		b[i] = mean_P[i] - a[i]*mean_I[i];
	}

}

__global__
void result_kernel(float *result, float *mean_a, float *I, float *mean_b, int height, int width){
//mean_a = mean_a.*I+mean_b
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		result[i] = mean_a[i]*I[i] + mean_b[i];
	}
}
void gfilter(float *result, float *I, float *P, int height, int width, dim3 blocks, dim3 grids){
	//I: guided image - origin gray scale image - 1 channel
	//P: imaged need to be filtered - transmission image - 1 channel
	//result: refined trans image - 1 channel

	/*float *tmp_dark;
	cudaMalloc((void **)(&tmp_dark), sizeof(float)*height*width);
	dark_channel_kernel<<<grids, blocks>>> ((float3 *)image, tmp_dark, height, width);
	
	int window = 7;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	prior_kernel<<<grids, blocks, shared_size>>>(tmp_dark, dark_channel, height, width, window);
	cudaFree(tmp_dark);*/
	int r = 60;
	//float eps = 10^-6;
	
	float *N;//
	float *ones;//
	float *mean_I;//
	// float *mean_P;//
	// float *mean_IP;
	// float *cov_IP;//
	// float *mean_II;
	// float *var_I;//
	// float *a;//
	// float *b;//
	// float *mean_a;//
	// float *mean_b;//

	cudaMalloc((void **)(&N), sizeof(float)*height*width);
	cudaMalloc((void **)(&ones), sizeof(float)*height*width);
	cudaMemset(&ones, 1, sizeof(float)*height*width);
	
	cudaMalloc((void **)(&mean_I), sizeof(float)*height*width);
	// cudaMalloc((void **)(&mean_P), sizeof(float)*height*width);
	// cudaMalloc((void **)(&mean_IP), sizeof(float)*height*width);
	// cudaMalloc((void **)(&cov_IP), sizeof(float)*height*width);
	// cudaMalloc((void **)(&mean_II), sizeof(float)*height*width);
	// cudaMalloc((void **)(&var_I), sizeof(float)*height*width);
	// cudaMalloc((void **)(&a), sizeof(float)*height*width);
	// cudaMalloc((void **)(&b), sizeof(float)*height*width);
	// cudaMalloc((void **)(&mean_a), sizeof(float)*height*width);
	// cudaMalloc((void **)(&mean_b), sizeof(float)*height*width);
	
	boxfilter_kernel<<<grids, blocks>>> (ones, N, ones, r, height, width);//compute N
	//cudaFree(ones);
	//I, mean_I
	//boxfilter_kernel<<<grids, blocks>>> (I, mean_I, ones, r, height, width);//compute mean_I
	 //boxfilter_kernel<<<grids, blocks>>> (P, mean_P, N, r, height, width);//compute mean_P


	 //float *ImulP;
	 //cudaMalloc((void **)(&ImulP), sizeof(float)*height*width);
	 //matmul_kernel<<<grids, blocks>>> (I, P, ImulP, height, width);// compute P = I.*P
	// boxfilter_kernel<<<grids, blocks>>> (ImulP, mean_IP, N, r);//compute mean_IP
	// cudaFree(ImulP);
	// var_kernel<<<grids, blocks>>> (mean_IP, mean_I, mean_P, cov_IP, height, width);//compute cov_IP=mean_Ip-mean_I*mean_P

	// float *ImulI;
	// cudaMalloc((void **)(&ImulI), sizeof(float)*height*width);
	// matmul_kernel<<<grids, blocks>>> (I, I, ImulI, height, width);// compute I = I*I
	// boxfilter_kernel<<<grids, blocks>>> (ImulI, mean_II, N, r);//compute mean_II
	// cudaFree(ImulI);
	// var_kernel<<<grids, blocks>>> (mean_II, mean_I, mean_I, var_I, height, width);//compute var_I=mean_II-mean_I^2

	// compab_kernel<<<grids, blocks>>>(a, b, cov_IP, var_I, mean_P, mean_I, height, width);//compute a&b
	// cudaFree(cov_IP);
	// cudaFree(var_I);
	// cudaFree(mean_I);
	// cudaFree(mean_P);
	// boxfilter_kernel<<<grids, blocks>>> (a, mean_a, N, r);//compute mean_II
	// boxfilter_kernel<<<grids, blocks>>> (b, mean_b, N, r);//compute mean_II
	// cudaFree(N);
	// cudaFree(a);
	// cudaFree(b);
	// result_kernel<<<grids, blocks>>> (result, mean_a, I, mean_b, height, width);//return result
	// cudaFree(mean_a);
	// cudaFree(mean_b);
	}
