#include "dehazing.h"
#include "stdio.h"




//convenient macros
#define IN_GRAPH(x,y,h,w) ((x>=0)&&(x<h)&&(y>=0)&&(y<w))
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#define WINDOW 7
#define R 15

/*
 * dark_channel host wrapper and kernel
 */
//first kernel calculate min of RGB

void printinfo(float *dark, int height, int width){
	float *xx = (float *)malloc(sizeof(float)*height*width);
	CUDA_CHECK_RETURN(cudaMemcpy(xx, dark, height * width * sizeof(float), cudaMemcpyDeviceToHost));
	for(int i=0;i<height*width;i++){printf("%.2f ", xx[i]);}
	

}

__global__
void dark_kernel1(float3 *image, float *img_grey, float *dark, int height, int width){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		dark[i] = min(image[i].x, min(image[i].y, image[i].z));
		img_grey[i] = image[i].x * 0.299 +  image[i].y * 0.587 + image[i].z * 0.114;
	}
}

__device__
void padding(float *buffer, float *dark,
	int x, int y,
	int tx, int ty,
	int window,
	int bdimx, int bdimy,
	int height, int width){
	const int si = (tx + window) * (bdimy + window * 2) + ty + window;
	int i = x*width + y;
	buffer[si] = dark[i];
	if(tx < window && IN_GRAPH(x-window, y, height, width) ){
		buffer[si - (bdimy + window * 2) * window] = dark[i - window * width];
		if(ty < window &&
			IN_GRAPH(x-window, y-window, height, width) ){
			buffer[si - (bdimy + window * 2) * window - window]
			= dark[i - window * width - window];
		}
		if(ty >= bdimy - window &&
			IN_GRAPH(x-window, y+window, height, width) ){
			buffer[si - (bdimy + window * 2) * window + window]
		       = dark[i - window * width + window];
		}
	}
	if(tx >= bdimx - window && IN_GRAPH(x+window, y, height, width) ){
		buffer[si + (bdimy + window * 2) * window] = dark[i + window * width];
		if(ty >= bdimy - window &&
			IN_GRAPH(x+window, y+window, height, width) ){
				buffer[si + (bdimy + window * 2) * window + window]
				       = dark[i + window * width + window];
		}
		if(ty < window &&
			IN_GRAPH(x+window, y-window, height, width) ){
				buffer[si + (bdimy + window * 2) * window - window]
				       = dark[i + window * width - window];
		}

	}
	if(ty >= bdimy - window && IN_GRAPH(x, y+window, height, width) ){
		buffer[si + window] = dark[i + window];
	}
	if(ty < window && IN_GRAPH(x, y-window, height, width) ){
		buffer[si - window] = dark[i - window];
	}

}


//second kernel calculate min of 15 X 15 ceil
__global__
void dark_kernel2(float *dark, float *new_dark, int height, int width, int window){
	extern __shared__ float buffer[];
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		
		//using shared memory
		padding(buffer, dark,
			x, y,
			threadIdx.x, threadIdx.y,
			window,
			blockDim.x, blockDim.y,
			height, width);

		__syncthreads();
		
		float minval = 255.0;
		for(int startx = 0; startx < window * 2 + 1; startx++){
			for(int starty = 0; starty < window * 2 + 1; starty++){
				if(IN_GRAPH(x-window+startx, y-window+starty, height, width)){
					int shared_row_index = (threadIdx.x+startx)*(blockDim.y + window * 2);
					int shared_index = shared_row_index + threadIdx.y + starty;
					minval = min(buffer[shared_index],minval);
				}
			}
		}
		new_dark[i] = minval;

		/*
		//using global memory
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
		new_dark[i] = buffer[threadIdx.x*blockDim.y + threadIdx.y];
		*/
	}
}

void dark_channel(float *image, float *img_grey, float *dark_channel, int height, int width, dim3 blocks, dim3 grids){
	
	float *tmp_dark;
	cudaMalloc((void **)(&tmp_dark), sizeof(float)*height*width);
	
	dark_kernel1<<<grids, blocks>>> ((float3 *)image, img_grey, tmp_dark, height, width);
	
	int window = WINDOW;
	int shared_size = (blocks.x + window * 2) * (blocks.y + window * 2) * sizeof(float);
	dark_kernel2<<<grids, blocks, shared_size>>>(tmp_dark, dark_channel, height, width, window);
	
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
		
		//using shared memory
		padding(buffer, dark,
			x, y,
			threadIdx.x, threadIdx.y,
			window,
			blockDim.x, blockDim.y,
			height, width);

		__syncthreads();
		
		float minval = 1.0;
		for(int startx = 0; startx < window * 2 + 1; startx++){
			for(int starty = 0; starty < window * 2 + 1; starty++){
				if(IN_GRAPH(x-window+startx, y-window+starty, height, width)){
					int shared_row_index = (threadIdx.x+startx)*(blockDim.y + window * 2);
					int shared_index = shared_row_index + threadIdx.y + starty;
					minval = min(buffer[shared_index], minval);
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
	int window = WINDOW;
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
void setones(float *img_in, int height, int width, float val){
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		img_in[i] = val;
	}
}

__global__
void boxfilter_kernel(float *img_in, float *img_res, float *patch, int r, int height, int width){//r: local window radius
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	extern __shared__ float buffer[];
	if(x < height && y < width){
		padding(
			buffer, img_in,
			x, y,
			threadIdx.x, threadIdx.y,
			r,
			blockDim.x, blockDim.y,
			height, width);

		__syncthreads();

		float val = 0.0;
		for(int startx = 0; startx < r * 2 + 1; startx++){
			for(int starty = 0; starty < r * 2 + 1; starty++){
				if(IN_GRAPH(x-r+startx, y-r+starty, height, width)){
					int shared_row_index = (threadIdx.x+startx)*(blockDim.y + r * 2);
					int shared_index = shared_row_index + threadIdx.y + starty;
					val += buffer[shared_index];
				}
			}
		}
		
		img_res[i] = val/patch[i];//((2*r+1)*(2*r+1));
	}
}

__global__
void boxfilter_kernel2(float *img_in,
	float *img_res,
	float *img_in2,
	float *img_res2,
	float *patch,
	int r,
	int height,
	int width){

	//r: local window radius
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	extern __shared__ float buffer[];
	float *buffer2 = buffer + (blockDim.x + r * 2) * (blockDim.y + r * 2);
	
	if(x < height && y < width){
		
		padding(
			buffer, img_in,
			x, y,
			threadIdx.x, threadIdx.y,
			r,
			blockDim.x, blockDim.y,
			height, width);

		padding(buffer2, img_in2,
			x, y,
			threadIdx.x, threadIdx.y,
			r,
			blockDim.x, blockDim.y,
			height, width);

		__syncthreads();

		float val = 0.0;
		float val2 = 0.0;
		for(int startx = 0; startx < r * 2 + 1; startx++){
			for(int starty = 0; starty < r * 2 + 1; starty++){
				if(IN_GRAPH(x-r+startx, y-r+starty, height, width)){
					int shared_row_index = (threadIdx.x+startx)*(blockDim.y + r * 2);
					int shared_index = shared_row_index + threadIdx.y + starty;
					val += buffer[shared_index];
					val2 += buffer2[shared_index];
				}
			}
		}

		img_res[i] = val/patch[i];
		img_res2[i] = val2/patch[i];
	}
}

__global__
void matmul_kernel(float *a, float *b, float *res1, float *res2, int height, int width){
//b=a.*b
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		res1[i] = a[i]*b[i];
		res2[i] = a[i]*a[i];
	}
}

__global__//(mean_IP, mean_II, mean_I, mean_P, cov_IP, var_I, height, width)
//(a, b, cov_IP, var_I, mean_P, mean_I, height, width)
void var_kernel(float *a, float *b, float *mean_IP, float *mean_II, float *mean_I, float *mean_P, float *cov_IP, float *var_I, int height, int width){
//d = a-b.*c

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		cov_IP[i] = mean_IP[i]-mean_I[i]*mean_P[i];
		var_I[i] = mean_II[i]-mean_I[i]*mean_I[i];
		a[i] = cov_IP[i]/(var_I[i] + 0.000001);
		b[i] = mean_P[i] - a[i]*mean_I[i];
	}
}
/*
__global__
void compab_kernel(float *a, float *b, float *cov_IP, float *var_I, float *mean_P, float *mean_I, int height, int width){
//a=cov_IP./(var_I.+eps);
	//eps = 10^-6
//b=mean_P-a.*mean_I;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = x * width + y;
	if(x < height && y < width){
		a[i] = cov_IP[i]/(var_I[i] + 0.000001);
		b[i] = mean_P[i] - a[i]*mean_I[i];
	}

}
*/
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

	int r = R;
	//float eps = 10^-6;
	
	float *N;
	float *ones;
	float *mean_I;
	float *mean_P;
	float *mean_IP;
	float *cov_IP;
	float *mean_II;
	float *var_I;
	float *a;
	float *b;
	float *mean_a;
	float *mean_b;
	
	//intermediate variables
	cudaMalloc((void **)(&N), sizeof(float)*height*width);
	cudaMalloc((void **)(&ones), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_I), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_P), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_IP), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_II), sizeof(float)*height*width);
	cudaMalloc((void **)(&a), sizeof(float)*height*width);
	cudaMalloc((void **)(&b), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_a), sizeof(float)*height*width);
	cudaMalloc((void **)(&mean_b), sizeof(float)*height*width);
	cudaMalloc((void **)(&cov_IP), sizeof(float)*height*width);
	cudaMalloc((void **)(&var_I), sizeof(float)*height*width);

	setones<<<grids, blocks>>> (ones, height, width, 1.0);
	//printinfo(ones, height, width);
	int shared_size = (blocks.x + r * 2) * (blocks.y + r * 2) * sizeof(float);
	int shared_size2 = 2 * shared_size;
	//compute N
	boxfilter_kernel<<<grids, blocks, shared_size>>> (
		ones, N, ones, r, height, width);
	
	cudaFree(ones);

	//compute mean_I and mean_P
	boxfilter_kernel2<<<grids, blocks, shared_size2>>> (
		I, mean_I, P, mean_P, N, r, height, width);



	float *ImulP;
	float *ImulI;
	cudaMalloc((void **)(&ImulP), sizeof(float)*height*width);
	cudaMalloc((void **)(&ImulI), sizeof(float)*height*width);
	matmul_kernel<<<grids, blocks>>> (I, P, ImulP, ImulI, height, width);// compute P = I.*P
	boxfilter_kernel2<<<grids, blocks, shared_size2>>> (ImulP, mean_IP, ImulI, mean_II, N, r, height, width);//compute mean_IP
	cudaFree(ImulP);
	
	//var_kernel<<<grids, blocks>>> (mean_IP, mean_I, mean_P, cov_IP, height, width);//compute cov_IP=mean_Ip-mean_I*mean_P

	//boxfilter_kernel<<<grids, blocks, shared_size>>> (ImulI, mean_II, N, r, height, width);//compute mean_II
	cudaFree(ImulI);
	//mean_IP
	var_kernel<<<grids, blocks>>> (a, b, mean_IP, mean_II, mean_I, mean_P, cov_IP, var_I, height, width);//compute var_I=mean_II-mean_I^2

	//compab_kernel<<<grids, blocks>>>(a, b, cov_IP, var_I, mean_P, mean_I, height, width);//compute a&b
	cudaFree(mean_I);
	cudaFree(mean_P);
	cudaFree(cov_IP);
	cudaFree(var_I);
	//compute mean_II
	boxfilter_kernel2<<<grids, blocks, shared_size2>>> (
		a, mean_a, b, mean_b, N, r, height, width);
	cudaFree(N);
	cudaFree(a);
	cudaFree(b);
	
	result_kernel<<<grids, blocks>>> (result, mean_a, I, mean_b, height, width);//return result
	cudaFree(mean_a);
	cudaFree(mean_b);
}
