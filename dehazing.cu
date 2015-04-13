#include "dehazing.h"

using namespace std;
using namespace cv;


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }



//convinent macros
#define IN_GRAPH(x,y,h,w) ((x>=0)&&(x<h)&&(y>=0)&&(y<w))
#define mymin(x,y) ((x<y)?x:y)
#define mymax(x,y) ((x>y)?x:y)

__global__ void kernel(
		gpu::DevMem2Df mat,
		gpu::DevMem2Df trans_mat,
		float airlight1,
		float airlight2,
		float airlight3,
		gpu::DevMem2Df dest,
		int height, int width, int t0)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	extern __shared__ float airlight[];
	airlight[0] = airlight1;
	airlight[1] = airlight2;
	airlight[2] = airlight3;
	if(x < height && y < width){

		unsigned index_grey = x * width + y;
		unsigned index = (index_grey) * 3;

		for(int i = 0; i < 3; i++){
			dest.data[index+i] =
				(mat.data[index+i] - airlight[i])/mymax(trans_mat.data[index_grey], t0)
				+ airlight[i];
		}
	}

}
void gpu_func(
		gpu::DevMem2Df mat,
		gpu::DevMem2Df trans_mat,
		Vec<float, 3> airlight,
		gpu::DevMem2Df dest,
		int _PriorSize,
		int height,
		int width,
		int t0)
{
	dim3 grid(height/_PriorSize+1,width/_PriorSize+1);
	dim3 block(_PriorSize,_PriorSize);
    kernel<<<grid,block>>>(
    		mat, trans_mat, airlight[0], airlight[1], airlight[2], dest, height, width, t0);
}

//Read Image
/*
void ReadImage(){
	Mat img=imread(img_name, CV_LOAD_IMAGE_COLOR);
	img = img/255;
	height = img.rows;
	width = img.cols;
	size = img.rows*img.cols;
	Mat real_img(img.rows,img.cols,CV_32FC3);
	img.convertTo(real_img,CV_32FC3);
	real_img=real_img/255;


	gpu::GpuMat gpu_img(real_img);
	gpu::GpuMat gpu_channel[3];
	gpu::split(gpu_img, gpu_channel);

	gpu::GpuMat dark_channel(gpu_channel[0]);

	gpu::min(dark_channel, gpu_channel[2], dark_channel);
	gpu::min(dark_channel, gpu_channel[1], dark_channel);
	gpu::erode(dark_channel, dark_channel, Mat::ones(_PriorSize,_PriorSize,0));

	//int n_bright=_topbright*size;
	Point maxLoc;
	gpu::minMaxLoc(dark_channel,0,0,0,&maxLoc);
	Vec<float,3> airlight = real_img.at<Vec<float,3>>(maxLoc);
	gpu::GpuMat gpu_al = gpu::GpuMat(airlight);

	Mat trans_img = imread(trans_name, 0);
	Mat real_trans_img(img.rows,img.cols,CV_32FC1);
	trans_img.convertTo(real_trans_img,CV_32FC1);
	real_trans_img=real_trans_img/255;

	gpu::GpuMat gpu_trans_img(real_trans_img);
	gpu::GpuMat gpu_dest(height, width, CV_32FC3);
	gpu_func(gpu_img, gpu_trans_img, gpu_al, gpu_dest);

	Mat dest(height, width, CV_32FC3);
	gpu_dest.download(dest);

	imwrite(out_name,free_img*255);
	Image *image = (Image *)malloc(size*sizeof(Image));

	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			image[i*width+j].red = img.at<float>(i,j,0);
			image[i*width+j].blue = img.at<float>(i,j,1);
			image[i*width+j].green = img.at<float>(i,j,2);
		}
	}

	return image;
}
*/
//Read TransImage
/*
TransImage* ReadTransImage(){
	Mat img=imread(img_name, 0);
	Mat real_img(img.rows,img.cols,CV_32FC1);
	img.convertTo(real_img,CV_32FC1);
	TransImage *image = (TransImage *)malloc(size*sizeof(TransImage));
	for(int i=0;i<height;i++){
		for(int j=0;j<width;j++){
			image[i*width+j].grey = real_img.at<float>(i,j);
		}
	}

	return image;
}
*/
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


//Calculate Airlight
/*
Vec<float,3> Airlight(Mat img, Mat dark)
{
	int n_bright=_topbright*size;
	Mat dark_1=dark.reshape(1,size);
	Vector<int> max_idx;
	float max_num=0;
	int max_pos=0;
	Vec<float,3> a;
	Vec<float,3> A(0,0,0);
	Mat RGBPixcels=Mat::ones(n_bright,1,CV_32FC3);
	Mat HLSPixcels=Mat::ones(n_bright,1,CV_32FC3);
	Mat IdxPixcels=Mat::ones(n_bright,1,CV_32SC1);


	for(int i=0;i<n_bright;i++)
	{
		max_num=0;
		max_idx.push_back(max_num);
		for(float * p = (float *)dark_1.datastart;p!=(float *)dark_1.dataend;p++)
		{
			if(*p>max_num)
			{
				max_num = *p;
				max_idx[i] = (p-(float *)dark_1.datastart);
				RGBPixcels.at<Vec<float,3>>(i,0) = ((Vec<float,3> *)img.data)[max_idx[i]];
				IdxPixcels.at<int>(i,0) = (p-(float *)dark_1.datastart);
				//((Vec<float,3> *)img.data)[max_idx[i]] = Vec<float,3>(0,0,1);
			}
		}
		((float *)dark_1.data)[max_idx[i]]=0;
	}

	float maxL=0.0;
	//int maxIdx=0;
	for(int j=0; j<n_bright; j++)
	{
		A[0]+=RGBPixcels.at<Vec<float,3>>(j,0)[0];
		A[1]+=RGBPixcels.at<Vec<float,3>>(j,0)[1];
		A[2]+=RGBPixcels.at<Vec<float,3>>(j,0)[2];
	}

	A[0]/=n_bright;
	A[1]/=n_bright;
	A[2]/=n_bright;

	return A;
}
*/

//Calculate Transmission Matrix
/*
Mat TransmissionMat(Mat dark)
{

	return 1-_w*dark;
}
*/

//Calculate Haze Free Image
/*
Mat hazefree(Mat img,Mat t,Vec<float,3> a,float exposure = 0)
{
	Mat freeimg=Mat::zeros(height,width,CV_32FC3);
	img.copyTo(freeimg);
	Vec<float,3> * p=(Vec<float,3> *)freeimg.datastart;
	float * q=(float *)t.datastart;
	for(;p<(Vec<float,3> *)freeimg.dataend && q<(float *)t.dataend;p++,q++)
	{
		(*p)[0]=((*p)[0]-a[0])/std::max(*q,t0)+a[0] + exposure;
		(*p)[1]=((*p)[1]-a[1])/std::max(*q,t0)+a[1] + exposure;
		(*p)[2]=((*p)[2]-a[2])/std::max(*q,t0)+a[2] + exposure;
	}
	return freeimg;
}
*/

//************* Utility Functions **********
//Print Matrix

	/*
	cout<<"Reading Image ..."<<endl;
	start=clock();

	//Read image
	Image *image = ReadImage();
	Image *device_image;
	cudaMalloc(&device_image, size*sizeof(Image));
	cudaMemcpy(device_image, image, size*sizeof(Image), cudaMemcpyHostToDevice);

	//Read Trans image
	TransImage *t_image = ReadImage();
	Image *device_t_image;
	cudaMalloc(&device_t_image, size*sizeof(TransImage));
	cudaMemcpy(device_t_image, t_image, size*sizeof(TransImage), cudaMemcpyHostToDevice);

	//finished
	finish=clock();
	duration1=( double )( finish - start )/ CLOCKS_PER_SEC ;
	cout<<"Time Cost: "<<duration1<<"s"<<endl;
	waitKey(1000);
	cout<<endl;

	//Calculate DarkChannelPrior
	cout<<"Calculating Dark Channel Prior ..."<<endl;
	start=clock();
	dark_channel=DarkChannelPrior(img);
	//imshow("Dark Channel Prior",dark_channel);
	//printMatInfo("dark_channel",dark_channel);
	finish=clock();
	duration3=( double )( finish - start )/ CLOCKS_PER_SEC ;
	cout<<"Time Cost: "<<duration3<<"s"<<endl;
	waitKey(1000);
	cout<<endl;

	//Calculate Airlight
	cout<<"Calculating Airlight ..."<<endl;
		start=clock();
	Vec<float,3> a=Airlight(img,dark_channel);
	cout<<"Airlight:\t"<<" B:"<<a[0]<<" G:"<<a[1]<<" R:"<<a[2]<<endl;
		finish=clock();
		duration4=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration4<<"s"<<endl;
	cout<<endl;

	//Reading Refine Trans
	cout<<"Reading Refine Transmission..."<<endl;
	trans_refine=ReadTransImage();
	printMatInfo("trans_refine",trans_refine);
	//imshow("Refined Transmission Mat",trans_refine);
	cout<<endl;

	//Haze Free
	cout<<"Calculating Haze Free Image ..."<<endl;
		start=clock();

	free_img=hazefree(img,trans_refine,a,0.2);
	//imshow("Haze Free",free_img);

	printMatInfo("free_img",free_img);
		finish=clock();
		duration7=( double )( finish - start )/ CLOCKS_PER_SEC ;
		cout<<"Time Cost: "<<duration7<<"s"<<endl;

		//cout<<"Total Time Cost: "<<duration1+duration2+duration3+duration4+duration5+duration6+duration7<<"s"<<endl;

	//Save Image
	//char img_name_dark[100]="Dark_";
	//char img_name_step[100]="Step_";
	//char img_name_free[100]="Hazefree_";
	//strcat(img_name_free,img_name);
	//strcat(img_name_step,img_name);
	//strcat(img_name_dark,img_name);
	imwrite(out_name,free_img*255);
	//imwrite(img_name_step,trans_refine*255);
	//imwrite(img_name_dark,trans*255);
	cout<<"Image saved as "<<out_name<<endl;
	//waitKey();
	cout<<endl;

	return 0;
	*/
