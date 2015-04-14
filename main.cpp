/*
 * main.cpp
 *
 *  Created on: Apr 12, 2015
 *      Author: river
 */

#ifdef __APPLE__
        #include <sys/uio.h>
#else
        #include <sys/io.h>
#endif
#include "iostream"
#include "time.h"
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include "limits.h"
#include <unistd.h>
#include "dehazing.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

using namespace cv;
using namespace std;

// Define Const
float lambda=0.0001;	//lambda
int _PriorSize=15;		//the window size of dark channel
double _topbright=0.01;//the top rate of bright pixel in dark channel
double _w=0.95;			//w
float t0=0.01;			//lowest transmission
int height=0;			//image Height
int width=0;			//image Width
int size=0;			//total number of pixels

// Define Rows and Cols index of L
int idx_l=0;					//total number of non-zero value of L

// Define fast convert table
int convert_table[25];

char img_name[100]="1.png";
char trans_name[100]="2.png";
char out_name[100]="3.png";

/*
 * dehazing procedures
 */

//read from img_name
Mat read_image(){
	Mat img=imread(img_name, CV_LOAD_IMAGE_COLOR);
	height = img.rows;
	width = img.cols;
	size = img.rows*img.cols;
	Mat real_img(img.rows,img.cols,CV_32FC3);
	img.convertTo(real_img,CV_32FC3);
	real_img=real_img/255;
	return real_img;
}

//************* Utility Functions **********
//Print Matrix
void printMat(char * name,Mat m)
{
	cout<<name<<"\n"<<m<<endl;
}

//Print Matrix Information
void printMatInfo(char * name,Mat m)
{
	cout<<name<<":"<<endl;
	cout<<"\t"<<"cols="<<m.cols<<endl;
	cout<<"\t"<<"rows="<<m.rows<<endl;
	cout<<"\t"<<"channels="<<m.channels()<<endl;
}

//Process Args from command line
void processArgs(int argc, char * argv[])
{
	cout<<"This is a dehazing algorithm"<<endl;
	for(int i=1;i<argc;i++)
	{
		if(strcmp(argv[i],"-o")==0)
		{
			i++;
			strcpy(out_name,argv[i]);
		}
		else if(strcmp(argv[i],"-t")==0)
		{
			i++;
			strcpy(trans_name,argv[i]);
		}
		else
		{
			strcpy(img_name,argv[i]);
		}
	}
}

//Main Function
int main(int argc, char * argv[])
{
	char filename[100];
	cout<<"here"<<endl;
	processArgs(argc,argv);

	while(access(img_name,0)!=0)
	{
		cout<<"The image "<<img_name<<" don't exist."<<endl<<"Please enter another one:"<<endl;
		cin>>filename;
		strcpy(img_name,filename);
	}

	clock_t start , finish ;
	//double duration1,duration2,duration3,duration4,duration5,duration6,duration7;

	//load into a openCV's mat object
	Mat img = read_image();

	/* load img into CPU float array and GPU float array */
	float* cpu_image = new float[(size+1) * 3];
	if (!cpu_image)
	{
		std::cout << "ERROR: Failed to allocate memory" << std::endl;
		return -1;
	}
	for (int i = 0; i < height; i++){
		for(int j = 0; j < width; j++)
		{
			for(int k = 0; k < 3; k++)
				cpu_image[i * width + j + k] = img.at<float>(i,j,k);
		}
	}
	cpu_image[size] = 0;
	cpu_image[size+1] = 0;
	cpu_image[size+2] = 0;

	float *gpu_image = NULL;
	float *dark_channel = NULL;
	//size+1 for storing the airlight
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&gpu_image), ((size+1) * 3) * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&dark_channel), size * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(gpu_image, cpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyHostToDevice));

	/*
	 * Dehazing Algorithm:
	 * 1. Calculate Dark Prior
	 * 2. Calculate Air Light
	 * 3. Get the image
	 */

	//define the block size and grid size
	int grid_size = (int)ceil(double((size) / 256.0));

	dim3 block(16, 16);
	dim3 grid(grid_size);
	dark_channel(gpu_image, dark_channel, size, block, grid);

	dim3 block_air(256);
	dim3 grid_air((int)ceil(double((grid_size) / 256.0)));
	air_light(gpu_image, dark_channel, size, block_air, grid_air);

	/*
	 * copy back to CPU memory
	 */
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_image, gpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpu_image));

	Mat dest(height, width, CV_32FC3);
	dest.data = (uchar *)cpu_image;

	imwrite(out_name, dest*255);

	/*
	gpu::GpuMat gpu_img(real_img);
	gpu::GpuMat gpu_channel[3];
	split(gpu_img, gpu_channel);

	gpu::GpuMat dark_channel(gpu_channel[0]);

	gpu::min(dark_channel, gpu_channel[2], dark_channel);
	gpu::min(dark_channel, gpu_channel[1], dark_channel);
	gpu::erode(dark_channel, dark_channel, Mat::ones(_PriorSize,_PriorSize,0));

	//int n_bright=_topbright*size;
	Point maxLoc;
	gpu::minMaxLoc(dark_channel,0,0,0,&maxLoc);
	Vec<float,3> airlight;
	airlight[0] = real_img.at<Vec<float,3> >(maxLoc)[0];
	airlight[1] = real_img.at<Vec<float,3> >(maxLoc)[1];
	airlight[2] = real_img.at<Vec<float,3> >(maxLoc)[2];

	Mat trans_img = imread(trans_name, 0);
	Mat real_trans_img(img.rows,img.cols,CV_32FC1);
	trans_img.convertTo(real_trans_img,CV_32FC1);
	real_trans_img=real_trans_img/255;

	gpu::GpuMat gpu_trans_img(real_trans_img);
	gpu::GpuMat gpu_dest(height, width, CV_32FC3);
	//gpu_func(gpu_img, gpu_trans_img, airlight, gpu_dest,_PriorSize,height,width,t0);
	*/

	return 0;
}
