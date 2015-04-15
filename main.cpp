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
#include "opencv2/opencv.hpp"
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
clock_t start , finish ;
float lambda=0.0001;	//lambda
int _PriorSize=15;		//the window size of dark channel
double _topbright=0.01;//the top rate of bright pixel in dark channel
double _w=0.95;			//w
float t0=0.01;			//lowest transmission
int height=0;			//image Height
int width=0;			//image Width
int size=0;			//total number of pixels

char img_name[100]="1.png";
char out_name[100]="3.png";

/*
 * dehazing procedures
 */

//read from img_name
Mat *read_image(){

	Mat *img=new Mat(imread(img_name, CV_LOAD_IMAGE_COLOR));
	height = img->rows;
	width = img->cols;
	size = img->rows*img->cols;
	Mat *real_img = new Mat(img->rows,img->cols,CV_32FC3);
	img->convertTo(*real_img,CV_32FC3);
	*real_img=(*real_img)/255;
	return real_img;
}

//************* Utility Functions **********
//Print Matrix
void printMat(char * name,Mat m)
{
	cout<<name<<"\n"<<m<<endl;
}

//Print Matrix Information
void printMatInfo(Mat *m)
{
	cout<<"\t"<<"cols="<<m->cols<<endl;
	cout<<"\t"<<"rows="<<m->rows<<endl;
	cout<<"\t"<<"channels="<<m->channels()<<endl;
}

//Process Args from command line
void processArgs(int argc, char * argv[])
{
	for(int i=1;i<argc;i++)
	{
		if(strcmp(argv[i], "-h")==0){
			printf("usage: -o output -i input.\n");
			exit(1);
		}
		else if(strcmp(argv[i],"-o")==0){
			i++;
			strcpy(out_name,argv[i]);
		}
		else if(strcmp(argv[i],"-i")==0){
			i++;
			strcpy(img_name,argv[i]);
		}
		else{
			printf("use -h to see usage.\n");
			exit(1);
		}
	}
}

void finish_clock(){
	finish=clock();
	double duration=( double )( finish - start )/ CLOCKS_PER_SEC * 1000;
	cout<<"Time Cost: "<<duration<<"ms"<<endl;
	waitKey(1000);
	cout<<endl;
}

void start_clock(){
	start=clock();
}

//Main Function
int main(int argc, char * argv[])
{
	char filename[100];
	processArgs(argc,argv);

	while(access(img_name,0)!=0)
	{
		cout<<"The image "<<img_name<<" don't exist."<<endl<<"Please enter another one:"<<endl;
		cin>>filename;
		strcpy(img_name,filename);
	}

	cout<<"Reading Image ..."<<endl;
	start_clock();
	//load into a openCV's mat object
	Mat *img = read_image();

	/* load img into CPU float array and GPU float array */
	float* cpu_image = (float *)malloc((size+1) * 3 * sizeof(float));
	if (!cpu_image)
	{
		std::cout << "ERROR: Failed to allocate memory" << std::endl;
		return -1;
	}
	for (int i = 0; i < height; i++){
		for(int j = 0; j < width; j++)
		{
			for(int k = 0; k < 3; k++){
				cpu_image[(i * width + j) * 3 + k] = img->at<float>(i,j,k);
			}
		}
	}
	cpu_image[size] = 0;
	cpu_image[size+1] = 0;
	cpu_image[size+2] = 0;

	float *gpu_image = NULL;
	float *dark = NULL;
	//size+1 for storing the airlight
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&gpu_image), ((size+1) * 3) * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&dark), size * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(gpu_image, cpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyHostToDevice));
	finish_clock();
	/*
	 * Dehazing Algorithm:
	 * 1. Calculate Dark Prior
	 * 2. Calculate Air Light
	 * 3. Get the image
	 */

	//define the block size and grid size
	cout<<"Calculating Dark Channel Prior ..."<<endl;
	start_clock();
	dim3 block(_PriorSize, _PriorSize);
	int grid_size_x = (int)ceil(double((height) / _PriorSize));
	int grid_size_y = (int)ceil(double((width) / _PriorSize));
	//printf("%d", grid_size);
	dim3 grid(grid_size_x, grid_size_y);
	dark_channel(gpu_image, dark, height, width, block, grid);
	finish_clock();

	cout<<"Calculating Airlight ..."<<endl;
	start_clock();
	dim3 block_air(1024);
	dim3 grid_air((int)ceil(double(size) / block_air.x));
	air_light(gpu_image, dark, height, width, block_air, grid_air);
	finish_clock();
	/*
	 * copy back to CPU memory
	 */
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_image, gpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpu_image));
	for(int i=0;i<size;i++){
		cpu_image[i*3] *= 255.f;
		cpu_image[i*3+1] *= 255.f;
		cpu_image[i*3+2] *= 255.f;
	}
	printf("%.2f %.2f %.2f\n", cpu_image[size*3], cpu_image[size*3+1],cpu_image[size*3+2]);

	Mat dest(height, width, CV_32FC3, cpu_image);
	imwrite(out_name, dest);

	return 0;
}
