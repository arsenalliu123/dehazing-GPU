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

using namespace cv;
using namespace std;

// Define Const
clock_t start , finish ;
float lambda=0.0001;	//lambda
double _w=0.95;			//w
int height=0;			//image Height
int width=0;			//image Width
int size=0;			//total number of pixels
int blockdim = 32;

char img_name[100]="1.png";
char out_name[100]="2.png";
char trans_name[100]="3.png";
char dark_name[100]="4.png";

/*
 * dehazing procedures
 */

//read from img_name
Mat read_image(){

	Mat img = imread(img_name);
	height = img.rows;
	width = img.cols;
	size = img.rows*img.cols;
	Mat real_img(img.rows,img.cols,CV_32FC3);
	img.convertTo(real_img,CV_32FC3);
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
			printf("usage: -o output -i input -r filtered_transmission -t transmission.\n");
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
		else if(strcmp(argv[i],"-t")==0){
			i++;
			strcpy(dark_name,argv[i]);
		}
		else if(strcmp(argv[i],"-r")==0){
			i++;
			strcpy(trans_name,argv[i]);
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
	Mat img = read_image();
	
	float* cpu_image = (float *)malloc((size+1) * 3 * sizeof(float));
	float *dark_image = (float *)malloc(size * sizeof(float));
	float *trans_image = (float *)malloc(size * sizeof(float));

	/* load img into CPU float array and GPU float array */
	if (!cpu_image)
	{
		std::cout << "ERROR: Failed to allocate memory" << std::endl;
		return -1;
	}
	for (int i = 0; i < height; i++){
		for(int j = 0; j < width; j++)
		{
			for(int k = 0; k < 3; k++){
				cpu_image[(i * width + j) * 3 + k] = img.at<Vec<float,3> >(i,j)[k];
			}
		}
	}
	cpu_image[size] = 0;
	cpu_image[size+1] = 0;
	cpu_image[size+2] = 0;

	
	float *gpu_image = NULL;
	float *dark = NULL;
	float *img_gray = NULL;
	//size+1 for storing the airlight
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&gpu_image), ((size+1) * 3) * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMalloc((void **)(&dark), size * sizeof(float)));
	
	CUDA_CHECK_RETURN(cudaMalloc((void **)(&img_gray),size * sizeof(float)));

	CUDA_CHECK_RETURN(cudaMemcpy(gpu_image, cpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyHostToDevice));
	
    	float *trans = NULL;
    	CUDA_CHECK_RETURN(cudaMalloc((void **)(&trans), size * sizeof(float)));

    	float *filter = NULL;
    	CUDA_CHECK_RETURN(cudaMalloc((void **)(&filter), size * sizeof(float)));
    	/////////////////
	printf("height: %d width: %d\n", height, width);

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
	dim3 block(blockdim, blockdim);
	int grid_size_x = CEIL(double(height) / blockdim);
	int grid_size_y = CEIL(double(width) / blockdim);
	dim3 grid(grid_size_x, grid_size_y);
	//dark channel: dark
	dark_channel(gpu_image, img_gray, dark, height, width, block, grid);
	finish_clock();

	cout<<"Calculating Airlight ..."<<endl;
	start_clock();
	dim3 block_air(1024);
	dim3 grid_air(CEIL(double(size) / block_air.x));
	//airlight: gpu_image[height*width]
	air_light(gpu_image, dark, height, width, block_air, grid_air);
	finish_clock();
    
	cout<<"Calculating transmission ..."<<endl;
	start_clock();
    	//t: transmission
    	transmission(gpu_image, trans, height, width, block, grid);
	finish_clock();

	cout<<"Refining transmission ..."<<endl;
	dim3 block_guide(blockdim, blockdim);
	int grid_size_x_guide = CEIL(double(height) / blockdim);
	int grid_size_y_guide = CEIL(double(width) / blockdim);
	dim3 grid_guide(grid_size_x_guide, grid_size_y_guide);
	//filter: guided imaging filter result
    	gfilter(filter, img_gray, trans, height, width, block_guide, grid_guide);
	finish_clock();
    
	cout<<"Calculating dehaze ..."<<endl;
    	start_clock();
    	dehaze(gpu_image, dark, filter, height, width, block, grid);//dehaze image: ori_image
    	finish_clock();
    

	/*
	 * copy back to CPU memory
	 */
	cout<<"Copy back to host memory ..."<<endl;
	start_clock();
	
	CUDA_CHECK_RETURN(cudaFree(dark));
	
	CUDA_CHECK_RETURN(cudaMemcpy(trans_image, filter, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(filter));
	
	CUDA_CHECK_RETURN(cudaMemcpy(dark_image, trans, size * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(trans));
	
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_image, gpu_image, ((size+1) * 3) * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(gpu_image));
	
	/*
	printf("air light: %.2f %.2f %.2f\n", 
		cpu_image[3*size], 
		cpu_image[3*size+1], 
		cpu_image[3*size+2]);;
	*/

	for(int i=0;i<size;i++){
		trans_image[i] *= 255.f;
		dark_image[i] *= 255.f;
	}

	Mat dest(height, width, CV_32FC3, cpu_image);
	Mat trans_dest(height, width, CV_32FC1, trans_image);
	Mat dark_dest(height, width, CV_32FC1, dark_image);
	
	imwrite(out_name, dest);
	imwrite(trans_name, trans_dest);
	imwrite(dark_name, dark_dest);
	
	free(cpu_image);
	free(trans_image);
	free(dark_image);

	free(cpu_image);
	free(trans_image);
	free(dark_image);
	
	finish_clock();
	return 0;
}
