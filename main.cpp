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

#include "dehazing.h"
#include "iostream"
#include "time.h"
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include "limits.h"
#include <unistd.h>

using namespace std;
using namespace cv;

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
int idx_x[INT_MAX];				//Rows
int idx_y[INT_MAX];				//Cols
double idx_v[INT_MAX]={0.0};	//Values
int idx_l=0;					//total number of non-zero value of L

// Define fast convert table
int convert_table[25];

char img_name[100]="example.bmp";
char trans_name[100]="printMatInfosdfasdfasdf;";
char out_name[100]="";
/*
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

//Write Matrix to File, so that Matlab can read
void writeMatToFile(char * filename,Mat m)
{
	FILE * fout = fopen(filename,"w");
	int count=0;
	for(float * p=(float *)m.datastart;p<(float *)m.dataend;p++)
	{
		fprintf(fout,"%f ",*p);
		count++;
		if(count%m.cols==0) fprintf(fout,"\n");
	}
	fclose(fout);
}

//Write L Sparse Matrix to File
void writeLFile()
{
	FILE * foutx = fopen("idx_x.txt","w");
	FILE * fouty = fopen("idx_y.txt","w");
	FILE * foutv = fopen("idx_v.txt","w");
	for(int i=0;i<idx_l;i++)
	{
		fprintf(foutx,"%d ",idx_x[i]+1);
		fprintf(fouty,"%d ",idx_y[i]+1);
		fprintf(foutv,"%f ",idx_v[i]);
	}
	fclose(foutx);
	fclose(fouty);
	fclose(foutv);
}

//Calculate Min and Max value of Matrix
MinMax MaxAndMinOfMatirx( Mat x )
{
	MinMax rtn;
	rtn.max=0;
	rtn.min=1000;
	for(float * p=(float *)x.datastart; p<(float *)x.dataend; p++)
	{
		if(*p>rtn.max)	rtn.max=*p;
		if(*p<rtn.min)  rtn.min=*p;
	}
	return rtn;
}
*/
//Process Args from CMD
void processArgs(int argc, char * argv[])
{
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

	processArgs(argc,argv);

	while(access(img_name,0)!=0)
	{
		cout<<"The image "<<img_name<<" don't exist."<<endl<<"Please enter another one:"<<endl;
		cin>>filename;
		//img_name=filename;
	}

	clock_t start , finish ;
	//double duration1,duration2,duration3,duration4,duration5,duration6,duration7;

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
	gpu_func(gpu_img, gpu_trans_img, airlight, gpu_dest,_PriorSize,height,width,t0);

	Mat dest(height, width, CV_32FC3);
	gpu_dest.download(dest);

	imwrite(out_name,dest*255);

	return 0;
}



