#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <opencv2/opencv.hpp> //openCV 사용
using namespace cv;
using namespace std;

void getActivation(Mat& Y){           //Sigmoid
    for(int k=0;k<Y.channels();k++){
        for(int i=0;i<Y.rows;i++){
            for(int j=0;j<Y.cols;j++){   
                if(Y.at<Vec3b>(i,j)[k]<0)     //x<0이면 y=0,이외엔 y=x
                   Y.at<Vec3b>(i,j)[k] = 1/(1+exp(Y.at<Vec3b>(i,j)[k]));
            }
        }
    } 
}

Mat MaxPooling(Mat& Y, int kernel, int st){
    double max=0;
    int PY_row=(Y.rows-kernel)/st+1;
    int PY_col=(Y.cols-kernel)/st+1;
    int st_row=0;
    int st_col=0;

    Mat PY(PY_row,PY_col,CV_8UC3);

    for(int k=0;k<Y.channels();k++){
        for(int m=0;m<PY.rows;m++){
            for(int n=0;n<PY.cols;n++){
                for(int i=0;i<kernel;i++){
                    for(int j=0;j<kernel;j++){
                        if(Y.at<Vec3b>(i+st_row,j+st_col)[k]>max)   //max value
                            max=Y.at<Vec3b>(i+st_row,j+st_col)[k];
                    }
                    if(i==kernel-1) st_col+=st;//비교가 끝나면 stride만큼 column 증가
                }
                PY.at<Vec3b>(m,n)[k]=max;           //최대값을 PY에 저장
                max=0;
                if(n==PY.cols-1){
                    st_row+=st;                     //stride만큼 row 증가
                    st_col=0;
                }
            }
            if(m==PY.rows-1) st_row=0;
        }
    }
    return PY;
}
Mat Convolution(Mat& W, Mat& F, const int P, const int S){
    int Y_row = (W.rows-F.rows+2*P)/S+1;
    int Y_col = (W.cols-F.cols+2*P)/S+1;
    int i,j,k,l,m,row_s=0,col_s=0;

    Mat WP(W.rows+2*P,W.cols+2*P,CV_64FC3); //패딩처리된 입력
    Mat Y(Y_row,Y_col,CV_64FC3);            //연산을 위해 double형으로 선언

    for(k=0;k<WP.channels();k++){   //패딩
        for(i=0;i<WP.rows;i++){ 
            for(j=0;j<WP.cols;j++){
                if((i<P)||(i>W.rows+P-1))
                    WP.at<Vec3d>(i,j)[k]=0;
                else if((j<P)||(j>W.cols+P-1))
                    WP.at<Vec3d>(i,j)[k]=0;
                else
                    WP.at<Vec3d>(i,j)[k]=W.at<Vec3b>(i-P,j-P)[k];
            }
        }
    }

    for(k=0;k<Y.channels();k++){                    //Convolution
        for(m=0;m<Y.rows;m++){
            for(i=0;i<Y.cols;i++){ 
                for(j=0;j<F.rows;j++){               //필터와 입력데이터 곱셈
                    for(l=0;l<F.cols;l++){
                        Y.at<Vec3d>(m,i)[k] += WP.at<Vec3d>(j+row_s,l+col_s)[k]*F.at<double>(j,l); //곱셈의 합
                    }
                    if(j==F.rows-1) col_s+=S;        //column이 stride만큼 증가 
                }
                if(i==Y.cols-1){
                    col_s=0;                        //column 초기화, row가 stride만큼 증가
                    row_s+=S;
                }           
            }
            if(m==Y.rows-1) row_s=0;                //row 초기화
        }
    }
    Mat outY;         //결과를 return할 변수(타입 변환)
    Y.convertTo(outY,CV_8UC3);

    return outY;
}

int main(){
    clock_t start = clock();            //프로그램 실행부터 현재까지의 시간

    int Filter_size = 5;    //필터의 사이즈
    int padding = 2;        //필터의 padding과 stride
    int stride = 1;
    int kernel = 2;         //풀링의 kernel과 stride
    int stride_pool = 1;
    int i,j,k;

    Mat img;
    img = imread("image00.jpg",IMREAD_COLOR);  //이미지 읽기,image는 CV_8U3S로 타입변환
    if(img.empty()){
        cout<<"Could not open of find the image"<<endl; //이미지를 못읽었을 경우 오류처리
        return -1;
    }

    Size size=Size(Filter_size,Filter_size);
    Mat imgF(Filter_size,Filter_size,CV_64F,Scalar(1.0/size.area()));   //blur filter 생성

    Mat Y = Convolution(img,imgF,padding,stride);   //convolution, blur
    //getActivation(Y);                             //activation function
    Mat imgY = MaxPooling(Y,kernel,stride_pool);    //max pooling
    imwrite("PATH/imageout0.jpg",Y);                //결과 이미지 생성

    printf("time: %0.5fs\n",(float)(clock() - start)/CLOCKS_PER_SEC);//알고리즘 시간측정

    Mat Y2;
    filter2D(img,Y2,img.depth(),imgF);  //openCV에 내장된 covolution 함수 실행
    imwrite("PATH/imageout1.jpg",Y2);

    Mat diffY(Y.rows,Y.cols,CV_8UC3);
    for(k=0;k<Y.channels();k++){
        for(i=0;i<Y.rows;i++){             //내가만든 conv와 내장함수 conv 와의 차이값
            for(j=0;j<Y.cols;j++){
                diffY.at<Vec3b>(i,j)[k]=abs(Y.at<Vec3b>(i,j)[k]-Y2.at<Vec3b>(i,j)[k]);
                if(i==2) printf("%d ",diffY.at<Vec3b>(i,j)[k]); //이미지 픽섹 한줄 뽑아서 확인
            }
        }
    }
    imshow("diif",diffY);
    waitKey();

    return 0;
}