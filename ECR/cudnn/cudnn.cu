#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>

#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda.h>
using namespace std;



int main() {
  

  // convolution
  const int pad_h = 0;
  const int pad_w = 0;
  const int str_h = 3;
  const int str_w = 3;
  const int dil_h = 1;
  const int dil_w = 1;
  
	string a[4995];
	int h[4995],w[4995];
	ifstream file_list("../../dataset/file_list");
	for(int i=0;i<4995;i++)
		file_list>>a[i];
	file_list.close();
	ifstream conv_shape("../../dataset/conv_shape");
	for(int i=0;i<4995;i++){
		conv_shape>>h[i];
		w[i] = h[i];
	}
	conv_shape.close();
//	printf("read success!\n");
	
	// per-experiment
	int len;
	
	
	 // filter
	const int filt_k = 1;
	const int filt_c = 1;
	const int filt_h = 3;
	const int filt_w = 3;
	
	const int kernelSize = filt_w*filt_h;
 	const float kernel[kernelSize] = { 1,0,1,0,1,1,0,1,1};
  
	 
	 for(int i=0;i<4995;i++){
	 
   // input
   const int in_n = 1;
   const int in_c = 1;
    int in_h = h[i];
    int in_w = w[i];

	int arraySize = in_h*in_w;
	//int o_w = (i_w-k_w)/stride +1;
	//int o_h = (i_h-k_h)/stride +1;
	
			
		
	float *feature = new float[arraySize];
	len = 0;
	
	ifstream conv_feature(("../../dataset/conv/"+a[i]).c_str());
	while(!conv_feature.eof())
		conv_feature>>feature[len++];
	conv_feature.close();	 
  	//printf("feature read success!\n");
	// output
	int out_n;
	int out_c;
	int out_h;
	int out_w;
  
  
	cudnnHandle_t cudnn;
   (cudnnCreate(&cudnn));
  cudnnTensorDescriptor_t in_desc;
   (cudnnCreateTensorDescriptor(&in_desc));
   (cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
   (cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));

 


  cudnnFilterDescriptor_t filt_desc;
   (cudnnCreateFilterDescriptor(&filt_desc));
   (cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
   (cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));




  cudnnConvolutionDescriptor_t conv_desc;
   (cudnnCreateConvolutionDescriptor(&conv_desc));
   (cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));


  
   (cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));



  cudnnTensorDescriptor_t out_desc;
   (cudnnCreateTensorDescriptor(&out_desc));
   (cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
   (cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  // = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST; // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
 (cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

 

  // workspace
  size_t ws_size;
   (cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
   (cudaMalloc(&ws_data, ws_size));



  // perform
  float alpha = 1.f;
  float beta = 0.f;
  cudaEvent_t start,stop;
  float elapsedTime1 = 0.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
	cudaMemcpy(in_data,feature,arraySize*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(filt_data,kernel,9*sizeof(float),cudaMemcpyHostToDevice);
	
       (cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime1, start, stop);
	cout << elapsedTime1<< endl; //ms
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

  // finalizing
	 (cudaFree(ws_data));
	 (cudaFree(out_data));
	 (cudnnDestroyTensorDescriptor(out_desc));
	 (cudnnDestroyConvolutionDescriptor(conv_desc));
	 (cudaFree(filt_data));
	 (cudnnDestroyFilterDescriptor(filt_desc));
	 (cudaFree(in_data));
	 (cudnnDestroyTensorDescriptor(in_desc));
	 (cudnnDestroy(cudnn));
  }
  
  
  return 0;
}
