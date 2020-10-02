#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <cuda.h>
#include <cudnn.h>
#include <assert.h>



using namespace std;


#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}


int main() {
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

for(int i=0;i<4995;i++){
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = h[i];
  const int in_w = w[i];


  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));


  float *data = (float*)malloc(in_n * in_c * in_h * in_w * sizeof(float));
  int len = 0;
  ifstream conv_feature(("../../dataset/conv/"+a[i]).c_str());
  while(!conv_feature.eof())
    conv_feature>>data[len++];
  conv_feature.close();

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));


  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 3;
  const int filt_w = 3;


  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));


  float kernel[9] = { 1,0,1,0,1,0,1,0,0};
  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));



  // convolution
  const int pad_h = 0;
  const int pad_w = 0;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));



  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo ;//=CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;//= CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED ;

  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
      cudnn,
      in_desc, filt_desc, conv_desc, out_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));



  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  float * con_result = (float*)malloc(out_n * out_c * out_h * out_w *sizeof(float));
  // perform
  float alpha = 1.f;
  float beta = 0.f;



 cudaMemcpy(filt_data,kernel,filt_k * filt_c * filt_h * filt_w *sizeof(float),cudaMemcpyHostToDevice);
 cudaMemcpy(in_data,data,in_n * in_c * in_h * in_w *sizeof(float),cudaMemcpyHostToDevice);

cudaEvent_t start, stop;
float elapsedTime1 = 0.0;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, algo, ws_data, ws_size,
      &beta, out_desc, out_data));

  // results

  CUDA_CALL(cudaMemcpy(
        con_result, out_data,
        out_n * out_c * out_h * out_w * sizeof(float),
        cudaMemcpyDeviceToHost));
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime1, start, stop);
cudaEventDestroy(start);
cudaEventDestroy(stop);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));


  cudnnHandle_t cudnn_p;
  cudnnPoolingDescriptor_t pooling_desc;
  CUDNN_CALL(cudnnCreatePoolingDescriptor(&pooling_desc));
  cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         CUDNN_POOLING_MAX,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         2,                       //window height
                                         2,                       //window width
                                         0,                       //vertical padding
                                         0,                       //horizontal padding
                                         1,                       //vertical stride
                                         1);
  cudnnTensorDescriptor_t in_p_desc;
  cudnnCreateTensorDescriptor(&in_p_desc);
  cudnnSetTensor4dDescriptor(in_p_desc,                  //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DATA_FLOAT,              //data type (precision)
                                        out_n,                        //number of images
                                        out_c,                        //number of channels
                                        out_h,                       //data height
                                        out_w);

  cudnnTensorDescriptor_t out_p_desc;
  cudnnCreateTensorDescriptor(&out_p_desc);
  cudnnSetTensor4dDescriptor(out_p_desc,                 //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DATA_FLOAT,              //data type (precision)
                                        1,                        //number of images
                                        1,                        //number of channels
                                        out_h-2 +1,                        //data height
                                        out_w-2 +1);
  float a = 1.0f;
  float be = 0.0f;

  float *gpu_in;
  float *gpu_out;

  cudaMalloc(&gpu_in,out_n * out_c * out_h * out_w *sizeof(float));
  cudaMalloc(&gpu_out, (out_h-2 +1)*(out_h-2 +1)*sizeof(float));
    cudaMemset(out_data,0,(out_h-2 +1)*(out_h-2 +1)*sizeof(float));
    cudaMemset(out_data,0,(out_h-2 +1)*(out_h-2 +1)*sizeof(float));
cudaEvent_t start1, stop1;
float elapsedTime2 = 0.0;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);
cudaEventRecord(start1,0);
  cudaMemcpy(gpu_in,con_result,out_n * out_c * out_h * out_w *sizeof(float),cudaMemcpyHostToDevice);



  cudnnPoolingForward(cudnn_p,         //cuDNN context handle
                                 pooling_desc,  //pooling descriptor handle
                                 &a,        //alpha scaling factor
                                 in_p_desc,       //input tensor descriptor
                                 gpu_in,       //input data pointer to GPU memory
                                 &be,         //beta scaling factor
                                 out_p_desc,      //output tensor descriptor
                                 gpu_out);
cudaEventRecord(stop1, 0);
cudaEventSynchronize(stop1);
cudaEventElapsedTime(&elapsedTime2, start1, stop1);
cout<<elapsedTime1+elapsedTime2<<endl;
cudaEventDestroy(stop1);
cudaEventDestroy(start1);


  cudnnDestroyTensorDescriptor(in_p_desc);
  cudnnDestroyTensorDescriptor(out_p_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  cudnnDestroy(cudnn_p);

  cudaFree(gpu_in);
  cudaFree(gpu_out);

  free(con_result);

  }
  return 0;
}
