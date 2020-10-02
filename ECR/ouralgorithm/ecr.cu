#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>

#include <assert.h>


// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


using  namespace std;




__global__ void func_new(float* input,float *kernel,float* output,
							int i_w, int i_h, int k_w,int k_h,int stride,
							int o_w,int o_h){
	const unsigned int thread_idx = threadIdx.x;
	const unsigned int block_idx  = blockIdx.x; 

	__shared__ float F_data[5120];
	__shared__ float K_data[5120];
	__shared__ float ptr[500];
	
	int temp=0;
	for(int i=0;i<k_h;i++){
		for(int j=0;j<k_w;j++){
			int offset = thread_idx*stride+	block_idx*i_w+i*i_w+j;
			float value = input[offset];
			float kvalue=kernel[i+j*k_w];
			if((value!=0)&&(kvalue!=0)){
				//float kvalue = kernel[i+j*k_w];
				//if(kvalue!=0){
				F_data[thread_idx*k_w*k_h+temp] = value;
				K_data[thread_idx*k_w*k_h+temp] = kvalue;
				temp++;	
				//}		
			}
		}

	}
	if(temp!=0)
		ptr[thread_idx] = thread_idx*k_w*k_h+temp-1;
	else
		ptr[thread_idx] = -1;
	__syncthreads();
	if(ptr[thread_idx]==-1)
		output[thread_idx+block_idx*blockDim.x] = 0;
	else
		for(int i=thread_idx*k_w*k_h;i<=ptr[thread_idx];i++){
			output[thread_idx+block_idx*blockDim.x] += F_data[i]*K_data[i];	
		}	
}



int main(){
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
	// per-experiment
	int len;
	const int k_w = 3;
	const int k_h = 3;
	const int stride = 1;
	const float kernel[9] = { 1,0,1,0,1,0,1,0,0};
	const int kernelSize = k_w*k_h;

	cudaError_t cudaStatus;
	
	
	
	
	for(int i=0;i<4995;i++){
		int i_w = w[i];
		int i_h = h[i];
		//cout<<i_w<<i_h<<endl;
		int arraySize = i_w*i_h;
		int o_w = (i_w-k_w)/stride +1;
		int o_h = (i_h-k_h)/stride +1;
		int outSize = o_w*o_h;
			
		
		float *feature = new float[arraySize];
		//float *features = new float[arraySize];	
		//read-feature
		
		len = 0;
		ifstream conv_feature(("../../dataset/conv/"+a[i]).c_str());
		while(!conv_feature.eof())
			conv_feature>>feature[len++];
		conv_feature.close();

		//for(int  j=0;j<arraySize;j++)
		//	feature[j] = (float)features[j];
		float * gpu_input;
		float * gpu_kernel;
		float * gpu_output;
		
		cudaStatus = cudaMalloc((void**)&gpu_input,arraySize*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!%s\n", cudaGetErrorString(cudaStatus));
		}
		
		cudaStatus = cudaMalloc((void**)&gpu_kernel,kernelSize*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!%s\n", cudaGetErrorString(cudaStatus));
		}
		
		cudaStatus = cudaMalloc((void**)&gpu_output,outSize*sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!%s\n", cudaGetErrorString(cudaStatus));
		}
		
		
		
		
		cudaStatus = cudaMemcpy(gpu_input,feature,arraySize*sizeof(float),cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!%s\n", cudaGetErrorString(cudaStatus));
		}
		
		cudaStatus = cudaMemcpy(gpu_kernel,kernel,kernelSize*sizeof(float),cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!%s\n", cudaGetErrorString(cudaStatus));
		}
			//time-start
		cudaEvent_t start,stop;
		float elapsedTime1 = 0.0;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
	
		
		//function-start
		
		func_new<<<(i_h-k_h)/stride+1,(i_w-k_w)/stride+1>>>(gpu_input,gpu_kernel,gpu_output,i_w,i_h,k_w,k_h,stride,o_w,o_h);

		//end-time
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime1, start, stop);
		cout << elapsedTime1<< endl; //ms
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		float *result = new float[outSize];
		cudaStatus = cudaMemcpy(result,gpu_output,outSize*sizeof(float),cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!%s\n", cudaGetErrorString(cudaStatus));
		}
		//for(int m=0;m<outSize;m++)
		//	cout<<result[m]<<" "<<endl;
		cudaStatus = cudaFree(gpu_input);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed!\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaFree(gpu_kernel);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed!\n", cudaGetErrorString(cudaStatus));
		}
		cudaStatus = cudaFree(gpu_output);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaFree failed!\n", cudaGetErrorString(cudaStatus));
			//cout<<feature[0]<<endl;
			delete[] feature;
			delete[] result;
		}
	
	}
	
	
	
	
	return 0;
}

 
