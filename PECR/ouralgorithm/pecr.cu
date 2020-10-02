#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <fstream>

#include <assert.h>

//CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>


using namespace std;


__global__ void func_pecr(float *input,float *kernel,int i_w,int i_h, int k_w, int k_h, int c_s, int p_s, int p_w,int p_h,float *maxvalue){
const unsigned int thread_idx = threadIdx.x;
const unsigned int block_idx = blockIdx.x;
const unsigned int block_dim = blockDim.x;


const unsigned int start = thread_idx *c_s*p_s +i_w*(block_idx*c_s);

__shared__ float data[1024];
__shared__ int index[1024];
__shared__ int count[200];
int cnt =0;
for (int n=0;n<p_w*p_h;n++){
int num = 0;
int n_start = n/(p_w)*c_s*i_w + n%p_h*c_s;
for (int i=0;i<k_w;i++)
for (int j=0;j<k_h;j++){

int offset = start + n_start+i*i_w+j;
if (input[offset] != 0){
data[cnt] = input[offset];
index[cnt] = i*j+i;
cnt +=1;
num +=1;

} 

}
count[n] = num;
}

int temp = 0;
float max = 0;
__shared__ float output[500];
for(int i=0;i<p_w*p_h;i++){
output[i] = 0;
for(int n=temp;n<temp+count[i];n++){
output[i] += data[n]*kernel[index[n]];
}
if (max<output[i])
	max = output[i];
temp += count[i];
}

maxvalue[thread_idx] = float(max);
//if (maxvalue[thread_idx] >0)
//printf("%f\n",maxvalue[thread_idx]);
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
		//int outSize = o_w*o_h;
		int p_w =2;
		int p_h =2;
		int op=(o_w-p_w)/stride +1;

		
		float *feature = new float[arraySize];
		len = 0;
//cout<<a[i]<<endl;
		ifstream conv_feature(("../../dataset/conv/"+a[i]).c_str());
		while(!conv_feature.eof())
			conv_feature>>feature[len++];
		conv_feature.close();

float * gpu_input;
float * gpu_kernel;
float * gpu_output;
cudaEvent_t start, stop;
			float elapsedTime1 = 0.0;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start,0);	
cudaStatus = cudaMalloc((void**)&gpu_input,arraySize*sizeof(float));
		if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!%s\n", cudaGetErrorString(cudaStatus));
		}

cudaStatus = cudaMalloc((void**)&gpu_kernel,kernelSize*sizeof(float));
		if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!%s\n", cudaGetErrorString(cudaStatus));
		}


cudaStatus = cudaMalloc((void**)&gpu_output,op*op*sizeof(float));
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
float *result = (float*)malloc(op*op*sizeof(float));
		//time-start


//function-start
func_pecr<<<op,op>>>(gpu_input,gpu_kernel,i_w,i_h,k_w,k_h,1,1,p_w,p_h,gpu_output);




cudaStatus = cudaMemcpy(result,gpu_output,op*op*sizeof(float),cudaMemcpyDeviceToHost);


if (cudaStatus != cudaSuccess) {
cout<<i<<endl;

fprintf(stderr, "cudaMemcpy failed!%s\n", cudaGetErrorString(cudaStatus));
return -1;
}
/*
for(int m=0;m<op*op;m++){
if (result[m]>0)
printf("%f",result[m]);
}
*/

free(feature);
free(result);
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

		}

//end-time
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsedTime1, start, stop);
cout << elapsedTime1<< endl; //ms
cudaEventDestroy(start);
cudaEventDestroy(stop);

}


return 0;
}



