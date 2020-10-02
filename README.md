# Accelerating convolutional neural network on GPUs

This is the source code of Accelerating convolutional neural network on GPUs. 

## Experiment environment

```
OS: Ubuntu 18.04
GPU: GTX 1080Ti
CUDA: 10.1, V10.1.243
CuDNN: 7.6
```
If you want to reproduct our experiment, please use same environment. We also the GTX 2080 to finishned our experiment. 

## Usage

First you need clone our code in your computer.
```
$ git lone https://github.com/milk2we/conv_pool_algorithm.git
$ cd conv_pool_algorithm
$ wget $(Datasets-link)
$ unzip datasets.zip
```
Data-Link can be found from https://drive.google.com/file/d/1Mc5R3-P-4zNucER1o39VVZqXcTQmZdJ6/view?usp=sharing

If you want to only using the code of convolution:

```
$ cd ECR
$ bash run_ecr_comparation.sh
```
Therefore, we can run the convolution and max pooling by the fellow steps:
```
$ cd PECR
$ bash run_pecr_comparation.sh
```

## Contributing

PRs accepted.

## License

MIT © Richard McRichface

