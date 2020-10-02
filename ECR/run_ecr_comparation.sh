#!/bin/bash

cd ouralgorithm
nvcc ecr.cu 
./a.out > time_cscc.txt

pwd

cd ../cudnn
pwd
make
./cudnn > time_cudnn.txt

cd ..

pwd




