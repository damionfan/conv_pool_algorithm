#!/bin/bash

cd ouralgorithm
nvcc ecr.cu 
./a.out > time_cscc.txt

cd ../cudnn
make
./cudnn > time_cudnn.txt

cd ..

pwd




