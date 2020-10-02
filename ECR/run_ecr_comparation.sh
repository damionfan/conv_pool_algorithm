#!/bin/bash

cd ouralgorithm

echo "Start ECR Algorithm!"
nvcc ecr.cu 
./a.out > time.txt

echo "Finashed ECR Algorithm!"


cd ../cudnn
echo "Start CuDNN Algorithm!"
make
./cudnn > time.txt
echo "Finashed CuDNN Algorithm!"
cd ..

pwd




