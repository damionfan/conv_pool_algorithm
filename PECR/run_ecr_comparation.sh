#!/bin/bash

cd ouralgorithm

echo "Start PECR Algorithm!"
nvcc pecr.cu 
./a.out > time.txt

echo "Finashed ECR Algorithm!"


cd ../cudnn
echo "Start CuDNN Algorithm!"
make
./cudnn > time.txt
echo "Finashed CuDNN Algorithm!"
cd ..

python3 result.py



