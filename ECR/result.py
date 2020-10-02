# -*- coding:utf-8 -*-
# author:Administrator
# datetime:2020/6/7 19:13
# software: PyCharm
# function : None

import os



# 0:3:67:131:259:387:643:899:1155:1411:1923:2435:2947:3459:3971:4483

# string = '0:3:67:131:259:387:643:899:1155:1411:1923:2435:2947:3459:3971:4483'
# numbers = string.split(':')
string = '3:67:131:259:387:643:899:1155:1411:1923:2435:2947:3459:3971:4483'
index = string.split(':')


def time_all(lines):
    times = []
    temp = 0
    count = 0
    for i, line in enumerate(lines):
        temp += float(line.strip())
        count += 1
        if str(i) in index:
            times.append(temp / count)
            temp = 0
            count = 0
    times.append(temp / count)

    return times


file = open('./ouralgorithm/time.txt', 'r', encoding='UTF-8')

lines = file.readlines()
file.close()
ecr_time = time_all(lines)

file = open('./cudnn/time.txt', 'r', encoding='UTF-8')
lines = file.readlines()
file.close()
cudnn_time = time_all(lines)



string = '   ECR :|'
for i in range(len(ecr_time)):
    string += str(ecr_time[i])+'  '
string += sum(ecr_time)
print(string)

string = ''
string = 'CuDNN: |'
for i in range(len(cudnn_time)):
    string += str(cudnn_time[i])+'  '
string += sum(cudnn_time)
print(string)

string = ''
string = 'Speedup:|'
for i in range(len(cudnn_time)):
    string += str(cudnn_time[i]/ecr_time[i])+'  '

string += sum(cudnn_time)/sum(ecr_time)
print(string)
