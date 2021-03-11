###
 # @Author: your name
 # @Date: 2020-05-31 16:56:04
 # @LastEditTime: 2020-06-02 10:04:54
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /CNN/srcOpenCL/cmd.sh
### 
g++ -std=c++11 -O1 main.cpp backward.cpp bmp.cpp cnn.cpp forward.cpp init.cpp math_functions.cpp mnist.cpp model.cpp predict.cpp train.cpp -o ../Release/opencl_cnn
cd ../Release
./opencl_cnn
