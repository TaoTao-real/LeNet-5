###
 # @Author: your name
 # @Date: 2020-05-31 00:48:31
 # @LastEditTime: 2020-05-31 01:41:38
 # @LastEditors: your name
 # @Description: In User Settings Edit
 # @FilePath: /CNN/srcUnrolling/cmd.sh
### 
g++ -O3 main.cpp backward.cpp bmp.cpp cnn.cpp forward.cpp init.cpp math_functions.cpp mnist.cpp model.cpp predict.cpp train.cpp -o ../Release/unrolling_cnn
cd ../Release
./unrolling_cnn
