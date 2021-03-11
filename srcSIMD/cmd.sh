###
 # @Author: your name
 # @Date: 2020-05-31 02:03:15
 # @LastEditTime: 2020-05-31 02:05:46
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /CNN/srcSIMD/cmd.sh
### 

g++ -O3 main.cpp backward.cpp bmp.cpp cnn.cpp forward.cpp init.cpp math_functions.cpp mnist.cpp model.cpp predict.cpp train.cpp -o ../Release/simd_cnn -mavx -mfma
cd ../Release
./simd_cnn