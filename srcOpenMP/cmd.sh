###
 # @Author: your name
 # @Date: 2020-05-31 16:11:46
 # @LastEditTime: 2020-05-31 16:17:10
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /CNN/srcOpenMP/cmd.sh
### 

clang++ -Xpreprocessor -fopenmp -O3 main.cpp backward.cpp bmp.cpp cnn.cpp forward.cpp init.cpp math_functions.cpp mnist.cpp model.cpp predict.cpp train.cpp -o ../Release/openmp_cnn -lomp -mavx -mfma
cd ../Release
./openmp_cnn