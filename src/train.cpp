#include <sstream>
#include <fstream>
#include <iostream>
#include <string.h>
#include "cnn.h"
using namespace std;

struct  timeval  lsBegin, lsEnd, tsBegin, tsEnd, ToltsBegin, ToltsEnd;
long  c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, opDuration, inDuration, t1Duration;

//x,y是卷积核或者图片的组内坐标，channel代表是第几个，width，height代表组的宽核长，depth代表位深度。总的排列顺序是一个（height*channel）*witdh的大矩阵来存储所有的卷积核或者图片
int CNN::get_index(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

bool CNN::train()
{
	string outForwardFilePath = "/Users/starwish/Desktop/CNN/Release/log/rawforwardtimedata.csv";
	string outbackwardFilePath = "/Users/starwish/Desktop/CNN/Release/log/rawbackwardtimedata.csv";
	string outweightFilePath = "/Users/starwish/Desktop/CNN/Release/log/rawweighttimedata.csv";
	//创建输出文件
	ofstream outforwardFile(outForwardFilePath);
	ofstream outbackwardFile(outbackwardFilePath);
	ofstream outweightFile(outweightFilePath);
	if(outforwardFile)
		outforwardFile << "C1" << "," << "S2" << "," << "C3" << "," << "S4" << "," << "C5" << "," << "output" << "," << "total" << endl;
	else
		cout << "outforwardFile failed" << endl;
	if(outbackwardFile)
		outbackwardFile << "C1" << "," << "S2" << "," << "C3" << "," << "S4" << "," << "C5" << "," << "intput" << "," << "output" << "," << "total" << endl;
	else 
		cout << "outbackwardFile failed" << endl;

	std::cout << "training" << std::endl;
	int iter = 0;
	for (iter = 0; iter < num_epochs_CNN; iter++) {
		std::cout << "epoch: " << iter + 1 << std::endl;
		gettimeofday(&ToltsBegin, NULL);
		for (int i = 0; i < num_patterns_train_CNN; i++) {

			//1 输入模式顺传播
			data_single_image = data_input_train + i * num_neuron_input_CNN;
			data_single_label = data_output_train + i * num_neuron_output_CNN;

			memcpy(neuron_input, data_single_image, num_neuron_input_CNN*sizeof(float));
			if (i % 1000 == 0) {
				gettimeofday(&tsBegin, NULL);
				gettimeofday(&lsBegin, NULL);
				Forward_C1();
				gettimeofday(&lsEnd, NULL);
				c1Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);
				
				gettimeofday(&lsBegin, NULL);
				Forward_S2();
				gettimeofday(&lsEnd, NULL);
				s2Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Forward_C3();
				gettimeofday(&lsEnd, NULL);
				c3Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Forward_S4();
				gettimeofday(&lsEnd, NULL);
				s4Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);;

				gettimeofday(&lsBegin, NULL);
				Forward_C5();
				gettimeofday(&lsEnd, NULL);
				c5Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Forward_output();
				gettimeofday(&lsEnd, NULL);
				opDuration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				
				//outforwardFile << "C1" << "," << "S2" << "," << "C3" << "," << "S4" << "," << "C5" << "," << "output" << "," << "total" << endl;
				outforwardFile << c1Duration << "," << s2Duration << "," << c3Duration << "," << s4Duration << "," << c5Duration << "," << opDuration << "," << t1Duration << endl;
				printf("%dth --> forward| C1: %ld ms, S2: %ld ms, C3: %ld ms, S4: %ld ms, C5: %ld ms, output: %ld ms, total: %ld ms\n", i, c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, opDuration, t1Duration);
			}else{
				Forward_C1();
				Forward_S2();
				Forward_C3();
				Forward_S4();
				Forward_C5();
				Forward_output();
			}

			//2 输出误差逆传播
			if(i % 1000 == 0){
				gettimeofday(&tsBegin, NULL);

				gettimeofday(&lsBegin, NULL);
				Backward_output();
				gettimeofday(&lsEnd, NULL);
				inDuration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Backward_C5();
				gettimeofday(&lsEnd, NULL);
				c5Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);
				
				gettimeofday(&lsBegin, NULL);
				Backward_S4();
				gettimeofday(&lsEnd, NULL);
				s4Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Backward_C3();
				gettimeofday(&lsEnd, NULL);
				c3Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Backward_S2();
				gettimeofday(&lsEnd, NULL);
				s2Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Backward_C1();
				gettimeofday(&lsEnd, NULL);
				c1Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&lsBegin, NULL);
				Backward_input();
				gettimeofday(&lsEnd, NULL);
				inDuration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);

				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);

				//outbackwardFile << "C1" << "," << "S2" << "," << "C3" << "," << "S4" << "," << "C5" << "," << "intput" << "," << "output" << "," << "total" << endl;
				outbackwardFile << c1Duration << "," << s2Duration << "," << c3Duration << "," << s4Duration << "," << c5Duration << "," << inDuration << "," << opDuration << "," << t1Duration << endl;

				printf("%dth --> backward| C1: %ld ms, S2: %ld ms, C3: %ld ms, S4: %ld ms, C5: %ld ms, input: %ld ms output: %ld ms, total: %ld ms\n", i, c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, inDuration, opDuration, t1Duration);
			}
			else{
				Backward_output();
				Backward_C5();
				Backward_S4();
				Backward_C3();
				Backward_S2();
				Backward_C1();
				Backward_input();
			}
			
			if (i % 1000 == 0) {
				gettimeofday(&tsBegin, NULL);

				UpdateWeights();

				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				outweightFile << t1Duration << endl;
				printf("%dth --> UpdateWeights| %ld ms\n", i, t1Duration);
			}else{
				UpdateWeights();
			}
		}   //3 循环记忆训练
		//4 学习结果判别
		float accuracyRate = test();
		std::cout << ",    accuray rate: " << accuracyRate << std::endl;
		if (accuracyRate > accuracy_rate_CNN) {
			saveModelFile("cnn.model");
			std::cout << "generate cnn model" << std::endl;
			break;
		}
		saveModelFile("cnn.model");
		std::cout << "generate cnn model" << std::endl;
		gettimeofday(&ToltsEnd, NULL);
		t1Duration = 1000000L * (ToltsEnd.tv_sec - ToltsBegin.tv_sec) + (ToltsEnd.tv_usec - ToltsBegin.tv_usec);
		printf(" *******  every epoch : %ld s ^_^ \n", t1Duration/1000000L);
	}

	if (iter == num_epochs_CNN) {
		saveModelFile("cnn.model");
		std::cout << "generate cnn model" << std::endl;
	}

	outforwardFile.close();
	outbackwardFile.close();
	outweightFile.close();
    return true;
}

void CNN::update_weights_bias(const float* delta, float* e_weight, float* weight, int len)
{
	for (int i = 0; i < len; i++) {
		e_weight[i] += delta[i] * delta[i];
		weight[i] -= learning_rate_CNN * delta[i] / (std::sqrt(e_weight[i]) + eps_CNN);
	}
}

bool CNN::UpdateWeights()
{
	update_weights_bias(delta_weight_C1, E_weight_C1, weight_C1, len_weight_C1_CNN);
	update_weights_bias(delta_bias_C1, E_bias_C1, bias_C1, len_bias_C1_CNN);

	update_weights_bias(delta_weight_S2, E_weight_S2, weight_S2, len_weight_S2_CNN);
	update_weights_bias(delta_bias_S2, E_bias_S2, bias_S2, len_bias_S2_CNN);

	update_weights_bias(delta_weight_C3, E_weight_C3, weight_C3, len_weight_C3_CNN);
	update_weights_bias(delta_bias_C3, E_bias_C3, bias_C3, len_bias_C3_CNN);

	update_weights_bias(delta_weight_S4, E_weight_S4, weight_S4, len_weight_S4_CNN);
	update_weights_bias(delta_bias_S4, E_bias_S4, bias_S4, len_bias_S4_CNN);

	update_weights_bias(delta_weight_C5, E_weight_C5, weight_C5, len_weight_C5_CNN);
	update_weights_bias(delta_bias_C5, E_bias_C5, bias_C5, len_bias_C5_CNN);

	update_weights_bias(delta_weight_output, E_weight_output, weight_output, len_weight_output_CNN);
	update_weights_bias(delta_bias_output, E_bias_output, bias_output, len_bias_output_CNN);

	return true;
}

float CNN::test()
{
	int count_accuracy = 0;

	for (int num = 0; num < num_patterns_test_CNN; num++) {
		data_single_image = data_input_test + num * num_neuron_input_CNN;
		data_single_label = data_output_test + num * num_neuron_output_CNN;

		memcpy(neuron_input, data_single_image, num_neuron_input_CNN*sizeof(float));

		Forward_C1();
		Forward_S2();
		Forward_C3();
		Forward_S4();
		Forward_C5();
		Forward_output();

		int pos_t = -1;
		int pos_y = -2;
		float max_value_t = -9999.0;
		float max_value_y = -9999.0;

		for (int i = 0; i < num_neuron_output_CNN; i++) {
			if (neuron_output[i] > max_value_y) {
				max_value_y = neuron_output[i];
				pos_y = i;
			}

			if (data_single_label[i] > max_value_t) {
				max_value_t = data_single_label[i];
				pos_t = i;
			}
		}

		if (pos_y == pos_t) {
			++count_accuracy;
		}
		// Copper Sleep(1);
	}
	return (count_accuracy * 1.0 / num_patterns_test_CNN);
}