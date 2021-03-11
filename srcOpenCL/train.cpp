#include "cnn.h"
#include <sstream>
#include <fstream>
#include <iostream>
#include <string.h>
using namespace std;
struct  timeval  lsBegin, lsEnd, tsBegin, tsEnd, ToltsBegin, ToltsEnd;
long  c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, opDuration, inDuration, t1Duration;

int CNN::get_index(int x, int y, int channel, int width, int height, int depth)
{
	assert(x >= 0 && x < width);
	assert(y >= 0 && y < height);
	assert(channel >= 0 && channel < depth);
	return (height * channel + y) * width + x;
}

bool CNN::train()
{
	string outForwardFilePath = "/Users/starwish/Desktop/CNN/Release/OpenCLlog/rawforwardtimedata.csv";
	string outbackwardFilePath = "/Users/starwish/Desktop/CNN/Release/OpenCLlog/rawbackwardtimedata.csv";
	string outweightFilePath = "/Users/starwish/Desktop/CNN/Release/OpenCLlog/rawweighttimedata.csv";
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

			if (i % 1000 == 0) {
				gettimeofday(&lsBegin, NULL);
				gettimeofday(&tsBegin, NULL);
			}
			//1 输入模式顺传播
			data_single_image = data_input_train + i * num_neuron_input_CNN;
			data_single_label = data_output_train + i * num_neuron_output_CNN;

			// memcpy(neuron_input, data_single_image, num_neuron_input_CNN*sizeof(float));

			// printf("Forward C1 %d:\n",i);
			Forward_C1(i * num_neuron_input_CNN,cl_data_input_train);
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Forward_S2();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				s2Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}
			
			Forward_C3();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c3Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}
			
			Forward_S4();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				s4Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}
			
			Forward_C5();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c5Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Forward_output();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				gettimeofday(&lsEnd, NULL);
				opDuration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				t1Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);
				outforwardFile << c1Duration << "," << s2Duration << "," << c3Duration << "," << s4Duration << "," << c5Duration << "," << opDuration << "," << t1Duration << endl;
				printf("%dth --> forward| C1: %ld ms, S2: %ld ms, C3: %ld ms, S4: %ld ms, C5: %ld ms, output: %ld ms, total: %ld ms\n", i, c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, opDuration, t1Duration);
				gettimeofday(&tsBegin, NULL);
			}

			//2 输出误差逆传播
			// printf("Backward C1 %d:\n",i);
			Backward_output(i * 10);
			if (i % 1000 == 0) {
				gettimeofday(&lsBegin, NULL);
				gettimeofday(&tsEnd, NULL);
				opDuration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}			

			Backward_C5();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c5Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Backward_S4();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				s4Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}			

			Backward_C3();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c3Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Backward_S2();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				s2Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Backward_C1();
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				c1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				gettimeofday(&tsBegin, NULL);
			}

			Backward_input(i * num_neuron_input_CNN);
			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				gettimeofday(&lsEnd, NULL);
				inDuration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				t1Duration = 1000000L * (lsEnd.tv_sec - lsBegin.tv_sec) + (lsEnd.tv_usec - lsBegin.tv_usec);
				outbackwardFile << c1Duration << "," << s2Duration << "," << c3Duration << "," << s4Duration << "," << c5Duration << "," << inDuration << "," << opDuration << "," << t1Duration << endl;
				printf("%dth --> backward| C1: %ld ms, S2: %ld ms, C3: %ld ms, S4: %ld ms, C5: %ld ms, input: %ld ms output: %ld ms, total: %ld ms\n", i, c1Duration, s2Duration, c3Duration, s4Duration, c5Duration, inDuration, opDuration, t1Duration);
				gettimeofday(&tsBegin, NULL);
			}

			UpdateWeights();

			if (i % 1000 == 0) {
				gettimeofday(&tsEnd, NULL);
				t1Duration = 1000000L * (tsEnd.tv_sec - tsBegin.tv_sec) + (tsEnd.tv_usec - tsBegin.tv_usec);
				outweightFile << t1Duration << endl;
				printf("%dth --> UpdateWeights| %ld ms\n", i, t1Duration);
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
	for(int i=0;i<FORWARD_NUM;i++){
		if (clSetKernelArg(Update_weights, 0, sizeof(cl_mem), &Backward_bias[5-i]) ||
			clSetKernelArg(Update_weights, 1, sizeof(cl_mem), &Update_bias[i]) ||
			clSetKernelArg(Update_weights, 2, sizeof(cl_mem), &Forward_bias[i]) != CL_SUCCESS)
		{
			printf("Unable to set kernel Update_weights bias %d arguments.\n",i);
			return false;
		}
		size_t global[1] = {for_mem_bw_len[i][0]};
		err = clEnqueueNDRangeKernel(command_queue, Update_weights, 1, NULL, global, NULL /*local*/, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Unable to enqueue kernel Update_weights bias %d. Error Code=%d\n",i, err); 
			return false;
		}

		if (clSetKernelArg(Update_weights, 0, sizeof(cl_mem), &Backward_weight[5-i]) ||
			clSetKernelArg(Update_weights, 1, sizeof(cl_mem), &Update_weight[i]) ||
			clSetKernelArg(Update_weights, 2, sizeof(cl_mem), &Forward_weight[i]) != CL_SUCCESS)
		{
			printf("Unable to set kernel Update_weights weight %d arguments.\n",i);
			return false;
		}

		size_t global_[1] = {for_mem_bw_len[i][1]};

		err = clEnqueueNDRangeKernel(command_queue, Update_weights, 1, NULL, global_, NULL /*local*/, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			printf("Unable to enqueue kernel Update_weights weight %d. Error Code=%d\n",i, err); 
			return false;
		}
	}
	clFinish(command_queue);
	return true;
}

float CNN::test()
{
	int count_accuracy = 0;


	for (int num = 0; num < num_patterns_test_CNN; num++) {
		data_single_image = data_input_test + num * num_neuron_input_CNN;
		data_single_label = data_output_test + num * num_neuron_output_CNN;

		Forward_C1(num * num_neuron_input_CNN, cl_data_input_test);
		Forward_S2();
		Forward_C3();
		Forward_S4();
		Forward_C5();
		Forward_output();

		int pos_t = -1;
		int pos_y = -2;
		float max_value_t = -9999.0;
		float max_value_y = -9999.0;

		clEnqueueReadBuffer(command_queue, Forward_out_mem, CL_TRUE, 0, num_neuron_output_CNN*sizeof(cl_float), neuron_output, 0, NULL, NULL);

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




