/*
 * @Author: your name
 * @Date: 2017-05-31 10:08:27
 * @LastEditTime: 2020-05-31 02:40:37
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /CNN/srcSIMD/math_functions.cpp
 */ 
/*
 * math_functions.cpp
 *
 *  Created on: Apr 29, 2017
 *      Author: copper
 */
#include <x86intrin.h>
#include "cnn.h"
using namespace std;


float CNN::activation_function_tanh(float x)
{
	float ep = std::exp(x);
	float em = std::exp(-x);

	return (ep - em) / (ep + em);
}

float CNN::activation_function_tanh_derivative(float x)
{
	return (1.0 - x * x);
}

__m256 CNN::m256_activation_function_tanh_derivative(__m256 x){
	__m256 num = _mm256_set1_ps(1.0);
	__m256 temp = _mm256_mul_ps(x, x);
	return _mm256_sub_ps(num, temp);
}

float CNN::activation_function_identity(float x)
{
	return x;
}

float CNN::activation_function_identity_derivative(float x)
{
	return 1;
}

float CNN::loss_function_mse(float y, float t)
{
	return (y - t) * (y - t) / 2;
}

float CNN::loss_function_mse_derivative(float y, float t)
{
	return (y - t);
}

void CNN::loss_function_gradient(const float* y, const float* t, float* dst, int len)
{
	for (int i = 0; i < len; i++) {
		dst[i] = loss_function_mse_derivative(y[i], t[i]);
	}
}

float CNN::dot_product(const float* s1, const float* s2, int len)
{
	float result = 0.0;

	for (int i = 0; i < len; i++) {
		result += s1[i] * s2[i];
	}

	return result;
}

bool CNN::muladd(const float* src, float c, int len, float* dst)
{
	for (int i = 0; i < len; i++) {
		dst[i] += (src[i] * c);
	}
	return true;
}

void CNN::init_variable(float* val, float c, int len)
{
	for (int i = 0; i < len; i++) {
		val[i] = c;
	}
}






