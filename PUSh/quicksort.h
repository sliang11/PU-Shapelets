#pragma once

#include "stack.h"

template <class T>
void swapElm(T &x, T &y){
	T tmp = x;
	x = y;
	y = tmp;
}

template <class T>
int partition(T *array, int *order, int low, int high){
	T x = array[order[high]];
	int i = low - 1, j;
	for (j = low; j < high; j++){
		if (array[order[j]] <= x){	//this can run into issues due to loss of precision
			i++;
			swapElm(order[i], order[j]);
		}
	}
	swapElm(order[i + 1], order[high]);
	return i + 1;
}

template <class T>
void getOrder(T *array, int *order, int low, int high, bool isAscend){
	int originalLow = low;
	int originalHigh = high;
	for (int i = low; i <= high; i++)
		order[i] = i;

	stackElm *top, *poppedElm;
	top = initStack(low, high);
	int numElm = 1;
	while (numElm){
		top = pop(top, low, high);
		numElm--;

		int pivot = partition(array, order, low, high);
		if (low < pivot - 1){
			top = push(top, low, pivot - 1);
			numElm++;
		}
		if (high > pivot + 1){
			top = push(top, pivot + 1, high);
			numElm++;
		}
	}

	if (!isAscend){
		low = originalLow;
		high = originalHigh;
		while (high > low)
			swapElm(order[low++], order[high--]);
	}
}
