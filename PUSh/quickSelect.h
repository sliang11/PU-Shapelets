#pragma once

template <class T>
void swapElm(T &x, T &y, T &tmp){
	tmp = x;
	x = y;
	y = tmp;
}

template <class T>
int partition(T *array, int *order, int low, int high, int seed, int stride){

	int tmp;
	swapElm(order[seed * stride], order[high * stride], tmp);
	T x = array[order[high * stride] * stride];
	int i = low - 1;
	for (int j = low; j < high; j++){
		if (array[order[j * stride] * stride] <= x){
			i++;
			swapElm(order[i * stride], order[j * stride], tmp);
		}
	}
	swapElm(order[(i + 1) * stride], order[high * stride], tmp);
	return i + 1;
}

template <class T>
T select(T *array, int *order, int target, int low, int high, float *seeds, int stride){
	for (int i = low; i <= high; i++)
		order[i * stride] = i;

	if (low == high)
		return array[order[low * stride] * stride];
	else{
		int seedId = 0;
		while (1){
			int seed = low + (high - low) * seeds[seedId++];
			int pivot = partition(array, order, low, high, seed, stride);
			int k = pivot - low + 1;
			if (target == k){
				return array[order[pivot * stride] * stride];
			}
			else if (target < k)
				high = pivot - 1;
			else{
				low = pivot + 1;
				target -= k;
			}

			if (low == high){
				return array[order[low * stride] * stride];
			}
		}
	}
}