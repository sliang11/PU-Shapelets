#pragma once
#include <math.h>

double ED(double *vec1, double *vec2, int len) {
	double ed = 0;
	for (int i = 0; i < len; i++) {
		ed += pow(vec1[i] - vec2[i], 2);
	}
	ed = sqrt(ed);
	return ed;
}

double ED2_early(double *vec1, double *vec2, int len, double th_ed2) {
	double ed2 = 0;
	for (int i = 0; i < len; i++) {
		ed2 += pow(vec1[i] - vec2[i], 2);
		if (ed2 >= th_ed2)
			break;
	}
	return ed2;
}