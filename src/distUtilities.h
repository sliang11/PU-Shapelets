#pragma once
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define INF 1e7
#define SQUARE (term1 - term2) * (term1 - term2)

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

__device__ void getMuSigma(double &mu, double &sigma, double *ts, const int tsLen){
	double term, s, s2;
	s = s2 = 0;
	for (int i = 0; i < tsLen; i++){
		term = ts[i];
		s += term;
		s2 += term * term;
	}
	mu = s / tsLen;
	sigma = s2 / tsLen > mu * mu ? sqrt(s2 / tsLen - mu * mu) : 1;
}

__device__ void getNormalizedTerm(double &term, double mu, double sigma){
	term = (term - mu) / sigma;
}

//Reference: Doruk Sart, Abdullah Mueen, Walid A. Najjar, Eamonn J. Keogh, Vit Niennattrakul:
//Accelerating Dynamic Time Warping Subsequence Search with GPUs and FPGAs. ICDM 2010: 1001-1006.
__device__ void dtw(double &dist, double *ts, double *query, double *tmp,
	const int tsLen, const int queryLen, const int maxWarp){

	double term1, term2, mu1, sigma1, mu2, sigma2;
	getMuSigma(mu1, sigma1, ts, tsLen);
	getMuSigma(mu2, sigma2, query, queryLen);

	if (tsLen == 1 && queryLen == 1){
		term1 = ts[0];
		getNormalizedTerm(term1, mu1, sigma1);
		term2 = query[0];
		getNormalizedTerm(term2, mu2, sigma2);
		dist = sqrt(SQUARE);
	}
	else{
		int i;
		for (i = 0; i < 2 * (2 * maxWarp + 2); i++)
			tmp[i] = INF;

		term1 = ts[0];
		getNormalizedTerm(term1, mu1, sigma1);
		term2 = query[0];
		getNormalizedTerm(term2, mu2, sigma2);
		tmp[maxWarp] = SQUARE;

		int lowerRight = maxWarp < tsLen - 1 ? 2 * maxWarp : tsLen + maxWarp - 1;
		for (i = maxWarp + 1; i <= lowerRight; i++){
			term1 = ts[i - maxWarp];
			getNormalizedTerm(term1, mu1, sigma1);
			term2 = query[0];
			getNormalizedTerm(term2, mu2, sigma2);
			tmp[i] = tmp[i - 1] + SQUARE;
		}

		if (queryLen == 1)
			dist = sqrt(tmp[lowerRight]);
		else{
			double selected;
			int lower, upper, id, j, t, upperRight;
			lower = 0;
			upper = 2 * maxWarp + 2;
			j = 1;
			while (1){
				id = j - maxWarp;
				term2 = query[j];
				getNormalizedTerm(term2, mu2, sigma2);
				if (id < 0)
					tmp[upper + 1] = INF;
				else{
					selected = tmp[lower] < tmp[lower + 1] ? tmp[lower] : tmp[lower + 1];
					term1 = ts[id];
					getNormalizedTerm(term1, mu1, sigma1);
					tmp[upper + 1] = selected + SQUARE;
				}
				upperRight = maxWarp < tsLen - j ? 2 * maxWarp + 1 : tsLen + maxWarp - j;
				for (i = 2; i <= upperRight; i++){
					id = i + j - maxWarp - 1;
					if (id < 0)
						tmp[upper + i] = INF;
					else{
						selected = tmp[upper + i - 1] < tmp[lower + i] ? tmp[upper + i - 1] : tmp[lower + i];
						selected = selected < tmp[lower + i - 1] ? selected : tmp[lower + i - 1];
						term1 = ts[id];
						getNormalizedTerm(term1, mu1, sigma1);
						tmp[upper + i] = selected + SQUARE;
					}
				}

				if (j == queryLen - 1)
					break;

				t = lower;
				lower = upper;
				upper = t;

				for (i = 0; i < 2 * maxWarp + 1; i++)
					tmp[lower + i] = tmp[lower + i + 1];
				tmp[lower + 2 * maxWarp + 1] = INF;
				j++;
			}
			dist = sqrt(tmp[upper + upperRight]);
		}
	}
}

//Sequential computation of DTW can be painfully slow. Therefore, we apply GPU acceleration.
//Reference: Doruk Sart, Abdullah Mueen, Walid A. Najjar, Eamonn J. Keogh, Vit Niennattrakul:
//Accelerating Dynamic Time Warping Subsequence Search with GPUs and FPGAs. ICDM 2010: 1001-1006.
__global__ void getPDists_DTW(double *trainTss_in, double *testTss_in, double *pDists_out,
	const int numTrain, const int numTest, const int tsLen, const double bandwidth){

	int blockSize = blockDim.x;
	int numThreadsPerGrid = gridDim.x * blockSize;
	int numQueriesPerThread = ceil((double)numTest / numThreadsPerGrid);
	int maxWarp = ceil(tsLen * bandwidth);
	double *tmp = new double[2 * (2 * maxWarp + 2)];
	double *query, *ts;
	int i, j, next;
	for (i = 0; i < numQueriesPerThread; i++){
		next = (blockIdx.x * blockSize + threadIdx.x) * numQueriesPerThread + i;
		if (next >= numTest)
			break;
		query = &testTss_in[next * tsLen];
		for (j = 0; j < numTrain; j++){
			ts = &trainTss_in[j * tsLen];
			dtw(pDists_out[next * numTrain + j], ts, query, tmp, tsLen, tsLen, maxWarp);
		}
	}
	delete tmp;
}