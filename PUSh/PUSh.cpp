//PU-Shapelets: Pattern Ensemble (PE) + Average Shapelet Precision Maximization (ASPM)

#include "utilities.h"
#include "quicksort.h"
#include "quickSelect.h"
#include "calcUtilities.h"
#include "distUtilities.h"
#include "evaluationUtilities.h"
#include <io.h>
#include <direct.h>
#include <time.h>
#include <string>
#include <vector>

#define INF 1e6
#define MAX_CHAR 10

//pre-calculations for getNN
void calcStats(double *tss, double *sumMtx, double *sum2Mtx, int numTrain, int tsLen) {

	double *ts;
	double term, s, s2;
	int idx;
	for (int i = 0; i < numTrain; i++) {
		ts = &tss[i * tsLen];
		sumMtx[i * (tsLen + 1)] = sum2Mtx[i * (tsLen + 1)] = 0;
		s = s2 = 0;
		for (int j = 0; j < tsLen; j++) {
			term = ts[j];
			s += term;
			s2 += term * term;

			idx = i * (tsLen + 1) + j + 1;	//padding zeros
			sumMtx[idx] = s;
			sum2Mtx[idx] = s2;
		}
	}
}

//getting the subsequence matching distances using the method proposed in
//Abdullah Mueen, Eamonn J. Keogh, Neal E. Young: 
//Logical-shapelets: an expressive primitive for time series classification. KDD 2011 : 1154-1162.
void getDistMtx(double *distMtx, double *tss, double *sumMtx, double *sum2Mtx, double *dotPrMtx, int numTrain, int tsLen, int sLen) {

	//calc dot products
	int numShPerTs = tsLen - sLen + 1;
	int numSh = numTrain * numShPerTs;

	double *ts1, *ts2;
	double *sumVec_1, *sumVec_2, *sum2Vec_1, *sum2Vec_2, s1, s2, s1_2, s2_2, var1, var2, sigma1, sigma2, dotPr, corr, curDist, nnDist;
	int i, j, u, v, idx, nnLoc;

	//initiate
	memset(distMtx, 0, numTrain * numShPerTs * numTrain * sizeof(double));
	for (i = 0; i < numTrain; i++) {
		ts1 = &tss[i * tsLen];
		sumVec_1 = &sumMtx[i * (tsLen + 1)];
		sum2Vec_1 = &sum2Mtx[i * (tsLen + 1)];
		for (j = 0; j < numTrain; j++) {
			if (j == i)
				continue;

			ts2 = &tss[j * tsLen];
			sumVec_2 = &sumMtx[j * (tsLen + 1)];
			sum2Vec_2 = &sum2Mtx[j * (tsLen + 1)];

			//dot products
			memset(dotPrMtx, 0, (tsLen + 1) * (tsLen + 1) * sizeof(double));
			for (u = 0; u < tsLen; u++) {

				//horizontal
				for (v = u; v < tsLen; v++) {
					idx = (u + 1) * (tsLen + 1) + v + 1;
					dotPrMtx[idx] = dotPrMtx[idx - tsLen - 2] + (double)ts1[u] * (double)ts2[v];
				}

				//vertical
				for (v = u + 1; v < tsLen; v++) {
					idx = (v + 1) * (tsLen + 1) + u + 1;
					dotPrMtx[idx] = dotPrMtx[idx - tsLen - 2] + (double)ts1[v] * (double)ts2[u];
				}
			}

			for (u = 0; u < numShPerTs; u++) {
				nnDist = INF;
				s1 = sumVec_1[u + sLen] - sumVec_1[u];
				s1_2 = sum2Vec_1[u + sLen] - sum2Vec_1[u];
				var1 = s1_2 / sLen - (s1 / sLen) * (s1 / sLen);
				sigma1 = var1 <= 0 ? 0 : sqrt(var1);


				for (v = 0; v < numShPerTs; v++) {
					s2 = sumVec_2[v + sLen] - sumVec_2[v];
					s2_2 = sum2Vec_2[v + sLen] - sum2Vec_2[v];
					var2 = s2_2 / sLen - (s2 / sLen) * (s2 / sLen);
					sigma2 = var2 <= 0 ? 0 : sqrt(var2);

					if (sigma1 == 0 && sigma2 == 0){
						curDist = INF;
					}
					else if (sigma1 == 0 || sigma2 == 0){
						curDist = INF;
					}
					else{
						dotPr = dotPrMtx[(u + sLen) * (tsLen + 1) + v + sLen]
							- dotPrMtx[u * (tsLen + 1) + v];

						corr = (dotPr - s1 * s2 / sLen) / (sLen * sigma1 * sigma2);
						if (corr > 1) {
							corr = 1;
						}
						curDist = sqrt(2 * (1 - corr));
					}

					if (curDist < nnDist) {
						nnDist = curDist;
						nnLoc = v;
					}
				}
				distMtx[(i * numShPerTs + u) * numTrain + j] = nnDist;
			}
		}
	}
}

void seedByIterToPreLabels(int *preLabels, int *seedByIter, int numTrain, int preNumP) {
	memset(preLabels, 0, numTrain * sizeof(int));
	for (int i = 0; i < preNumP; i++) {
		preLabels[seedByIter[i]] = 1;
	}
}

//Used in getIndMinDiff()
void getIndMinDiffCandidate(int &indCand, double &diffCand, int pInd, double *distVec, int *order, int *argOrder, int *isSeed, const int numTrain) {

	int pIndOnOrderline = argOrder[pInd];
	double pivotVal = distVec[pInd];

	double diffLeft;
	int leftIndOriginal;
	if (pIndOnOrderline == 0) {
		diffLeft = INF;
	}
	else {
		leftIndOriginal = order[pIndOnOrderline - 1];
		if (isSeed[leftIndOriginal])
			diffLeft = INF;
		else
			diffLeft = pivotVal - distVec[leftIndOriginal];
	}

	double diffRight;
	int rightIndOriginal;
	if (pIndOnOrderline == numTrain - 1) {
		diffRight = INF;
	}
	else {
		rightIndOriginal = order[pIndOnOrderline + 1];
		if (isSeed[rightIndOriginal])
			diffRight = INF;
		else
			diffRight = distVec[rightIndOriginal] - pivotVal;
	}

	if (diffLeft == INF && diffRight == INF) {
		diffCand = INF;
		indCand = -1;
	}
	else if (diffLeft < diffRight) {
		diffCand = diffLeft;
		indCand = leftIndOriginal;
	}
	else {
		diffCand = diffRight;
		indCand = rightIndOriginal;
	}

}

//Propagating 1NN
void getIndMinDiff(int &indMinDiff, double *distVec, int *order, int *argOrder, int *seed, int *isSeed, const int numTrain, const int numPLabeled) {

	if (distVec[order[1]] == INF){
		indMinDiff = -1;
	}
	else{
		//Fast nearest neighbor discovery
		int indCand;
		double diffCand, minDiff_P = INF;
		for (int i = 0; i < numPLabeled; i++) {
			getIndMinDiffCandidate(indCand, diffCand, seed[i], distVec, order, argOrder, isSeed, numTrain);
			if (minDiff_P > diffCand) {
				minDiff_P = diffCand;
				indMinDiff = indCand;
			}
		}
	}
}

//get shapelet precision and gap at the orderline split point
void getShapeletPrecision(double &sp, double &spGap, double *distVec, int *order, int *preLabels_g,
	int *preLabels_l, int *preLabels_r, int numTrain, int numPLabeled, int iter) {

	if (distVec[order[1]] == INF){
		sp = -1;
		spGap = -1;
	}
	else{
		memset(preLabels_l, 0, numTrain * sizeof(int));
		memset(preLabels_r, 0, numTrain * sizeof(int));
		int curPreNumP = numPLabeled + iter + 1;
		for (int i = 0; i < curPreNumP; i++) {
			preLabels_l[order[i]] = 1;
			preLabels_r[order[numTrain - i - 1]] = 1;
		}

		double sp_l = precision(preLabels_g, preLabels_l, numTrain);
		double sp_r = precision(preLabels_g, preLabels_r, numTrain);
		if (sp_l > sp_r) {
			sp = sp_l;
			spGap = distVec[order[curPreNumP]] - distVec[order[curPreNumP - 1]];	//We use a different deinition of gap than the Logical Shapelets paper.
		}
		else {
			sp = sp_r;
			spGap = distVec[order[numTrain - curPreNumP]] - distVec[order[numTrain - curPreNumP - 1]];	//We use a different deinition of gap than the Logical Shapelets paper.
		}
	}
}

//get the rankings of (assumed) shapelets by primary and secondary metrics
void getShRanking(int *shRanking, double *primaryComplete, double *secondaryComplete, int *tmpOrder, float *randSeeds, std::vector<double> candPrimary,
	std::vector<double> candSecondary, std::vector<int> candShIds, std::vector<int> orderPrimary, std::vector<int> orderSecondary,
	const int numShTotal, const int maxNumSh) {

	//coarse ranking by primary
	double primary, primaryTh = select(primaryComplete, tmpOrder, numShTotal - maxNumSh + 1, 0, numShTotal - 1, randSeeds, 1);
	candPrimary.clear(); candShIds.clear();
	for (int i = 0; i < numShTotal; i++) {
		primary = primaryComplete[i];
		if (primary >= primaryTh) {
			candPrimary.push_back(primary);
			candShIds.push_back(i);
		}
	}

	int numPrimary = candPrimary.size();
	if (numPrimary == 1)
		shRanking[0] = candShIds[0];
	else {
		if (orderPrimary.size() < numPrimary)
			orderPrimary.resize(numPrimary);
		getOrder(&candPrimary[0], &orderPrimary[0], 0, numPrimary - 1, false);

		//fine ranking by secondary
		int cnt, offset, numSecondary, breakFlg = 0;
		cnt = offset = 0;
		double curPrimary = candPrimary[orderPrimary[0]];
		candSecondary.clear(); candSecondary.push_back(secondaryComplete[candShIds[orderPrimary[0]]]);
		for (int i = 1; i < numPrimary; i++) {
			primary = candPrimary[orderPrimary[i]];
			if (curPrimary != primary || (curPrimary == primary && i == numPrimary - 1)) {
				if (curPrimary == primary)
					candSecondary.push_back(secondaryComplete[candShIds[orderPrimary[i]]]);
				else
					curPrimary = primary;

				//sort the current secondaries
				numSecondary = candSecondary.size();
				if (orderSecondary.size() < numSecondary)
					orderSecondary.resize(numSecondary);
				getOrder(&candSecondary[0], &orderSecondary[0], 0, numSecondary - 1, false);
				candSecondary.clear();

				//add to the current shRankings
				for (int j = 0; j < numSecondary; j++) {
					shRanking[cnt] = candShIds[orderPrimary[offset + orderSecondary[j]]];
					cnt++;
					if (cnt == maxNumSh) {	//break when enough assumed shapelets are selected
						breakFlg = 1;
						break;
					}
				}
				offset += numSecondary;
			}
			if (breakFlg)
				break;

			if (i == numPrimary - 1) {
				shRanking[cnt] = candShIds[orderPrimary[i]];
			}
			else
				candSecondary.push_back(secondaryComplete[candShIds[orderPrimary[i]]]);
		}
	}
}

//Get shapelet information
void getShInfo(int &tsId, int &pos, int &sLen, int targetShIdTotal,
	const int *numShAll, const int numSLens, const int minSLen, const int maxSLen, const int sLenStep, const int numTrain, const int tsLen) {

	int cnt = 0;
	sLen = minSLen;
	int numSh;
	for (int i = 0; i < numSLens; i++) {
		numSh = numShAll[i];
		if (cnt + numSh > targetShIdTotal) {
			break;
		}
		sLen += sLenStep;
		cnt += numSh;
	}

	int targetShId = targetShIdTotal - cnt;
	int numShPerTs = tsLen - sLen + 1;
	tsId = (targetShId + 1) / numShPerTs - 1;
	pos = targetShId - (tsId + 1) * numShPerTs;
	if (pos == -1)
		pos = numShPerTs - 1;
	else
		tsId++;
}

int main(int argc, char **argv) {

	//parameter settings
	if (argc < 7)
		exit(1);
	std::string datasetName = argv[1];
	const int numTrain = atoi(argv[2]);
	const int numP = atoi(argv[3]);
	const int numPLabeled = atoi(argv[4]);
	const int tsLen = atoi(argv[5]);
	const int seedId = atoi(argv[6]);
	const int maxNumSh = argc > 7 ? atoi(argv[7]) : 200;
	const int minNumIters = argc > 8 ? atoi(argv[8]) : (numP >= 10 ? 5 : 1);
	const int maxNumIters = argc > 9 ? atoi(argv[9]) : numTrain * 2 / 3 - numPLabeled;
	const int minSLen = argc > 10 ? atoi(argv[10]) : 10;
	const int maxSLen = argc > 11 ? atoi(argv[11]) : tsLen;
	const int sLenStep = argc > 12 ? atoi(argv[12]) : ((maxSLen - minSLen) / 10 > 0 ? (maxSLen - minSLen) / 10 : 1);
	const std::string path = argc > 13 ? argv[13] : "..\\sample_data\\" + datasetName;
	const std::string outputPath = argc > 14 ? argv[14] : "..\\results";

	char s_seedId[MAX_CHAR], s_maxNumSh[MAX_CHAR], s_minNumIters[MAX_CHAR], s_maxNumIters[MAX_CHAR];
	_itoa(maxNumSh, s_maxNumSh, 10);
	_itoa(seedId, s_seedId, 10);
	_itoa(minNumIters, s_minNumIters, 10);
	_itoa(maxNumIters, s_maxNumIters, 10);

	//load time series
	long long trainTssBytes = numTrain * tsLen * sizeof(double);
	double *trainTss = (double*)malloc(trainTssBytes);
	long long trainLabelsBytes = numTrain * sizeof(int);
	int *trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, path, datasetName, "TRAIN", numTrain, tsLen);
	relabel(trainLabels, numTrain, 1);

	//load seeds
	const int numSeeds = numP > 10 ? 10 : numP;
	if (seedId >= numSeeds)
		exit(1);
	int *seeds = (int*)malloc(numSeeds * numPLabeled * sizeof(int));
	std::string fName = path + "\\seeds_" + datasetName + ".txt";
	importMatrix(seeds, fName, numSeeds, numPLabeled, 1);
	for (int i = 0; i < numSeeds * numPLabeled; i++)
		seeds[i]--;	//matlab -> c
	int *seed = (int *)malloc(numPLabeled * sizeof(int));
	memcpy(seed, seeds + seedId * numPLabeled, numPLabeled * sizeof(int));
	free(seeds);

	//start timer
	clock_t tic = clock();

	//number of shapelet candidates by subsequence length
	int numSLens = (maxSLen - minSLen) / sLenStep + 1;
	int *numShAll = (int *)malloc(numSLens * sizeof(int));
	int sLen, cnt = 0;
	int numShTotal = 0;
	for (sLen = minSLen; sLen <= maxSLen; sLen += sLenStep) {
		numShAll[cnt] = numTrain * (tsLen - sLen + 1);
		numShTotal += numShAll[cnt];
		cnt++;
	}

	//distance computations
	double *sumMtx = (double *)malloc(numTrain * (tsLen + 1) * sizeof(double));
	double *sum2Mtx = (double *)malloc(numTrain * (tsLen + 1) * sizeof(double));
	calcStats(trainTss, sumMtx, sum2Mtx, numTrain, tsLen);

	double *dotPrMtx = (double *)malloc((tsLen + 1) * (tsLen + 1) * sizeof(double));
	double *distMtxComplete = (double *)malloc(numShTotal * numTrain * sizeof(double));
	double *distMtx = distMtxComplete;
	cnt = 0;
	for (sLen = minSLen; sLen <= maxSLen; sLen += sLenStep) {
		getDistMtx(distMtx, trainTss, sumMtx, sum2Mtx, dotPrMtx, numTrain, tsLen, sLen);
		distMtx += numShAll[cnt] * numTrain;
		cnt++;
	}
	free(sumMtx);
	free(sum2Mtx);
	free(dotPrMtx);

	clock_t toc_dist = clock();	//single out distance computation time

	//the orderlines
	int *ordersComplete = (int *)malloc(numShTotal * numTrain * sizeof(int));	//<idxOrderline, idxOriginal>
	int *argOrdersComplete = (int *)malloc(numShTotal * numTrain * sizeof(int));	//<idxOriginal, idxOrderline>
	int *order = ordersComplete;
	int *argOrder = argOrdersComplete;
	double *distVec = distMtxComplete;
	int shIdTotal;
	for (shIdTotal = 0; shIdTotal < numShTotal; shIdTotal++) {
		getOrder(distVec, order, 0, numTrain - 1, true);
		for (int i = 0; i < numTrain; i++) {
			argOrder[order[i]] = i;
		}
		distVec += numTrain;
		order += numTrain;
		argOrder += numTrain;
	}

	//positive labeled examples and the predicted labels
	int *seedByIter = (int *)malloc((numPLabeled + maxNumIters) * sizeof(int));
	memcpy(seedByIter, seed, numPLabeled * sizeof(int));
	int *preLabels = (int *)malloc(numTrain * sizeof(int));
	seedByIterToPreLabels(preLabels, seedByIter, numTrain, numPLabeled);

	//vote counter
	int *votes = (int *)malloc(numTrain * sizeof(int));
	memset(votes, 0, numTrain * sizeof(int));
	for (int i = 0; i < numPLabeled; i++) {
		votes[seed[i]] = -1;
	}

	//For stopping criterion
	double cumSp;
	double *avgSpByIter = (double *)malloc(maxNumSh * maxNumIters * sizeof(double));

	//for shapelet precision and information gain calculation
	double *primaryComplete = (double *)malloc(numShTotal * sizeof(double));
	double *secondaryComplete = (double *)malloc(numShTotal * sizeof(double));
	double splitPt;
	int numByLabel[]{ numTrain - numPLabeled, numPLabeled }, numByLabelIn[2], numByLabelOut[2];

	double spTh;
	int *shRanking = (int *)malloc(maxNumSh * sizeof(int));
	std::vector<double> candPrimary, candSecondary;
	std::vector<int> candShIds, orderByPrimary, orderBySecondary;

	int *tmpOrder = (int *)malloc(numShTotal * sizeof(int));
	float *randSeeds = (float *)malloc(numShTotal * sizeof(float));	//randomized seeds for quick select
	srand((int)time(0));
	for (int i = 0; i < numShTotal; i++)
		randSeeds[i] = (float)rand() / RAND_MAX;

	//for sp calculation
	int *preLabels_l = (int *)malloc(numTrain * sizeof(int));
	int *preLabels_r = (int *)malloc(numTrain * sizeof(int));

	int iter, indMinDiff, nextVotes, nextP, ind;
	for (iter = 0; iter < maxNumIters; iter++) {

		//initialize the vote counters
		for (int i = 0; i < numTrain; i++) {
			if (votes[i] != -1)
				votes[i] = 0;
		}

		//voting
		for (shIdTotal = 0; shIdTotal < numShTotal; shIdTotal++) {
			distVec = distMtxComplete + shIdTotal * numTrain;
			order = ordersComplete + shIdTotal * numTrain;
			argOrder = argOrdersComplete + shIdTotal * numTrain;
			getIndMinDiff(indMinDiff, distVec, order, argOrder, seedByIter, preLabels, numTrain, numPLabeled + iter);

			if (indMinDiff >= 0){
				votes[indMinDiff]++;
			}
		}
		max(nextVotes, nextP, votes, numTrain);
		seedByIter[numPLabeled + iter] = nextP;
		preLabels[nextP] = 1;
		votes[nextP] = -1;

		//select assumed shapelets
		numByLabel[0]--; numByLabel[1]++;
		for (shIdTotal = 0; shIdTotal < numShTotal; shIdTotal++) {
			distVec = distMtxComplete + shIdTotal * numTrain;
			order = ordersComplete + shIdTotal * numTrain;
			getShapeletPrecision(primaryComplete[shIdTotal], secondaryComplete[shIdTotal], distVec, order, preLabels, preLabels_l, preLabels_r, numTrain, numPLabeled, iter);
		}
		getShRanking(shRanking, primaryComplete, secondaryComplete, tmpOrder, randSeeds, candPrimary, candSecondary, candShIds,
			orderByPrimary, orderBySecondary, numShTotal, maxNumSh);

		//average sp calculation
		cumSp = 0;
		for (int i = 0; i < maxNumSh; i++) {
			cumSp += primaryComplete[shRanking[i]];
			avgSpByIter[i * maxNumIters + iter] = cumSp / (i + 1);
		}
	}

	//find the best stopping points	
	int minIter = minNumIters - 1;
	int maxIter = maxNumIters - 1;
	int maxItersConsidered = maxIter - minIter + 1;

	double *maxSps = (double *)malloc(maxNumSh * sizeof(double));
	int *numMaxSps = (int *)malloc(maxNumSh * sizeof(int));
	int *spStopIter = (int *)malloc(maxNumSh * maxItersConsidered * sizeof(int));
	int *spStopIter_final = (int *)malloc(maxNumSh * sizeof(int));
	int *gaps = (int *)malloc(maxNumSh * sizeof(int));

	//stopping point for each number of assumed shapelets
	double *curAvgSpByIter, maxSpBelow;

	int *curSpStopIterBelow = minIter > 0 ? (int *)malloc(minIter * sizeof(int)) : NULL;
	//int *curSpStopIterAll = (int *)malloc(maxNumIters * sizeof(int));
	int *curSpStopIter, numMaxSpBelow, lastIterBelow, curSpStopIter_final, gap, maxGap, gapTh, prevGap, nextGap;
	for (int i = 0; i < maxNumSh; i++) {
		curAvgSpByIter = avgSpByIter + i * maxNumIters;
		curSpStopIter = spStopIter + i * maxItersConsidered;
		maxWithTies(maxSps[i], curSpStopIter, numMaxSps[i], curAvgSpByIter + minIter, maxItersConsidered);
		for (int j = 0; j < numMaxSps[i]; j++) {
			curSpStopIter[j] += minIter;
		}

		if (minIter > 0)
			maxWithTies(maxSpBelow, curSpStopIterBelow, numMaxSpBelow, curAvgSpByIter, minIter);	//first minNumIter-1=minIter iterations
		else
			maxSpBelow = -1;

		if (maxSpBelow >= maxSps[i]) {

			for (int j = 0; j < minIter; j++){
				if (curAvgSpByIter[j] >= maxSps[i]) {
					lastIterBelow = j;
				}
			}
		}
		else
			lastIterBelow = -1;

		if (numMaxSps[i] == 1) {	//only one option
			spStopIter_final[i] = curSpStopIter[0];
			gaps[i] = 0;
		}
		else {
			//the gaps
			maxGap = 0;
			for (int j = 1; j < numMaxSps[i]; j++) {
				gap = curSpStopIter[j] - curSpStopIter[j - 1] - 1;
				if (gap < 2)
					continue;
				if (maxGap < gap) {
					maxGap = gap;
				}
			}
			gapTh = max((int)(ceil((double)maxGap / 2)), 2);

			for (int j = 1; j < numMaxSps[i]; j++) {
				gap = curSpStopIter[j] - curSpStopIter[j - 1] - 1;
				if (gap >= gapTh) {
					if (j - 1 == 0) {
						if (lastIterBelow == -1)
							prevGap = gapTh;
						else {
							prevGap = curSpStopIter[0] - lastIterBelow - 1;
						}
					}
					else
						prevGap = curSpStopIter[j - 1] - curSpStopIter[j - 2] - 1;

					if (j == numMaxSps[i] - 1)
						nextGap = gapTh;
					else
						nextGap = curSpStopIter[j + 1] - curSpStopIter[j] - 1;

					if (nextGap >= gapTh) {	//isolated point after gap
						spStopIter_final[i] = curSpStopIter[j];
					}
					else {
						if (prevGap >= gapTh) {	//isolated point before gap, no isolated point after gap
							spStopIter_final[i] = curSpStopIter[j];
						}
						else {	//neither the point before or after the gap is isolated
							spStopIter_final[i] = curSpStopIter[j - 1];
						}
					}
					gaps[i] = gap;
					break;
				}
				else if (j == numMaxSps[i] - 1) {	//no gap
					spStopIter_final[i] = curSpStopIter[j];
					gaps[i] = 0;
				}
			}
		}
	}
	if (curSpStopIterBelow != NULL)
		free(curSpStopIterBelow);

	//stopping point for the best number of assumed shapelets
	int num_numAS_maxGap;
	int *inds_maxGap = (int *)malloc(maxNumSh * sizeof(int));
	maxWithTies(maxGap, inds_maxGap, num_numAS_maxGap, gaps, maxNumSh);
	int *spStopIters_maxGap = (int *)malloc(num_numAS_maxGap * sizeof(int));
	for (int i = 0; i < num_numAS_maxGap; i++) {
		spStopIters_maxGap[i] = spStopIter_final[inds_maxGap[i]];
	}
	int latestStop, num_numAS_lateStop;
	int *inds_lateStop = (int *)malloc(num_numAS_maxGap * sizeof(int));
	maxWithTies(latestStop, inds_lateStop, num_numAS_lateStop, spStopIters_maxGap, num_numAS_maxGap);
	int numAS = inds_maxGap[inds_lateStop[num_numAS_lateStop - 1]] + 1;
	int preNumP = numPLabeled + spStopIter_final[numAS - 1] + 1;	//the predicted number of positive training examples

	//the predicted labels of the training set
	seedByIterToPreLabels(preLabels, seedByIter, numTrain, preNumP);

	//the shapelets and the shapelet-transformed training data
	numByLabel[0] = numTrain - preNumP; numByLabel[1] = preNumP;
	for (shIdTotal = 0; shIdTotal < numShTotal; shIdTotal++) {
		distVec = distMtxComplete + shIdTotal * numTrain;
		order = ordersComplete + shIdTotal * numTrain;

		if (distVec[order[1]] == INF){
			primaryComplete[shIdTotal] = secondaryComplete[shIdTotal] = -1;
			continue;
		}

		infoGain(primaryComplete[shIdTotal], splitPt, secondaryComplete[shIdTotal], distVec, preLabels, order, numByLabel, numByLabelIn, numByLabelOut, numTrain, 2);
	}
	getShRanking(shRanking, primaryComplete, secondaryComplete, tmpOrder, randSeeds, candPrimary, candSecondary, candShIds,
		orderByPrimary, orderBySecondary, numShTotal, maxNumSh);

	int shTsId, shPos, shSLen;
	int *shInfo = (int *)malloc(maxNumSh * 3 * sizeof(int));
	double *shTransTrainTss = (double *)malloc(numTrain * maxNumSh * sizeof(double));
	for (int i = 0; i < maxNumSh; i++) {
		shIdTotal = shRanking[i];
		getShInfo(shTsId, shPos, shSLen, shIdTotal, numShAll, numSLens, minSLen, maxSLen, sLenStep, numTrain, tsLen);
		shInfo[3 * i] = shTsId;
		shInfo[3 * i + 1] = shPos;
		shInfo[3 * i + 2] = shSLen;

		for (int j = 0; j < numTrain; j++) {
			shTransTrainTss[j * maxNumSh + i] = distMtxComplete[shIdTotal * numTrain + j];
		}
	}

	//end timer
	clock_t toc = clock();

	//Evaluation
	double precision, recall, fscore;
	prfWithSeed(precision, recall, fscore, trainLabels, preLabels, seed, numTrain, numPLabeled);
	printf("fscore = %f\n", fscore);
	double time_total = (double)(toc - tic) / ((double)CLOCKS_PER_SEC);
	double time_dist = (double)(toc_dist - tic) / ((double)CLOCKS_PER_SEC);
	printf("Running time (Total / Distance computation): %f / %f\n", time_total, time_dist);

	//Output
	std::ofstream fout;
	fName = outputPath + "\\" + datasetName + "_PUSh_train_performance_" + s_seedId + +"_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	fout << "Precision: " << precision << std::endl;
	fout << "Recall: " << recall << std::endl;
	fout << "Fscore: " << fscore << std::endl;
	fout << "Running time (Total / Distance computation): " << time_total << " / " << time_dist;
	fout.close();
	//fout << precision << std::endl << recall << std::endl << fscore << std::endl << time_total << std::endl << time_dist << std::endl;
	fout.close();

	fName = outputPath + "\\" + datasetName + "_PUSh_trainPreLabels_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	for (int i = 0; i < numTrain; i++) {
		fout << preLabels[i] << " ";
	}
	fout.close();

	fName = outputPath + "\\" + datasetName + "_PUSh_shInfo_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	for (int i = 0; i < maxNumSh; i++) {
		for (int j = 0; j < 3; j++) {
			fout << shInfo[3 * i + j] << " ";
		}
		fout << std::endl;
	}
	fout.close();

	fName = outputPath + "\\" + datasetName + "_PUSh_shTransTrainTss_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	fout.precision(15);
	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < maxNumSh; j++) {
			fout << shTransTrainTss[i * maxNumSh + j] << " ";
		}
		fout << std::endl;
	}
	fout.close();

	free(distMtxComplete);
	free(numShAll);
	free(votes);
	free(seedByIter);
	free(ordersComplete);
	free(argOrdersComplete);
	free(preLabels_l);
	free(preLabels_r);
	free(tmpOrder);
	free(avgSpByIter);
	free(maxSps);
	free(numMaxSps);
	free(spStopIters_maxGap);
	free(inds_maxGap);
	free(inds_lateStop);
	free(shRanking);
	free(randSeeds);
	free(primaryComplete);
	free(secondaryComplete);
	free(spStopIter);
	free(spStopIter_final);
	free(gaps);
	free(shInfo);
	free(shTransTrainTss);
	free(trainTss);
	free(trainLabels);
	free(seed);
	free(preLabels);

	return 0;
}