//Baseline methods: Whole-stream based Propagating 1NN algorithms
//Three distance measures: ED, DTW, DTW-D
//Eight stopping criteria: WK (W), RW (R), BHRK (B), GBTRM 1-5 (G1-G5)

#include "utilities.h"
#include "calcUtilities.h"
#include "distUtilities.h"
#include "evaluationUtilities.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <io.h>
#include <direct.h>
#include <string>
#include <vector>
#include <tuple>

#define INF 1e6
#define EPS 1e-16
#define MAX_CHAR 10

void getPDists_DTW_D(double *pDistMtx_DTW_D, double *pDistMtx_ED, double *pDistMtx_DTW, int numTrain) {
	for (int i = 0; i < numTrain; i++) {
		pDistMtx_DTW_D[i * numTrain + i] = INF;
		for (int j = i + 1; j < numTrain; j++) {
			pDistMtx_DTW_D[i * numTrain + j] = pDistMtx_DTW_D[j * numTrain + i] =
				pDistMtx_DTW[i * numTrain + j] / (pDistMtx_ED[i * numTrain + j] + EPS);
		}
	}
}

void getNNs(double *nnDists, int *nnInds, double *pDistMtx, int numTrain) {
	double *pDistVec;
	for (int i = 0; i < numTrain; i++) {
		pDistVec = pDistMtx + i * numTrain;
		min(nnDists[i], nnInds[i], pDistVec, numTrain);
	}
}

bool checkNNIsRanked(int *rankedInds, int nnInd, int numRanked) {
	for (int i = 0; i < numRanked; i++) {
		if (nnInd == rankedInds[i])
			return true;
	}
	return false;
}

void update(int *rankedInds, double *minNNDists, double *pDistMtx, bool *ranked, int numTrain, int numRanked, int numPLabeled) {
	double *pDistVec, nnDistU, dist, minNNDist = INF;
	int curInd, nnIndU, minNNInd;
	for (int i = 0; i < numRanked; i++) {
		curInd = rankedInds[i];
		pDistVec = pDistMtx + curInd * numTrain;

		nnDistU = INF;
		for (int j = 0; j < numTrain; j++) {
			if (ranked[j])
				continue;

			dist = pDistVec[j];
			if (dist < nnDistU) {
				nnIndU = j;
				nnDistU = dist;
			}
		}

		if (nnDistU < minNNDist) {
			minNNInd = nnIndU;
			minNNDist = nnDistU;
		}
	}
	rankedInds[numRanked] = minNNInd;
	minNNDists[numRanked - numPLabeled] = minNNDist;
	ranked[minNNInd] = true;
}

void rankTrainInds(int *rankedInds, double *minNNDists, int *seed, double *pDistMtx, bool *ranked, int numTrain, int numPLabeled) {

	memcpy(rankedInds, seed, numPLabeled * sizeof(int));
	memset(ranked, 0, numTrain * sizeof(bool));
	for (int i = 0; i < numTrain; i++) {
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seed[j]) {
				ranked[i] = true;
				break;
			}
		}
	}
	for (int i = numPLabeled; i < numTrain; i++) {
		update(rankedInds, minNNDists, pDistMtx, ranked, numTrain, i, numPLabeled);
	}

}

void getPrfsByIter(double *precisions, double *recalls, double *fscores,
	int *rankedInds, int *realLabels, int *preLabels, int numTrain, int numPLabeled) {

	memset(preLabels, 0, numTrain * sizeof(int));
	for (int i = 0; i < numTrain; i++) {
		preLabels[rankedInds[i]] = 1;
		if (i >= numPLabeled) {
			prfWithSeed(precisions[i - numPLabeled], recalls[i - numPLabeled], fscores[i - numPLabeled],
				realLabels, preLabels, rankedInds, numTrain, numPLabeled);
		}
	}
}

//Li Wei, Eamonn J.Keogh: Semi-supervised time series classification. KDD 2006: 748-753
int sc_WK(int *rankedInds, int *nnInds, int minNumP, int maxNumP) {

	int curInd, curNNInd;
	for (int i = minNumP; i < maxNumP; i++) {
		curInd = rankedInds[i];
		curNNInd = nnInds[curInd];
		if (!checkNNIsRanked(rankedInds, curNNInd, i))
			return i;
	}
	return maxNumP;
}

//Chotirat Ann Ratanamahatana, Dechawut Wanichsan: Stopping Criterion Selection for Efficient Semi-supervised Time Series Classification.Software Engineering, Artificial Intelligence, Networking and Parallel / Distributed Computing 2008: 1-14
int sc_RW(double *minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled) {

	//initialization
	double minNNDist, sum, sum2;
	sum = sum2 = 0;
	for (int i = 0; i < minNumP - numPLabeled + 1; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;
	}

	double diff, std, scc, maxScc = -INF;
	int preNumP, initNumU = numTrain - numPLabeled;
	for (int i = minNumP - numPLabeled + 1; i < max(maxNumP + 2, numTrain) - numPLabeled; i++) {
		minNNDist = minNNDists[i];
		sum += minNNDist;
		sum2 += minNNDist * minNNDist;

		diff = abs(minNNDists[i] - minNNDists[i - 1]);
		std = sum2 / (i + 1) - sum * sum / ((i + 1) * (i + 1));
		std = std > 0 ? sqrt(std) : 1;
		scc = diff / std * (initNumU - i) / initNumU;

		if (scc > maxScc) {
			maxScc = scc;
			preNumP = numPLabeled + i - 1;
		}
	}
	return preNumP;
}

void discretize(int *seq, double *ts, int tsLen, int card) {

	double minVal = min(ts, tsLen);
	double maxVal = max(ts, tsLen);
	if (minVal == maxVal)
		maxVal++;

	for (int i = 0; i < tsLen; i++) {
		seq[i] = round((ts[i] - minVal) / (maxVal - minVal) * (card - 1)) + 1;
	}
}

int getRdl(int *hypoSeq, int &cumNumMiss, double *nextTs, int *nextSeq, int numTrain, int numRanked, int tsLen, int card){
	double minVal = min(nextTs, tsLen);
	double maxVal = max(nextTs, tsLen);
	discretize(nextSeq, nextTs, tsLen, card);
	for (int i = 0; i < tsLen; i++){
		if (nextSeq[i] != hypoSeq[i])
			cumNumMiss++;
	}

	return ceil((numTrain - numRanked + 1) * tsLen * log2(card)
		+ cumNumMiss * (log2(card) + ceil(log2(tsLen))));
}

//Nurjahan Begum, Bing Hu, Thanawin Rakthanmanon, Eamonn J.Keogh: A Minimum Description Length Technique for Semi-Supervised Time Series Classification. IRI 2013: 171 - 192
int sc_BHRK(double *tss, int *rankedInds, int *hypoSeq, int *nextSeq,
	int minNumP, int maxNumP, int numTrain, int numPLabeled, int tsLen, int card) {

	double *ts;
	int cumNumMiss, preNumP, optPreNumP, curRdl, prevRdl, minRdl = INF;
	for (int i = 0; i < numPLabeled; i++) {
		ts = tss + rankedInds[i] * tsLen;
		discretize(hypoSeq, ts, tsLen, card);
		prevRdl = INF;
		cumNumMiss = 0;
		preNumP = 0;
		for (int j = numPLabeled; j < maxNumP; j++) {
			ts = tss + rankedInds[j] * tsLen;
			curRdl = getRdl(hypoSeq, cumNumMiss, ts, nextSeq, numTrain, j + 1, tsLen, card);
			if (j < minNumP || curRdl < prevRdl) {
				prevRdl = curRdl;
			}
			else {
				preNumP = j;
				break;
			}
		}
		if (!preNumP)
			preNumP = maxNumP;
		if (prevRdl < minRdl) {
			minRdl = prevRdl;
			optPreNumP = preNumP;
		}
	}
	return optPreNumP;
}

std::vector<std::tuple<int, int, int, int>> getIntervals(double *minNNDists, int numTrain, int numPLabeled, double beta) {

	std::vector<std::tuple<int, int, int, int>> intervals;	//start, ad, ds, finish
	int start, ad, ds, finish, instTrend, prevTrend, curTrend;
	prevTrend = 0; curTrend = -1; //1 for ascend, -1 for descend, 0 for stable
	double minNNDist, prevMinNNDist, diff, hd, lb, ub;
	prevMinNNDist = minNNDists[0];
	hd = 0; lb = INF; ub = -INF;
	start = INF;
	for (int i = 1; i < numTrain - numPLabeled; i++) {

		minNNDist = minNNDists[i];
		if (minNNDist > lb && minNNDist < ub)
			instTrend = 0;
		else {
			diff = minNNDist - prevMinNNDist;
			instTrend = sign(diff);
			if (!instTrend) {
				instTrend = curTrend;
			}
		}

		if (curTrend != instTrend) {
			curTrend = instTrend;
		}

		if (prevTrend == 0) {
			if (curTrend == 0) {
				//do nothing
			}
			else if (curTrend == 1) {
				finish = i - 1;
				if (start < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
				}
				prevTrend = curTrend;
				start = i - 1;
				hd = 0; lb = INF; ub = -INF;
			}
			else {
				finish = i - 1;
				if (start < finish) {
					intervals.push_back(std::make_tuple(start, ad, ds, finish));
					prevTrend = curTrend;
					start = i;
				}
				hd = 0; lb = INF; ub = -INF;
			}
		}
		else if (prevTrend == 1) {
			if (curTrend == -1) {
				ad = i - 1;
				prevTrend = curTrend;
				hd -= diff;
			}
		}
		else {
			if (curTrend == 0) {
				//do nothing
			}
			else if (curTrend == 1) {
				lb = prevMinNNDist - beta * hd;
				ub = prevMinNNDist + beta * hd;
				if (minNNDist > lb && minNNDist < ub) {
					curTrend = 0;
					ds = i - 1;
				}
				else {
					start = i - 1;
					hd = 0; lb = INF; ub = -INF;
				}
				prevTrend = curTrend;
			}
			else {
				hd -= diff;
			}
		}
		prevMinNNDist = minNNDist;
	}
	return intervals;
}

//Mabel Gonz��lez Castellanos, Christoph Bergmeir, Isaac Triguero, Yanet Rodr��guez, Jos�� Manuel Ben��tez: On the stopping criteria for k - Nearest Neighbor in positive unlabeled time series classification problems. Inf.Sci. 328: 42-59 (2016)
void sc_GBTRM(int *preNumPs, double *minNNDists, int minNumP, int maxNumP, int numTrain, int numPLabeled, double beta) {

	int initNumU = numTrain - numPLabeled;
	double max_minNNDists = max(minNNDists, initNumU);

	int start, finish, ad, ds, ip, ws;
	double maxScs[5], scs[5], ha, hd, max_interval, lw;
	for (int i = 0; i < 5; i++)
		preNumPs[i] = maxScs[i] = -INF;

	std::tuple<int, int, int, int> curInterval;
	std::vector<std::tuple<int, int, int, int>> intervals = getIntervals(minNNDists, numTrain, numPLabeled, beta);
	for (int i = 0; i < intervals.size(); i++) {

		curInterval = intervals[i];
		start = std::get<0>(curInterval);
		finish = std::get<3>(curInterval);

		if (finish < minNumP - numPLabeled - 1 || start > maxNumP - numPLabeled - 1)
			continue;

		ad = std::get<1>(curInterval);
		ds = std::get<2>(curInterval);
		ha = minNNDists[ad] - minNNDists[start];
		hd = minNNDists[ad] - minNNDists[ds];
		ws = finish - ds;
		max(max_interval, ip, minNNDists + start, finish - start + 1);
		ip += start;
		lw = (double)(initNumU - ip) / initNumU;

		scs[0] = hd * lw;
		scs[1] = ha * lw;
		scs[2] = ws * lw;
		scs[3] = max(ha, hd) * lw;
		scs[4] = max(ha / max_minNNDists, (double)ws / (initNumU - 1)) * lw;
		for (int j = 0; j < 5; j++) {
			if (scs[j] > maxScs[j]) {
				maxScs[j] = scs[j];
				preNumPs[j] = numPLabeled + ip;
			}
		}
	}
}

void getTransPreLabels(int *transPreLabels, int *rankedInds, int numTrain, int preNumP) {

	memset(transPreLabels, 0, numTrain * sizeof(int));
	for (int i = 0; i < preNumP; i++) {
		transPreLabels[rankedInds[i]] = 1;
	}

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
	const int minNumIters = argc > 7 ? atoi(argv[7]) : 5;
	const int maxNumIters = argc > 8 ? atoi(argv[8]) : numTrain * 2 / 3 - numPLabeled;
	const double warp = argc > 9 ? atof(argv[9]) : 0;	//reset to UCR settings if set to 0
	const int maxThreadsPerBlock = argc > 10 ? atoi(argv[10]) : 8;
	const int maxBlocksPerGrid = argc > 11 ? atoi(argv[11]) : 8;
	const std::string dataInfoPath = argc > 12 ? argv[12] : "..\\sample_data";	//information of all UCR datasets (Ver. 2015)
	const std::string path = argc > 13 ? argv[13] : dataInfoPath + "\\" + datasetName;

	/*std::string datasetName = "Car";
	const int numTrain = 60;
	const int numP = 16;
	const int numPLabeled = 2;
	const int tsLen = 577;
	const int seedId = 0;
	const int minNumIters = 5;
	const int maxNumIters = numTrain * 2 / 3 - numPLabeled;
	const double warp = 0;
	const int maxThreadsPerBlock = 8;
	const int maxBlocksPerGrid = 8;
	const std::string dataInfoPath = "..\\sample_data";	//information of all UCR datasets (Ver. 2015)
	const std::string path = dataInfoPath + "\\" + datasetName;*/

	const int numSeeds = numP < 10 ? numP : 10;
	if (seedId >= numSeeds)
		exit(1);

	std::string fName;

	//load time series
	long long trainTssBytes = numTrain * tsLen * sizeof(double);
	double *trainTss = (double*)malloc(trainTssBytes);
	long long trainLabelsBytes = numTrain * sizeof(int);
	int *trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, path, datasetName, "TRAIN", numTrain, tsLen);
	relabel(trainLabels, numTrain, 1);

	//load seeds
	long long seedBytes = numPLabeled * sizeof(int);
	int *seeds = (int*)malloc(numSeeds * seedBytes);
	fName = path + "\\seeds_" + datasetName + ".txt";
	importMatrix(seeds, fName, numSeeds, numPLabeled, 1);
	for (int i = 0; i < numSeeds * numPLabeled; i++)
		seeds[i]--;	//matlab -> c
	int *seed = (int *)malloc(seedBytes);
	memcpy(seed, seeds + seedId * numPLabeled, numPLabeled * sizeof(int));

	//DTW warping window
	std::vector<double> warps;
	if (warp == 0){
		std::vector<double> defaultWarps;
		for (int i = 0; i < 10; i++){
			defaultWarps.push_back((double)i / 100);
		}

		fName = dataInfoPath + "\\InfoAll";
		std::ifstream fin;
		fin.open(fName);
		char buf[MAX_CHAR_PER_LINE];
		char *tmp;
		bool flg = true;
		double errorRates[3];
		double bestWarp;
		while (flg) {
			fin.getline(buf, MAX_CHAR_PER_LINE, '\n');
			tmp = strtok(buf, " ,\r\n\t");
			if (!strcmp(tmp, datasetName.c_str())) {
				flg = false;
				for (int j = 0; j < 5; j++) {
					tmp = strtok(NULL, " ,\r\n\t");
				}
				errorRates[0] = atof(tmp);
				tmp = strtok(NULL, " ,\r\n\t");
				errorRates[1] = atof(tmp);
				tmp = strtok(NULL, " ,\r\n\t");
				bestWarp = atof(tmp) / 100;
				tmp = strtok(NULL, " ,\r\n\t");
				errorRates[2] = atof(tmp);
			}
		}
		if (errorRates[2] < errorRates[1])
			warps.push_back(1);
		else if (bestWarp == 0)
			warps = defaultWarps;
		else
			warps.push_back(bestWarp);
		fin.close();
	}
	else{
		warps.push_back(warp);
	}

	int minNumP = minNumIters + numPLabeled;
	int maxNumP = maxNumIters + numPLabeled;
	int *transPreLabelsAll = (int *)malloc(27 * numTrain * sizeof(int));	//3 distance measures * 9 stopping criteria (including oracle)
	int *warpsAll = (int *)malloc(27 * sizeof(int));
	int idxTrans = 0;

	char s_seedId[MAX_CHAR], s_minNumIters[MAX_CHAR], s_maxNumIters[MAX_CHAR];
	_itoa(seedId, s_seedId, 10);
	_itoa(minNumIters, s_minNumIters, 10);
	_itoa(maxNumIters, s_maxNumIters, 10);

	//ED
	double *trainTss_in;
	cudaMalloc(&trainTss_in, trainTssBytes);
	cudaMemcpy(trainTss_in, trainTss, trainTssBytes, cudaMemcpyHostToDevice);
	double *pDists_out;
	long long pDistsBytes = numTrain * numTrain * sizeof(double);
	cudaMalloc(&pDists_out, pDistsBytes);
	int blockSize = numTrain < maxThreadsPerBlock ? numTrain : maxThreadsPerBlock;
	int gridSize = ceil((double)numTrain / blockSize) < maxBlocksPerGrid ? ceil((double)numTrain / blockSize) : maxBlocksPerGrid;
	getPDists_DTW << <gridSize, blockSize >> > (trainTss_in, trainTss_in, pDists_out, numTrain, numTrain, tsLen, 0);	//ED is DTW with zero warp.
	double *pDistMtx_ED = (double *)malloc(pDistsBytes);
	cudaMemcpy(pDistMtx_ED, pDists_out, pDistsBytes, cudaMemcpyDeviceToHost);

	int *rankedInds = (int *)malloc(numTrain *  sizeof(int));
	double *minNNDists = (double *)malloc((numTrain - numPLabeled) * sizeof(double));
	bool *ranked = (bool *)malloc(numTrain * sizeof(bool));
	rankTrainInds(rankedInds, minNNDists, seed, pDistMtx_ED, ranked, numTrain, numPLabeled);

	double *precisions = (double *)malloc((numTrain - numPLabeled) * sizeof(double));
	double *recalls = (double *)malloc((numTrain - numPLabeled) * sizeof(double));
	double *fscores = (double *)malloc((numTrain - numPLabeled) * sizeof(double));
	int *preLabels = (int *)malloc(numTrain * sizeof(int));
	getPrfsByIter(precisions, recalls, fscores, rankedInds, trainLabels, preLabels, numTrain, numPLabeled);

	double ED_f_oracle, ED_f_WK, ED_f_RW, ED_f_BHRK, ED_f_GBTRM[5], DTW_f_oracle, DTW_f_WK, DTW_f_RW, DTW_f_BHRK, DTW_f_GBTRM[5], DTW_D_f_oracle, DTW_D_f_WK, DTW_D_f_RW, DTW_D_f_BHRK, DTW_D_f_GBTRM[5],
		best_DTW_f_oracle, best_DTW_f_WK, best_DTW_f_RW, best_DTW_f_BHRK, best_DTW_f_GBTRM[5], best_DTW_D_f_oracle, best_DTW_D_f_WK, best_DTW_D_f_RW, best_DTW_D_f_BHRK, best_DTW_D_f_GBTRM[5];
	//////Oracle
	int preNumP = numP;
	ED_f_oracle = fscores[preNumP - numPLabeled - 1];
	getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumP);
	warpsAll[idxTrans] = 0;
	idxTrans++;
	//////WK
	double *nnDists = (double *)malloc(numTrain * sizeof(double));
	int *nnInds = (int *)malloc(numTrain * sizeof(int));
	getNNs(nnDists, nnInds, pDistMtx_ED, numTrain);
	preNumP = sc_WK(rankedInds, nnInds, minNumP, maxNumP);
	ED_f_WK = -INF;
	int tmpPreNumP;
	for (int i = minNumP - numPLabeled - 1; i < preNumP - numPLabeled; i++) {
		if (ED_f_WK <= fscores[i]) {
			ED_f_WK = fscores[i];
			tmpPreNumP = i + numPLabeled + 1;
		}
	}
	preNumP = tmpPreNumP;
	getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumP);
	warpsAll[idxTrans] = 0;
	idxTrans++;
	//////RW
	preNumP = sc_RW(minNNDists, minNumP, maxNumP, numTrain, numPLabeled);
	ED_f_RW = fscores[preNumP - numPLabeled - 1];
	getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumP);
	warpsAll[idxTrans] = 0;
	idxTrans++;
	//////BHRK
	int *hypoSeq = (int *)malloc(tsLen * sizeof(int));
	int *nextSeq = (int *)malloc(tsLen * sizeof(int));
	preNumP = sc_BHRK(trainTss, rankedInds, hypoSeq, nextSeq, minNumP, maxNumP, numTrain, numPLabeled, tsLen, 16);
	ED_f_BHRK = fscores[preNumP - numPLabeled - 1];
	getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumP);
	warpsAll[idxTrans] = 0;
	idxTrans++;
	//////GBTRM
	int preNumPs[5];
	sc_GBTRM(preNumPs, minNNDists, minNumP, maxNumP, numTrain, numPLabeled, 0.3);
	for (int i = 0; i < 5; i++) {
		if (preNumPs[i] < 0) {

			ED_f_GBTRM[i] = -1;
			for (int j = 0; j < numTrain; j++) {
				transPreLabelsAll[idxTrans * numTrain + j] = -1;
			}
			warpsAll[idxTrans] = -1;
		}
		else {
			ED_f_GBTRM[i] = fscores[preNumPs[i] - numPLabeled - 1];
			getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumPs[i]);
			warpsAll[idxTrans] = 0;
		}
		idxTrans++;
	}

	//initiation for DTW and DTW-D
	best_DTW_f_oracle = -INF;
	best_DTW_f_WK = -INF;
	best_DTW_f_RW = -INF;
	best_DTW_f_BHRK = -INF;
	for (int i = 0; i < 5; i++)
		best_DTW_f_GBTRM[i] = -INF;
	best_DTW_D_f_oracle = -INF;
	best_DTW_D_f_WK = -INF;
	best_DTW_D_f_RW = -INF;
	best_DTW_D_f_BHRK = -INF;
	for (int i = 0; i < 5; i++)
		best_DTW_D_f_GBTRM[i] = -INF;

	double bestWarps[18], bestFs[18];
	int bestPreNumPs[18];
	int *bestRankedInds = (int *)malloc(18 * numTrain * sizeof(int));

	double *pDistMtx_DTW = (double *)malloc(pDistsBytes);
	double *pDistMtx_DTW_D = (double *)malloc(pDistsBytes);
	for (int i = 0; i < warps.size(); i++) {

		getPDists_DTW << <gridSize, blockSize >> > (trainTss_in, trainTss_in, pDists_out, numTrain, numTrain, tsLen, warps[i]);
		cudaMemcpy(pDistMtx_DTW, pDists_out, pDistsBytes, cudaMemcpyDeviceToHost);
		rankTrainInds(rankedInds, minNNDists, seed, pDistMtx_DTW, ranked, numTrain, numPLabeled);
		getPrfsByIter(precisions, recalls, fscores, rankedInds, trainLabels, preLabels, numTrain, numPLabeled);

		//////Oracle
		int localIdx = 0;
		preNumP = numP;
		DTW_f_oracle = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_f_oracle < DTW_f_oracle) {
			bestFs[localIdx] = best_DTW_f_oracle = DTW_f_oracle;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////WK
		getNNs(nnDists, nnInds, pDistMtx_DTW, numTrain);
		preNumP = sc_WK(rankedInds, nnInds, minNumP, maxNumP);
		DTW_f_WK = -INF;
		for (int i = minNumP - numPLabeled - 1; i < preNumP - numPLabeled; i++) {
			DTW_f_WK = DTW_f_WK < fscores[i] ? fscores[i] : DTW_f_WK;
		}
		if (best_DTW_f_WK < DTW_f_WK) {
			bestFs[localIdx] = best_DTW_f_WK = DTW_f_WK;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////RW
		preNumP = sc_RW(minNNDists, minNumP, maxNumP, numTrain, numPLabeled);
		DTW_f_RW = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_f_RW < DTW_f_RW) {
			bestFs[localIdx] = best_DTW_f_RW = DTW_f_RW;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////BHRK
		preNumP = sc_BHRK(trainTss, rankedInds, hypoSeq, nextSeq, minNumP, maxNumP, numTrain, numPLabeled, tsLen, 64);
		DTW_f_BHRK = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_f_BHRK < DTW_f_BHRK) {
			bestFs[localIdx] = best_DTW_f_BHRK = DTW_f_BHRK;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////GBTRM
		sc_GBTRM(preNumPs, minNNDists, minNumP, maxNumP, numTrain, numPLabeled, 0.3);
		for (int j = 0; j < 5; j++) {
			if (preNumPs[j] < 0)
				DTW_f_GBTRM[j] = -1;
			else {
				DTW_f_GBTRM[j] = fscores[preNumPs[j] - numPLabeled - 1];
			}
			if (best_DTW_f_GBTRM[j] < DTW_f_GBTRM[j]) {
				bestFs[localIdx] = best_DTW_f_GBTRM[j] = DTW_f_GBTRM[j];
				bestPreNumPs[localIdx] = preNumPs[j];
				bestWarps[localIdx] = warps[i];
				memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
				localIdx++;
			}
		}

		//DTW-D
		getPDists_DTW_D(pDistMtx_DTW_D, pDistMtx_ED, pDistMtx_DTW, numTrain);
		rankTrainInds(rankedInds, minNNDists, seed, pDistMtx_DTW_D, ranked, numTrain, numPLabeled);
		getPrfsByIter(precisions, recalls, fscores, rankedInds, trainLabels, preLabels, numTrain, numPLabeled);

		//////Oracle
		preNumP = numP;
		DTW_D_f_oracle = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_D_f_oracle < DTW_D_f_oracle) {
			bestFs[localIdx] = best_DTW_D_f_oracle = DTW_D_f_oracle;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////WK
		getNNs(nnDists, nnInds, pDistMtx_DTW_D, numTrain);
		preNumP = sc_WK(rankedInds, nnInds, minNumP, maxNumP);
		DTW_D_f_WK = -INF;
		for (int i = minNumP - numPLabeled - 1; i < preNumP - numPLabeled; i++) {
			DTW_D_f_WK = DTW_D_f_WK < fscores[i] ? fscores[i] : DTW_D_f_WK;
		}
		if (best_DTW_D_f_WK < DTW_D_f_WK) {
			bestFs[localIdx] = best_DTW_D_f_WK = DTW_D_f_WK;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////RW
		preNumP = sc_RW(minNNDists, minNumP, maxNumP, numTrain, numPLabeled);
		DTW_D_f_RW = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_D_f_RW < DTW_D_f_RW) {
			bestFs[localIdx] = best_DTW_D_f_RW = DTW_D_f_RW;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////BHRK
		preNumP = sc_BHRK(trainTss, rankedInds, hypoSeq, nextSeq, minNumP, maxNumP, numTrain, numPLabeled, tsLen, 64);
		DTW_D_f_BHRK = fscores[preNumP - numPLabeled - 1];
		if (best_DTW_D_f_BHRK < DTW_D_f_BHRK) {
			bestFs[localIdx] = best_DTW_D_f_BHRK = DTW_D_f_BHRK;
			bestPreNumPs[localIdx] = preNumP;
			bestWarps[localIdx] = warps[i];
			memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
			localIdx++;
		}
		//////GBTRM
		sc_GBTRM(preNumPs, minNNDists, minNumP, maxNumP, numTrain, numPLabeled, 0.3);
		for (int j = 0; j < 5; j++) {
			if (preNumPs[j] < 0)
				DTW_D_f_GBTRM[j] = -1;
			else {
				DTW_D_f_GBTRM[j] = fscores[preNumPs[j] - numPLabeled - 1];
			}
			if (best_DTW_D_f_GBTRM[j] < DTW_D_f_GBTRM[j]) {
				bestFs[localIdx] = best_DTW_D_f_GBTRM[j] = DTW_D_f_GBTRM[j];
				bestPreNumPs[localIdx] = preNumPs[j];
				bestWarps[localIdx] = warps[i];
				memcpy(bestRankedInds + localIdx * numTrain, rankedInds, numTrain * sizeof(int));
				localIdx++;
			}
		}

	}

	for (int i = 0; i < 18; i++) {
		memcpy(rankedInds, bestRankedInds + i * numTrain, numTrain * sizeof(int));
		preNumP = bestPreNumPs[i];
		getTransPreLabels(transPreLabelsAll + idxTrans * numTrain, rankedInds, numTrain, preNumP);
		warpsAll[idxTrans] = bestFs[i] >= 0 ? bestWarps[i] * 100 : -1;
		idxTrans++;
	}

	//Save to file
	std::ofstream fout;
	fName = path + "\\" + datasetName + "_P1NN_warps_and_preLabels_" + +s_seedId + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	for (int i = 0; i < 27; i++) {
		fout << warpsAll[i] << " ";
		for (int j = 0; j < numTrain; j++) {
			fout << transPreLabelsAll[i * numTrain + j] << " ";
		}
		fout << std::endl;
	}
	fout.close();

	fName = path + "\\" + datasetName + "_P1NN_train_fscores_" + s_seedId + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	fout.open(fName);
	fout << "ED_oracle: " << ED_f_oracle << std::endl;
	fout << "ED_WK: " << ED_f_WK << std::endl;
	fout << "ED_RW: " << ED_f_RW << std::endl;
	fout << "ED_BHRK: " << ED_f_BHRK << std::endl;
	fout << "ED_GBTRM_1: " << ED_f_GBTRM[0] << std::endl;
	fout << "ED_GBTRM_2: " << ED_f_GBTRM[1] << std::endl;
	fout << "ED_GBTRM_3: " << ED_f_GBTRM[2] << std::endl;
	fout << "ED_GBTRM_4: " << ED_f_GBTRM[3] << std::endl;
	fout << "ED_GBTRM_5: " << ED_f_GBTRM[4] << std::endl;

	fout << "DTW_oracle: " << best_DTW_f_oracle << std::endl;
	fout << "DTW_WK: " << best_DTW_f_WK << std::endl;
	fout << "DTW_RW: " << best_DTW_f_RW << std::endl;
	fout << "DTW_BHRK: " << best_DTW_f_BHRK << std::endl;
	fout << "DTW_GBTRM_1: " << best_DTW_f_GBTRM[0] << std::endl;
	fout << "DTW_GBTRM_2: " << best_DTW_f_GBTRM[1] << std::endl;
	fout << "DTW_GBTRM_3: " << best_DTW_f_GBTRM[2] << std::endl;
	fout << "DTW_GBTRM_4: " << best_DTW_f_GBTRM[3] << std::endl;
	fout << "DTW_GBTRM_5: " << best_DTW_f_GBTRM[4] << std::endl;
	
	fout << "DTW_D_oracle: " << best_DTW_D_f_oracle << std::endl;
	fout << "DTW_D_WK: " << best_DTW_D_f_WK << std::endl;
	fout << "DTW_D_RW: " << best_DTW_D_f_RW << std::endl;
	fout << "DTW_D_BHRK: " << best_DTW_D_f_BHRK << std::endl;
	fout << "DTW_D_GBTRM_1: " << best_DTW_D_f_GBTRM[0] << std::endl;
	fout << "DTW_D_GBTRM_2: " << best_DTW_D_f_GBTRM[1] << std::endl;
	fout << "DTW_D_GBTRM_3: " << best_DTW_D_f_GBTRM[2] << std::endl;
	fout << "DTW_D_GBTRM_4: " << best_DTW_D_f_GBTRM[3] << std::endl;
	fout << "DTW_D_GBTRM_5: " << best_DTW_D_f_GBTRM[4] << std::endl;

	fout.close();

	std::cout << "F-scores:" << std::endl;
	std::cout << "ED_oracle: " << ED_f_oracle << std::endl;
	std::cout << "ED_WK: " << ED_f_WK << std::endl;
	std::cout << "ED_RW: " << ED_f_RW << std::endl;
	std::cout << "ED_BHRK: " << ED_f_BHRK << std::endl;
	std::cout << "ED_GBTRM_1: " << ED_f_GBTRM[0] << std::endl;
	std::cout << "ED_GBTRM_2: " << ED_f_GBTRM[1] << std::endl;
	std::cout << "ED_GBTRM_3: " << ED_f_GBTRM[2] << std::endl;
	std::cout << "ED_GBTRM_4: " << ED_f_GBTRM[3] << std::endl;
	std::cout << "ED_GBTRM_5: " << ED_f_GBTRM[4] << std::endl;

	std::cout << "DTW_oracle: " << best_DTW_f_oracle << std::endl;
	std::cout << "DTW_WK: " << best_DTW_f_WK << std::endl;
	std::cout << "DTW_RW: " << best_DTW_f_RW << std::endl;
	std::cout << "DTW_BHRK: " << best_DTW_f_BHRK << std::endl;
	std::cout << "DTW_GBTRM_1: " << best_DTW_f_GBTRM[0] << std::endl;
	std::cout << "DTW_GBTRM_2: " << best_DTW_f_GBTRM[1] << std::endl;
	std::cout << "DTW_GBTRM_3: " << best_DTW_f_GBTRM[2] << std::endl;
	std::cout << "DTW_GBTRM_4: " << best_DTW_f_GBTRM[3] << std::endl;
	std::cout << "DTW_GBTRM_5: " << best_DTW_f_GBTRM[4] << std::endl;

	std::cout << "DTW_D_oracle: " << best_DTW_D_f_oracle << std::endl;
	std::cout << "DTW_D_WK: " << best_DTW_D_f_WK << std::endl;
	std::cout << "DTW_D_RW: " << best_DTW_D_f_RW << std::endl;
	std::cout << "DTW_D_BHRK: " << best_DTW_D_f_BHRK << std::endl;
	std::cout << "DTW_D_GBTRM_1: " << best_DTW_D_f_GBTRM[0] << std::endl;
	std::cout << "DTW_D_GBTRM_2: " << best_DTW_D_f_GBTRM[1] << std::endl;
	std::cout << "DTW_D_GBTRM_3: " << best_DTW_D_f_GBTRM[2] << std::endl;
	std::cout << "DTW_D_GBTRM_4: " << best_DTW_D_f_GBTRM[3] << std::endl;
	std::cout << "DTW_D_GBTRM_5: " << best_DTW_D_f_GBTRM[4] << std::endl;

	cudaFree(trainTss_in);
	cudaFree(pDists_out);
	free(trainTss);
	free(trainLabels);
	free(preLabels);
	free(seeds);
	free(seed);
	free(pDistMtx_ED);
	free(pDistMtx_DTW);
	free(pDistMtx_DTW_D);
	free(rankedInds);
	free(minNNDists);
	free(ranked);
	free(precisions);
	free(recalls);
	free(fscores);
	free(nnDists);
	free(nnInds);
	free(hypoSeq);
	free(nextSeq);
	free(transPreLabelsAll);
	free(warpsAll);
	free(bestRankedInds);
	return 0;
}