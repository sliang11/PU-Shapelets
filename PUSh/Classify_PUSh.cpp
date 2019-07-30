//Classification with PUSh
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

#define INF 1e6
#define MAX_CHAR 10

double getNNDist(double *ts, int tsLen, double *query, int sLen, double *zQuery, double *zWin) {

	double std_q = stdv(query, sLen); //this can run into issues due to loss of precision
	if(std_q > 0)
		zscore(zQuery, query, sLen); //this can run into issues due to loss of precision

	double *win, std_w, dist, nnDist = INF;	//the sliding window
	for (int i = 0; i < tsLen - sLen + 1; i++) {
		win = ts + i;

		std_w = stdv(win, sLen); //this can run into issues due to loss of precision
		if (std_q == 0 && std_w == 0){ //this can run into issues due to loss of precision
			//dist = abs(query[0] - win[0]);
			dist = INF;
		}
		else if (std_q == 0 || std_w == 0){ //this can run into issues due to loss of precision
			//dist = 1;
			dist = INF;
		}
		else{
			zscore(zWin, win, sLen); //this can run into issues due to loss of precision
			dist = ED2_early(zQuery, zWin, sLen, nnDist);
		}

		if (dist < nnDist) {
			nnDist = dist;
		}

	}
	nnDist = sqrt(nnDist / sLen);
	return nnDist;

}

void getShTransTs(double *shTransTs, double *ts, double *zSh, double *zWin, double *trainTss, int *shInfo, int numSh, int tsLen) {

	int tsId, pos, sLen;
	double *shapelet;
	for (int i = 0; i < numSh; i++) {
		tsId = shInfo[3 * i];
		pos = shInfo[3 * i + 1];
		sLen = shInfo[3 * i + 2];
		shapelet = trainTss + tsId * tsLen + pos;
		shTransTs[i] = getNNDist(ts, tsLen, shapelet, sLen, zSh, zWin);
	}
}

int classify_nn_ed(double *shTransTestTs, double *shTransTrainTss, int *groundTruthLabels, int numSh, int maxNumSh, int numTrain) {
	double *shTransTrainTs, dist, nnDist = INF;
	int nnInd;
	for (int i = 0; i < numTrain; i++) {
		shTransTrainTs = shTransTrainTss + i * maxNumSh;
		dist = ED2_early(shTransTestTs, shTransTrainTs, numSh, nnDist);

		if (dist < nnDist) {
			nnDist = dist;
			nnInd = i;
		}
	}

	return groundTruthLabels[nnInd];

}

int main(int argc, char **argv) {

	//parameter settings
	if (argc < 8)
		exit(1);
	std::string datasetName = argv[1];
	const int numTrain = atoi(argv[2]);
	const int numP = atoi(argv[3]);
	const int numPLabeled = atoi(argv[4]);
	const int tsLen = atoi(argv[5]);
	const int numTest = atoi(argv[6]);
	const int seedId = atoi(argv[7]);
	const int numSh = argc > 8 ? atoi(argv[8]) : 10;	//The actual number of shapelets used
	const int maxNumSh = argc > 9 ? atoi(argv[9]) : 200;	//Just for reading the correct file name
	const int minNumIters = argc > 10 ? atoi(argv[10]) : (numP >= 10 ? 5 : 1);
	const int maxNumIters = argc > 11 ? atoi(argv[11]) : numTrain * 2 / 3 - numPLabeled;
	const int maxSLen = argc > 12 ? atoi(argv[12]) : tsLen;
	const std::string path = argc > 13 ? argv[13] : "..\\sample_data\\" + datasetName;
	const std::string outputPath = argc > 14 ? argv[14] : "..\\results";

	const int numSeeds = numP > 10 ? 10 : numP;

	printf("seedId = %d\n", seedId);

	char s_seedId[MAX_CHAR], s_maxNumSh[MAX_CHAR], s_numSh[MAX_CHAR], s_minNumIters[MAX_CHAR], s_maxNumIters[MAX_CHAR];
	_itoa(seedId, s_seedId, 10);
	_itoa(maxNumSh, s_maxNumSh, 10);
	_itoa(numSh, s_numSh, 10);
	_itoa(minNumIters, s_minNumIters, 10);
	_itoa(maxNumIters, s_maxNumIters, 10);

	//load time series
	long long trainTssBytes = numTrain * tsLen * sizeof(double);
	double *trainTss = (double*)malloc(trainTssBytes);
	long long trainLabelsBytes = numTrain * sizeof(int);
	int *trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, path, datasetName, "TRAIN", numTrain, tsLen);

	//load test data
	long long testTssBytes = numTest * tsLen * sizeof(double);
	double *testTss = (double*)malloc(testTssBytes);
	long long testLabelsBytes = numTest * sizeof(int);
	int *testLabels = (int*)malloc(testLabelsBytes);
	importTimeSeries(testTss, testLabels, path, datasetName, "TEST", numTest, tsLen);
	relabel(testLabels, numTest, 1);

	//load shInfo
	int *shInfo = (int *)malloc(3 * maxNumSh * sizeof(int));
	std::string fName = outputPath + "\\" + datasetName + "_PUSh_shInfo_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	importMatrix(shInfo, fName, maxNumSh, 3, true);

	//load shapelet transformed training data
	double *shTransTrainTss = (double *)malloc(numTrain * maxNumSh * sizeof(double));
	fName = outputPath + "\\" + datasetName + "_PUSh_shTransTrainTss_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	importMatrix(shTransTrainTss, fName, numTrain, maxNumSh, false);

	//load train preLabels. Note that the classification process is based on predicted training labels by PE+ASPM, not on actual training labels
	int *trainPreLabels = (int *)malloc(numTrain * sizeof(int));
	fName = outputPath + "\\" + datasetName + "_PUSh_trainPreLabels_" + s_seedId + "_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	importMatrix(trainPreLabels, fName, 1, numTrain, true);

	int *testPreLabels = (int *)malloc(numTest * sizeof(int));
	double precision, recall, fscore;
	
	double *testTs, *shTransTestTs = (double *)malloc(maxNumSh * sizeof(double));
	double *zSh = (double *)malloc(maxSLen * sizeof(double));
	double *zWin = (double *)malloc(maxSLen * sizeof(double));
	clock_t tic, toc;
	double runningTime = 0;
	for (int i = 0; i < numTest; i++) {
		testTs = testTss + i * tsLen;
		
		tic = clock();
		getShTransTs(shTransTestTs, testTs, zSh, zWin, trainTss, shInfo, numSh, tsLen);
		testPreLabels[i] = classify_nn_ed(shTransTestTs, shTransTrainTss, trainPreLabels, numSh, maxNumSh, numTrain);
		toc = clock();
		runningTime += (double)(toc - tic) / ((double)CLOCKS_PER_SEC);
	}
	prf(precision, recall, fscore, testLabels, testPreLabels, numTest);
	runningTime /= numTest;
	
	printf("fscore = %f\n", fscore);
	printf("Avg time per test example = %f\n", runningTime);

	//Output
	std::ofstream fout;
	fName = outputPath + "\\" + datasetName + "_PUSh_test_performance_" + s_seedId + +"_" + s_maxNumSh + "_" + s_minNumIters + "_" + s_maxNumIters + "_" + s_numSh + ".txt";
	fout.open(fName);
	fout << "Precision: " << precision << std::endl;
	fout << "Recall: " << recall << std::endl;
	fout << "Fscore: " << fscore << std::endl;
	fout << "Avg time per test example: " << runningTime;
	/*fout << precision << std::endl;
	fout << recall << std::endl;
	fout << fscore << std::endl;
	fout << runningTime;*/
	fout.close();

	free(trainTss);
	free(trainLabels);
	free(testTss);
	free(testLabels);
	free(shInfo);
	free(shTransTrainTss);
	free(trainPreLabels);
	free(testPreLabels);
	free(shTransTestTs);
	free(zSh);
	free(zWin);
	return 0;
}
