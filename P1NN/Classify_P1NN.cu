//1NN classifier based on output of Propagating_1NN

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

#define INF 1e6
#define MAX_CHAR 10
#define MAX_CHAR_PER_LINE 200000

void getDists_DTW_D(double *distMtx_DTW_D, double *distMtx_ED, double *distMtx_DTW, int numTrain, int numTest) {
	for (int i = 0; i < numTest; i++) {
		for (int j = 0; j < numTrain; j++) {

			if (distMtx_ED[i * numTrain + j] == 0) //this can run into issues due to loss of precision
				distMtx_DTW_D[i * numTrain + j] = 0;
			else
				distMtx_DTW_D[i * numTrain + j] = distMtx_DTW[i * numTrain + j] / distMtx_ED[i * numTrain + j];

		}
	}
}

void classifyNN(int *preLabels, int *groundTruthLabels, double *distMtx, int numTrain, int numTest) {
	double minDist;
	int nnInd;
	for (int i = 0; i < numTest; i++) {
		min(minDist, nnInd, distMtx + i * numTrain, numTrain);
		preLabels[i] = groundTruthLabels[nnInd];
	}
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
	const int minNumIters = argc > 8 ? atoi(argv[8]) : (numP >= 10 ? 5 : 1);
	const int maxNumIters = argc > 9 ? atoi(argv[9]) : numTrain * 2 / 3 - numPLabeled;
	const int maxThreadsPerBlock = argc > 10 ? atoi(argv[10]) : 8;
	const int maxBlocksPerGrid = argc > 11 ? atoi(argv[11]) : 8;
	const std::string dataInfoPath = argc > 12 ? argv[12] : "..\\sample_data\\";	//information of all UCR datasets (Ver. 2015)
	const std::string path = argc > 13 ? argv[13] : "..\\sample_data\\" + datasetName;
	const std::string outputPath = argc > 14 ? argv[14] : "..\\results";

	const int numSeeds = numP < 10 ? numP : 10;
	if (seedId >= numSeeds)
		exit(1);

	//DTW warping window
	std::vector<double> warps;
	std::vector<double> defaultWarps;
	for (int i = 1; i <= 10; i++){
		defaultWarps.push_back((double)i / 100);
	}
	std::string fName = dataInfoPath + "\\InfoAll";
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

	char s_seedId[MAX_CHAR], s_minNumIters[MAX_CHAR], s_maxNumIters[MAX_CHAR];
	_itoa(seedId, s_seedId, 10);
	_itoa(minNumIters, s_minNumIters, 10);
	_itoa(maxNumIters, s_maxNumIters, 10);

	//training data
	long long trainTssBytes = numTrain * tsLen * sizeof(double);
	double *trainTss = (double*)malloc(trainTssBytes);
	long long trainLabelsBytes = numTrain * sizeof(int);
	int *trainLabels = (int*)malloc(trainLabelsBytes);
	importTimeSeries(trainTss, trainLabels, path, datasetName, "TRAIN", numTrain, tsLen);
	relabel(trainLabels, numTrain, 1);

	//testing data
	long long testTssBytes = numTest * tsLen * sizeof(double);
	double *testTss = (double*)malloc(testTssBytes);
	long long testLabelsBytes = numTest * sizeof(int);
	int *testLabels = (int*)malloc(testLabelsBytes);
	importTimeSeries(testTss, testLabels, path, datasetName, "TEST", numTest, tsLen);
	relabel(testLabels, numTest, 1);

	//warp and transPreLabels
	int *warpAndLabels = (int *)malloc(27 * (numTrain + 1) * sizeof(int));
	fName = outputPath + "\\" + datasetName + "_P1NN_warps_and_preLabels_" + s_seedId + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	importMatrix(warpAndLabels, fName, 27, numTrain + 1, 1);

	//ED distMtx
	double *trainTss_in;
	cudaMalloc(&trainTss_in, trainTssBytes);
	cudaMemcpy(trainTss_in, trainTss, trainTssBytes, cudaMemcpyHostToDevice);
	double *testTss_in;
	cudaMalloc(&testTss_in, testTssBytes);
	cudaMemcpy(testTss_in, testTss, testTssBytes, cudaMemcpyHostToDevice);
	double *dists_out;
	long long pDistsBytes = numTest * numTrain * sizeof(double);
	cudaMalloc(&dists_out, pDistsBytes);
	int blockSize = numTest < maxThreadsPerBlock ? numTest : maxThreadsPerBlock;
	int gridSize = ceil((double)numTest / blockSize) < maxBlocksPerGrid ? ceil((double)numTest / blockSize) : maxBlocksPerGrid;
	getPDists_DTW << <gridSize, blockSize >> > (trainTss_in, testTss_in, dists_out, numTrain, numTest, tsLen, 0);	//ED is DTW with zero warp.
	cudaError_t cudaerr = cudaThreadSynchronize();
	if (cudaerr != cudaSuccess){
		exit(1);
	}
	double *distMtx_ED = (double *)malloc(pDistsBytes);
	cudaMemcpy(distMtx_ED, dists_out, pDistsBytes, cudaMemcpyDeviceToHost);

	std::vector<double> curWarps;
	double warp, prevWarp, p, r, f;
	int intWarp;
	int *transPreLabels, *inducPreLabels = (int *)malloc(numTest * sizeof(int));
	double *inducFscores = (double *)malloc(27 * sizeof(double));
	double *distMtx = (double *)malloc(pDistsBytes);
	double* distMtx_DTW = (double *)malloc(pDistsBytes);
	char s_warp[MAX_CHAR];
	prevWarp = -1;
	for (int i = 0; i < 27; i++) {
		intWarp = warpAndLabels[i * (numTrain + 1)];
		if (intWarp < 0) {
			curWarps = warps;
		}
		else{
			curWarps.clear();
			curWarps.push_back((double)intWarp / 100);
		}
		transPreLabels = warpAndLabels + i * (numTrain + 1) + 1;

		if(transPreLabels[0] == -1){
			inducFscores[i] = -2;
			continue;
		}
		
		if (i < 9){	//ED
			memcpy(distMtx, distMtx_ED, pDistsBytes);
			classifyNN(inducPreLabels, transPreLabels, distMtx, numTrain, numTest);
			prf(p, r, f, testLabels, inducPreLabels, numTest);
			inducFscores[i] = f;

		}
		else {
			inducFscores[i] = -INF;
			for (int j = 0; j < curWarps.size(); j++){
				warp = curWarps[j];
				if (warp != prevWarp){
					getPDists_DTW << <gridSize, blockSize >> > (trainTss_in, testTss_in, dists_out, numTrain, numTest, tsLen, warp);
					cudaError_t cudaerr = cudaThreadSynchronize();
					if (cudaerr != cudaSuccess){
						exit(1);
					}
					cudaMemcpy(distMtx_DTW, dists_out, pDistsBytes, cudaMemcpyDeviceToHost);
					prevWarp = warp;
				}

				if (i < 18) {	//DTW
					memcpy(distMtx, distMtx_DTW, pDistsBytes);

				}
				else {	//DTW-D
					getDists_DTW_D(distMtx, distMtx_ED, distMtx_DTW, numTrain, numTest);
				}
				classifyNN(inducPreLabels, transPreLabels, distMtx, numTrain, numTest);
				prf(p, r, f, testLabels, inducPreLabels, numTest);

				if(f > inducFscores[i])
					inducFscores[i] = f;
			}
		}
	}

	std::string methods[]{"ED_oracle", "ED_WK", "ED_RW", "ED_BHRK", "ED_GBTRM_1", "ED_GBTRM_2", "ED_GBTRM_3", "ED_GBTRM_4", "ED_GBTRM_5", 
		"DTW_oracle", "DTW_WK", "DTW_RW", "DTW_BHRK", "DTW_GBTRM_1", "DTW_GBTRM_2", "DTW_GBTRM_3", "DTW_GBTRM_4", "DTW_GBTRM_5", 
		"DTWD_oracle", "DTWD_WK", "DTWD_RW", "DTWD_BHRK", "DTWD_GBTRM_1", "DTWD_GBTRM_2", "DTWD_GBTRM_3", "DTWD_GBTRM_4", "DTWD_GBTRM_5"};

	fName = outputPath + "\\" + datasetName + "_P1NN_test_fscores_" + s_seedId + "_" + s_minNumIters + "_" + s_maxNumIters + ".txt";
	std::ofstream fout;
	fout.open(fName);
	for (int i = 0; i < 27; i++) {
		std::cout << methods[i] << ": " << inducFscores[i] << std::endl;
		fout << methods[i] << ": " << inducFscores[i] << std::endl;
	}
	fout.close();
	std::cout << std::endl;

	cudaFree(trainTss_in);
	cudaFree(testTss_in);
	cudaFree(dists_out);
	free(trainTss);
	free(trainLabels);
	free(testTss);
	free(testLabels);
	free(warpAndLabels);
	free(inducPreLabels);
	free(inducFscores);
	free(distMtx_ED);
	free(distMtx_DTW);
	free(distMtx);

	return 0;
}
