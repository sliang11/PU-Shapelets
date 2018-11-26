#pragma once

//#include <math.h>
//#include <vector>

#define INF 1e6

double entropy(int *numByLabel, int numElm, int numLabels) {

	if (!numElm)
		return 0;

	double p, ent = 0;
	for (int i = 0; i < numLabels; i++) {
		if (!numByLabel[i])
			continue;

		p = (double)numByLabel[i] / numElm;
		ent -= p * log2(p);
	}
	return ent;
}

double infoGain(double ent, double entIn, double entOut, int numElm, int cIn, int cOut) {
	double pIn = (double)cIn / numElm;
	double pOut = (double)cOut / numElm;
	double gain = ent - pIn * entIn - pOut * entOut;
	return gain;
}

//Supports multiple classes. Must relabel first!
template<class T>
void infoGain(double &maxGain, T &splitPt, T &gapAtSplit, T *vec, int *labels, int *order, int *numByLabel, int *numByLabelIn, int *numByLabelOut,
	const int numElm, const int numLabels, const int minCIn = 1, int maxCIn = INF, const int stride = 1) {

	if (maxCIn > numElm - 1)
		maxCIn = numElm - 1;

	double ent = entropy(numByLabel, numElm, numLabels);
	int cIn, cOut, optCIn, nextLabel;
	double entIn, entOut, gain;

	memset(numByLabelIn, 0, numLabels * sizeof(int));
	memcpy(numByLabelOut, numByLabel, numLabels * sizeof(int));
	maxGain = -INF;
	for (cIn = minCIn; cIn <= maxCIn; cIn++) {
		cOut = numElm - cIn;
		nextLabel = labels[order[cIn - 1]];
		numByLabelIn[nextLabel] += 1;
		numByLabelOut[nextLabel] -= 1;
		entIn = entropy(numByLabelIn, cIn, numLabels);
		entOut = entropy(numByLabelOut, cOut, numLabels);
		gain = infoGain(ent, entIn, entOut, numElm, cIn, cOut);
		if (maxGain < gain) {
			maxGain = gain;
			optCIn = cIn;
		}
	}
	splitPt = (vec[order[optCIn - 1]] + vec[order[optCIn]]) / 2;
	gapAtSplit = vec[order[optCIn]] - vec[order[optCIn - 1]];
}

template<class T>
void gapOrderline(double &maxGap, T &splitPt, T &caliSplitPt, T *vec, int *order, T *vec2, T *s, T *s2, const int numElm, const int minCIn = 1, int maxCIn = INF, const int stride = 1) {
	if (maxCIn > numElm - 1)
		maxCIn = numElm - 1;

	cumsum(s, vec, order, numElm);
	dotProduct(vec2, vec, vec, numElm);
	cumsum(s2, vec2, order, numElm);

	int cIn, cOut, optCIn, caliCIn;
	double sIn, s2In, sOut, s2Out, meanIn, meanOut, stdIn, stdOut, gap;
	maxGap = -INF;
	for (cIn = minCIn; cIn <= maxCIn; cIn++) {
		sIn = (double)s[cIn - 1];
		s2In = (double)s2[cIn - 1];
		meanIn = sIn / cIn;
		stdIn = s2In / cIn - (sIn * sIn) / (cIn * cIn);
		stdIn = stdIn > 0 ? sqrt(stdIn) : 0;

		sOut = (double)(s[numElm - 1] - s[cIn - 1]);
		s2Out = (double)(s2[numElm - 1] - s2[cIn - 1]);
		cOut = numElm - cIn;
		meanOut = sOut / cOut;
		stdOut = s2Out / cOut - (sOut * sOut) / (cOut * cOut);
		stdOut = stdOut > 0 ? sqrt(stdOut) : 0;

		gap = meanOut - meanIn - stdIn - stdOut;
		if (maxGap < gap) {
			maxGap = gap;
			optCIn = cIn;
			caliSplitPt = meanIn + stdIn;
		}
	}
	splitPt = (vec[order[optCIn - 1]] + vec[order[optCIn]]) / 2;

}

double precision(int *realLabels, int *preLabels, int numObj) {
	int cm[4];
	memset(cm, 0, 4 * sizeof(int));
	int realLabel, preLabel;
	for (int i = 0; i < numObj; i++) {
		realLabel = realLabels[i];
		preLabel = preLabels[i];
		if (realLabel && preLabel)
			cm[0]++;
		else if (realLabel && !preLabel)
			cm[1]++;
		else if (!realLabel && preLabel)
			cm[2]++;
		else
			cm[3]++;
	}

	double p;
	if (!(cm[0] || cm[2]))
		p = -1;
	else
		p = (double)cm[0] / (cm[0] + cm[2]);
	return p;
}

double precisionWithSeeds(int *realLabels, int *preLabels, int *seeds, int numObj, int numPLabeled) {
	int cm[4];
	memset(cm, 0, 4 * sizeof(int));
	int realLabel, preLabel;
	bool isSeed;
	for (int i = 0; i < numObj; i++) {

		isSeed = false;
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seeds[j]) {
				isSeed = true;
				break;
			}
		}
		if (isSeed)
			continue;

		realLabel = realLabels[i];
		preLabel = preLabels[i];
		if (realLabel && preLabel)
			cm[0]++;
		else if (realLabel && !preLabel)
			cm[1]++;
		else if (!realLabel && preLabel)
			cm[2]++;
		else
			cm[3]++;
	}

	double p;
	if (!(cm[0] || cm[2]))
		p = -1;
	else
		p = (double)cm[0] / (cm[0] + cm[2]);
	return p;
}

//prf excluding the initial labeled examples
void prfWithSeed(double &precision, double &recall, double &fscore,
	int *realLabels, int *preLabels, int *seeds, int numObj, int numPLabeled) {

	int cm[4];
	memset(cm, 0, 4 * sizeof(int));
	int realLabel, preLabel;
	bool isSeed;
	for (int i = 0; i < numObj; i++) {

		isSeed = false;
		for (int j = 0; j < numPLabeled; j++) {
			if (i == seeds[j]) {
				isSeed = true;
				break;
			}
		}
		if (isSeed)
			continue;

		realLabel = realLabels[i];
		preLabel = preLabels[i];
		if (realLabel && preLabel)
			cm[0]++;
		else if (realLabel && !preLabel)
			cm[1]++;
		else if (!realLabel && preLabel)
			cm[2]++;
		else
			cm[3]++;
	}

	if (!(cm[0] || cm[2]))
		precision = -1;
	else
		precision = (double)cm[0] / (cm[0] + cm[2]);

	if (!(cm[0] || cm[1]))
		recall = -1;
	else
		recall = (double)cm[0] / (cm[0] + cm[1]);

	if (precision == -1 || recall == -1 || !(precision || recall))
		fscore = -1;
	else
		fscore = 2 * precision * recall / (precision + recall);

}

//For two classes only. Labels are either 0 or 1. 1 first, 0 second.
void prf(double &precision, double &recall, double &fscore, int *realLabels, int *preLabels, int numObj){
	int cm[4];
	memset(cm, 0, 4 * sizeof(int));
	int realLabel, preLabel;
	for (int i = 0; i < numObj; i++){
		realLabel = realLabels[i];
		preLabel = preLabels[i];
		if (realLabel && preLabel)
			cm[0]++;
		else if (realLabel && !preLabel)
			cm[1]++;
		else if (!realLabel && preLabel)
			cm[2]++;
		else
			cm[3]++;
	}

	if (!(cm[0] || cm[2]))
		precision = -1;
	else
		precision = (double)cm[0] / (cm[0] + cm[2]);

	if (!(cm[0] || cm[1]))
		recall = -1;
	else
		recall = (double)cm[0] / (cm[0] + cm[1]);

	if (precision == -1 || recall == -1 || !(precision || recall))
		fscore = -1;
	else
		fscore = 2 * precision * recall / (precision + recall);
}