#pragma once

#define INF 1e6

//Precision is set to -1 in divide-by-zero cases. Switch this value to other values (0 or 1) as needed.
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

//Precision is set to -1 in divide-by-zero cases. Switch this value to other values (0 or 1) as needed.
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
//P, R, F are set to -1 in divide-by-zero cases. Switch this value to other values (0 or 1) as needed.
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

//P, R, F are set to -1 in divide-by-zero cases. Switch this value to other values (0 or 1) as needed.
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