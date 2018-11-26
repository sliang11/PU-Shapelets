#pragma once

#include <iostream>
#include <fstream>

#define MAX_CHAR_PER_LINE 200000
#define BITS_PER_BYTE 8

template <class T>
void importTimeSeries(T *tss, int *labels, std::string datasetPath, std::string datasetName,
	std::string pfx, int numTs, int tsLen){

	std::string fName = datasetPath + "\\" + datasetName + "_" + pfx;
	//std::cout << fName << std::endl;
	//system("pause");
	std::ifstream fin;
	fin.open(fName);
	if (!fin)
		exit(1);

	char buf[MAX_CHAR_PER_LINE];
	char *tmp;
	for (int i = 0; i < numTs; i++){
		fin.getline(buf, MAX_CHAR_PER_LINE, '\n');
		tmp = strtok(buf, " ,\r\n");
		labels[i] = atoi(tmp);
		for (int j = 0; j < tsLen; j++){
			tmp = strtok(NULL, " ,\r\n");
			tss[i * tsLen + j] = atof(tmp);
		}
	}
	fin.close();
}

void relabel(int *labels, int numTs, int pLabel){
	for (int i = 0; i < numTs; i++){
		if (labels[i] == pLabel)
			labels[i] = 1;
		else
			labels[i] = 0;
	}
}

template <class T>
void importMatrix(T *output, std::string fName, int numRow, int numCol, bool isInteger){
	std::ifstream fin;
	fin.open(fName);
	if (!fin)
		exit(1);

	char buf[MAX_CHAR_PER_LINE];
	char *tmp;
	for (int i = 0; i < numRow; i++){
		fin.getline(buf, MAX_CHAR_PER_LINE, '\n');
		tmp = strtok(buf, " ,\r\n\t");

		if (isInteger)
			output[i * numCol] = atoi(tmp);
		else
			output[i * numCol] = atof(tmp);
		for (int j = 1; j < numCol; j++){
			tmp = strtok(NULL, " ,\r\n\t");

			if (isInteger)
				output[i * numCol + j] = atoi(tmp);
			else {
				output[i * numCol + j] = atof(tmp);
			}
		}
	}
	fin.close();
}

template <class T>
void importMatrixTransposed(T *output, std::string fName, int numRowInFile, int numColInFile, bool isInteger){

	std::ifstream fin;
	fin.open(fName);
	if (!fin)
		exit(1);

	char buf[MAX_CHAR_PER_LINE];
	char *tmp;
	for (int i = 0; i < numRowInFile; i++){
		fin.getline(buf, MAX_CHAR_PER_LINE, '\n');
		tmp = strtok(buf, " ,\r\n\t");
		if (isInteger)
			output[i] = atoi(tmp);
		else
			output[i] = atof(tmp);
		for (int j = 1; j < numColInFile; j++){
			tmp = strtok(NULL, " ,\r\n\t");
			if (isInteger)
				output[i + j * numRowInFile] = atoi(tmp);
			else
				output[i + j * numRowInFile] = atof(tmp);
		}
	}
	fin.close();

}

