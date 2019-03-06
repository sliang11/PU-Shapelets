#pragma once

#include <iostream>
#include <fstream>

#define BUF_SIZE 200000
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

	char buf[BUF_SIZE];
	char *temp;
	for (int i = 0; i < numTs; i++){
		fin.getline(buf, BUF_SIZE, '\n');
		temp = strtok(buf, " ,\r\n");
		labels[i] = atoi(temp);
		for (int j = 0; j < tsLen; j++){
			temp = strtok(NULL, " ,\r\n");
			tss[i * tsLen + j] = atof(temp);
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

	char buf[BUF_SIZE];
	char *temp;
	for (int i = 0; i < numRow; i++){
		fin.getline(buf, BUF_SIZE, '\n');
		temp = strtok(buf, " ,\r\n\t");

		if (isInteger)
			output[i * numCol] = atoi(temp);
		else
			output[i * numCol] = atof(temp);
		for (int j = 1; j < numCol; j++){
			temp = strtok(NULL, " ,\r\n\t");

			if (isInteger)
				output[i * numCol + j] = atoi(temp);
			else {
				output[i * numCol + j] = atof(temp);
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

	char buf[BUF_SIZE];
	char *temp;
	for (int i = 0; i < numRowInFile; i++){
		fin.getline(buf, BUF_SIZE, '\n');
		temp = strtok(buf, " ,\r\n\t");
		if (isInteger)
			output[i] = atoi(temp);
		else
			output[i] = atof(temp);
		for (int j = 1; j < numColInFile; j++){
			temp = strtok(NULL, " ,\r\n\t");
			if (isInteger)
				output[i + j * numRowInFile] = atoi(temp);
			else
				output[i + j * numRowInFile] = atof(temp);
		}
	}
	fin.close();

}

