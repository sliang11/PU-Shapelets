#pragma once

#include <stdlib.h>

struct stackElm{
	int low;
	int high;
	stackElm *next;
};

stackElm *initStack(int low, int high){
	stackElm *top = (stackElm *)malloc(sizeof(stackElm));
	top->low = low;
	top->high = high;
	top->next = NULL;
	return top;
}

stackElm *push(stackElm *top, int low, int high){
	stackElm *newElm = (stackElm *)malloc(sizeof(stackElm));
	newElm->low = low;
	newElm->high = high;
	newElm->next = top;
	return newElm;
}

stackElm* pop(stackElm *top, int &low, int &high){
	low = top->low;
	high = top->high;
	stackElm *ret = top->next;
	free(top);
	return ret;
}