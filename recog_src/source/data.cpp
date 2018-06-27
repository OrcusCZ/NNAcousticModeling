#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

CData::CData() {
	numRows=0; //num of Frames
	numColumns=0; //dim
	data=NULL;
}

CData::~CData() {
	Delete();
}

void CData::Delete() {
	if (data != NULL) delete[] data;
	numRows = 0; //num of Frames
	numColumns = 0; //dim
	data = NULL;
}

int CData::LoadBin(char *filename) {
	Delete();

	FILE *in = NULL;
	if ((in = fopen(filename, "rb")) == NULL) {
		printf("Cannot open file %s\n", filename);
		return RETURN_FAIL;
	}

	if (fread((void*)&numRows, sizeof(int), 1, in) != 1)
		return RETURN_FAIL;

	if (fread((void*)&numColumns, sizeof(int), 1, in) != 1)
		return RETURN_FAIL;

	data = new float[numRows*numColumns];

	if (fread(data, sizeof(float)*numColumns, numRows, in) != numRows)
		return RETURN_FAIL;
	fclose(in);
	return RETURN_OK;
}

int CData::SaveBin(char *filename) {

	FILE *fid = NULL;
	if ((fid = fopen(filename, "wb")) == NULL) {
		printf("Cannot open file %s\n", filename);
		return RETURN_FAIL;
	}

	if (fwrite((void*)&numRows, sizeof(int), 1, fid) != 1)
		return RETURN_FAIL;

	if (fwrite((void*)&numColumns, sizeof(int), 1, fid) != 1)
		return RETURN_FAIL;

	if (fwrite(data, sizeof(float)*numColumns, numRows, fid) != numRows)
		return RETURN_FAIL;
	fclose(fid);
	return RETURN_OK;
}

int CData::ImportByCopy(float *data, int numRows, int numColumns) {
	Delete();
	this->data = new float[numRows*numColumns];
	this->numRows = numRows;
	this->numColumns = numColumns;
	this->data = new float[numRows*numColumns];
	if (memcpy(this->data, data, numRows * numColumns * sizeof(float)) == NULL)
		return RETURN_FAIL;
	return RETURN_OK;
}

int CData::ImportByRef(float *data, int numRows, int numColumns) {
	this->numRows = numRows;
	this->numColumns = numColumns;
	this->data = data;
	return RETURN_OK;
}

int CData::Create(CData *ref) {
	return Create(ref->numRows, ref->numColumns);
}

int CData::Create(int numRows, int numColumns) {
	Delete();
	this->numRows = numRows;
	this->numColumns = numColumns;
	this->data = new float[numRows*numColumns];
	return RETURN_OK;
}
