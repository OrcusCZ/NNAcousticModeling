#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mlf_io.h"

#define CHECK_ERROR(e) {int iRet = (e); if(iRet!=0) printf("Error #%d (file:%s, line:%d)\n", iRet, __FILE__, __LINE__); return iRet;}

CMlfWriter::CMlfWriter() {
	fid = NULL;
}

CMlfWriter::~CMlfWriter() {
	if (fid != NULL) WriteEnd();
}

int CMlfWriter::WriteStart(char *filename) {
	if (fid != NULL) WriteEnd();
	fid = fopen(filename, "w");
	if (fid == NULL) {
		printf("Cannot open file %s\n", filename);
		return RETURN_FAIL;
	}
	fprintf(fid, "#!MLF!#\n");
	return RETURN_OK;
}

int CMlfWriter::WriteOne(char * uttName, std::vector<sResult> &result) {
	if (fid == NULL) return RETURN_FAIL;
	fprintf(fid, "\"*/%s.rec\"\n", uttName);
	for (int n = 0; n < result.size(); n++) {
		fprintf(fid, "%d00000 %d00000 %s\n", result[n].begin, result[n].end, result[n].phone);
	}
	fprintf(fid, ".\n");
	return RETURN_OK;
}

void CMlfWriter::DeleteResult(std::vector<sResult> &result) {
	int n = 0;
	for (int n = 0; n < result.size(); n++) {
		delete result[n].phone;
	}
	result.clear();
}

int CMlfWriter::WriteOneAndClear(char * uttName, std::vector<sResult> &result) {
	CHECK_ERROR(WriteOne(uttName, result));
	DeleteResult(result);
	return RETURN_OK;
}

int CMlfWriter::WriteEnd() {
	fclose(fid);
	fid = NULL;
	return RETURN_OK;
}

char *allocAndCopyPhone(char *in) {
	char *out = new char[strlen(in) + 1];
	strcpy(out, in);
	return out;
}