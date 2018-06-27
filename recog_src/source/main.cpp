#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "my_stopwatch.h"
#include "phone_recog.h"
#include "OL_FileList.h"

void catchError() {
	printf("Error catching possible at line %d\n", __LINE__);
}

#define CHECK_ERROR(e) {int iRet = (e); if(iRet!=0) {printf("Error #%d (file:%s, line:%d)\n", iRet, __FILE__, __LINE__); catchError(); return iRet;}}
#define CHECK_ERROR_PRINT_ONLY(e) {int iRet = (e); if(iRet!=0) {printf("Error #%d (file:%s, line:%d)\n", iRet, __FILE__, __LINE__); catchError();}}

char * baseNameCopy(char*in);

int main(int argc, char *argv[]) {

	if (argc != 6) {
		printf("\nPhoneRecog\nBuild date %s\n\n", __DATE__);
		printf("Use: nPhoneRecog.exe test.scp image.bin output.mlf PIP LMW\n");
		printf("test.scp - file per line, binary float32 format of AM log-likelihoods\n");
		return 0;
	}

	float PIP = atof(argv[4]);
	float LMW = atof(argv[5]);

	CPhoneRecog phoneRecog;
	CHECK_ERROR(phoneRecog.Init(argv[2], PIP, LMW));

	CFileList list;
	CHECK_ERROR(list.ReadList(argv[1]));

	char *filename = NULL;
	std::vector<sResult> *results = new std::vector<sResult>[list.ListLength()];
	
	//copy a list of filenames into vector
	std::vector<char *> filenames;
	list.Rewind();
	filenames.reserve(list.ListLength());
	char *foo;
	while (list.GetItemName(&foo)) {
		filenames.insert(filenames.end(), foo);
	}

	MYSTOPWATCH_FIRST_START;

	//main loop
#pragma omp parallel for
	for (int f = 0; f < filenames.size(); f++) {
		CData data;
		//read a file
		CHECK_ERROR_PRINT_ONLY(data.LoadBin(filenames[f]));

		std::vector<sResult> result;
		//recog One
		CHECK_ERROR_PRINT_ONLY(phoneRecog.RecogOne(&data, results[f]));

		data.Delete();
		printf(".");

	}
	printf("\nWriting results\n");

	//store results
	CMlfWriter mlfWriter;
	CHECK_ERROR(mlfWriter.WriteStart(argv[3]));
	for (int f = 0; f < filenames.size(); f++) {
		//store results
		char *uttName = baseNameCopy(filenames[f]);
		CHECK_ERROR(mlfWriter.WriteOneAndClear(uttName, results[f]));

		//print progress
		//printf("%s\n", uttName);
		delete[] uttName;
	}
	CHECK_ERROR(mlfWriter.WriteEnd());
	delete[] results;

	MYSTOPWATCH_STOP_AND_PRINT;

	return 0;
}

char * baseNameCopy(char*in) {
	int b = 0, e = strlen(in);
	for (int i = 0; i < strlen(in); i++) {
		if (in[i] == '/' || in[i] == '\\') b = i + 1;
		if (in[i] == '.') e = i - 1;
	}
	char *out = new char[2 + e - b];
	strncpy(out, in + b, 1 + e - b);
	out[1 + e - b] = '\0';
	return out;
}
