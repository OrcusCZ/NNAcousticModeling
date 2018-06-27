#include <stdio.h>
#include <string.h>
#include <math.h>

#include "phone_recog.h"

#define READ_INT_CHECK(var) {if(fread(&var, sizeof(int), 1, fid) != 1) { printf("Error reading image file (file:%s, line:%d)\n", __FILE__, __LINE__); return RETURN_FAIL;}}
#define ALLOC_READ_CHECK(var, type, count) {var = new type[count]; if(fread(var, sizeof(type), count, fid) != count) { printf("Error reading image file (file:%s, line:%d)\n", __FILE__, __LINE__); return RETURN_FAIL;}}

CPhoneRecog::CPhoneRecog() {
	numPhones=0;
	numBiphones = 0;
	numTriphones = 0;
	numStates = 0;

	startIdxs=NULL;
	endIdxs = NULL;

	biphoneIdxIn = NULL;
	biphoneIdxOut = NULL;

	stateIdxs = NULL;

	tpSelfloop = NULL;
	tpTransition = NULL;
	
	triphoneOutListOffsets = NULL;
	triphoneOutListValues = NULL;

	lm = NULL;
	PIP = 0;

	phoneTable = NULL;
}

CPhoneRecog::~CPhoneRecog() {
	Delete();
}

void CPhoneRecog::Delete() {
	delete[] startIdxs;
	delete[] endIdxs;

	delete[] biphoneIdxIn;
	delete[] biphoneIdxOut;

	delete[] stateIdxs;

	delete[] tpSelfloop;
	delete[] tpTransition;

	delete[] triphoneOutListOffsets;
	delete[] triphoneOutListValues;

	delete[] lm;

	if (phoneTable != NULL) delete phoneTable[0];
	delete[] phoneTable;
	numPhones = 0;
	numBiphones = 0;
	numTriphones = 0;
	numStates = 0;

	startIdxs = NULL;
	endIdxs = NULL;

	biphoneIdxIn = NULL;
	biphoneIdxOut = NULL;

	stateIdxs = NULL;

	tpSelfloop = NULL;
	tpTransition = NULL;

	triphoneOutListOffsets = NULL;
	triphoneOutListValues = NULL;

	lm = NULL;
	PIP = 0;

	phoneTable = NULL;
}

int CPhoneRecog::Init(char *imgFile, float PIP, float LMW) {
	Delete();

	FILE *fid = fopen(imgFile, "rb");
	if (fid == NULL) {
		printf("Cannot open file %s\n", imgFile);
		return RETURN_FAIL;
	}

	READ_INT_CHECK(numPhones);
	READ_INT_CHECK(numBiphones);
	READ_INT_CHECK(numTriphones);
	READ_INT_CHECK(numStates);

	ALLOC_READ_CHECK(startIdxs, int, numPhones);
	ALLOC_READ_CHECK(endIdxs, int, numPhones);

	ALLOC_READ_CHECK(biphoneIdxIn, int, numTriphones);
	ALLOC_READ_CHECK(biphoneIdxOut, int, numTriphones);

	ALLOC_READ_CHECK(stateIdxs, int, 3*numTriphones);
	
	ALLOC_READ_CHECK(tpSelfloop, float, 3 * numTriphones);
	ALLOC_READ_CHECK(tpTransition, float, 3 * numTriphones);

	ALLOC_READ_CHECK(triphoneOutListOffsets, int, 1 + numBiphones);
	ALLOC_READ_CHECK(triphoneOutListValues, int, triphoneOutListOffsets[numBiphones]);

	ALLOC_READ_CHECK(lm, float, numBiphones);
	for (int i = 0; i < numBiphones; i++) lm[i] *= LMW;

	int numChars = 0;
	READ_INT_CHECK(numChars);
	phoneTable = new char*[numTriphones];
	phoneTable[0] = new char[numChars + 1];
	if (fread(phoneTable[0], sizeof(char), numChars, fid) != numChars) {
		printf("Error reading image file (file:%s, line:%d)\n", __FILE__, __LINE__);
		return RETURN_FAIL;
	}
	phoneTable[0][numChars] = '\0';
	int n = 0;
	for (int i = 1; i < numChars; i++) { //split string-phones and set pointers
		if (phoneTable[0][i] == ' ') {
			phoneTable[0][i] = '\0';
			phoneTable[++n] = &(phoneTable[0][i+1]);
		}
	}

	fclose(fid);

	this->PIP = PIP;
	return RETURN_OK;
}

int CPhoneRecog::RecogOne(CData *lp, std::vector<sResult> &result) {
	if (lp->numColumns != numStates) {
		printf("Num states in log-probs file and model image differ!\n");
		return RETURN_FAIL;
	}

	float Inf = 1e+20;
	
	float *p = new float[numBiphones];
	for (int i = 0; i < numBiphones; i++) p[i] = -Inf;
	for (int i = 0; i < numPhones; i++) p[startIdxs[i]] = lm[startIdxs[i]] - log((float)numPhones);

	int N = lp->numRows;

	int *bestI = new int[N * numBiphones];
	int *bestT = new int[N * numBiphones];
	//float *bestP = new float[N * numBiphones];
	memset(bestI, 0, N * numBiphones * sizeof(int));
	memset(bestT, 0, N * numBiphones * sizeof(int));
	//memset(bestP, 0, N * numBiphones * sizeof(float));

	int *tt = new int[3 * numTriphones];
	memset(tt, 0, 3 * numTriphones * sizeof(int));

	float *pp = new float[3 * numTriphones];
	for (int i = 0; i < 3 * numTriphones; i++) pp[i] = -Inf;

	//main loop
	float *lp1 = lp->data;
	for (int k = 0; k < N; k++) {
		
		//triphone loop
		for (int n = 0; n < numTriphones; n++) {
			//prepare input:
			float pIn = p[biphoneIdxIn[n]] + PIP;
			int tIn = (k - 1);

			//internal process:
			float pStay[3] = { pp[n] + tpSelfloop[n * 3], pp[n + numTriphones] + tpSelfloop[n * 3 + 1], pp[n + 2 * numTriphones] + tpSelfloop[n * 3 + 2] };
			float pGo[3] = { pIn, pp[n] + tpTransition[n * 3], pp[n + numTriphones] + tpTransition[n * 3 + 1] };
			int tStay[3] = { tt[n], tt[n + numTriphones], tt[n + 2*numTriphones] };
			int tGo[3] = { tIn, tt[n], tt[n + numTriphones] };
			for (int s = 2; s >= 0; s--) {
				if (pGo[s] > pStay[s]) {
					pp[n + s * numTriphones] = pGo[s] + lp1[stateIdxs[n * 3 + s]];
					tt[n + s * numTriphones] = tGo[s];
				}
				else {
					pp[n + s * numTriphones] = pStay[s] + lp1[stateIdxs[n * 3 + s]];
					tt[n + s * numTriphones] = tStay[s];
				}
			}
		} //for n < numTriphones

		//reset init values
		if (k == 0) {
			for (int i = 0; i < numBiphones; i++) p[i] = -Inf;
		}

		//export
		for (int i = 0; i < numBiphones; i++) {
			int mxi = 0;
			float mx = -Inf;
			for (int s = triphoneOutListOffsets[i]; s < triphoneOutListOffsets[i + 1]; s++) {
				int ids = triphoneOutListValues[s];
				float pOut = pp[ids + 2 * numTriphones] + tpTransition[ids * 3 + 2];
				if (mx < pOut) {
					mx = pOut;
					mxi = ids;
				}
			}
			p[i] = mx + lm[i];
			bestI[k * numBiphones + i] = mxi;
			bestT[k * numBiphones + i] = tt[mxi + 2 * numTriphones];
		}
	
		//memcpy(bestP + k*numBiphones, p, sizeof(float) * numBiphones);
		lp1 += lp->numColumns;
	} //for k < N

	//debug:
	//CData dd;
	//dd.ImportByRef(bestP, N, numBiphones);
	//dd.SaveBin("__bestP.bin");
	
	//back-track
	int mxi = 0;
	float mx = -Inf;
	for (int i = 0; i < numPhones; i++) {
		int ids = endIdxs[i];
		if (mx < p[ids]) {
			mx = p[ids];
			mxi = ids;
		}
	}
	sResult new_result;
	result.clear();
	int t = N - 1;
	while (t >= 0) {
		int id = bestI[t * numBiphones + mxi];
		new_result.end = t + 1;
		t = bestT[t * numBiphones + mxi];
		new_result.begin = t + 1;
		new_result.phone = allocAndCopyPhone(phoneTable[id]);
		mxi = biphoneIdxIn[id];
		result.insert(result.begin(), new_result);
	}

	delete[] p;
	delete[] bestI;
	delete[] bestT;
	//delete[] bestP;
	delete[] tt;
	delete[] pp;

	return RETURN_OK;
}