#ifndef __PHONE_RECOG
#define __PHONE_RECOG

#include <vector>
#include "data.h"
#include "mlf_io.h"

class CPhoneRecog {
	int numPhones;
	int numBiphones;
	int numTriphones;
	int numStates;

	int *startIdxs;
	int *endIdxs;

	int *biphoneIdxIn;
	int *biphoneIdxOut;

	int *stateIdxs;

	float *tpSelfloop;
	float *tpTransition;

	int *triphoneOutListOffsets;
	int *triphoneOutListValues;
	
	float *lm;

	char **phoneTable;

	float PIP;

public:

	CPhoneRecog();
	~CPhoneRecog();

	int Init(char *imgFile, float PIP, float LMW);
	int RecogOne(CData *lp, std::vector<sResult> &result);
	void Delete();

};

#endif //__PHONE_RECOG
