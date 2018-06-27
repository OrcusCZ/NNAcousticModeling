#ifndef __MLF_IO
#define __MLF_IO

#include <vector>
#include "stdio.h"
#ifndef RETURN_OK
#define RETURN_OK 0
#define RETURN_FAIL -1
#endif

char *allocAndCopyPhone(char *);

struct sResult {
	int begin; //time in frames
	int end; //time in frames
	char *phone;
};

class CMlfWriter {
	FILE *fid;
public:

	CMlfWriter();
	~CMlfWriter();
	int WriteStart(char *filename);
	int WriteOne(char * uttName, std::vector<sResult> &result);
	void DeleteResult(std::vector<sResult> &result);
	int WriteOneAndClear(char * uttName, std::vector<sResult> &result);
	int WriteEnd();
};

#endif //__MLF_IO