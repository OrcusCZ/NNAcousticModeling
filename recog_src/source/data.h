#ifndef __DATA
#define __DATA

#ifndef RETURN_OK
#define RETURN_OK 0
#define RETURN_FAIL -1
#endif

class CData { //row-major order
public:
	int numRows; //num of Frames
	int numColumns; //dim
	float *data;

	CData();
	~CData();
	int Create(CData *ref);
	int Create(int numRows, int numCols);
	int LoadBin(char *filename);
	int SaveBin(char *filename);
	int ImportByCopy(float *data, int numRows, int numColumns);
	int ImportByRef(float *data, int numRows, int numColumns);
	void Delete();
};

#endif //__DATA