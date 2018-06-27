// 
// Copyright (c) 2007 Ales Padrta
// Copyright (c) 2007 Jan Vanek
// 
// University of West Bohemia, Department of Cybernetics, 
// Plzen, Czech Repulic
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products
//    derived from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

#ifndef _OL_FILE_LIST_
#define _OL_FILE_LIST_

#include <stdio.h>

//buffer pro nacitani souboru
#ifndef FILE_BUFFER_SIZE
	#define FILE_BUFFER_SIZE 500
#endif

//max. delka souboru
#ifndef FILE_NAME_LEN
	#define FILE_NAME_LEN	500
#endif

//max. delka souboru
#ifndef DIR_NAME_LEN
	#define DIR_NAME_LEN	1000
#endif

//************************************************************************************************

//struktura pro ulozeni seznamu nazvu souboru do pameti
class COneFileName{
private:
	char Name[FILE_NAME_LEN+1];	//jmeno souboru
	COneFileName *p_next;
	COneFileName *p_prev;

public:
	//vytvoreni prvku z nazvem FileName
	COneFileName(const char *FileName);

	inline char const* getName() {return Name;} 

	friend class CFileList;
}; //class COneFileName

//************************************************************************************************

class CFileList{
private:
	char Name[FILE_NAME_LEN+1];	//jmeno seznamu
	unsigned int Number;

	//ukazatele na seznam objektu
	COneFileName	*p_first;
	COneFileName	*p_akt;
	COneFileName	*p_last;

	//nacteni souboru z adresare (vrati 0 pokud je vse v poradku)
	int ReadSubDir(const char *DirName, const char *Mask);

	//adresare souboru z adresare (vrati 0 pokud je vse v poradku)
	int ReadSubDirectories(const char *DirName, const char *Mask);
public:

	//konstruktor - inicializace promennych
	CFileList(const char *Name = "UNNAMED");

	//destruktor - uvoleni polozky z pameti
	~CFileList();

	//nacteni souboru obsahujiciho seznam souboru (vrati 0 pokud je vse v poradku)
	int ReadList(const char *FileName);

	//nacteni souboru z adresare (vrati 0 pokud je vse v poradku)
	int ReadDir(const char *DirName, const char *Mask, bool FilesFromSubDirs = false);

	//nacte nazvy podadresaru ze zadaneho adresare (vrati 0 pokud je vse v poradku)
	int ReadSubdirNames(const char *DirName, const char *Mask, bool FilesFromSubDirs = false);

	//ulozeni seznamu do souboru (ve forme nactitelne pomoci ReadList)
	int WriteList(const char *FileName);

	//ulozeni seznamu do souboru na jeho konec, pokud uz soubor exietuje (ve forme nactitelne pomoci ReadList)
	int AppendList(char *FileName);

	//vypis seznamu do zvoleneho proudu
	void ShowList(FILE *stream);

	//pridani prvku do seznamu
	void AddItem(const char *ItemName);

	//vrati 1 pokud retezec (str) je v seznamu, jinak vrati 0
	int IsInList(const char *str);

	//vrati delku seznamu
	unsigned int ListLength(void);

	//vrati jmeno Index-teho prvku v seznamu (nebo NULL, v pripade neplatneho indexu)
	char *GetItemName(unsigned int Index);

	//LMA 9.5.2009
	//nastavi aktualni pointer na prvni v seznamu
	inline void Rewind() {p_akt=p_first;}
	
	//ulozi jmeno prvku v seznamu s aktualnim pointrem, vrati false na konci seznamu
	bool GetItemName(char **name);

	// presune prvky z jineho FileListu do teto tridy
	void AppendFileList(CFileList *list);
	
	// vrati pointer na prvni a posledni prvek a uvolni tyto pointry z tridy
	void ReleaseItems(COneFileName **first, COneFileName **last, int *iNum);

	// vymaze list
	void Clear();

	void push(COneFileName *item);
	COneFileName *pop();


}; //class CFileList

#endif
