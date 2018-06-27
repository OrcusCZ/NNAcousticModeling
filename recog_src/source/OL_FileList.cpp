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

#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(WIN32) || defined(WIN64)
#	include "io.h"
#else
#	include "general/my_inttypes.h"
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#	include <dirent.h>
#endif
#include "OL_FileList.h"

#ifndef REMOVE_ENTER
#define REMOVE_ENTER

char *remove_enter(char *input){
	int n;

	n = strlen(input);
	if (input[n-1] == '\n'){
		input[n-1] = '\0';
	}
	return(input);
} //remove enter 

#endif

//************************************************************************************************
// class COneFileName
//************************************************************************************************

COneFileName::COneFileName(const char *FileName){
	this->p_next = NULL;
	this->p_prev = NULL;
	strcpy(this->Name, FileName);
} //konstruktor

//************************************************************************************************
// class CFileList
//************************************************************************************************

CFileList::CFileList(const char *Name){
	this->Number = 0;
	strcpy(this->Name, Name);
	this->p_akt = NULL;
	this->p_first = NULL;
	this->p_last = NULL;
} //konstruktor

//================================================================================================

CFileList::~CFileList(){
	Clear();
} //destruktor

//================================================================================================

void CFileList::Clear() {
	while (p_first != NULL){
		p_akt = p_first;
		p_first = p_first->p_next;
		delete p_akt;
	} //while

	p_akt = NULL;
	p_first = NULL;
	p_last = NULL;
	Number = 0;
}
//================================================================================================

int CFileList::ReadList(const char *filename){
	FILE *fr;
	char buffer[FILE_BUFFER_SIZE];

	if ((fr = fopen(filename, "r")) == NULL){
		printf("Cannot open file '%s'\n", filename);
		return(1);
	} //fopen

	while (fgets(buffer, FILE_BUFFER_SIZE, fr) != NULL){
		remove_enter(buffer);
		this->AddItem(buffer);
	} //while

	if (fclose(fr) == EOF){
		printf("Cannot close file '%s'\n", filename);
	} //fclose

	return(0);
} //ReadList

//================================================================================================

int CFileList::WriteList(const char *FileName){
	FILE *fw;

	if ((fw = fopen(FileName, "w")) == NULL){
		printf("Cannot open file '%s'\n", FileName);
		return(1);
	} //fopen

	p_akt = p_first;
	while (p_akt != NULL){
		fprintf(fw, "%s\n", p_akt->Name);
		p_akt = p_akt->p_next;		
	} //while

	if (fclose(fw) == EOF){
		printf("Cannot close file '%s'\n", FileName);
	} //fclose

	return(0);
} //WriteList
//================================================================================================

int CFileList::AppendList(char *FileName){
	FILE *fw;

	if ((fw = fopen(FileName, "a+")) == NULL){
		printf("Cannot open file '%s'\n", FileName);
		return(1);
	} //fopen

	p_akt = p_first;
	while (p_akt != NULL){
		fprintf(fw, "%s\n", p_akt->Name);
		p_akt = p_akt->p_next;		
	} //while

	if (fclose(fw) == EOF){
		printf("Cannot close file '%s'\n", FileName);
	} //fclose

	return(0);
} //AppendList

//================================================================================================

void CFileList::ShowList(FILE *stream){
	fprintf(stream, "File list '%s' contains %d items:\n", this->Name, this->Number);
	p_akt = p_first;
	while (p_akt != NULL){
		fprintf(stream, "%s\n", p_akt->Name);
		p_akt = p_akt->p_next;		
	} //while
} //ShowList

//================================================================================================

int CFileList::IsInList(const char *str){
	p_akt = p_first;
	while (p_akt != NULL){
		if (strcmp(p_akt->Name, str) == 0){
			return(1); //nalezeno
		}
		p_akt = p_akt->p_next;
	} //while
	return(0); //nenalezeno
} //is_in_list


//================================================================================================

unsigned int CFileList::ListLength(void){
	return(Number);
} //length

//================================================================================================

char *CFileList::GetItemName(unsigned int Index){
	unsigned int i;

	if (Index > Number){
		printf("Desired Index is greater than number of items!");
		return(NULL);
	} //if

	p_akt = p_first;
	for (i=1;i<Index;i++){
		p_akt = p_akt->p_next;
	} //for i

	return(p_akt->Name);
} //GetItemName

//LMa 9.5.2009
bool CFileList::GetItemName(char **Name){	
	if(p_akt == NULL) {
		*Name = NULL;
		return false;
	}

	*Name = p_akt->Name;
	p_akt = p_akt->p_next;
	return true;
}
//================================================================================================

int CFileList::ReadSubDir(const char *DirName, const char *Mask){
	//nacteni souboru z daneho adresare
	char kde[DIR_NAME_LEN+1];
	char item[DIR_NAME_LEN+1];

	//vygenerování cesty
	if (strlen(DirName) == 0){
		sprintf(kde, "%s", Mask);		
	}
	else{
		sprintf(kde, "%s\\%s", DirName, Mask);
	}

#if defined(WIN32) || defined(WIN64)
	struct _finddata_t c_file;
	int hFile;

	//hledani souboru
	if( (hFile = _findfirst(kde, &c_file )) == -1L ){
		//adresar je prazdny
		return(0);
	}
	else{
		if (strlen(DirName) == 0){
			sprintf(item, "%s", c_file.name);
		}
		else{
			sprintf(item, "%s\\%s", DirName, c_file.name);
		}
		if ((c_file.attrib & _A_SUBDIR) == 0){
			//jde o soubor
			//pridani do seznamu
			this->AddItem(item);
		} 

		while( _findnext( hFile, &c_file ) == 0 ){
			if (strlen(DirName) == 0){
				sprintf(item, "%s", c_file.name);
			}
			else{
				sprintf(item, "%s\\%s", DirName, c_file.name);
			}
			if ((c_file.attrib & _A_SUBDIR) == 0){
				//jde o soubor
				//pridani do seznamu
				this->AddItem(item);
			} 
		} //while
		_findclose( hFile );
	} //if findfirst

#else //LINUX
	//otevreni adresare
	const char *pmask = strchr(Mask, '.');

	//printf("xxx\n"); //LLLLLLLLLL
	DIR *actDBdir = opendir(DirName);
	if(actDBdir != NULL) {
		struct dirent *dirEntry;
		//nacitani postupne jednotlivych souboru
		while((dirEntry=readdir(actDBdir))) {
			//			if(!isDir(dirEntry->d_name)) {
			//pokud neodpovida masce pokracujeme:
			//printf("Soubor %s\n", dirEntry->d_name); //LLLLLLLLLLLLL
			if (strstr(dirEntry->d_name, pmask) == NULL) continue;
			if ( (strcmp(dirEntry->d_name, ".") != 0) && (strcmp(dirEntry->d_name, "..") != 0) && (strchr(dirEntry->d_name, '.') != NULL) ) {
				//pokud neni adresar ulozime jmeno souboru
				if (strlen(DirName) == 0){
					sprintf(item, "%s", dirEntry->d_name);
				}else{
					sprintf(item, "%s/%s", DirName, dirEntry->d_name);
				}			
				//pridani do seznamu
				//printf("Pridavame soubor %s\n", item); //LLLLLLLLLLL
				this->AddItem(item);
			}
		}
		closedir(actDBdir);
	}
#endif //WIN vs. LINUX

	return(0);
} //ReadSubDir

//================================================================================================

int CFileList::ReadDir(const char *DirName, const char *Mask, bool FilesFromSubDirs){
	char kde[DIR_NAME_LEN+1];
	char item[DIR_NAME_LEN+1];

	//nacteni souboru z aktualniho adresare
	if (ReadSubDir(DirName, Mask) != 0) return(1);

	//projiti podaresaru - je-li zadano
	if (FilesFromSubDirs == false){
		return(0);
	}

	//vygenerování cesty
	if (strlen(DirName) == 0){
		sprintf(kde, "%s", "*.*");		
	}
	else{
		sprintf(kde, "%s\\%s", DirName, "*.*");
	}

#if defined(WIN32) || defined(WIN64)
	struct _finddata_t c_file;
	int hFile;

	//hledani souboru
	if( (hFile = _findfirst(kde, &c_file )) == -1L ){
		//adresar je prazdny
		return(0);
	}
	else{
		if (strlen(DirName) == 0){
			sprintf(item, "%s", c_file.name);
		}
		else{
			sprintf(item, "%s\\%s", DirName, c_file.name);
		}
		if ((c_file.attrib & _A_SUBDIR) != 0){
			//jde o adresar
			//rekurzivni volani
			if ((strcmp(c_file.name, ".") != 0) && (strcmp(c_file.name, "..") != 0))
				this->ReadDir(item, Mask, FilesFromSubDirs);
		} //if adresar

		while( _findnext( hFile, &c_file ) == 0 ){
			if (strlen(DirName) == 0){
				sprintf(item, "%s", c_file.name);
			}
			else{
				sprintf(item, "%s\\%s", DirName, c_file.name);
			}
			if ((c_file.attrib & _A_SUBDIR) != 0){
				//jde o adresar
				//rekurzivni volani
				if ((strcmp(c_file.name, ".") != 0) && (strcmp(c_file.name, "..") != 0))
					this->ReadDir(item, Mask, FilesFromSubDirs);
			} //if adresar
		} //while
		_findclose( hFile );
	} //if findfirst

#else //LINUX
	//otevreni adresare
	//printf("oooooooo\n"); //LLLLLLLLL
	DIR *actDBdir = opendir(DirName);
	if(actDBdir != NULL) {
		struct dirent *dirEntry;
		//nacitani postupne jednotlivych souboru
		while((dirEntry=readdir(actDBdir))) {
			//if(isDir(dirEntry->d_name)) {
			//printf("Testujeme %s\n", dirEntry->d_name); //LLLLLLLLLLLLLLLLLLLL
			if ((strcmp(dirEntry->d_name, ".") != 0) && (strcmp(dirEntry->d_name, "..") != 0) && (strchr(dirEntry->d_name, '.') == NULL)) {
				//pokud je adresar zavolame ho rekurentne
				if (strlen(DirName) == 0){
					sprintf(item, "%s", dirEntry->d_name);
				}else{
					sprintf(item, "%s/%s", DirName, dirEntry->d_name);
				}			
				//rekurzivni volani
				//printf("Rekurzivni volani pro %s\n", item);  //LLLLLLLLLLLLLLLLL
				this->ReadDir(item, Mask, FilesFromSubDirs);
			}
		}
		closedir(actDBdir);
	}

#endif //WIN vs. LINUX

	return(0);
} //ReadDir

//================================================================================================

int CFileList::ReadSubDirectories(const char *DirName, const char *Mask){
	//nacteni souboru z daneho adresare
	char kde[DIR_NAME_LEN+1];
	char item[DIR_NAME_LEN+1];

	//vygenerování cesty
	if (strlen(DirName) == 0){
		sprintf(kde, "%s", Mask);		
	}
	else{
		sprintf(kde, "%s\\%s", DirName, Mask);
	}

#if defined(WIN32) || defined(WIN64)
	struct _finddata_t c_file;
	int hFile;

	//hledani souboru
	if( (hFile = _findfirst(kde, &c_file )) == -1L ){
		//adresar je prazdny
		return(0);
	}
	else{
		if (strlen(DirName) == 0){
			sprintf(item, "%s", c_file.name);
		}
		else{
			sprintf(item, "%s\\%s", DirName, c_file.name);
		}
		if ((c_file.attrib & _A_SUBDIR) != 0){
			//jde o adresar
			//pridani do seznamu
			if ((item[strlen(item)-1]) != '.'){
				this->AddItem(item);
			}
		} 

		while( _findnext( hFile, &c_file ) == 0 ){
			if (strlen(DirName) == 0){
				sprintf(item, "%s", c_file.name);
			}
			else{
				sprintf(item, "%s\\%s", DirName, c_file.name);
			}
			if ((c_file.attrib & _A_SUBDIR) != 0){
				//jde o adresar
				//pridani do seznamu
				if ((item[strlen(item)-1]) != '.'){
					this->AddItem(item);
				}
			} 
		} //while
		_findclose( hFile );
	} //if findfirst
#else //LINUX
	const char *pmask = strchr(Mask, '.');
	//otevreni adresare
	DIR *actDBdir = opendir(DirName);
	if(actDBdir != NULL) {
		struct dirent *dirEntry;
		//nacitani postupne jednotlivych souboru
		while((dirEntry=readdir(actDBdir))) {
			//if(isDir(dirEntry->d_name)) {
			if (strstr(dirEntry->d_name, pmask) == NULL) continue;
			if ((strcmp(dirEntry->d_name, ".") != 0) && (strcmp(dirEntry->d_name, "..") != 0) && (strchr(dirEntry->d_name, '.') == NULL)) {
				//pokud je adresar ulozime ho
				if (strlen(DirName) == 0){
					sprintf(item, "%s", dirEntry->d_name);
				}else{
					sprintf(item, "%s/%s", DirName, dirEntry->d_name);
				}			
				//pridani do seznamu
				this->AddItem(item);
			}
		}
		closedir(actDBdir);
	}	
#endif //WIN vs. LINUX

	return(0);
} //ReadSubDirectories

//================================================================================================

int CFileList::ReadSubdirNames(const char *DirName, const char *Mask, bool FilesFromSubDirs){
	char kde[DIR_NAME_LEN+1];
	char item[DIR_NAME_LEN+1];

	//nacteni souboru z aktualniho adresare
	if (ReadSubDirectories(DirName, Mask) != 0) return(1);

	//projiti podaresaru - je-li zadano
	if (FilesFromSubDirs == false){
		return(0);
	}

	//vygenerování cesty
	if (strlen(DirName) == 0){
		sprintf(kde, "%s", "*.*");		
	}
	else{
		sprintf(kde, "%s\\%s", DirName, "*.*");
	}

#if defined(WIN32) || defined(WIN64)
	struct _finddata_t c_file;
	int hFile;

	//hledani souboru
	if( (hFile = _findfirst(kde, &c_file )) == -1L ){
		//adresar je prazdny
		return(0);
	}
	else{
		if (strlen(DirName) == 0){
			sprintf(item, "%s", c_file.name);
		}
		else{
			sprintf(item, "%s\\%s", DirName, c_file.name);
		}
		if ((c_file.attrib & _A_SUBDIR) != 0){
			//jde o adresar
			//rekurzivni volani
			if ((strcmp(c_file.name, ".") != 0) && (strcmp(c_file.name, "..") != 0))
				this->ReadSubdirNames(item, Mask, FilesFromSubDirs);
		} //if adresar

		while( _findnext( hFile, &c_file ) == 0 ){
			if (strlen(DirName) == 0){
				sprintf(item, "%s", c_file.name);
			}
			else{
				sprintf(item, "%s\\%s", DirName, c_file.name);

				if ((c_file.attrib & _A_SUBDIR) != 0){
					//jde o adresar
					//rekurzivni volani
					if ((strcmp(c_file.name, ".") != 0) && (strcmp(c_file.name, "..") != 0))
						this->ReadSubdirNames(item, Mask, FilesFromSubDirs);
				} //if adresar
			} //while
			_findclose( hFile );
		} //if findfirst
	}


#else //LINUX
	//otevreni adresare
	DIR *actDBdir = opendir(DirName);
	if(actDBdir != NULL) {
		struct dirent *dirEntry;
		//nacitani postupne jednotlivych souboru
		while((dirEntry=readdir(actDBdir))) {
			//if(isDir(dirEntry->d_name)) {
			if ((strcmp(dirEntry->d_name, ".") != 0) && (strcmp(dirEntry->d_name, "..") != 0) && (strchr(dirEntry->d_name, '.') == NULL)) {
				//pokud je adresar zavolame rekurzi
				if (strlen(DirName) == 0){
					sprintf(item, "%s", dirEntry->d_name);
				}else{
					sprintf(item, "%s/%s", DirName, dirEntry->d_name);
				}			
				//rekurzivni volani
				this->ReadSubdirNames(item, Mask, FilesFromSubDirs);
			}
		}
		closedir(actDBdir);
	}	
#endif //WIN vs. LINUX

	return(0);
} //ReadSubdirNames

//================================================================================================

void CFileList::AddItem(const char *ItemName){
	if ((p_akt = new COneFileName(ItemName)) == NULL){
		printf("Cannot allocate memory");
		exit(1);
	} //if		
	// zarazeni do seznamu		
	if (p_first == NULL){
		p_first = p_akt;
		p_last = p_akt;
		p_akt->p_next = NULL;
		p_akt->p_prev = NULL;
	}
	else{
		p_last->p_next = p_akt;
		p_akt->p_prev = p_last;
		p_akt->p_next = NULL; //LMa 9.5.2009
		p_last = p_akt;		
	}
	Number++;
} //AddItem
//************************************************************************************************
// presune prvky z jineho FileListu do teto tridy
void CFileList::AppendFileList(CFileList *list) {
	int iNum;
	COneFileName *first, *last;

	list->ReleaseItems(&first, &last, &iNum);

	if(p_first == NULL) {
		p_first = first;
		p_last = last;
		p_akt = p_first;		
	}
	else {
		p_last->p_next = first;
		p_last = last;
	}
	Number += iNum;
}

//************************************************************************************************
// vrati pointer na prvni a posledni prvek a uvolni tyto pointry z tridy
void CFileList::ReleaseItems(COneFileName **first, COneFileName **last, int *iNum) {
	*first = p_first;
	*last = p_last;
	*iNum = Number;

	p_first = NULL;
	p_last = NULL;
	p_akt = NULL;
	Number = 0;
}

//************************************************************************************************
void CFileList::push(COneFileName *item) {
	//zarazeni do seznamu		
	if (p_first == NULL) {
		p_first = item;
		p_last = item;
		item->p_next = NULL;
		item->p_prev = NULL;
	}
	else {
		p_last->p_next = item;
		item->p_prev = p_last;
		item->p_next = NULL;
		p_last = item;		
	}
	Number++;
}

//************************************************************************************************
COneFileName *CFileList::pop() {
	if (p_first == NULL)
		return NULL;

	COneFileName *p_ret = p_first;
	
	p_first = p_first->p_next;
	if(p_ret == p_last)
		p_last = NULL;

	Number--;

	return p_ret;
}
//************************************************************************************************
