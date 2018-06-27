# system
import numpy as np
# util
from kw_utils import loadMlf

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def levenshtein2(s1, s2, flip=False):
    if len(s1) < len(s2):
        return levenshtein2(s2, s1, True)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    path_idx = np.ones([len(s1), len(s2)], dtype=np.int32) * (len(s1) + len(s2))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
            path_idx[i, j] = np.argmin(np.asarray([insertions, deletions, substitutions]))
        previous_row = current_row

    #back track
    i = len(s1) - 1
    j = len(s2) - 1
    H = 0
    D = 0
    I = 0
    S = 0
    while True:
        if i < 0 or j < 0:
            break
        if path_idx[i, j] == 0:
            I += 1
            i -= 1
            continue
        if path_idx[i, j] == 1:
            D += 1
            j -= 1
            continue
        if path_idx[i, j] == 2:
            if s1[i] == s2[j]:
                H += 1
            else:
                S += 1
            i -= 1
            j -= 1
            continue

    if flip:
        foo = I
        I = D
        D = foo
        N = len(s1)
    else:
        N = len(s2)

    return N, H, D, I, S

#TIMIT compatible Phone Error Rate: 'sil' is optional
def levenshtein_nist(s1, s2, flip=False):
    if len(s1) < len(s2):
        return levenshtein_nist(s2, s1, True)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    path_idx = np.ones([len(s1), len(s2)], dtype=np.int32) * (len(s1) + len(s2))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            optiSil = False
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            if flip and s1[i] == 'sil': #nist compatible - sil is optional
                insertions -= 1
                optiSil = True
            if not flip and s2[j] == 'sil': #nist compatible - sil is optional
                deletions -= 1
                optiSil = True
            current_row.append(min(insertions, deletions, substitutions))
            path_idx[i, j] = np.argmin(np.asarray([substitutions, insertions, deletions]))
            if path_idx[i, j] > 0 and insertions == deletions: #if same, prefere that with optional sil
                if flip:
                    path_idx[i, j] = 1
                else:
                    path_idx[i, j] = 2

        previous_row = current_row

    #back track
    i = len(s1) - 1
    j = len(s2) - 1
    H = 0
    D = 0
    I = 0
    S = 0
    NSkips = 0
    while True:
        if i < 0 or j < 0:
            break
        if path_idx[i, j] == 0:
            if s1[i] == s2[j]:
                H += 1
            else:
                S += 1
            i -= 1
            j -= 1
            continue
        if path_idx[i, j] == 1:
            I += 1
            if flip and s1[i] == 'sil':
                I -= 1 #nist compatible - sil is optional
                NSkips += 1
            i -= 1
            continue
        if path_idx[i, j] == 2:
            D += 1
            if not flip and s2[j] == 'sil':
                D -= 1 #nist compatible - sil is optional
                NSkips += 1
            j -= 1
            continue

    if flip:
        foo = I
        I = D
        D = foo
        N = len(s1)
    else:
        N = len(s2)
    H += NSkips

    return N, H, D, I, S

#s1 = ['A', 'C', 'D', 'E', 'F']
#s2 = ['A', 'B', 'C', 'E']
#print levenshtein2(s1, s2)

def computeWER(testMlfFile, refMlfFile, NISTvariant=False):
    if isinstance(testMlfFile, str):
        test = loadMlf(testMlfFile)
    else:
        test = testMlfFile
    if isinstance(refMlfFile, str):
        ref = loadMlf(refMlfFile)
    else:
        ref = refMlfFile

    TH = 0
    TD = 0
    TS = 0
    TI = 0
    TN = 0

    uttList = test.keys()
    uttOrigOrder = [test[utt][3] for utt in uttList]
    uttOrigOrderList = [x for (y, x) in sorted(zip(uttOrigOrder, uttList))]


    for k in uttOrigOrderList:
        if k not in ref:
            raise('Utterance' + k + ' not in the refence mlf')
        if NISTvariant:
            N, H, D, I, S = levenshtein_nist(test[k][0], ref[k][0]) #[0] = word list
        else:
            N, H, D, I, S = levenshtein2(test[k][0], ref[k][0]) #[0] = word list

        TH += H
        TD += D
        TS += S
        TI += I
        TN += N
    TN += (TN == 0)
    #corr = (100.0 * TH)/TN
    #acc = (100.0 * (TH - TI))/TN
    err = (100.0 * (TD + TI + TS))/TN

    return err


