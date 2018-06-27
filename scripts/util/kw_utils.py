# system
import numpy as np

def saveBin(filename, x):
    fid = open(filename, 'wb')
    dims = np.asarray(x.shape, dtype=np.uint32)
    if len(dims) == 1:
        dims = np.resize(dims, 2)
        dims[1] = 1
    dims.tofile(fid, sep="")
    x.tofile(fid, sep="")
    fid.close()

def loadBin(filename, dtype):
    fid = open(filename, 'rb')
    dims = np.fromfile(fid, dtype=np.uint32, sep="", count=2)
    if dims[1] > 1:
        x = np.fromfile(fid, dtype=dtype, sep="").reshape(dims)
    else:
        x = np.fromfile(fid, dtype=dtype, sep="")
    fid.close()
    return x

def splicing(data, iShifts):
    [N, M] = data.shape
    numShifts = len(iShifts)
    x = np.zeros([N, M*numShifts], dtype=np.float32)
    for idx in range(0, N):
        for wi, w in enumerate(iShifts):
            w = w + idx
            if w < 0:
                w = 0
            if w >= N:
                w = N-1
            x[idx][wi*M:(wi+1)*M] = data[w]
    return x

def logsum(lp, axis=0):
    Inf = 1e+20
    mx = np.max(lp, axis=axis).reshape([lp.shape[0], 1])
    lps = mx + np.log(np.sum(np.exp(lp-mx), axis=axis)).reshape([lp.shape[0], 1])
    lps[np.isnan(lps)] = -Inf;
    return lps

def loadMlf(filename):
    mlf = {}
    origId = 0
    consumeEnd = False
    for line in open(filename).readlines():
        if not line or line[0] == '#':
            continue
        if line[0] == '"':
            id = line[1:].split('.')[0]
            if id[0] == '*':
                id = id[2:]
            words = []
            begins = []
            ends = []
            consumeEnd = False
            continue
        if line[0] == '.':
            #store the actual utterance
            mlf[id] = [words, begins, ends, origId]
            origId += 1
            consumeEnd = True
            continue
        if consumeEnd:
            continue
        #add a word
        if len(line.split()) == 3:
            sp = line.split()
            words.append(sp[2])
            begins.append(int(sp[0]))
            ends.append(int(sp[1]))
        else:
            words.append(line.strip())

    return mlf